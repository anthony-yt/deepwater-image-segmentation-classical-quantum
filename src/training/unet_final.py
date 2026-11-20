"""Task 4.8: corridas finales del U-Net bajo el mismo protocolo que SegFormer y UWSegFormer.

La tabla comparativa la arma `src.training.final_retraining.aggregate_shared_results`, que exige
los tres modelos sobre el mismo manifiesto, con las mismas columnas y el mismo sha256. Este modulo
es el lado del U-Net de ese contrato: reutiliza las piezas de Dayana en vez de reimplementarlas,
para que las filas se puedan concatenar sin traducciones.

De `src.training.final_retraining` se reutiliza, sin copiar nada:

- `validate_manifest`  — el manifiesto y los 62 pares imagen/mascara, con su sha256.
- `iteration_budget`   — presupuesto en pasadas, no en epocas (decision 3 del protocolo).
- `materialize`        — copia sin perdidas del fold; el test nunca se expone como validacion.
- `score_predictions`  — las cinco metricas mas miou/mdice, con el arnes de la task 5.7.
- `summarize`          — una puntuacion por semilla y video, luego la SD entre semillas.
- `validate_output`    — la salida cubre exactamente el test y los metadatos coinciden.

Preflight en CPU (valida manifiesto, pares, plan y hashes; no entrena):

    python -m src.training.unet_final --settings configs/final_retraining_unet.example.json

Ejecucion con GPU:

    python -m src.training.unet_final --settings mi_config.json --run

La tabla de los tres modelos, una vez que existan las corridas de los tres en `results_dir`:

    python -m src.training.final_retraining --settings <config de Dayana> --table

No se elige el mejor checkpoint: se entrena el presupuesto acordado y se mide con los pesos
finales (decision 3). Un `select_on` aqui volveria a mirar el test, que es justo lo que la
task 4.2 mostro que infla el resultado.

torch se importa dentro de las funciones que entrenan, para que el preflight corra en una
maquina sin GPU ni torch instalado.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.training.final_retraining import (digest_json, iteration_budget, materialize,
                                           score_predictions, sha256, summarize,
                                           validate_manifest, validate_output)

MODEL_KEY = "unet"
MODEL_NAME = "U-Net"
LOSSES = ("ce_dice", "ce", "dice", "focal", "tversky")
CODE_FILES = ("src/training/unet_final.py", "src/models/unet.py", "src/data/pipe_dataset.py",
              "src/training/losses.py", "src/training/predict.py",
              "src/evaluation/segmentation_metrics.py")


class ResizePair:
    """Lleva el par al tamano de trabajo: imagen bilineal, mascara vecino mas cercano.

    Sin padding, asi que no aparece `IGNORE_INDEX`. La augmentacion de escala se estudia en la
    task 5.11 y no entra en las corridas finales: el protocolo fija el presupuesto, no el
    preprocesamiento, y mezclar las dos cosas haria incomparables las columnas.
    """

    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, image, mask):
        return {
            "image": np.asarray(Image.fromarray(image).resize((self.size, self.size),
                                                              Image.BILINEAR)),
            "mask": np.asarray(Image.fromarray(mask).resize((self.size, self.size),
                                                            Image.NEAREST)),
        }


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preflight(settings: dict):
    """Valida la configuracion y el manifiesto, y devuelve (settings, manifiesto, pares, plan)."""
    s = dict(settings)
    if s.get("mode") not in {"pilot", "final"}:
        raise ValueError("mode debe ser pilot o final")
    if s.get("passes") is None:
        raise ValueError("Falta acordar passes sin consultar el test final")
    if s.get("models", [MODEL_KEY]) != [MODEL_KEY]:
        raise ValueError(f"Este modulo corre solo {MODEL_KEY}; los SegFormer van por "
                         f"src.training.final_retraining")
    seeds = s.get("seeds", [])
    if not seeds or any(type(x) is not int or x < 0 for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Semillas enteras no negativas y unicas requeridas")
    if s["mode"] == "final" and (not s.get("protocol_confirmed") or not 3 <= len(seeds) <= 5):
        raise ValueError("Final requiere protocolo confirmado y 3-5 semillas")

    df, records = validate_manifest(s["manifest"], s.get("manifest_sha256"),
                                    s["frames_dir"], s["masks_dir"])
    counts = df.drop_duplicates("frame_id").video.value_counts().to_dict()
    if s.get("expected_counts") is not None and counts != s["expected_counts"]:
        raise ValueError(f"Conteos de mascaras distintos al acuerdo: {counts}")
    videos = sorted(df.video.unique())
    if s.get("expected_videos") != videos:
        raise ValueError(f"Confirmar expected_videos = {videos}")
    folds = s.get("folds") or videos
    if len(set(folds)) != len(folds) or not set(folds) <= set(videos):
        raise ValueError("Folds desconocidos o duplicados")
    if s["mode"] == "final" and set(folds) != set(videos):
        raise ValueError("Final requiere todos los folds")
    s["folds"] = folds

    if type(s.get("batch_size")) is not int or s["batch_size"] < 1:
        raise ValueError("batch_size entero positivo requerido")
    if not math.isfinite(s.get("learning_rate", float("nan"))) or s["learning_rate"] <= 0:
        raise ValueError("learning_rate positivo requerido")
    if s.get("loss") not in LOSSES:
        raise ValueError(f"loss debe ser una de {LOSSES}")
    if not isinstance(s.get("attention"), bool):
        raise ValueError("Declarar attention explicitamente (True = Attention U-Net)")
    if s.get("scheduler") not in {"cosine", "constant"}:
        raise ValueError("scheduler debe ser cosine o constant")
    if s.get("augment", False):
        raise ValueError("Las corridas finales van sin augmentacion; eso es la task 5.11")
    if type(s.get("img_size")) is not int or s["img_size"] < 64 or s["img_size"] % 32:
        raise ValueError("img_size entero, multiplo de 32 y >= 64 (el U-Net baja 5 veces)")

    plan = []
    for fold in folds:
        n = int(((df.fold == fold) & (df.split == "train")).sum())
        if n < s["batch_size"]:
            raise ValueError(f"batch_size mayor que n_train en {fold}: drop_last lo dejaria sin pasos")
        iters = iteration_budget(s["passes"], n, s["batch_size"])
        for seed in seeds:
            plan.append(dict(model=MODEL_KEY, fold=fold, seed=seed, n_train=n, max_iters=iters,
                             effective_passes=iters * s["batch_size"] / n))
    s["dataset_sha256"] = digest_json([
        {k: v for k, v in record.items() if k not in {"image", "mask"}} for record in records])
    root = Path(__file__).resolve().parents[2]
    s["code_sha256"] = digest_json({p: sha256(root / p) for p in CODE_FILES})
    return s, df, records, plan


def train_unet(dataset_root, s: dict, seed: int, max_iters: int, device: str):
    """Entrena el presupuesto acordado y devuelve (modelo, bitacora). Sin seleccion de checkpoint."""
    import torch
    from torch.utils.data import DataLoader

    from src.data.pipe_dataset import PipeDataset, list_pairs
    from src.models.unet import build_unet
    from src.training.losses import make_criterion

    set_seed(seed)
    pairs = list_pairs(dataset_root, "train")
    if not pairs:
        raise ValueError(f"sin pares de entrenamiento en {dataset_root}")
    loader = DataLoader(PipeDataset(pairs, ResizePair(s["img_size"])), batch_size=s["batch_size"],
                        shuffle=True, drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    if len(loader) == 0:
        raise ValueError("drop_last dejo el loader vacio; baja batch_size")

    model = build_unet(attention=s["attention"], device=device)
    criterion = make_criterion(s["loss"])
    optimizer = torch.optim.Adam(model.parameters(), lr=s["learning_rate"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
                 if s["scheduler"] == "cosine" else None)

    model.train()
    losses, step = [], 0
    while step < max_iters:
        for image, mask in loader:
            if step >= max_iters:
                break
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(image), mask)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss))
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"  paso {step:4d}/{max_iters} | perdida {np.mean(losses[-50:]):.4f}")
    return model, {"pasos": step, "perdida_inicial": float(losses[0]),
                   "perdida_final_50": float(np.mean(losses[-50:]))}


def score_fold(model, dataset_root, img_size: int, device: str) -> pd.DataFrame:
    """Predice el test a la resolucion original del frame y puntua con el arnes compartido."""
    from src.data.pipe_dataset import list_pairs
    from src.training.predict import predict_masks

    pairs = list_pairs(dataset_root, "test")
    if not pairs:
        raise ValueError(f"sin pares de test en {dataset_root}")
    preds, gts = predict_masks(model, pairs, device, img_size)
    return score_predictions(preds, gts)


def execute(settings: dict, run: bool = False):
    s, df, records, plan = preflight(settings)
    print(pd.DataFrame(plan).to_string(index=False))
    print(f"{len(records)} pares; {len(plan)} corridas. mode={s['mode']} passes={s['passes']}")
    print(f"manifest_sha256 = {s['manifest_sha256']}")
    if not run:
        return plan

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("AVISO: sin GPU; el U-Net en CPU tarda horas por corrida.")
    result_root = Path(s["results_dir"]) / s["mode"]
    frames = []
    for entry in plan:
        identity = {**s, **entry, "model_name": MODEL_NAME}
        run_hash = digest_json({k: v for k, v in identity.items() if k != "model_name"})
        work = result_root / f"{MODEL_KEY}_{entry['fold']}_seed{entry['seed']}"
        done, output = work / "completed.json", work / "per_frame.csv"

        if done.exists():
            saved = json.loads(done.read_text())
            if saved["run_sha256"] != run_hash or not output.exists() \
                    or saved["output_sha256"] != sha256(output):
                raise ValueError(f"Resultado incompatible o modificado: {work}; usa otra results_dir")
            print(f"{work.name}: ya estaba, se reutiliza")
        else:
            if work.exists():
                raise ValueError(f"Corrida parcial en {work}; mueve esa carpeta antes de reintentar")
            work.mkdir(parents=True)
            dataset = work / "dataset"
            materialize(df[df.fold == entry["fold"]], records, dataset)
            identity.update(dataset_root=str(dataset.resolve()), work_dir=str(work.resolve()),
                            run_sha256=run_hash)
            (work / "run_settings.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")

            print(f"--- {work.name}: {entry['max_iters']} pasos sobre {entry['n_train']} imagenes")
            model, log = train_unet(dataset, s, entry["seed"], entry["max_iters"], device)
            frame = score_fold(model, dataset, s["img_size"], device)
            frame = frame.assign(modelo=MODEL_NAME, fold=entry["fold"], seed=entry["seed"],
                                 pasadas=s["passes"], effective_passes=entry["effective_passes"],
                                 manifest_sha256=s["manifest_sha256"],
                                 dataset_sha256=s["dataset_sha256"], run_sha256=run_hash,
                                 mode=s["mode"])
            frame.to_csv(output, index=False)
            (work / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            done.write_text(json.dumps(dict(run_sha256=run_hash, output_sha256=sha256(output)),
                                       indent=2), encoding="utf-8")

        frame = pd.read_csv(output)
        expected = df[(df.fold == entry["fold"]) & (df.split == "test")].frame_id
        validate_output(frame, {**identity, "run_sha256": run_hash}, expected)
        frames.append(frame)

        if s["mode"] == "final":
            # Mismo nombre y mismas columnas que exporta final_retraining para los SegFormer.
            export = Path(s["results_dir"]) / f"per_frame_{MODEL_KEY}_{entry['fold']}_seed{entry['seed']}.csv"
            if export.exists() and sha256(export) != sha256(output):
                raise ValueError(f"Ya existe otro resultado en {export}; usa otra results_dir")
            export.write_bytes(output.read_bytes())

    summary = summarize(pd.concat(frames, ignore_index=True))
    summary.to_csv(result_root / "tabla_unet_4_8.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--run", action="store_true",
                        help="entrena; sin esta opcion solo valida y muestra el plan")
    args = parser.parse_args()
    execute(json.loads(args.settings.read_text(encoding="utf-8")), run=args.run)


if __name__ == "__main__":
    main()
