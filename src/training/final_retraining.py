"""Task 4.8: strict LOVO inputs, isolated runs and shared final tables.

CPU preflight: python -m src.training.final_retraining --settings settings.json
GPU execution: add --run (single GPU, sequential runs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.evaluation.segmentation_metrics import evaluate

MODELS = {'segformer': 'SegFormer', 'uwsegformer': 'UWSegFormer'}
METRICS = ['iou', 'dice', 'miou', 'mdice', 'boundary_f1']
UWS_COMMIT = 'b61fa450bad58137b3ab8fecd1e7f7cedba4ce57'
HARNESS_ORIGIN = 'd818aa938507913749de989250daacff3b05c807'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_mask(path):
    # Resolve palette colors, including inverted palettes; never treat palette indices as labels.
    with Image.open(path) as image:
        arr = np.asarray(image.convert('RGB'))
    if not np.array_equal(arr[..., 0], arr[..., 1]) or not np.array_equal(arr[..., 0], arr[..., 2]):
        raise ValueError(f'Máscara de colores no binaria: {path}')
    arr = arr[..., 0]
    values = set(np.unique(arr).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f'Valores ambiguos {values} en {path}; reconciliar antes de entrenar')
    return (arr > 0).astype(np.uint8)


def validate_manifest(path, expected_hash, frames_dir, masks_dir):
    if not expected_hash or sha256(path) != expected_hash:
        raise ValueError('Falta el SHA-256 acordado o el manifiesto no coincide')
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'fold', 'frame_id', 'video', 'split'}
    if not required <= set(df.columns) or df.empty:
        raise ValueError('Manifiesto vacío o sin fold, frame_id, video, split')
    if df.duplicated(['fold', 'frame_id']).any() or set(df.split) != {'train', 'test'}:
        raise ValueError('Filas duplicadas o splits distintos de train/test')
    for col in ['fold', 'video']:
        if not df[col].str.fullmatch(r'Video[1-9]\d*').all():
            raise ValueError(f'{col} debe usar Video1, Video2, etc.')
    for row in df.itertuples():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row.frame_id):
            raise ValueError(f'frame_id inseguro: {row.frame_id}')
        match = re.search(r'Video(\d+)(?:_|$)', row.frame_id)
        if not match or f'Video{match[1]}' != row.video:
            raise ValueError(f'Video inconsistente: {row.frame_id}')
    if df.groupby('frame_id').video.nunique().max() != 1:
        raise ValueError('Un frame pertenece a varios videos')
    all_ids = set(df.frame_id)
    for fold, rows in df.groupby('fold'):
        if set(rows.frame_id) != all_ids or set(rows.split) != {'train', 'test'}:
            raise ValueError(f'Fold incompleto: {fold}')
        if not ((rows.video == fold) == (rows.split == 'test')).all():
            raise ValueError(f'Fuga entre videos en {fold}')
    if set(df.fold) != set(df.video):
        raise ValueError('Se requiere un fold de prueba para cada video incluido')

    records = []
    for row in df.drop_duplicates('frame_id').sort_values('frame_id').itertuples():
        candidates = [p for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
                      if (p := Path(frames_dir) / f'{row.frame_id}{ext}').is_file()]
        mask_path = Path(masks_dir) / f'{row.frame_id}_mask.png'
        if len(candidates) != 1 or not mask_path.is_file():
            raise ValueError(f'Par ausente/ambiguo: {row.frame_id}; no se omiten frames')
        mask = read_mask(mask_path)
        with Image.open(candidates[0]) as image:
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f'Dimensiones distintas: {row.frame_id}')
        records.append(dict(frame_id=row.frame_id, video=row.video,
                            image=str(candidates[0].resolve()), mask=str(mask_path.resolve()),
                            image_sha256=sha256(candidates[0]), mask_sha256=sha256(mask_path)))
    return df, records


def iteration_budget(passes, n_train, batch):
    if not math.isfinite(passes) or passes <= 0 or n_train < 1 or batch < 1:
        raise ValueError('Pasadas, n_train y batch deben ser positivos')
    return math.ceil(passes * n_train / batch)


def materialize(rows, records, destination):
    """Fresh lossless snapshot; test is never exposed as validation during training."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    index = {r['frame_id']: r for r in records}
    for row in rows.itertuples():
        record = index[row.frame_id]
        if any(sha256(record[k]) != record[f'{k}_sha256'] for k in ('image', 'mask')):
            raise ValueError(f'Los datos cambiaron después del preflight: {row.frame_id}')
        for kind in ('images', 'annotations'):
            (destination / kind / row.split).mkdir(parents=True, exist_ok=True)
        with Image.open(record['image']) as image:
            image.convert('RGB').save(destination / 'images' / row.split / f'{row.frame_id}.png')
        Image.fromarray(read_mask(record['mask'])).save(
            destination / 'annotations' / row.split / f'{row.frame_id}.png')


def restore_prediction(prediction, image_shape, original_shape):
    """Remove right/bottom padding and resize discrete classes using nearest neighbor."""
    pred = np.asarray(prediction)
    h, w = image_shape[:2]
    if pred.ndim != 2 or pred.shape[0] < h or pred.shape[1] < w:
        raise ValueError('Predicción con dimensiones incompatibles')
    if not set(np.unique(pred).tolist()) <= {0, 1}:
        raise ValueError('Predicción no binaria')
    resized = Image.fromarray(pred[:h, :w].astype(np.uint8)).resize(
        (original_shape[1], original_shape[0]), Image.Resampling.NEAREST)
    return np.asarray(resized)


def score_predictions(preds, gts):
    if not preds or set(preds) != set(gts):
        raise ValueError('Predicciones incompletas o frames adicionales')
    per_frame, _ = evaluate(preds, gts, tol=2, n_boot=1000)
    background, _ = evaluate({k: 1 - v for k, v in preds.items()},
                             {k: 1 - v for k, v in gts.items()}, tol=2, n_boot=1000)
    background = background.set_index('frame_id')
    for metric in ('iou', 'dice'):
        per_frame[f'{metric}_background'] = per_frame.frame_id.map(background[metric])
        per_frame[f'm{metric}'] = (per_frame[metric] + per_frame[f'{metric}_background']) / 2
    return per_frame


def summarize(frame):
    """One score per seed/video, then SD across seeds; global gives videos equal weight."""
    grouped = frame.groupby(['modelo', 'video', 'seed'])[METRICS].mean().reset_index()
    global_scores = grouped.groupby(['modelo', 'seed'])[METRICS].mean().reset_index()
    global_scores['video'] = 'GLOBAL_macro_videos'
    scores = pd.concat([grouped, global_scores], ignore_index=True)
    result = scores.groupby(['modelo', 'video'])[METRICS].agg(['mean', 'std'])
    result.columns = [f'{metric}_{stat}' for metric, stat in result.columns]
    result['n_seeds'] = scores.groupby(['modelo', 'video']).seed.nunique()
    return result.reset_index()


def preflight(settings):
    s = dict(settings)
    if s.get('mode') not in {'pilot', 'final'}:
        raise ValueError('mode debe ser pilot o final')
    if s.get('passes') is None:
        raise ValueError('Falta acordar passes sin consultar el test final')
    seeds = s.get('seeds', [])
    if not seeds or any(type(x) is not int or x < 0 for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Semillas enteras no negativas y únicas requeridas')
    models = s.get('models', [])
    if not models or len(set(models)) != len(models) or not set(models) <= set(MODELS):
        raise ValueError('models debe contener segformer y/o uwsegformer sin duplicados')
    if s['mode'] == 'final' and (not s.get('protocol_confirmed') or not 3 <= len(seeds) <= 5):
        raise ValueError('Final requiere protocolo confirmado y 3–5 semillas')
    df, records = validate_manifest(s['manifest'], s.get('manifest_sha256'),
                                    s['frames_dir'], s['masks_dir'])
    counts = df.drop_duplicates('frame_id').video.value_counts().to_dict()
    if s.get('expected_counts') is not None and counts != s['expected_counts']:
        raise ValueError(f'Conteos de máscaras distintos al acuerdo: {counts}')
    videos = sorted(df.video.unique())
    if s.get('expected_videos') != videos:
        raise ValueError(f'Confirmar expected_videos = {videos}; no se presume que hay 68 máscaras')
    folds = s.get('folds') or videos
    if len(set(folds)) != len(folds) or not set(folds) <= set(videos):
        raise ValueError('Folds desconocidos o duplicados')
    if s['mode'] == 'final' and set(folds) != set(videos):
        raise ValueError('Final requiere todos los folds')
    s['folds'] = folds
    if type(s.get('batch_size')) is not int or s['batch_size'] < 1:
        raise ValueError('batch_size entero positivo requerido (una GPU)')
    if not isinstance(s.get('freeze_backbone'), bool):
        raise ValueError('Declarar freeze_backbone explícitamente')
    if not math.isfinite(s['learning_rate']) or s['learning_rate'] <= 0:
        raise ValueError('learning_rate positivo requerido')
    plan = []
    for model in models:
        checkpoint = s['checkpoints'][model]
        for fold in folds:
            n = int(((df.fold == fold) & (df.split == 'train')).sum())
            if n < s['batch_size']:
                raise ValueError('El dataloader legacy descarta el último batch: batch_size > n_train')
            iters = iteration_budget(s['passes'], n, s['batch_size'])
            for seed in seeds:
                plan.append(dict(model=model, fold=fold, seed=seed, n_train=n,
                                 max_iters=iters, effective_passes=iters * s['batch_size'] / n,
                                 checkpoint_sha256=sha256(checkpoint)))
    s['dataset_sha256'] = digest_json([
        {k: v for k, v in record.items() if k not in {'image', 'mask'}} for record in records])
    s['harness_origin'] = HARNESS_ORIGIN
    s['uws_commit'] = UWS_COMMIT
    s['code_sha256'] = digest_json({p: sha256(Path(__file__).resolve().parents[2] / p) for p in (
        'src/training/final_retraining.py', 'src/training/mmseg_final_backend.py',
        'src/evaluation/segmentation_metrics.py')})
    return s, df, records, plan


def validate_output(frame, identity, expected_ids):
    required = {'frame_id', 'video', 'modelo', 'fold', 'seed', 'manifest_sha256',
                'run_sha256', 'mode', *METRICS}
    if not required <= set(frame.columns) or frame.frame_id.duplicated().any():
        raise ValueError('Salida incompleta o duplicada')
    if set(frame.frame_id) != set(expected_ids):
        raise ValueError('La salida no cubre exactamente el test')
    for col, expected in dict(modelo=identity['model_name'], fold=identity['fold'],
                              video=identity['fold'], seed=identity['seed'],
                              manifest_sha256=identity['manifest_sha256'],
                              run_sha256=identity['run_sha256'], mode=identity['mode']).items():
        if not frame[col].eq(expected).all():
            raise ValueError(f'Metadatos de salida incompatibles: {col}')
    values = frame[METRICS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Métricas no finitas o fuera de [0,1]')


def aggregate_shared_results(settings):
    """All three models, exact fold/seed/frame coverage; never silently publish a partial table."""
    s, manifest, _, _ = preflight(settings)
    if s['mode'] != 'final':
        raise ValueError('La tabla compartida requiere mode=final')
    files = sorted(Path(s['results_dir']).glob('per_frame_*.csv'))
    if not files:
        raise ValueError('No hay resultados compartidos')
    frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    required = {'modelo', 'fold', 'seed', 'frame_id', 'video', 'manifest_sha256', *METRICS}
    if not required <= set(frame.columns):
        raise ValueError('Faltan columnas (U-Net también debe exportar miou/mdice con score_predictions)')
    if set(frame.modelo) != {'U-Net', 'SegFormer', 'UWSegFormer'}:
        raise ValueError('La tabla final requiere los tres modelos')
    if frame.duplicated(['modelo', 'fold', 'seed', 'frame_id']).any():
        raise ValueError('Resultados duplicados; revisar la carpeta compartida')
    if not frame.manifest_sha256.eq(s['manifest_sha256']).all():
        raise ValueError('Resultados con manifiestos diferentes')
    if set(frame.seed) != set(s['seeds']) or set(frame.fold) != set(s['folds']):
        raise ValueError('Resultados con semillas o folds adicionales/ausentes')
    for model in ('U-Net', 'SegFormer', 'UWSegFormer'):
        for fold in s['folds']:
            ids = set(manifest[(manifest.fold == fold) & (manifest.split == 'test')].frame_id)
            for seed in s['seeds']:
                rows = frame[(frame.modelo == model) & (frame.fold == fold) & (frame.seed == seed)]
                if set(rows.frame_id) != ids or not rows.video.eq(fold).all():
                    raise ValueError(f'Cobertura incompleta: {model}, {fold}, semilla {seed}')
    values = frame[METRICS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Métricas ausentes/incorrectas; no se omiten al promediar')
    table = summarize(frame)
    table.to_csv(Path(s['results_dir']) / 'tabla_comparativa_final_4_8.csv', index=False)
    return table


def execute(settings, run=False):
    s, df, records, plan = preflight(settings)
    print(pd.DataFrame(plan).drop(columns='checkpoint_sha256').to_string(index=False))
    print(f"{len(records)} pares; {len(plan)} corridas. mode={s['mode']}")
    if not run:
        return plan
    backend_root = Path(s['uws_root']).resolve()
    actual = subprocess.check_output(['git', '-C', str(backend_root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != UWS_COMMIT:
        raise ValueError('El checkout de UWSegFormer no está en el commit fijado')
    # Include local compatibility patches, not just upstream HEAD, in run identity.
    s['uws_patch_sha256'] = digest_json({str(p.relative_to(backend_root)): sha256(p)
                                        for p in sorted((backend_root / 'mmseg').rglob('*.py'))})
    result_root = Path(s['results_dir']) / s['mode']
    all_results = []
    for entry in plan:
        identity = {**s, **entry}
        run_hash = digest_json(identity)
        work = result_root / f"{entry['model']}_{entry['fold']}_seed{entry['seed']}"
        done = work / 'completed.json'
        output = work / 'per_frame.csv'
        if done.exists():
            saved = json.loads(done.read_text())
            if saved['run_sha256'] != run_hash or not output.exists() or saved['output_sha256'] != sha256(output):
                raise ValueError(f'Resultado incompatible o modificado: {work}; usa otra results_dir')
        else:
            if work.exists():
                raise ValueError(f'Corrida parcial en {work}; conserva o mueve esa carpeta antes de reintentar')
            work.mkdir(parents=True)
            dataset = work / 'dataset'
            rows = df[df.fold == entry['fold']]
            materialize(rows, records, dataset)
            identity.update(dataset_root=str(dataset.resolve()), work_dir=str(work.resolve()),
                            checkpoint=str(Path(s['checkpoints'][entry['model']]).resolve()),
                            model_name=MODELS[entry['model']], run_sha256=run_hash)
            config_path = work / 'run_settings.json'
            config_path.write_text(json.dumps(identity, indent=2))
            subprocess.run([sys.executable, '-m', 'src.training.mmseg_final_backend',
                            '--settings', str(config_path.resolve())], check=True)
            if not output.exists():
                raise RuntimeError(f'El backend no exportó {output}')
            validate_output(pd.read_csv(output), identity, rows[rows.split == 'test'].frame_id)
            done.write_text(json.dumps(dict(run_sha256=run_hash, output_sha256=sha256(output)), indent=2))
        frame = pd.read_csv(output)
        validate_output(frame, {**identity, 'run_sha256': run_hash, 'model_name': MODELS[entry['model']]},
                        df[(df.fold == entry['fold']) & (df.split == 'test')].frame_id)
        all_results.append(frame)
        if s['mode'] == 'final':
            # Same filenames/columns consumed by Ariana's 3_4_final_retraining.ipynb.
            export = Path(s['results_dir']) / f"per_frame_{entry['model']}_{entry['fold']}_seed{entry['seed']}.csv"
            if export.exists() and sha256(export) != sha256(output):
                raise ValueError(f'Ya existe otro resultado en {export}; usa otra results_dir')
            export.write_bytes(output.read_bytes())
    summary = summarize(pd.concat(all_results, ignore_index=True))
    summary.to_csv(result_root / 'tabla_segformers_4_8.csv', index=False)
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--table', action='store_true', help='Validar y unir resultados de los tres modelos')
    parser.add_argument('--run', action='store_true', help='Entrenar; sin esta opción solo valida y muestra el plan')
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    if args.table:
        if args.run:
            parser.error('--table y --run son excluyentes')
        print(aggregate_shared_results(settings).to_string(index=False))
    else:
        execute(settings, run=args.run)


if __name__ == '__main__':
    main()
