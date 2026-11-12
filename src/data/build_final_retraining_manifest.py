"""Create or verify the shared 64-mask Task 4.8 LOVO manifest, without overwriting it."""
import argparse
from pathlib import Path
import re

import pandas as pd

from src.training.final_retraining import sha256, validate_manifest

EXPECTED_COUNTS = {'Video1': 24, **{f'Video{i}': 8 for i in range(2, 7)}}


def prepare_manifest(frames_dir, masks_dir, out):
    masks_dir, out = Path(masks_dir), Path(out)
    inventory, excluded = [], []
    for path in sorted(masks_dir.glob('*.png')):
        frame_id = path.stem[:-5] if path.stem.endswith('_mask') else path.stem
        match = re.search(r'Video(\d+)(?:_|$)', frame_id)
        if not match:
            raise ValueError(f'No se reconoce el video de {path.name}')
        video = f'Video{match[1]}'
        if video not in EXPECTED_COUNTS:
            excluded.append(path.name)
            continue
        inventory.append({'frame_id': frame_id, 'video': video})
    counts = pd.Series([r['video'] for r in inventory], dtype=str).value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f'Inventario distinto al acordado. Esperado={EXPECTED_COUNTS}; real={counts}')
    expected_ids = {r['frame_id'] for r in inventory}
    if len(expected_ids) != 64:
        raise ValueError('IDs duplicados en el inventario')
    if out.exists():
        df, _ = validate_manifest(out, sha256(out), frames_dir, masks_dir)
        if set(df.frame_id) != expected_ids:
            raise ValueError('El manifiesto existente no contiene exactamente las 64 máscaras; no se modifica')
        print('Manifiesto existente validado; no se modifica.')
    else:
        rows = [{**record, 'fold': fold, 'split': 'test' if record['video'] == fold else 'train'}
                for fold in EXPECTED_COUNTS for record in inventory]
        df = pd.DataFrame(rows, columns=['fold', 'frame_id', 'video', 'split'])
        # Validate bytes and every real pair before publishing the shared CSV.
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(out.name + '.pending')
        created = False
        try:
            with temporary.open('x') as stream:
                created = True
                stream.write(df.to_csv(index=False, lineterminator='\n'))
            validate_manifest(temporary, sha256(temporary), frames_dir, masks_dir)
            with out.open('xb') as stream:
                stream.write(temporary.read_bytes())
        finally:
            if created and temporary.exists():
                temporary.unlink()
        print('Manifiesto creado: compartir este mismo archivo con Ariana y Anthony.')
    print('Máscaras por video:', counts)
    print(df.groupby(['fold', 'split']).size().unstack().to_string())
    if excluded:
        print(f'Fuera del acuerdo: {len(excluded)} máscaras de otros videos; no se incluyen.')
    print('Archivo:', out)
    print('SHA-256:', sha256(out))
    return sha256(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames-dir', required=True, type=Path)
    parser.add_argument('--masks-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    prepare_manifest(args.frames_dir, args.masks_dir, args.out)
