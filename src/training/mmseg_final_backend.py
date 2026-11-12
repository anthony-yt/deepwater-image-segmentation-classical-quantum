"""Single-GPU MMSeg 0.11 backend, invoked in the legacy uws environment.

Architecture and ADE20K key conversion retained from notebooks 4 and 5.
No validation hook: the held-out video is read only after the last update.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
# Legacy UWSegFormer compatibility.
np.float = float
np.int = int

def convert_modern_mit_backbone(source):
    converted, consumed = {}, set()
    for key, value in source.items():
        patch = re.fullmatch(r'backbone\.layers\.(\d+)\.0\.(projection|norm)\.(weight|bias)', key)
        if patch:
            stage, module, parameter = patch.groups(); stage = int(stage) + 1
            legacy_module = 'proj' if module == 'projection' else 'norm'
            converted[f'backbone.patch_embed{stage}.{legacy_module}.{parameter}'] = value
            consumed.add(key); continue
        stage_norm = re.fullmatch(r'backbone\.layers\.(\d+)\.2\.(weight|bias)', key)
        if stage_norm:
            stage, parameter = stage_norm.groups(); stage = int(stage) + 1
            converted[f'backbone.norm{stage}.{parameter}'] = value
            consumed.add(key); continue
        block = re.fullmatch(r'backbone\.layers\.(\d+)\.1\.(\d+)\.(.+)', key)
        if not block:
            continue
        stage, index, suffix = block.groups(); prefix = f'backbone.block{int(stage)+1}.{index}'
        if suffix in {'attn.attn.in_proj_weight', 'attn.attn.in_proj_bias'}:
            parameter = 'weight' if suffix.endswith('weight') else 'bias'
            dim = value.shape[1] if parameter == 'weight' else value.numel() // 3
            converted[f'{prefix}.attn.q.{parameter}'] = value[:dim].clone()
            converted[f'{prefix}.attn.kv.{parameter}'] = value[dim:].clone()
        else:
            replacements = {
                'attn.attn.out_proj.': 'attn.proj.',
                'attn.sr.': 'attn.sr.',
                'attn.norm.': 'attn.norm.',
                'ffn.layers.0.': 'mlp.fc1.',
                'ffn.layers.1.': 'mlp.dwconv.dwconv.',
                'ffn.layers.4.': 'mlp.fc2.',
                'norm1.': 'norm1.', 'norm2.': 'norm2.',
            }
            target_suffix = None
            for modern, legacy in replacements.items():
                if suffix.startswith(modern):
                    target_suffix = legacy + suffix[len(modern):]
                    break
            if target_suffix is None:
                continue
            converted[f'{prefix}.{target_suffix}'] = value
        consumed.add(key)
    unconsumed = sorted(k for k in source if k.startswith('backbone.') and k not in consumed)
    return converted, unconsumed



def build_config(s, Config):
    root = Path(s['uws_root'])
    if s['model'] == 'uwsegformer':
        cfg = Config.fromfile(str(root / 'local_configs/uwsegformer/segformer_e+PPsegHead1.b0.640x480.suim.160k.py'))
        cfg.model.pretrained = None
        cfg.model.decode_head.num_classes = 2
        cfg.model.decode_head.loss_decode = dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    else:
        cfg = Config(dict(model=dict(
            type='EncoderDecoder', pretrained=None,
            backbone=dict(type='mit_b0', style='pytorch'),
            decode_head=dict(type='SegFormerHead', in_channels=[32, 64, 160, 256],
                in_index=[0, 1, 2, 3], feature_strides=[4, 8, 16, 32], channels=256,
                decoder_params=dict(embed_dim=256), dropout_ratio=0.1, num_classes=2,
                norm_cfg=dict(type='BN', requires_grad=True), align_corners=False,
                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            train_cfg=dict(), test_cfg=dict(mode='whole'))))
    norm = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    crop = (480, 640)
    train_pipeline = [
        dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', img_scale=(640, 480), ratio_range=(0.8, 1.2)),
        dict(type='RandomCrop', crop_size=crop, cat_max_ratio=0.90),
        dict(type='RandomFlip', prob=0.5), dict(type='Normalize', **norm),
        dict(type='Pad', size=crop, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'), dict(type='Collect', keys=['img', 'gt_semantic_seg'])]
    test_pipeline = [dict(type='LoadImageFromFile'),
        dict(type='MultiScaleFlipAug', img_scale=(640, 480), flip=False, transforms=[
            dict(type='Resize', keep_ratio=True), dict(type='RandomFlip'),
            dict(type='Normalize', **norm), dict(type='Pad', size=crop, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']), dict(type='Collect', keys=['img'])])]
    common = dict(type='CustomDataset', data_root=s['dataset_root'], img_suffix='.png',
                  seg_map_suffix='.png', classes=('background', 'structure'),
                  palette=[[0, 0, 0], [255, 80, 40]])
    # val/test descriptions support inference only; train_segmentor(validate=False).
    test = dict(**common, img_dir='images/test', ann_dir='annotations/test', pipeline=test_pipeline)
    cfg.data = dict(samples_per_gpu=s['batch_size'], workers_per_gpu=0,
                    train=dict(**common, img_dir='images/train', ann_dir='annotations/train', pipeline=train_pipeline),
                    val=test, test=test)
    cfg.optimizer = dict(type='AdamW', lr=s['learning_rate'], betas=(0.9, 0.999), weight_decay=0.01)
    cfg.optimizer_config = dict()
    cfg.lr_config = dict(policy='poly', warmup='linear', warmup_iters=min(100, s['max_iters']//10),
                         warmup_ratio=1e-3, power=1.0, min_lr=1e-7, by_epoch=False)
    cfg.runner = dict(type='IterBasedRunner', max_iters=s['max_iters'])
    cfg.checkpoint_config = dict(by_epoch=False, interval=s['max_iters'], max_keep_ckpts=1)
    cfg.pop('evaluation', None)
    cfg.log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
    cfg.workflow = [('train', 1)]
    cfg.work_dir = s['work_dir']
    cfg.gpu_ids = [0]
    cfg.seed = s['seed']
    cfg.resume_from = None
    cfg.load_from = None
    cfg.log_level = 'INFO'
    cfg.find_unused_parameters = True
    return cfg


def load_pretrained(model, s, torch):
    from src.training.final_retraining import sha256
    if sha256(s['checkpoint']) != s['checkpoint_sha256']:
        raise ValueError('El checkpoint cambió después del preflight')
    checkpoint = torch.load(s['checkpoint'], map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    current = model.state_dict()
    if s['model'] == 'segformer':
        state, unused = convert_modern_mit_backbone(state)
        if unused:
            raise ValueError('Claves de encoder no convertidas: ' + str(unused))
    else:
        for key in ('decode_head.conv_seg.weight', 'decode_head.conv_seg.bias'):
            state.pop(key, None)
    for key in list(state):
        if key in current and state[key].ndim == 4 and current[key].ndim == 2 and state[key].shape[2:] == (1, 1):
            state[key] = state[key].squeeze(-1).squeeze(-1)
    mismatches = [k for k in state if k in current and state[k].shape != current[k].shape]
    missing = [k for k in current if k.startswith('backbone.') and k not in state]
    if missing or mismatches:
        raise ValueError('Checkpoint incompatible; missing=' + str(missing) + '; shapes=' + str(mismatches))
    incompatible = model.load_state_dict(state, strict=False)
    if s['model'] == 'uwsegformer' and set(incompatible.missing_keys) != {
            'decode_head.conv_seg.weight', 'decode_head.conv_seg.bias'}:
        raise ValueError('UWSegFormer: solo puede faltar el clasificador binario: ' + str(incompatible))
    if s['freeze_backbone']:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    return dict(checkpoint_sha256=s['checkpoint_sha256'],
                missing_keys=list(incompatible.missing_keys), unexpected_keys=list(incompatible.unexpected_keys),
                encoder_frozen=s['freeze_backbone'],
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    args = parser.parse_args()
    s = json.loads(args.settings.read_text())
    sys.path.insert(0, str(Path(s['uws_root']).resolve()))
    import torch
    from PIL import Image
    import mmcv
    import mmseg
    from mmcv import Config
    from mmseg.apis import set_random_seed, train_segmentor
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from src.training.final_retraining import read_mask, restore_prediction, score_predictions
    if not torch.cuda.is_available():
        raise RuntimeError('La ejecución requiere una GPU CUDA; el preflight funciona en CPU')
    set_random_seed(s['seed'], deterministic=True)
    torch.backends.cudnn.benchmark = False
    cfg = build_config(s, Config)
    model = build_segmentor(cfg.model)
    loading = load_pretrained(model, s, torch)
    work = Path(s['work_dir'])
    runtime = dict(python=sys.version, torch=torch.__version__, mmcv=mmcv.__version__,
                   mmseg=mmseg.__version__, numpy=np.__version__,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    (work / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    (work / 'pretrained_loading.json').write_text(json.dumps(loading, indent=2))
    dataset = build_dataset(cfg.data.train)
    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE
    cfg.checkpoint_config['meta'] = dict(CLASSES=dataset.CLASSES, PALETTE=dataset.PALETTE,
                                         manifest_sha256=s['manifest_sha256'], run_sha256=s['run_sha256'])
    cfg.dump(str(work / 'final_config.py'))
    model.cfg = cfg
    train_segmentor(model, [dataset], cfg, distributed=False, validate=False,
                    timestamp=time.strftime('%Y%m%d_%H%M%S'), meta=dict(seed=s['seed'], run_sha256=s['run_sha256']))
    checkpoint = work / ('iter_%s.pth' % s['max_iters'])
    if not checkpoint.is_file():
        raise RuntimeError('No existe el checkpoint de la última iteración')
    # Explicitly load the final iteration, never best_mIoU.pth.
    from mmcv.runner import load_checkpoint
    load_checkpoint(model, str(checkpoint), map_location='cpu')
    model.cuda().eval()
    preds, gts = {}, {}
    prediction_dir = work / 'predictions'
    prediction_dir.mkdir()
    root = Path(s['dataset_root'])
    for image_path in sorted((root / 'images/test').glob('*.png')):
        fid = image_path.stem
        gt = read_mask(root / 'annotations/test' / image_path.name)
        # Obtain scores at network resolution, then resize the discrete labels explicitly.
        # rescale=False prevents implicit bilinear score resizing to the original frame.
        from mmseg.apis.inference import LoadImage
        from mmseg.datasets.pipelines import Compose
        from mmcv.parallel import collate, scatter
        pipeline = Compose([LoadImage()] + model.cfg.data.test.pipeline[1:])
        data = collate([pipeline(dict(img=str(image_path)))], samples_per_gpu=1)
        data = scatter(data, [next(model.parameters()).device])[0]
        with torch.no_grad():
            pred = model(return_loss=False, rescale=False, **data)[0]
        # Remove padding before restoring the original discrete mask.
        meta = data['img_metas'][0][0]
        pred = restore_prediction(pred, meta['img_shape'], gt.shape)
        preds[fid], gts[fid] = pred, gt
        Image.fromarray(pred).save(prediction_dir / image_path.name)
    frame = score_predictions(preds, gts)
    for key, value in dict(modelo=s['model_name'], fold=s['fold'], seed=s['seed'],
                           pasadas=s['passes'], effective_passes=s['effective_passes'],
                           manifest_sha256=s['manifest_sha256'], dataset_sha256=s['dataset_sha256'],
                           run_sha256=s['run_sha256'], mode=s['mode']).items():
        frame[key] = value
    frame.to_csv(work / 'per_frame.csv', index=False)


if __name__ == '__main__':
    main()
