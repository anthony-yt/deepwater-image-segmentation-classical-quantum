# Data Layout

Large datasets, image files, masks, videos, and model checkpoints are intentionally not tracked in Git. Keep working copies locally or in the project Drive, and reproduce the required local/Colab layout with the scripts in this repository.

## Project Drive

Current processed project data appears under:

```text
/content/drive/MyDrive/Under_Water_Dataset/Dataset_Processed/
  Dataset_Final_512/
  Frames_CLAHE/
  Frames_Crudos/
  Frames_MSRCR/
  Frames_UDCP/
```

Use these folders as internal test images for qualitative inference, especially `Frames_Crudos` or `Dataset_Final_512`.

## Local MMSegmentation Benchmark Layout

The current local SUIM benchmark prepared for development is:

```text
data/SUIM/
  images/
    training/
    validation/
  annotations/
    training/
    validation/
```

This folder is ignored by Git.

## Optional Drive Benchmark Layout

For shared Colab execution, reproduce the same structure in the project Drive:

```text
/content/drive/MyDrive/Under_Water_Dataset/benchmarks/
  SUIM/
    images/
      training/
      validation/
    annotations/
      training/
      validation/
  DUT/
    images/
      training/
      validation/
    annotations/
      training/
      validation/
```

Images and annotation masks must share the same stem where possible, for example:

```text
images/training/sample_001.jpg
annotations/training/sample_001.png
```

The official UWSegFormer README asks for the same logical layout under a local
`data/SUIM` or `data/DUT` folder. In Colab, either copy/symlink the Drive
benchmark folder into the cloned UWSegFormer repo's `data/` directory, or mount
Drive and point configs/scripts to the mounted paths.

## Dataset Roles

Use the project Drive folders shown above for internal qualitative testing.
They do not replace SUIM/DUT for benchmark setup unless they include matching
semantic segmentation masks.

Use SUIM/DUT for public benchmark integration because those datasets include
image/mask pairs needed by MMSegmentation evaluation and fine-tuning.

## What Goes In Git

Track:

- notebooks
- source scripts
- configuration templates
- dataset preparation instructions

Do not track:

- raw images
- generated images
- masks
- videos
- `.pth`, `.pt`, `.onnx`, or other checkpoints
