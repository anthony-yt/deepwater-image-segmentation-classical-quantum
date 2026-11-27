# Task 5.2 — VQC ablation: results and conclusion

Goal: attribute the v1 to v3 improvement to a concrete factor, since v3 changed labels,
features, circuit and split at once. Numbers below are on the test split, averaged over the
three seeds, from the v3 run (metric: per-video balanced accuracy, bal_acc_macro_video).

## 1. Encoding (angle vs amplitude): this is what drove the improvement

The angle-encoding VQC beats the amplitude VQC (the v1 baseline) by about 30 points on every
filter, and the amplitude VQC sits at chance.

| Filter | VQC-angles | VQC-amplitudes (v1) |
|---|---|---|
| Crudos | 0.797 | 0.504 |
| CLAHE | 0.814 | 0.524 |
| MSRCR | 0.775 | 0.503 |
| UDCP | 0.831 | 0.492 |
| Antirruido | 0.806 | 0.473 |

The amplitude VQC barely predicts the pipe at all (F1 near zero, e.g. 0.031 on Crudos). So the
jump from v1 to v3 came from switching to angle encoding with data re-uploading over
validation-selected features, not from the circuit being "quantum".

## 2. Does the VQC add anything over an SVM with the same 8 features? Not clearly

In-split the two are about even (Crudos: VQC 0.797 vs SVM 0.752 on per-video balanced accuracy,
but on plain balanced accuracy the SVM is ahead, 0.867 vs 0.786). On leave-one-video-out, which
measures generalization to an unseen video, the SVM matches or beats the VQC (holding Video1 out:
VQC 0.33 vs SVM 0.63). The run's own config states "No se demuestra ventaja cuántica", and these
numbers agree: the VQC is comparable to a classical SVM on the same inputs, with no quantum
advantage.

## 3. Per video (Crudos, test)

| Model | Video1 | Video6 | Video7 |
|---|---|---|---|
| VQC-angles (bal. acc) | 0.962 | 0.801 | 0.628 |
| SVM-angles (bal. acc) | 0.982 | 0.808 | 0.466 |

Both solve Video1. Video6 is the hard thin pipe: both over-predict it (precision ~0.11 to 0.17).
Video7 is where the VQC is clearly better than the SVM.

## 4. Augmentation (RandomResizedCrop): does not help

Forward selection chose not to use RRC on any of the five filters (validation balanced accuracy
without RRC was equal or higher than with it). So the crop/zoom augmentation added nothing here.

## Still open (needs one feature-saving run of the ablation script)

- Number of layers (1 / 2 / 4 / 6), holding encoding and features fixed.
- Number of features (4 / 8 / 12).
- Task 5.16 (quantum feature extractor) in full.

These need the per-patch feature matrices, which the run did not save. The plan is to save
features.npz from one pass up to the feature-selection cell, then run the layer/feature sweep and
5.16 from those features.

## Part (d): curated labels (5.1)

Pending Anthony's curated labels. When available, rerun the ablation with them and check whether
the conclusions above still hold against SAM-guided labels.
