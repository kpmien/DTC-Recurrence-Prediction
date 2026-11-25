# PROGRESS — image study (thyroid nodule ultrasound)

Working log of the image study. It follows a written specification for empirical studies: question and decision
rules fixed before modelling, phase gates, splits by patient or group, no tuning on test data, negative results
reported as such. The tabular recurrence study in
`notebooks/` and `results/` is a separate, finished analysis and is not covered here.

## Section 0: pre-specified protocol (locked 2026-11-23)

Changes after this date go under "Deviations" with date and reason; the text below is not edited.

| Placeholder | Value |
|---|---|
| PROJECT | DTC image study (name to be chosen) |
| QUESTION | How much of the cross-hospital drop of benign/malignant thyroid nodule classifiers is explained by burned-in markers, crop construction, frame geometry, prevalence and the reference standard (histology or cytology), and does reliance on non-nodule cues differ between DINOv3 + LoRA, frozen encoders and an ImageNet CNN? |
| UNIT | Patient where identifiers exist; duplicate group (TN5000); image (TN3K) |
| GROUP_KEY | `group_id`: patient ID, else byte-identical duplicate group, else image ID |
| PRIMARY_METRIC | AUROC on external datasets, bootstrap over `group_id` (2,000 resamples, 95% percentile interval), paired deltas |
| DATASETS | TN5000, TN3K, ThyUS2Path batch 2, ThyroidXL (included only if its audit passes) |
| EXTERNAL | DINOv3 ViT-B/16 and ViT-L/16 (on disk), DINOv2 ViT-S/14, one ImageNet CNN (fixed at the phase 2 gate) |

Must-not-expand list: recurrence prediction from images; segmentation or detection as a task; new architectures;
DDTI and TN-SCUI as evaluation sets.

### Hypotheses and decision rules

"Internal" means the held-out test part of the training dataset; "external" means every other dataset. A hypothesis
is called supported only if its rule is met; otherwise the result is reported as not supported, with the interval.

| | Hypothesis | Test | Supported if |
|---|---|---|---|
| H1 | Information outside the image content predicts the label. | Logistic regression on image size, border statistics and box/mask geometry only; grouped cross-validation inside each dataset. | 95% interval of AUROC lies above 0.5 in the dataset; reported per dataset. |
| H2 | Burned-in markers are unevenly distributed between benign and malignant. | Marker presence per image, labelled blind to the class; difference of proportions per dataset. | 95% interval of the difference excludes 0. |
| H3 | Classifiers discriminate with the nodule hidden. | Train and test with the released box (TN5000) or mask (TN3K, ThyroidXL) filled; internal test. | 95% interval of AUROC lies above 0.5. |
| H4 (primary) | Standardising crop construction and removing markers narrows the external drop. | Drop = internal AUROC minus external AUROC. Paired comparison of the drop between the default and the standardised pipeline, same model, same test cases. | 95% interval of the reduction in drop lies above 0. |
| H5 | Reliance on non-nodule cues differs between model families. | Paired differences between DINOv3 + LoRA and each other family in (a) nodule-hidden AUROC and (b) change in AUROC when markers are removed. | Interval of the paired difference excludes 0 after Holm correction over the comparisons. |

Fixed limits of the tests:
- ThyUS2Path batch 2 is scored per patient and on full frames only; it takes no part in H3 or in the crop part of H4.
- Prevalence and reference standard cannot be changed by an intervention. Prevalence is handled by reporting AUROC
  (and by reweighting for threshold metrics); the reference standard is reported as an unresolved part of the drop.
- H1 to H3 and H5 are secondary. No claim rests on a single seed; the number of seeds is fixed at the phase 2 gate.

Not locked here, to be fixed at their phase gates and before any external dataset is scored: splits and the handling
of TN5000 duplicates (phase 1), the marker labelling procedure and its check on a hand-labelled sample (phase 1),
the choice of CNN, the hyperparameter search space and the number of seeds (phase 2).

## Phase status

| Phase | State | Gate |
|---|---|---|
| 0. Bootstrap | Partly done: access audit written. No package, CLI, tests or environment record yet. | Not passed |
| 1. Data | Four datasets verified; manifest, fixed splits and leakage audit done (audit passed). Box and mask checks done. Caliper detector built; check against hand labels failed on TN3K and ThyUS2Path batch 2 (2026-11-25), passed on TN5000 and ThyroidXL. Still open: unit tests. | Not passed |
| 2 to 7 | Not started | |

## Log

### 2026-11-22
- Decision: image study is thyroid ultrasound, benign vs malignant. No public ultrasound dataset has recurrence labels.
- Literature and dataset survey; the key findings are re-checked against the sources below.
- TN5000 downloaded and audited (`scripts/image_study/audit_tn5000.py`, `dups_tn5000.py`).

### 2026-11-23
- Decision: recurrence from images dropped; work stays in this repository.
- Data kept in `data/external/`, excluded through `.git/info/exclude` (local, not committed).
- ThyUS2Path downloaded and audited (`audit_thyus.py`). Batch 1 image CSV and filenames use different patient
  numberings. Check with frozen DINOv3 features (`idspace_check.py`, CPU, one run): labels follow the CSV numbering.
  Batch 1 is blocked; batch 2 is usable.
- TN3K downloaded (images, masks from the TRFE repository; labels from the ACL Google Drive archive) and audited
  (`audit_cross.py` plus an inline label check, not yet saved as a script).
- No image shared between TN5000, TN3K and ThyUS2Path (exact or dHash-256 Hamming <= 12).
- Access audit written with sources, file counts and archive checksums (kept locally with the data).
- Papers read in full: Saleem et al. 2026 (TN5000 to ThyroidXL), Kiani Kalejahi and Rajabi 2026 (BMC audit),
  Sherif et al. 2026 (RCAF), Sadib et al. 2026 (TN5000 to TN3K, read in summary, quotes spot-checked), Yao et al. 2020
  (read in summary), and the README of the anonymous shortcut-audit repository.
- ThyroidXL: download attempted at revision b15fe293; refused: access to the gated dataset had not
  yet been granted.

- Dataset papers read in full: TN5000 descriptor, ACL/TNCD (arXiv 2207.00807), ThyroidXL. The file first saved
  as the TN3K paper was the ACL paper (wrong reference); the TN3K/TRFE paper and the ThyUS2Path paper were read
  afterwards (entries below).
- TN5000 descriptor: crosshair markers on 40% of images (17% outside the box, 21% intersecting, 2% inside), no
  breakdown by class, not removed; the authors state their influence is limited. None of the dataset papers gives
  marker prevalence by class or the Bethesda-to-label mapping.
- TNCD (classification archive) and TN3K (segmentation release) are the same 3,493 images, byte-identical (own check).

- TN3K dataset paper (Gong et al., ISBI 2021, TRFE-Net) read in full: 3,493 images from 2,421 patients, January
  2016 to August 2020, chosen from over 30,000 images of one unnamed partner hospital; one image kept per view;
  converted to greyscale and cropped; train/test split by patient (2,879 / 614). The paper has no benign/malignant
  labels, no scanner and no marker information.
- ThyUS2Path paper (Hou et al., Sci Data 2024) read in full: one hospital (Second Affiliated Hospital of Jiaxing
  University), 2019 to 2022, Esaote MyLab; all patients had thyroidectomy, labels are histology, nodules up to
  2.5 cm; labels are per patient ("different nodules of the same patient may have different labels"); no boxes or
  masks; batch 2 is called external but comes from the same hospital. Batch 2 counts match the files (2,503
  images, 241 patients, 100 benign / 141 malignant). Authors' model trained on batch 1 scores AUROC 0.735 on batch 2.
- Authors' code (github.com/TomHardy1997/ThyUS2Path, commit fa0c7d7) takes the patient of each image from the image
  CSV, which agrees with the numbering check; it contains nothing that maps the released filenames to the CSV.

Consequences for the protocol (taken into Section 0 at the lock):
- ThyUS2Path batch 2 can only be scored per patient (image scores aggregated by a rule fixed in advance) and only
  on full frames; the nodule-hidden and crop-standardisation tests cannot run there.
- Reference standards differ: histology of operated patients (ThyUS2Path), cytology (TN3K, ThyroidXL), and the
  TN5000 mix. This is a further source of the cross-dataset drop that the question must name.

- Section 0 locked, with the reference standard added to the question and decision
  rules written for H1 to H5.
- ThyroidXL access granted; download started at revision b15fe293.

- ThyroidXL downloaded (46,567 files) and audited (`audit_thyroidxl.py`, output in `reports/logs/`): 11,635 images,
  4,093 patients (train 3,354: 2,477 benign / 877 malignant; test 739: 386 / 353). Files are consistent, no patient
  crosses the split, duplicates stay inside one patient, nothing is shared with the other three datasets. The
  dataset card miscounts images and patients. The basis of each label (histopathology or FNAC) is recorded per
  patient: 96.7% of benign patients rest on FNAC only, 57.6% of malignant patients have histopathology.
  Malignancy is 26.1% of patients in train and 47.8% in test. ThyroidXL passes the audit and joins the datasets.

- Manifest and splits built (`build_manifest.py`; summary and checksums in `reports/splits_image_study.json`; the
  manifest itself stays in `data/manifests/`, not committed, because it carries labels of a gated dataset).
  22,631 rows. Decisions, all fixed before any model is trained:
  - `test` = official test part of TN5000 (1,000), TN3K (614) and ThyroidXL (2,094 images, 739 patients), kept
    exactly as released. No model is ever fitted or selected on a `test` row of any dataset; external scoring of a
    model uses the `test` parts of the other datasets.
  - `shift_test` = ThyUS2Path batch 2 (2,495 images, 241 patients); no model is trained on it. One patient
    (8 images) has no label and is excluded.
  - TN5000: 44 train/val files that are byte-identical copies of a test image and 1 train file near-identical to a
    test image (dHash-256 Hamming <= 6) are excluded; a duplicate group spanning train and val goes to train.
    Official val is kept as the selection set.
  - TN3K and ThyroidXL have no official validation part: 20% of the dev groups per class go to val, chosen by
    rank of sha256(seed|group_id), seed 20261123.
- Leakage audit (`audit_leakage.py`, `reports/leakage_audit.json`): all ten checks pass; hashes recomputed from the
  files. No group, patient or identical file in two partitions or two datasets. One near-identical pair remains
  inside TN3K dev (train/val, `0647.jpg` benign and `0928.jpg` malignant, opposite labels); to be looked at.
- Limitation recorded: TN5000 releases no patient identifiers and its split is by image, so images of one patient
  can sit in both train and test (adjacent file numbers share the label 95.1% of the time). The internal TN5000
  AUROC is therefore likely optimistic, which inflates the "drop" of H4 for models trained on TN5000. TN3K and
  ThyroidXL, whose splits are by patient, are the cleaner training datasets for H4.

- Annotation check (`audit_annotations.py`, `reports/annotation_audit.json`): every TN3K and ThyroidXL mask has the
  size of its image and is not empty; ThyroidXL masks agree with the released boxes within 3 px. TN5000: 2 XML
  files state a size different from the image and 1 box (`003813`) reaches outside the image; boxes are to be
  clipped at loading. TN3K masks are JPEG (values near 0 and near 255), to be thresholded at 127.
- The nodule box covers a median of 3.4% of the frame in TN5000, 6.4% in ThyroidXL and 12.0% in TN3K: frames are
  cropped differently per dataset, which is the "crop construction" factor of H4.

### 2026-11-24

- Caliper detector for H2 (`scripts/image_study/markers.py`, `train_marker_net.py`, `score_markers.py`,
  `finalize_markers.py`). A small convolutional network (115,345 parameters) is trained only on synthetic calipers
  drawn on training frames; it reads no diagnosis label and no hand label. Weights
  `models/image_study/marker_net.pt` (sha256 2bc06e7d...). Two hand-written detectors were tried first and dropped
  (template matching: background and calipers scored alike; four-arm cross filter: fired on speckle and on a corner
  icon, 93.5% of TN5000 flagged).
- Assumption used in training: ThyroidXL training frames carry no calipers (none seen in the frames viewed), so
  detector peaks on them were reused as hard negatives. The 150 hand-labelled ThyroidXL images test this assumption.
- Image rule: peaks in 16 px cells where the detector fires in at least 5% of the images of one frame format
  (dataset, width, height; formats with at least 40 images) are treated as fixed scanner furniture and dropped;
  an image is flagged when the second strongest remaining peak reaches the threshold (calipers come in pairs).
  The furniture map uses all images of a format, test included; it reads no label. TN3K has no format with 40
  images, so no furniture filter applies there.
- Threshold fixed at 0.7 on training images only (`dev_a`, role train), before any hand label exists. Share
  flagged on training images: TN5000 45.6%, TN3K 39.3%, ThyroidXL 0.3%. All images
  (`reports/marker_flags_summary.json`; flags in `reports/marker_flags.csv`, sha256 ec8bf820...): TN5000 45.6%,
  TN3K 42.0%, ThyroidXL 0.3%, ThyUS2Path batch 2 69.4%. The TN5000 descriptor reports 40%.
- Viewed by eye, 20 random training images per cell (not a measurement): TN5000 flagged 19 of 20 with real
  calipers (1 body-mark icon), not flagged 0 of 20 with calipers, dropped by the furniture filter 0 of 20 with
  calipers; ThyroidXL flagged 0 of 20 with calipers (depth-scale ticks in rare formats, bright tissue); TN3K
  flagged about 17 of 20 real (dotted lines and a dotted box in the others), not flagged at least 1 of 20 with
  faint calipers. ThyUS2Path batch 2 not viewed (shift test).
- Check sample: `scripts/image_study/make_marker_check.py` drew 150 random groups per dataset (600 images,
  neutral names, class hidden) into `data/marker_check/` (not in git). Acceptance rule fixed in
  `eval_marker_check.py` before labelling: sensitivity and specificity at least 0.90 in every dataset
  (sensitivity not judged below 10 positives). Not yet evaluated: `marker_labels.csv` does not exist.
- The detector flags measurement calipers only. Text, body marks and Doppler boxes are not covered; H2 as tested
  is therefore about calipers.

### 2026-11-25

- Hand labels of the check sample received (`data/marker_check/marker_labels.csv`, sha256 47701bc2..., 600 images:
  185 yes, 408 no, 7 cannot tell; one labeller, class hidden). Detector compared at the fixed threshold 0.7
  (`reports/marker_check_eval.json`, log `reports/logs/eval_marker_check.txt`). Result: NOT ACCEPTED.

  | Dataset | Hand prevalence | Sensitivity (95% Wilson) | Specificity (95% Wilson) | Rule |
  |---|---|---|---|---|
  | TN5000 | 38.7% | 58/58 = 1.00 (0.94-1.00) | 88/92 = 0.96 (0.89-0.98) | passed |
  | ThyroidXL | 0.7% | 0/1, not judged | 148/149 = 0.99 (0.96-1.00) | passed |
  | TN3K | 44.4% | 40/64 = 0.63 (0.50-0.73) | 70/80 = 0.88 (0.79-0.93) | failed |
  | ThyUS2Path batch 2 | 41.6% | 62/62 = 1.00 (0.94-1.00) | 49/87 = 0.56 (0.46-0.66) | failed |

- Consequences: detector flags are not used for TN3K or ThyUS2Path batch 2. The assumption "ThyroidXL carries no
  calipers" holds on the sample except for 1 image of 150. The threshold is not changed after seeing the labels.
  Any revised detector needs a new hand-labelled sample; this one has been seen.
- Labelling convention stated by the labeller after the check: colour flow (Doppler) patches and dotted boxes
  without '+' or 'x' were labelled "no". This matches the definition given (calipers only).
- Causes looked at on development images only (TN3K `dev_a` images of the first check sample, ThyUS2Path batch 1):
  TN3K misses are tiny, faint marks in enlarged crops; TN3K false alarms are bright specks and dotted boxes;
  on batch 1 the detector fires on body pictograms, scale ticks, text and colour flow patches.
- Detector revisions (`train_marker_net.py` version 2: tiny and faint synthetic marks, specks, dotted boxes and
  colour patches as distractors, enlarged ThyroidXL crops, fixed furniture as hard negatives). Weights
  `marker_net_v2.pt` (3 hard-negative rounds) and `marker_net_v3.pt` (6 more). The `dev_a` part of the first
  check sample is used from here on as a development set, no longer as a check. Pair rule without furniture
  filter, TN3K `dev_a` (48 marked, 75 unmarked): version 1 at 0.7 found 29/48 and left 66/75 alone; version 3
  at 0.7 43/48 and 51/75, at 0.8 42/48 and 57/75, at 0.9 38/48 and 61/75, at 0.95 34/48 and 64/75. No threshold
  reaches 0.90 on both. Files `reports/marker_scores_v{2,3}_devcheck.csv`, `reports/marker_scores_v{2,3}_design.csv`.
- Conclusion: the automatic detector is not adequate for TN3K; revisions stopped. Version 1 at 0.7 stays the
  detector for TN5000 and ThyroidXL (passed the check). ThyUS2Path could not be measured on development data
  (no hand labels on batch 1). How marker presence is obtained for TN3K and ThyUS2Path batch 2 is open.
- Decision (2026-11-25): marker presence for TN3K and ThyUS2Path batch 2 comes from hand labels.
  `MARKER_ALL=1 make_marker_check.py` wrote `data/marker_label_all/` (not in git): 5,688 images (TN3K 3,343,
  ThyUS2Path batch 2 2,345), i.e. every used image not in the first check sample, in random order, class hidden,
  key sha256 64237308.... Same definition as the first round (a '+' or 'x' of any colour is yes; dotted boxes or
  lines alone, colour flow patches, text, icons and the depth scale are no). One labeller; a second labeller on
  a subset is still to be arranged. Labelling not started.

## Deviations from the specification

- 2026-11-23: existing repository layout kept; spec directories are added only as needed (`reports/`, `scripts/image_study/`).
- 2026-11-23: manifest is CSV, not Parquet (no pandas/pyarrow in the environment, and no package was added to it). No `calibration` partition: the primary metric is AUROC; if threshold metrics are reported the
  threshold comes from `val`. `dev_a` carries a `dev_role` column (train or val); `dev_b` is not used (no stacking).
- 2026-11-23: access audit written before the package and smoke test exist (phase 0 order inverted) because dataset
  choice had to be settled first.
