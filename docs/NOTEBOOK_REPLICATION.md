# PC-SSL Notebook Replication Guide

The attached notebook `SD SEED V (1).ipynb` is the author's working implementation. It is **not** the same as the public GitHub README alone, but it matches what the paper reports when run end-to-end.

## What the notebook does

### 1. Preprocessing
- Loads SEED-V `.npz` DE features
- Builds consecutive `(past, future)` segment pairs per subject
- Saves `past_by_subject_DE.pkl`, `future_by_subject_DE.pkl`, label pickles

### 2. Cross-validation (critical)
- Splits each subject's segment list into **15 equal contiguous blocks**
- 3 folds hold out blocks `[0-4]`, `[5-9]`, `[10-14]`
- This is **segment-block CV**, not session-aware trial-ID CV

Our earlier replication used a different split and therefore trained/evaluated on different data.

### 3. Model (`CNNPredictiveCodingDE_Attn`)
- CBAM-style band/channel attention (avg+max pool + 2-layer MLP gates)
- Encoder: Conv 1→16→32 with MaxPool `(2,1)`
- Bottleneck: Conv 32→64
- Decoder: ConvTranspose2d + Upsample to `(62, 5)`
- Input/output shape: `(B, 62, 5)`

This differs from the earlier reconstructed model (simple linear gates, bilinear resize decoder, no separate bottleneck).

### 4. Pretraining
- 30 epochs, batch 256, Adam `lr=5e-4`, `weight_decay=1e-5`, grad clip 5.0
- MSE reconstruction loss
- One fresh model per subject/fold

### 5. Downstream classification
- Load each pretrained checkpoint into `EEGEmotionClassifier`
- Freeze first **8 encoder layers**
- Head: `Flatten → Dropout(0.3) → Linear(flat_dim, 128) → ReLU → Linear(128, 5)`
- Fine-tune 25 epochs, Adam `lr=8e-5`, `weight_decay=1e-6`, StepLR(5, 0.5)
- Uses a **global 80/20 stratified split** over all subjects' segments for fine-tuning

### 6. Reported evaluation (92.39%)
- Segment-level accuracy on each model's held-out fold test indices
- Aggregates predictions from all 48 subject/fold fine-tuned models
- Notebook output: `Mean Validation Accuracy: 0.9239`

## Code aligned to the notebook

Updated in `PC-SSL/`:
- `src/models/attention.py`
- `src/models/predictive_coding.py`
- `src/models/classifier.py` (`EEGEmotionClassifier`)
- `src/data/splits.py` (notebook 15-block trial split)
- `src/data/dataset.py` (tensor shape `62 x 5`)
- `configs/seed_v_notebook.yaml`
- `scripts/finetune_classifier_notebook.py`
- `scripts/evaluate_notebook_protocol.py`

## Run full SEED-V replication

From `PC-SSL/` with the venv activated:

```bash
# 1) Regenerate notebook-compatible splits (already run once if processed data exists)
.venv/bin/python - <<'PY'
from src.data.splits import CrossValidationSplitter
s = CrossValidationSplitter('data/processed', 'data/processed')
past, labels, trial_idx = s.load_data()
folds = s.create_trial_based_splits(past, labels, trial_idx)
s.save_splits(folds, 'folds_by_subject_trial_DE.pkl')
s.create_train_val_files(past, labels, folds, 'trial')
PY

# 2) Pretrain all subject/fold reconstruction models (~hours on GPU)
.venv/bin/python scripts/train_model.py --config configs/seed_v_notebook.yaml --device cuda

# 3) Fine-tune classifiers with notebook protocol
.venv/bin/python scripts/finetune_classifier_notebook.py --config configs/seed_v_notebook.yaml --device cuda

# 4) Evaluate segment-level accuracy (target ~92.39%)
.venv/bin/python scripts/evaluate_notebook_protocol.py \
  --data_dir data/processed \
  --model_dir classifier_models_notebook
```

Quick smoke test (subject 1, fold 1):

```bash
.venv/bin/python scripts/train_model.py --config configs/seed_v_notebook.yaml --subject 1 --fold 0 --device cuda
# temporarily set training.epochs: 1 in config for faster smoke

.venv/bin/python scripts/finetune_classifier_notebook.py --config configs/seed_v_notebook.yaml --subject 1 --fold 0 --epochs 2 --device cuda

.venv/bin/python scripts/evaluate_notebook_protocol.py --model_dir classifier_models_notebook
```

## Status

| Step | Before notebook | After notebook alignment |
| --- | --- | --- |
| Model | Reconstructed approximation | Author architecture |
| Splits | Session/trial-ID based | 15 contiguous blocks |
| Classifier | Full encoder freeze + 256/128 MLP | Partial encoder freeze + 128 MLP |
| Fine-tune data | Fold-specific | Global 80/20 split |
| Evaluation | Mixed protocols | Segment-level fold aggregation |
| SEED-V accuracy | ~40% (paper split) / ~65% (baseline) | **Run step 4 to measure** |

The public GitHub repo still does not ship `src/models/`; the notebook is the missing implementation source.
