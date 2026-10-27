# Missing In Action (MIA) — Membership Inference Attack

This project implements a **membership inference attack (MIA)** against a fine-tuned GPT-2 language model. Given text samples, the attack estimates whether each sample was part of the target model's training data.

The notebook covers shadow-model training, attack-score generation, evaluation, and packaging predictions for submission.

## How It Works

The attack uses per-sample language-model loss as a signal of membership:

1. **Prepare the test data** and tokenize each text sample.
2. **Load the target model**, a GPT-2 model fine-tuned with LoRA.
3. **Load shadow models** and calculate the same features for each sample.
4. **Calculate membership scores** using z-score normalization. Samples with target-model loss lower than the shadow-model average receive stronger membership evidence.
5. **Evaluate or export predictions**, depending on the selected phase.

The notebook extracts average per-token cross-entropy loss and perplexity. Its attack function uses the target and shadow-model loss features to calculate the membership scores.

## Project Structure

The notebook expects the repository to have the following structure:

```text
18734-17731_Project_Phase2_3/
├── data/
│   ├── shadow/                 # Data used to train shadow models
│   ├── train/                  # Training-phase evaluation data/labels
│   ├── val/                    # Validation test data and predictions
│   └── final/                  # Final test data and predictions
├── models/
│   ├── train/                  # Target model for training-phase evaluation
│   ├── val/                    # Target model for validation
│   ├── final/                  # Target model for final evaluation
│   └── shadow/                 # Trained shadow models or their ZIP archives
├── requirements.txt
└── project_submission.zip      # Generated submission archive
```

Exact contents depend on the course-provided repository and data. The notebook's configured target-model directory is `models/<PHASE>/gpt2_3_lora32_adamw_b8_lr2`.

## Requirements

- Python
- PyTorch
- Hugging Face `transformers` and `datasets`
- `peft` for LoRA models
- NumPy
- scikit-learn
- Matplotlib
- Seaborn (used by the optional score-distribution visualization)

The notebook installs dependencies from the repository's `requirements.txt`. Google Colab may already have many of these packages installed.

## Setup

The first notebook section clones the project repository into `/content` and changes into it:

```bash
git clone https://github.com/2020pyfcrawl/18734-17731_Project_Phase2_3.git
cd 18734-17731_Project_Phase2_3
pip install -r requirements.txt
```

In the notebook, run the initialization/setup cells before running the remaining sections. The notebook uses `/content/18734-17731_Project_Phase2_3` as its repository root and automatically selects CUDA when a GPU is available.

> **Important:** If you are running outside Google Colab, update `ROOT` and any setup paths to match your local checkout. Do not run the notebook's `%cd /content` commands unchanged on a machine that does not use that directory.

## Running the Notebook

Run the notebook cells in order, checking that the expected data and model files are present before starting.

### 1. Configure the phase

Set the `PHASE` variable to the intended mode:

```python
PHASE = "final"  # Options: "train", "val", "final"
```

- **`train`**: loads labels and computes evaluation metrics.
- **`val`** and **`final`**: writes membership scores to `prediction.csv`.

The notebook sets paths for the test input, target model, labels, and output based on this value.

### 2. Prepare data and tokenizer

The notebook reads `data/<PHASE>/test.json`, filters empty text samples, tokenizes the text with the target model's tokenizer, truncates sequences to a maximum length of 512 tokens, and creates a PyTorch `DataLoader`.

### 3. Train or prepare shadow models

The shadow-model section defines a GPT-2/LoRA training pipeline and uses the seeds:

```python
SEEDS = [104, 105, 106, 107]
```

The shadow-model setup uses WikiText-103 text data, filters short or empty samples, and configures LoRA fine-tuning. The notebook's training configuration includes 3 epochs, a per-device batch size of 8, learning rate `2e-4`, LoRA rank 32, LoRA alpha 64, and dropout 0.05.

This step can be computationally expensive. If shadow models have already been trained, use the existing model artifacts instead of retraining them.

### 4. Place shadow-model archives

The notebook expects shadow-model ZIP files under:

```text
models/shadow/
```

The unzipping cell extracts ZIP archives into that directory and **deletes each successfully extracted ZIP file**. Keep a separate backup of the archives if you need to retain them.

### 5. Run the attack

The attack computes per-token losses for the target model and each available shadow model, then uses z-score normalization to generate a membership score for every test sample. Higher scores are intended to indicate a greater likelihood that the sample was included in the target model's training data.

The attack expects LoRA model artifacts that can be loaded on top of the GPT-2 base model. Ensure the target-model directory and shadow-model directories contain the expected files before running this section.

### 6. Evaluate or save predictions

For `PHASE = "train"`, the notebook loads `data/train/test_label.json` and reports:

- ROC AUC
- True positive rate (TPR) at a false positive rate (FPR) below 0.01

For `PHASE = "val"` or `"final"`, it writes one score per line to:

```text
data/<PHASE>/prediction.csv
```

Despite the `.csv` extension, the notebook writes each score as a JSON-encoded floating-point value on its own line, rather than as a comma-separated table.

### 7. Package the submission

The packaging cell creates `project_submission.zip` and includes the available prediction files at:

```text
val/prediction.csv
final/prediction.csv
```

Both phase prediction files must exist for the packaging cell to complete successfully.

## Visualizations

The notebook includes plots for:

- A log-scale ROC curve, emphasizing low false-positive rates
- Member versus non-member score distributions, including a threshold based on the 99th percentile of non-member scores
- A standard ROC curve

The visualization cells that use `y_true`, `scores`, `fpr`, `tpr`, and `auc_score` require those variables to have been created by the training-phase evaluation. They may not run directly after a `val` or `final` run without the corresponding labels and metrics.

## Configuration Reference

| Variable | Purpose |
|---|---|
| `PHASE` | Selects `train`, `val`, or `final` mode |
| `BATCH_SIZE` | Batch size for test-data loading (50) |
| `BLOCK_SIZE` | Maximum tokenized sequence length (512) |
| `DEVICE` | CUDA when available; otherwise CPU |
| `TARGET_MODEL_DIR` | Directory containing the target model's LoRA artifacts |
| `SHADOW_MODELS_ROOT` | Directory containing shadow models |
| `TEST_PATH` | Test input JSON file |
| `LABEL_PATH` | Training-phase ground-truth labels |
| `PREDICTION_PATH` | Output file for membership scores |

## Troubleshooting

- **Missing data/model files:** Verify the repository paths and that the course-provided datasets and model artifacts are in the expected directories.
- **CUDA out-of-memory:** Use a GPU with more memory or reduce the batch size. The notebook's feature extraction processes samples individually inside each batch, which can be slow.
- **No shadow models found:** Confirm that archives were extracted under `models/shadow/` and that model directories match the notebook's `shadow_model` naming filter.
- **Visualization errors in validation/final mode:** The ROC and member/non-member plots require ground-truth labels and evaluation variables, which are only initialized in the training evaluation branch.
- **Submission ZIP fails:** Check that both `data/val/prediction.csv` and `data/final/prediction.csv` exist before packaging.

## Notes

This README describes the workflow implemented in the notebook. Use the dataset, target-model artifacts, and submission requirements provided with the original course repository. The notebook itself does not include the underlying datasets or trained model weights.
