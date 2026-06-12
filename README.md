SonoEye: Ocular Ultrasound Foundation Model Utilities

This repository contains PyTorch research code for ocular B-scan ultrasound image-text modeling, image-level disease evaluation, Eye-RADS-style risk mapping, and report generation.

The disease and risk-category terminology follows the project reference paper style:

- normal
- low-vision risk
- high-vision risk
- tumor risk

This code is intended for research and engineering experiments only. It is not a clinical diagnosis system.

Repository Contents

This repository contains entrypoints and modules for:

USCLIP training
Fine-tuning an image-text dual encoder on ocular ultrasound images and report text.

Image-level evaluation
Prototype-based and prompt-assisted Top-k disease prediction with per-class metrics, confusion matrices, Eye-RADS risk voting, and Vitreous Opacity subtype handling.

Report generation
CSV-to-text report generation using English disease labels, tumor subtype labels, and Eye-RADS risk categories.

Custom input pipeline
Dataset loading, report tokenization, and image-text processor wrappers for the USCLIP model.

Environment

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

The pinned requirements include CUDA 12.9 PyTorch builds:

```text
torch==2.8.0+cu129
torchvision==0.23.0+cu129
```

If your CUDA version is different, install the matching PyTorch and torchvision builds from the official PyTorch index, then install the remaining packages from `requirements.txt`.

Data

Training dataset input

`src/data/train/dataset.py` expects a nested folder structure like:

```text
root_dir/
  group_or_batch_id/
    patient_id/
      exam_date/
        image_or_eye_folder/
          image.jpg | image.jpeg | image.bmp
          metadata.json
```


Image-level evaluation input

`img_level_final.py` expects support and query folders where each disease is a directory name:

```text
support_dir/
  Retinal Detachment/
    image_001.jpg
  No Abnormalities/
    image_002.bmp

query_dir/
  Retinal Detachment/
    image_101.jpg
```

Report CSV input

`report_generate.py` expects a CSV with:

```text
patient_id,eye_side,label1,label2,...
```

Optional column:

```text
tumor_classification
```

Tumor subtype values should use English labels such as:

```text
Choroidal Hemangioma
Choroidal Nevus
Choroidal Melanoma
Retinoblastoma
Metastasis
Others
```

File-by-File Usage

`requirements.txt`
Lists the Python dependencies used by the training, evaluation, plotting, and report-generation scripts.

Install with:

```bash
pip install -r requirements.txt
```

`Finetune.py`
Main training entrypoint for fine-tuning `USCLIP` with Hugging Face `Trainer`.

Before running:

- Replace every `local_path` with your actual model, training data, and validation data paths.
- Confirm the dataset import points to the dataset class you want to use.
- Confirm the tokenizer in `src/input_pipeline/Custom_tokenizer.py` loads a real Hugging Face tokenizer.

Run:

```bash
python Finetune.py
```

Main outputs:

```text
output/
logs/
output/best_model/
```

`img_level_final.py`
Image-level evaluation and inference script. It combines prototype matching, disease knowledge prompts, optimized prompts, Top-k metrics, Vitreous Opacity subtype mapping, and Eye-RADS risk voting.

Before running:

- Replace `model_path` and `opacity_model_path`.
- Replace `support_dir` and `query_dir`.
- Replace `opacity_support_dir` and `opacity_query_dir`.
- Make sure support/query folder names exactly match the English disease labels in `candidate_texts`.

Run:

```bash
python img_level_final.py
```

Main outputs:

```text
disease_knowledge.json
optimized_disease_prompts.json
topk_results/
topk_results/metrics_top{k}.json
topk_results/topk_summary.csv
topk_results/per_class_topk_accuracy.csv
topk_results/per_image_top3_predictions.csv
topk_results/confusion_matrix_top{k}.png
topk_results/per_class_accuracy_top{k}.png
topk_results/severity_confusion_matrix_top{k}.png
topk_{k}_prototype_results.csv
topk_{k}_prototype_reasoning_details.json
```

`report_generate.py`
Generates one English text report per patient row from a CSV file.

Before running:

- Replace `csv_file_path`.
- Replace `output_dir`.
- Ensure the CSV includes `patient_id` and `eye_side`.
- Add one or more diagnosis columns named with the `label` prefix, such as `label1`, `label2`, `label3`.

Run:

```bash
python report_generate.py
```

Programmatic usage:

```python
from report_generate import generate_report, process_csv_to_reports

report_text = generate_report(
    patient_id="P001",
    eye_side="OD",
    diagnoses=["Retinal Detachment", "Vitreous Opacity"],
)

process_csv_to_reports("input.csv", "reports")
```

Typical Workflow

Step 1: Install dependencies.

```bash
pip install -r requirements.txt
```

Step 2: Prepare model paths and data paths.

Replace all `local_path` placeholders in:

```text
Finetune.py
img_level_final.py
report_generate.py
```

Step 3: Train or load a USCLIP model.

```bash
python Finetune.py
```

Step 4: Run image-level evaluation.

```bash
python img_level_final.py
```

Step 5: Generate English reports from a CSV file.

```bash
python report_generate.py
```

Known Setup Notes

- This repository is research code and currently uses direct script-level path variables rather than command-line arguments.
- Several scripts contain `local_path` placeholders that must be replaced before running.
- `EyeReportTokenizer` must be connected to a real Hugging Face tokenizer before fresh training.

License

AGPL-3.0 License: See the LICENSE file for more details.

Copyright 2026 IMMU Lab (for modified portions)

Disclaimer

This repository contains research code provided "AS IS" without warranty of any kind. By using this code, you expressly agree that:

1.Not a Medical Device: This code is not intended for use as a medical device under any jurisdiction.

2.Non-Clinical Use: This code shall not be used for clinical diagnosis, treatment, triage, or prognosis.

3.Clinical Review Required: Any generated prediction or report must be reviewed by a qualified ophthalmologist.

Liability Notice: The authors, copyright holders, and contributors are not liable for clinical, legal, economic, or other consequences arising from use of this code.
