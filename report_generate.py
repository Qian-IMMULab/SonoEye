import os

import pandas as pd

from src.label_translations import (
    canonical_disease_label,
    canonical_risk_label,
    canonical_tumor_class,
    canonical_tumor_type,
)


DISEASE_CLASSIFICATION = {
    "normal": ["No Abnormalities", "Asteroid Hyalosis"],
    "low-vision risk": [
        "Posterior Vitreous Detachment",
        "Posterior Staphyloma",
        "Optic Disc Calcification",
        "Silicone Oil",
        "Mild Vitreous Opacity",
    ],
    "high-vision risk": [
        "Retinal Detachment",
        "Choroidal Detachment",
        "Phthisis Bulbi",
        "Choroidal Defect",
        "Suprachoroidal Hemorrhage",
        "Intraocular Foreign Body",
        "Lens Dislocation",
        "Globe Wall Abnormality",
        "Optic Disc Edema",
        "Retinal Tear",
        "Marked Vitreous Opacity",
    ],
    "tumor risk": ["Intraocular Tumor"],
}

DISEASE_SEVERITY_SCORE = {
    "Intraocular Tumor": 1,
    "Phthisis Bulbi": 2,
    "Silicone Oil": 3,
    "Retinal Detachment": 4,
    "Choroidal Detachment": 4,
    "Suprachoroidal Hemorrhage": 4,
    "Choroidal Defect": 5,
    "Intraocular Foreign Body": 5,
    "Lens Dislocation": 5,
    "Retinal Tear": 5,
    "Optic Disc Edema": 5,
    "Marked Vitreous Opacity": 6,
    "Mild Vitreous Opacity": 7,
    "Globe Wall Abnormality": 7,
    "Posterior Vitreous Detachment": 7,
    "Posterior Staphyloma": 7,
    "Optic Disc Calcification": 7,
    "No Abnormalities": 8,
    "Asteroid Hyalosis": 8,
}

URGENCY_RECOMMENDATIONS = {
    "normal": "Routine follow-up is recommended.",
    "low-vision risk": "Elective ophthalmology consultation is recommended for further evaluation.",
    "high-vision risk": "Prompt ophthalmology evaluation is recommended.",
    "tumor risk": "Prompt ophthalmology evaluation is recommended to characterize the lesion.",
}

TUMOR_CLASSIFICATION = {
    "Benign": ["Choroidal Hemangioma", "Choroidal Nevus", "Others"],
    "Malignant": ["Choroidal Melanoma", "Retinoblastoma", "Metastasis"],
}


def get_tumor_malignancy(tumor_type):
    """Return the benign or malignant class for a tumor subtype."""
    if not tumor_type or pd.isna(tumor_type):
        return "Unknown"

    tumor_type_str = canonical_tumor_type(tumor_type)

    for malignancy, tumor_list in TUMOR_CLASSIFICATION.items():
        if tumor_type_str in tumor_list:
            return malignancy

    return "Unknown"


def generate_tumor_recommendation(tumor_type):
    """Generate a clinical recommendation for an intraocular tumor finding."""
    if not tumor_type or pd.isna(tumor_type):
        return (
            "The image suggests an intraocular mass lesion. Further ophthalmology "
            "evaluation is needed to confirm the diagnosis, characterize tumor "
            "type, and guide timely treatment."
        )

    tumor_type_str = canonical_tumor_type(tumor_type)
    malignancy = canonical_tumor_class(get_tumor_malignancy(tumor_type_str))

    base_text = f"The image suggests a possible tumor subtype of {tumor_type_str}"

    if malignancy == "Malignant":
        return (
            f"{base_text}. This is classified as malignant, and prompt ophthalmology "
            "evaluation plus systemic workup is recommended."
        )
    if malignancy == "Benign":
        return (
            f"{base_text}. This is classified as benign, and ophthalmology evaluation "
            "is recommended to determine observation or treatment."
        )
    return (
        f"{base_text}. Further ophthalmology evaluation is needed to characterize "
        "tumor type and treatment priority."
    )

# you can add more descriptions
DISEASE_DESCRIPTIONS = {
    "No Abnormalities": {
        "description": "Ophthalmic ultrasound shows no obvious posterior segment abnormality, but this does not exclude all ocular disease.",
        "recommendation": "Routine follow-up is reasonable if asymptomatic. Seek ophthalmology care if vision loss, visual field defect, redness, or pain occurs.",
    },
}


def get_disease_classification(diagnosis):
    """Return the Eye-RADS category for a diagnosis."""
    diagnosis = canonical_disease_label(diagnosis)
    for level, diseases in DISEASE_CLASSIFICATION.items():
        if diagnosis in diseases:
            return level
    return "unclassified"


def get_severity_priority(diagnosis_list):
    """Sort diagnoses by severity priority."""
    diagnosis_with_priority = []
    for diagnosis in diagnosis_list:
        diagnosis = canonical_disease_label(diagnosis)
        classification = get_disease_classification(diagnosis)
        severity_score = DISEASE_SEVERITY_SCORE.get(diagnosis, 999)
        diagnosis_with_priority.append((diagnosis, severity_score, classification))

    diagnosis_with_priority.sort(key=lambda x: x[1])
    return diagnosis_with_priority


def generate_report(patient_id, eye_side, diagnoses, tumor_classification=None):
    """Generate one English report."""
    valid_diagnoses = [canonical_disease_label(d) for d in diagnoses if str(d).strip()]
    if not valid_diagnoses:
        return None

    sorted_diagnoses = get_severity_priority(valid_diagnoses)
    highest_classification = sorted_diagnoses[0][2] if sorted_diagnoses else "unclassified"
    highest_classification = canonical_risk_label(highest_classification)

    report = [
        f"Patient ID: {patient_id}",
        f"Eye Side: {eye_side}",
        "",
        "Diagnosis - ultrasound findings by severity:",
    ]

    for i, (diagnosis, _, _) in enumerate(sorted_diagnoses, 1):
        report.append(f"  {i}. {diagnosis}")

    report.extend([
        "",
        "Classification - Eye-RADS risk stratification:",
        f"  Eye-RADS category: {highest_classification}",
        f"  Urgency: {URGENCY_RECOMMENDATIONS.get(highest_classification, 'Urgency recommendation is not configured.')}",
        "",
        "Description & Recommendations",
    ])

    for i, (diagnosis, _, _) in enumerate(sorted_diagnoses, 1):
        disease_info = DISEASE_DESCRIPTIONS.get(
            diagnosis,
            {
                "description": "Disease description is not configured.",
                "recommendation": "Clinical recommendation is not configured.",
            },
        )
        if diagnosis == "Intraocular Tumor":
            recommendation = generate_tumor_recommendation(tumor_classification)
        else:
            recommendation = disease_info["recommendation"]

        report.extend([
            f"  {i}. {diagnosis}:",
            f"    {disease_info['description']}",
            f"    {recommendation}",
        ])

    report.extend([
        "",
        "Disclaimer:",
        "  This report is an imaging interpretation of ocular B-scan ultrasound, not a clinical diagnosis.",
        "  Please share this report with an ophthalmologist for further evaluation and treatment.",
        "  The report is AI-assisted and is for reference only; final diagnosis and treatment decisions should be made by a qualified ophthalmologist.",
    ])

    return "\n".join(report)


def process_csv_to_reports(csv_file_path, output_dir):
    """Generate one text report per patient row from a CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        required_columns = ['patient_id', 'eye_side']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: CSV file is missing required column: {col}")
                return

        label_columns = [col for col in df.columns if col.startswith('label') and col != 'eye_side']
        has_tumor_classification = 'tumor_classification' in df.columns

        for _, row in df.iterrows():
            patient_id = row['patient_id']
            eye_side = row['eye_side']

            diagnoses = [str(row[col]).strip() for col in label_columns if pd.notna(row[col]) and str(row[col]).strip()]
            tumor_cls = row['tumor_classification'] if has_tumor_classification else None

            report_text = generate_report(patient_id, eye_side, diagnoses, tumor_cls)
            if report_text:
                filename = f"{patient_id}_{eye_side}.txt"
                file_path = os.path.join(output_dir, filename)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)

        print(f"Successfully generated {len(df)} individual reports in: {output_dir}")

    except Exception as e:
        print(f"Error while processing file: {e}")


if __name__ == "__main__":
    csv_file_path = "local_path"
    output_dir = "local_path"
    process_csv_to_reports(csv_file_path, output_dir)
