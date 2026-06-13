import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoProcessor, AutoConfig, AutoModel, ChineseCLIPTextConfig, ChineseCLIPTextModel
from src.input_pipeline.Custom_processor import EyeReportProcessor
from src.model.usclip_swinv2_cnbert import USCLIP
from PIL import Image
import os
import torch
from collections import defaultdict, Counter
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
plt.rcParams['font.sans-serif'] = ['SimHei']  # Font fallback.
plt.rcParams['axes.unicode_minus'] = False  # Render minus signs normally.
from statsmodels.stats.proportion import proportion_confint

def compute_binomial_ci(success, n, alpha=0.05, method='wilson'):
    lower, upper = proportion_confint(success, n, alpha=alpha, method=method)
    return lower, upper

def topk_accuracy(predictions, y_true, k=1):
    # Get the top-k prediction indices.
    top_k_pred = np.argsort(predictions, axis=1)[:, -k:]

    # Check whether the true label is in the top-k predictions.
    correct = 0
    for i, true_label_idx in enumerate(y_true):
        if true_label_idx in top_k_pred[i]:
            correct += 1

    return correct / len(y_true)


def mean_reciprocal_rank(predictions, y_true):
    mrr = 0
    for i, true_label_idx in enumerate(y_true):
        # Rank labels by descending probability.
        ranked_indices = np.argsort(predictions[i])[::-1]
        # Find the true-label rank, using one-based ranks.
        rank = np.where(ranked_indices == true_label_idx)[0][0] + 1
        mrr += 1.0 / rank

    return mrr / len(y_true)


def plot_confusion_matrix(cm, labels, k, save_path=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix (Top-{k})', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def compute_per_class_topk_accuracy(predictions, y_true, candidate_texts, k=1):
    per_class_accuracy = {}
    per_class_counts = {}

    # Initialize per-class counters.
    for i, label in enumerate(candidate_texts):
        per_class_accuracy[label] = 0.0
        per_class_counts[label] = 0

    # Count accuracy for each class.
    for true_idx in range(len(candidate_texts)):
        # Find all samples belonging to the current class.
        class_mask = (y_true == true_idx)
        class_samples = np.where(class_mask)[0]

        if len(class_samples) == 0:
            continue

        class_label = candidate_texts[true_idx]
        per_class_counts[class_label] = len(class_samples)

        # Compute top-k accuracy for this class.
        correct_count = 0
        for sample_idx in class_samples:
            # Get this sample's top-k predictions.
            top_k_pred = np.argsort(predictions[sample_idx])[-k:]
            if true_idx in top_k_pred:
                correct_count += 1

        per_class_accuracy[class_label] = correct_count / len(class_samples)

    return per_class_accuracy, per_class_counts


def plot_per_class_accuracy(per_class_accuracy, per_class_counts, k, save_path=None):
    # Prepare plotting data.
    labels = list(per_class_accuracy.keys())
    accuracies = list(per_class_accuracy.values())
    counts = [per_class_counts[label] for label in labels]

    # Create the figure.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Accuracy bar chart.
    bars1 = ax1.bar(range(len(labels)), accuracies, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Disease class')
    ax1.set_ylabel(f'Top-{k} accuracy')
    ax1.set_title(f'Per-disease Top-{k} accuracy')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)

    # Add value labels to the bars.
    for i, (bar, acc, count) in enumerate(zip(bars1, accuracies, counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{acc:.3f}\n(n={count})',
                 ha='center', va='bottom', fontsize=8)

    # Sample-count bar chart.
    bars2 = ax2.bar(range(len(labels)), counts, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Disease class')
    ax2.set_ylabel('Sample count')
    ax2.set_title('Sample Count Distribution by Disease Class')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Add value labels to the bars.
    for bar, count in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{count}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def compute_topk_metrics(predictions, y_true_single, candidate_texts, k=1):
    # Top-k accuracy.
    topk_acc = topk_accuracy(predictions, y_true_single, k)

    # MRR
    mrr = mean_reciprocal_rank(predictions, y_true_single)

    # Convert top-k predictions to a binary matrix for precision, recall, and F1.
    y_pred_topk = np.zeros_like(predictions)
    for i in range(predictions.shape[0]):
        top_k_indices = np.argsort(predictions[i])[-k:]
        y_pred_topk[i, top_k_indices] = 1

    # Convert single labels to a multilabel format for metric calculation.
    y_true_multilabel = np.zeros_like(predictions)
    for i, label_idx in enumerate(y_true_single):
        if 0 <= label_idx < len(candidate_texts):
            y_true_multilabel[i, label_idx] = 1

    # Compute macro- and micro-averaged metrics.
    precision_macro = precision_score(y_true_multilabel, y_pred_topk, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_multilabel, y_pred_topk, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_multilabel, y_pred_topk, average='macro', zero_division=0)

    precision_micro = precision_score(y_true_multilabel, y_pred_topk, average='micro', zero_division=0)
    recall_micro = recall_score(y_true_multilabel, y_pred_topk, average='micro', zero_division=0)
    f1_micro = f1_score(y_true_multilabel, y_pred_topk, average='micro', zero_division=0)

    # Compute the confusion matrix from the highest-probability prediction.
    y_pred_single = np.argmax(predictions, axis=1)
    cm = confusion_matrix(y_true_single, y_pred_single, labels=range(len(candidate_texts)))

    return {
        'top_k_accuracy': topk_acc,
        'mrr': mrr,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'confusion_matrix': cm
    }


def plot_metrics_comparison(metrics_by_k, save_path=None):
    k_values = sorted(metrics_by_k.keys())

    # Extract metrics.
    metrics_names = ['top_k_accuracy', 'mrr', 'precision_macro', 'recall_macro', 'f1_macro']
    metrics_labels = ['Top-k Accuracy', 'MRR', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (metric_name, metric_label) in enumerate(zip(metrics_names, metrics_labels)):
        values = [metrics_by_k[k][metric_name] for k in k_values]
        axes[i].plot(k_values, values, marker='o', linewidth=2, markersize=6)
        axes[i].set_title(metric_label, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('k')
        axes[i].set_ylabel(metric_label)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(k_values)

    # Hide the unused subplot.
    axes[-1].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()



def map_to_severity_category_voting(diseases, disease_categories, scores=None):
    if not diseases:
        return "normal"

    # Count votes for each risk category.
    category_votes = defaultdict(float)

    for i, disease in enumerate(diseases):
        # Find the category for this disease.
        for category, category_diseases in disease_categories.items():
            if disease in category_diseases:
                # Use weighted votes when scores are available.
                weight = scores[i] if scores is not None else 1.0
                category_votes[category] += weight
                break

    # If no category matches, return "normal".
    if not category_votes:
        return "normal"

    # Return the highest-vote category.
    return max(category_votes.items(), key=lambda x: x[1])[0]


def binary_classification_metrics(y_prob, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    auc = roc_auc_score(y_true, y_prob)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "accuracy": accuracy,
        "auc": auc,
        "sen": sensitivity,
        "spe": specificity,
    }


def topk_classification_test(predictions, y_true, candidate_texts, k=3):
    num_classes = len(candidate_texts)
    binary_metrics = {}

    # For top-k evaluation, redefine y_pred as a multilabel matrix.
    y_pred = np.zeros_like(predictions)

    # Select the top-k predictions for each sample.
    for i in range(predictions.shape[0]):
        # Get top-k indices.
        top_k_indices = np.argsort(predictions[i])[-k:]
        y_pred[i, top_k_indices] = 1

    for i, candidate_label in enumerate(candidate_texts):
        metrics = binary_classification_metrics(predictions[:, i], y_true[:, i], y_pred[:, i])
        binary_metrics[candidate_label] = metrics

    return binary_metrics


def get_topk_predictions(predictions, candidate_texts, k=3):
    topk_results = []

    for i in range(predictions.shape[0]):
        # Get the current sample's predicted probabilities.
        sample_probs = predictions[i]

        # Get the top-k indices, disease names, and scores.
        top_k_indices = np.argsort(sample_probs)[-k:][::-1]  # Descending order.

        sample_topk = []
        for idx in top_k_indices:
            disease = candidate_texts[idx]
            score = sample_probs[idx]
            sample_topk.append((disease, score))

        topk_results.append(sample_topk)

    return topk_results


def extract_features(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")

    # Get the model device.
    device = next(model.parameters()).device

    # Preprocess and move inputs to the correct device.
    inputs = processor(images=image, return_tensors="pt")
    if isinstance(inputs["pixel_values"], list):
        inputs["pixel_values"] = inputs["pixel_values"][0]

    # Add a batch dimension when the tensor is 3D.
    if inputs["pixel_values"].ndim == 3:
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run the image encoder.
    outputs = model.get_image_features(**{k: v for k, v in inputs.items() if k in ['pixel_values']})

    with torch.no_grad():
        outputs = model.get_image_features(**{k: v for k, v in inputs.items() if k in ['pixel_values']})
        return outputs.cpu().numpy()


# Load the disease knowledge base.
def load_disease_knowledge_base(knowledge_path="disease_knowledge.json"):
    if not os.path.exists(knowledge_path):
        # Create a default knowledge base when none exists. You can add more in knowledge base
        knowledge_base = {
            "Retinal Detachment": {
                "findings": ["V-shaped or Y-shaped band-like strong echo in the vitreous cavity connected to the optic disc; tent-like band-like echo near the globe wall adhesion"],
                "key_features": [""],
                "differential_points": [""]
            },
        }

        # Save the default knowledge base as JSON.
        with open(knowledge_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=4)
    else:
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)

    return knowledge_base


def normalize(scores):
    values = list(scores.values())

    # Handle empty scores or all-zero scores.
    if len(values) == 0 or sum(values) == 0:
        return {disease: 1.0 / len(scores) for disease in scores}

    # Compute the total score.
    total = sum(values)

    # Normalize scores.
    normalized_scores = {
        disease: score / total
        for disease, score in scores.items()
    }

    return normalized_scores


def compute_prototypes(support_images, support_labels, model, processor, candidate_texts, prototypes):
    from sklearn.cluster import KMeans
    device = next(model.parameters()).device
    features_by_class = defaultdict(list)
    image_paths_by_class = defaultdict(list)

    # Extract features for all support-set images.
    print("Extracting support-set features...")
    for i, (img_path, label) in enumerate(zip(support_images, support_labels)):
        if i % 10 == 0:
            print(f"Processing support-set image {i + 1}/{len(support_images)}")

        if label in candidate_texts:
            features = extract_features(model, processor, img_path)
            features_by_class[label].append(features[0])
            image_paths_by_class[label].append(img_path)
    n_prototypes_per_class = {}
    for label in candidate_texts:
        n_prototypes_per_class[label] = prototypes[label]
    # Compute multiple prototypes for each class.
    prototypes = {}
    prototype_source_images = {}  # Store source images for debugging and interpretability.

    for label, features in features_by_class.items():
        if not features:
            continue

        features_array = np.array(features)
        # Use all samples as prototypes when the sample count is small.
        if len(features) <= n_prototypes_per_class[label]:
            prototypes[label] = features_array
            prototype_source_images[label] = image_paths_by_class[label]
        else:
            # Use KMeans to generate multiple prototypes.
            kmeans = KMeans(n_clusters=n_prototypes_per_class[label], random_state=42)
            kmeans.fit(features_array)

            # Use cluster centers as prototypes.
            prototypes[label] = kmeans.cluster_centers_

            # Find the real sample closest to each cluster center.
            closest_samples = []
            for cluster_idx in range(n_prototypes_per_class[label]):
                cluster_samples = features_array[kmeans.labels_ == cluster_idx]
                cluster_sample_indices = np.where(kmeans.labels_ == cluster_idx)[0]

                if len(cluster_samples) > 0:
                    # Compute distances to the cluster center.
                    distances = np.linalg.norm(cluster_samples - kmeans.cluster_centers_[cluster_idx], axis=1)
                    closest_idx = cluster_sample_indices[np.argmin(distances)]
                    closest_samples.append(image_paths_by_class[label][closest_idx])

            prototype_source_images[label] = closest_samples

    print("Prototype source images:")
    for label, sources in prototype_source_images.items():
        print(f"  Class '{label}': {len(sources)} prototypes")
        for i, src in enumerate(sources):
            print(f"    Prototype {i + 1}: {os.path.basename(src)}")

    return prototypes


def compute_prototypes_mean(support_images, support_labels, model, processor, candidate_texts):
    features_by_class = defaultdict(list)

    # Extract features for all support-set images.
    print("Extracting support-set features...")
    for i, (img_path, label) in enumerate(zip(support_images, support_labels)):
        if i % 10 == 0:
            print(f"Processing support-set image {i + 1}/{len(support_images)}")

        if label in candidate_texts:  # Only process candidate classes.
            features = extract_features(model, processor, img_path)
            features_by_class[label].append(features[0])

    # Compute class prototypes.
    prototypes = {}
    for label, features in features_by_class.items():
        if features:  # Ensure this class has samples.
            prototypes[label] = np.mean(features, axis=0)

    return prototypes


def optimize_prompts_for_classes(knowledge_base, model, processor, gold_standard_images):
    best_prompts = {}
    device = next(model.parameters()).device  # Get the model device.
    model.eval()

    for disease in tqdm(knowledge_base.keys(), desc="Optimizing disease prompts"):
        if disease not in gold_standard_images or not gold_standard_images[disease]:
            print(f"Skipping {disease}: no gold-standard images available")
            continue

        # Generate candidate prompts for this disease.
        candidate_prompts = generate_candidate_prompts(disease, knowledge_base)
        if not candidate_prompts:
            print(f"Skipping {disease}: no candidate prompts available")
            continue

        # Process all gold-standard images for this disease.
        images = gold_standard_images[disease]
        image_features_list = []

        for image in images:
            try:
                # Create image inputs and move them to the correct device.
                image_inputs = processor(images=image, return_tensors="pt")
                if isinstance(image_inputs["pixel_values"], list):
                    image_inputs["pixel_values"] = image_inputs["pixel_values"][0]

                # Add a batch dimension when the tensor is 3D.
                if image_inputs["pixel_values"].ndim == 3:
                    image_inputs["pixel_values"] = image_inputs["pixel_values"].unsqueeze(0)

                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

                with torch.no_grad():
                    image_features = model.get_image_features(
                        **{k: v for k, v in image_inputs.items() if k in ['pixel_values']})
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    image_features_list.append(image_features)
            except Exception as e:
                print(f"Error while processing image: {e}")
                continue

        if not image_features_list:
            print(f"Skipping {disease}: no images could be processed")
            continue

        # Compute the average similarity score for each candidate prompt.
        prompt_avg_scores = {}

        for prompt in candidate_prompts:
            try:
                # Process text inputs and move them to the correct device.
                text_inputs = processor(text=[prompt], padding=True, return_tensors="pt")
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                with torch.no_grad():
                    text_features_inputs = {k: v for k, v in text_inputs.items() if
                                            k in ['input_ids', 'attention_mask']}
                    text_features = model.get_text_features(**text_features_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    # Compute the average similarity between this prompt and all images.
                    similarities = []
                    for img_feat in image_features_list:
                        similarity = torch.matmul(img_feat, text_features.transpose(0, 1)).item()
                        similarities.append(similarity)

                    prompt_avg_scores[prompt] = sum(similarities) / len(similarities)
            except Exception as e:
                print(f"Error while processing prompt '{prompt}' failed: {e}")
                continue

        if not prompt_avg_scores:
            print(f"Skipping {disease}: could not score any prompt")
            continue

        # Select the prompt with the highest average score.
        best_prompt = max(prompt_avg_scores.items(), key=lambda x: x[1])[0]
        best_score = prompt_avg_scores[best_prompt]
        best_prompts[disease] = best_prompt

        print(f"Disease: {disease}")
        print(f"Best prompt: {best_prompt}")
        print(f"Best score: {best_score:.4f}")
        print("-" * 50)

    return best_prompts


def generate_candidate_prompts(disease, knowledge_base):
    """Generate candidate prompts for one disease."""
    prompts = []

    if disease not in knowledge_base:
        return prompts

    # 1. Direct finding prompts.
    if knowledge_base[disease].get('findings'):
        for desc in knowledge_base[disease]['findings']:
            feature_prompt = f"Ophthalmic ultrasound image shows: {desc}"
            prompts.append(feature_prompt)

        # Combine all findings.
        all_desc_prompt = f"Ophthalmic ultrasound image shows: {', '.join(knowledge_base[disease]['findings'])}"
        prompts.append(all_desc_prompt)

    # 2. Key-feature prompts.
    key_features = knowledge_base[disease].get('key_features', [])
    if key_features:
        for feature in key_features:
            key_prompt = f"In ophthalmic ultrasound, {disease} key features: {feature}"
            prompts.append(key_prompt)

        # Combine all key features.
        all_key_prompt = f"In ophthalmic ultrasound, {disease} key features: {', '.join(key_features)}"
        prompts.append(all_key_prompt)

    # 3. Confirmation prompt.
    confirm_prompt = f"This ophthalmic ultrasound image shows {disease}"
    prompts.append(confirm_prompt)

    # 4. Differential-point prompts.
    if knowledge_base[disease].get('differential_points'):
        for point in knowledge_base[disease]['differential_points']:
            diff_prompt = f"In ophthalmic ultrasound, {disease} differential points: {point}"
            prompts.append(diff_prompt)

        # Combine all differential points.
        all_diff_prompt = f"In ophthalmic ultrasound, {disease} differential points: {', '.join(knowledge_base[disease]['differential_points'])}"
        prompts.append(all_diff_prompt)

    # 5. Combined prompt.
    if knowledge_base[disease].get('findings'):
        combined_prompt = f"{disease} ophthalmic ultrasound features: {', '.join(knowledge_base[disease]['findings'])}"
        prompts.append(combined_prompt)

    # 6. Hybrid prompt.
    if key_features and knowledge_base[disease].get('findings'):
        hybrid_prompt = f"{disease}: {', '.join(knowledge_base[disease]['findings'])}; key features: {', '.join(key_features)}"
        prompts.append(hybrid_prompt)

    return prompts


def load_gold_standard_images(base_dir, candidate_texts):
    gold_standard_images = {}

    # Iterate over all disease-class directories.
    for class_dir in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_dir)

        # Keep only directories that are candidate classes.
        if not os.path.isdir(class_path) or class_dir not in candidate_texts:
            continue

        disease_name = class_dir  # The directory name is the disease name.
        gold_standard_images[disease_name] = []

        # Load all images for this disease.
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)

            # Keep image files only.
            if os.path.isfile(img_path) and img_path.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                try:
                    # Load the image.
                    image = Image.open(img_path).convert('RGB')
                    gold_standard_images[disease_name].append(image)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return gold_standard_images


def compute_distances(query_feature, prototypes):
    distances = {}
    prototype_indices = {}  # Store the selected prototype index for each class.

    for label, class_prototypes in prototypes.items():
        if len(class_prototypes.shape) == 1:  # Single-prototype case.
            # Use cosine similarity and convert it to distance: 1 - similarity.
            similarity = np.dot(query_feature, class_prototypes) / (
                    np.linalg.norm(query_feature) * np.linalg.norm(class_prototypes))
            distance = 1 - similarity
            distances[label] = distance
            prototype_indices[label] = 0
        else:  # Multi-prototype case.
            # Compute cosine similarity to each prototype.
            similarities = []
            for prototype in class_prototypes:
                similarity = np.dot(query_feature, prototype) / (
                        np.linalg.norm(query_feature) * np.linalg.norm(prototype))
                similarities.append(similarity)

            # Select the most similar prototype.
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            distances[label] = 1 - best_similarity
            prototype_indices[label] = best_idx

    return distances, prototype_indices


def distance_to_probability(distances, temperature=0.1):
    # Convert distances to similarities; smaller distance means higher similarity.
    similarities = {k: np.exp(-v / temperature) for k, v in distances.items()}

    # Normalize to probabilities.
    total = sum(similarities.values())
    probabilities = {k: v / total for k, v in similarities.items()}

    return probabilities


def load_images_from_directory(base_dir, candidate_texts):
    # Get all label folders.
    label_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Store image paths and labels.
    images = []
    labels = []
    image_names = []

    # Iterate over label folders.
    for folder in label_folders:
        # Get the label name.
        label = folder
        folder_path = os.path.join(base_dir, folder)

        # Iterate over images in the folder.
        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".bmp")):
                img_path = os.path.join(folder_path, file)
                images.append(img_path)
                labels.append(label)
                image_names.append(file)

    return images, labels, image_names


def select_diverse_support_set(support_images, support_labels, candidate_texts, n_per_class=2):
    selected_images = []
    selected_labels = []

    # Try to get n_per_class examples for each candidate disease
    for disease in candidate_texts:
        count = 0
        for i, label in enumerate(support_labels):
            if label == disease and count < n_per_class:
                selected_images.append(support_images[i])
                selected_labels.append(label)
                count += 1

    return selected_images, selected_labels


# Hybrid reasoning that combines prototype matching with image-text similarity.
def hybrid_reasoning(image_feature, prototypes, knowledge_base, candidate_texts, model, processor, best_prompts=None):
    # Part 1: prototype-based similarity analysis.
    distances, prototype_indices = compute_distances(image_feature, prototypes)
    prototype_probabilities = distance_to_probability(distances, temperature=0.1)

    # Part 2: image-text similarity analysis with optimized prompts.
    device = next(model.parameters()).device
    image_feature_tensor = torch.from_numpy(image_feature).to(device)
    normalized_image_feature = image_feature_tensor / image_feature_tensor.norm(dim=-1, keepdim=True)

    text_similarity_scores = {}
    prompt_details = {}  # Store the prompt used and its score.

    for disease in candidate_texts:
        if disease not in knowledge_base:
            text_similarity_scores[disease] = 0.0
            continue

        # Use an optimized prompt if available; otherwise generate default prompts.
        if best_prompts and disease in best_prompts:
            prompts = [best_prompts[disease]]
            prompt_details[disease] = {"type": "optimized", "prompt": best_prompts[disease]}
        else:
            # Build multiple prompts for each disease.
            prompts = []

            # 1. Direct finding prompt.
            if knowledge_base[disease]['findings']:
                feature_prompt = f"Ophthalmic ultrasound image shows: {', '.join(knowledge_base[disease]['findings'])}"
                prompts.append(feature_prompt)

            # 2. Key-feature prompt.
            key_features = knowledge_base[disease].get('key_features', [''])
            if key_features[0]:
                key_prompt = f"In ophthalmic ultrasound, {disease} key features: {', '.join(key_features)}"
                prompts.append(key_prompt)

            # 3. Confirmation prompt.
            confirm_prompt = f"This ophthalmic ultrasound image shows {disease}"
            prompts.append(confirm_prompt)

            prompt_details[disease] = {"type": "default", "prompts": prompts}

        # Compute the similarity score for each prompt.
        prompt_scores = []

        for prompt in prompts:
            text_inputs = processor(text=[prompt], padding=True, return_tensors="pt")
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            with torch.no_grad():
                text_features = model.get_text_features(
                    **{k: v for k, v in text_inputs.items() if k in ['input_ids', 'attention_mask']})
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = torch.matmul(normalized_image_feature, text_features.transpose(0, 1)).item()
                prompt_scores.append(similarity)

        # Use the average as the text-similarity score.
        text_similarity_scores[disease] = sum(prompt_scores) / len(prompt_scores) if prompt_scores else 0
        if "prompts" in prompt_details[disease]:
            prompt_details[disease]["scores"] = prompt_scores

    # Avoid zero totals.
    if sum(text_similarity_scores.values()) > 0:
        total = sum(text_similarity_scores.values())
        text_probabilities = {k: v / total for k, v in text_similarity_scores.items()}
    else:
        text_probabilities = {k: 1.0 / len(text_similarity_scores) for k in text_similarity_scores}

    # Part 3: combine both scoring paths.
    final_scores = {}
    for disease in candidate_texts:
        prototype_score = prototype_probabilities.get(disease, 0)
        text_score = text_probabilities.get(disease, 0)

        # Adjust weights.
        prototype_confidence = 1.0 - min(distances.values()) if distances else 0
        prototype_weight = 0.7 if prototype_confidence > 0.3 else 0.5
        prompt_weight = 1.0 - prototype_weight

        # Weighted fusion.
        final_scores[disease] = prototype_weight * prototype_score + prompt_weight * text_score

    # Normalize final scores.
    final_scores = normalize(final_scores)

    # Sort by final score.
    final_ranking = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    final_disease, final_score = final_ranking[0]

    # Build the reasoning payload.
    reasoning_process = {
        "image_analysis": [],
        "reasoning_process": [],
        "differential_diagnoses": [],
        "final_diagnosis": "",
        "prototype_reference": {},  # Store which prototype was used.
        "prompt_info": prompt_details  # Store prompt metadata.
    }

    # Add image analysis from the top-ranked diseases.
    top_features = set()
    for disease, _ in final_ranking[:2]:
        if disease in knowledge_base and 'key_features' in knowledge_base[disease]:
            for feature in knowledge_base[disease]['key_features']:
                if feature:
                    top_features.add(feature)

    if top_features:
        reasoning_process["image_analysis"].append(f"Primary imaging features: {', '.join(list(top_features)[:3])}")
    else:
        reasoning_process["image_analysis"].append("Imaging features are unclear")

    # Store the prototype index used for each disease.
    for disease in candidate_texts:
        if disease in prototype_indices:
            reasoning_process["prototype_reference"][disease] = int(prototype_indices[disease])

    # Add detailed reasoning for the top three diseases.
    for disease, score in final_ranking[:3]:
        if disease in knowledge_base:
            # Get prototype and text-similarity scores.
            prototype_score = prototype_probabilities.get(disease, 0)
            text_score = text_similarity_scores.get(disease, 0)

            # Get the selected prototype index.
            prototype_idx = prototype_indices.get(disease, 0)

            reasoning = f"Consider {disease} probability({score:.2f}):"
            reasoning += f"\n- Feature similarity: {prototype_score:.2f} (Prototype variant #{prototype_idx + 1})"
            reasoning += f"\n- Text similarity: {text_score:.2f}"

            # Add prompt metadata.
            if disease in prompt_details:
                if prompt_details[disease]["type"] == "optimized":
                    reasoning += f"\n  Using optimized prompt: \"{prompt_details[disease]['prompt']}\""
                else:
                    reasoning += f"\n  Using default prompt set ({len(prompt_details[disease]['prompts'])} prompts)"

            # Add key-feature analysis.
            if "key_features" in knowledge_base[disease] and knowledge_base[disease]["key_features"][0]:
                feature_match = []
                for feature in knowledge_base[disease]["key_features"]:
                    # Infer the match degree from the final score.
                    match_degree = "high match" if score > 0.2 else "partial match" if score > 0.08 else "no match"
                    feature_match.append(f"- {feature}:{match_degree}")

                reasoning += "\nKey feature analysis:\n" + "\n".join(feature_match)

            # Add differential points.
            if "differential_points" in knowledge_base[disease] and knowledge_base[disease]["differential_points"][0]:
                reasoning += f"\ndifferential_points: {knowledge_base[disease]['differential_points'][0]}"

            reasoning_process["reasoning_process"].append(reasoning)
            reasoning_process["differential_diagnoses"].append(f"{disease} (confidence: {score:.2f})")

    # Update the final diagnosis.
    reasoning_process["final_diagnosis"] = f"{final_disease} (final confidence: {final_score:.2f})"

    # Build confidence scores for all candidate diseases.
    confidence_scores = np.zeros(len(candidate_texts))
    for i, disease in enumerate(candidate_texts):
        if disease in final_scores:
            confidence_scores[i] = final_scores[disease]

    return final_disease, reasoning_process, confidence_scores, prototype_probabilities, text_probabilities
def build_mapped_topk_for_sample(pred_row, candidate_texts, vitreous_idx, choose_label_fn, sample_idx, k):
    top_k_indices = np.argsort(pred_row)[-k:][::-1]
    out = []
    for idx in top_k_indices:
        if idx == vitreous_idx:
            mapped_label = choose_label_fn(sample_idx)
            out.append((mapped_label, float(pred_row[idx])))
        else:
            out.append((candidate_texts[idx], float(pred_row[idx])))
    return out

def build_full_rank_labels_mapped(pred_row, candidate_texts, vitreous_idx, choose_label_fn, sample_idx):
    order = np.argsort(pred_row)[::-1]
    labels = []
    for idx in order:
        if idx == vitreous_idx:
            labels.append(choose_label_fn(sample_idx))
        else:
            labels.append(candidate_texts[idx])
    return labels

# Main program flow.
def main():
    # Register custom configuration classes.
    AutoConfig.register('chinese_clip_text_model', ChineseCLIPTextConfig)
    AutoModel.register(ChineseCLIPTextConfig, ChineseCLIPTextModel)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Configure paths.
    model_path = "local_path"
    opacity_model_path = "local_path"
    model = USCLIP.from_pretrained(model_path).to(device)
    opacity_model = USCLIP.from_pretrained(opacity_model_path).to(device)
    processor = EyeReportProcessor.from_pretrained(model_path)

    # Define candidate labels.
    candidate_texts = [
        "Retinal Detachment",
        "Retinal Tear",
        "Choroidal Detachment",
        "Intraocular Tumor",
        "Globe Wall Abnormality",
        "Vitreous Opacity",
        "No Abnormalities",
        "Posterior Vitreous Detachment",
        "Posterior Staphyloma",
        "Optic Disc Edema",
        "Silicone Oil",
        "Phthisis Bulbi",
        "Choroidal Defect",
        "Suprachoroidal Hemorrhage",
        "Optic Disc Calcification",
        "Intraocular Foreign Body",
        "Lens Dislocation",
        "Asteroid Hyalosis",
    ]
    # Support-set and query-set directories.
    support_dir = "local_path"  # Support-set directory.
    query_dir = "local_path"  # Query-set directory.

    # Load the support and query sets.
    support_images, support_labels, support_image_names = load_images_from_directory(support_dir, candidate_texts)
    query_images, query_labels, query_image_names = load_images_from_directory(query_dir, candidate_texts)

    print(f"Support set size: {len(support_images)} images")
    print(f"Query set size: {len(query_images)} images")

    # Count support-set samples for each class.
    support_class_counts = {}
    for label in support_labels:
        if label in candidate_texts:
            support_class_counts[label] = support_class_counts.get(label, 0) + 1

    print("Support-set class distribution:")
    for label, count in support_class_counts.items():
        print(f"  {label}: {count} images")

    n_prototypes_per_class = {}
    for label, count in support_class_counts.items():
            n_prototypes_per_class[label] = 3  # Use three prototypes.

    print("Prototype count per class:")
    for label, n_proto in n_prototypes_per_class.items():
        print(f"  {label}: {n_proto}prototypes")

    # Compute class prototypes with the multi-prototype method.
    prototypes = compute_prototypes(support_images, support_labels, model, processor, candidate_texts,
                                    n_prototypes_per_class)
    print(f"Created {len(prototypes)} class prototypes")

    # Load the disease knowledge base.
    knowledge_base = load_disease_knowledge_base("disease_knowledge.json")
    print(f"Loaded disease knowledge base with {len(knowledge_base)} diseases")

    # Load gold-standard images for prompt optimization.
    print("Loading gold-standard images for prompt optimization...")
    gold_standard_images = {}
    for disease in candidate_texts:
        disease_images = [Image.open(img).convert('RGB') for img, label in zip(support_images, support_labels) if
                          label == disease]
        if disease_images:
            gold_standard_images[disease] = disease_images
            print(f"  {disease}: found {len(disease_images)} gold-standard images")

    # Optimize prompts for each disease.
    print("Optimizing disease prompts...")
    best_prompts = optimize_prompts_for_classes(knowledge_base, model, processor, gold_standard_images)
    print(f"Successfully optimized prompts for {len(best_prompts)} diseases")

    # Save optimized prompts.
    import json
    with open("optimized_disease_prompts.json", 'w', encoding='utf-8') as f:
        json.dump(best_prompts, f, ensure_ascii=False, indent=4)
    print("Optimized prompts saved to optimized_disease_prompts.json")

    # Evaluate the query set.
    print("Evaluating query set...")
    predictions = np.zeros((len(query_images), len(candidate_texts)))
    results_mapping = {}  # Store prediction results for each image.
    reasoning_results = {}  # Store the reasoning payload for each image.

    for i, img_path in enumerate(query_images):
        if i % 10 == 0:
            print(f"Processing query-set image {i + 1}/{len(query_images)}")

        # Extract query-image features.
        query_feature = extract_features(model, processor, img_path)[0]

        # Use hybrid reasoning with optimized prompts.
        predicted_disease, reasoning_process, confidence_scores, prototype_scores, text_scores = hybrid_reasoning(
            query_feature, prototypes, knowledge_base, candidate_texts, model, processor, best_prompts
        )

        # Save inference results.
        img_name = query_image_names[i]
        true_label = query_labels[i]

        # Save predicted probabilities.
        predictions[i] = confidence_scores

        # Store the prediction result, true label, and reasoning payload for this image.
        results_mapping[img_name] = {
            "path": img_path,
            "true_label": true_label,
            "predicted_label": predicted_disease,
            "predicted_probs": confidence_scores,
            "prototype_scores": prototype_scores,  # Save prototype scores
            "text_scores": text_scores  # Save text scores
        }

        reasoning_results[img_name] = reasoning_process

    y_true_single = np.array([candidate_texts.index(label) if label in candidate_texts else -1
                              for label in query_labels])

    # Remove samples outside the candidate labels.
    valid_indices = y_true_single >= 0
    y_true_single = y_true_single[valid_indices]
    predictions_filtered = predictions[valid_indices]
    query_images_filtered = [img for i, img in enumerate(query_images) if valid_indices[i]]
    query_labels_filtered = [label for i, label in enumerate(query_labels) if valid_indices[i]]
    query_image_names_filtered = [name for i, name in enumerate(query_image_names) if valid_indices[i]]

    print(f"Valid sample count: {len(y_true_single)}")

    # Store results for different k values.
    metrics_by_k = {}
    k_values = [1, 3, 5, 7, 9]  # Adjust as needed.

    # Create the output directory.
    results_dir = "topk_results"
    os.makedirs(results_dir, exist_ok=True)
    # ========= Vitreous Opacity subtype preprocessing; run once before the k loop. =========
    opacity_types = ["Marked Vitreous Opacity", "Mild Vitreous Opacity"]
    opacity_support_dir = "local_path"
    opacity_query_dir = "local_path"

    # Subtype support and query sets.
    opacity_support_images, opacity_support_labels, opacity_support_image_names = load_images_from_directory(
        opacity_support_dir, opacity_types)
    opacity_query_images, opacity_query_labels, opacity_query_image_names = load_images_from_directory(
        opacity_query_dir, opacity_types)

    # Mean subtype prototypes.
    opacity_prototypes = compute_prototypes_mean(
        opacity_support_images, opacity_support_labels, opacity_model, processor, opacity_types
    )

    # Image name to subtype ground truth, used to replace true "Vitreous Opacity" labels.
    imgname2opacity = {}
    for img_name, label in zip(opacity_query_image_names, opacity_query_labels):
        if label in opacity_types:
            imgname2opacity[img_name] = label

    # Index of "Vitreous Opacity" in the main label list and the report label list.
    assert "Vitreous Opacity" in candidate_texts
    vitreous_idx = candidate_texts.index("Vitreous Opacity")
    candidate_texts_report = (
            candidate_texts[:vitreous_idx]
            + ["Marked Vitreous Opacity", "Mild Vitreous Opacity"]
            + candidate_texts[vitreous_idx + 1:]
    )
    print("Report label list:", candidate_texts_report)

    # Precompute subtype probabilities for filtered query samples for reuse.
    opacity_probs_all = np.zeros((len(query_images_filtered), 2), dtype=np.float32)  # [:,0]=marked, [:,1]=mild
    for i, img_path in enumerate(query_images_filtered):
        q_feat = extract_features(opacity_model, processor, img_path)[0]
        distances, _ = compute_distances(q_feat, opacity_prototypes)
        probs = distance_to_probability(distances, temperature=0.1)
        opacity_probs_all[i, 0] = float(probs.get("Marked Vitreous Opacity", 0.5))
        opacity_probs_all[i, 1] = float(probs.get("Mild Vitreous Opacity", 0.5))

    # Choose the subtype from subtype probabilities.
    def choose_opacity_label_for_sample(sample_idx: int) -> str:
        return "Marked Vitreous Opacity" if opacity_probs_all[sample_idx, 0] >= 0.4 else "Mild Vitreous Opacity"

    # Build sample validity and ground truth for report evaluation after subtype mapping.
    report_valid_mask = np.zeros(len(query_labels_filtered), dtype=bool)
    true_labels_mapped = [None] * len(query_labels_filtered)
    for i, (lbl, img_name) in enumerate(zip(query_labels_filtered, query_image_names_filtered)):
        if lbl == "Vitreous Opacity":
            # Evaluate only samples with subtype ground truth.
            if img_name in imgname2opacity:
                true_labels_mapped[i] = imgname2opacity[img_name]
                report_valid_mask[i] = True
            else:
                report_valid_mask[i] = False
        else:
            true_labels_mapped[i] = lbl
            report_valid_mask[i] = True

    valid_indices_report = np.where(report_valid_mask)[0]
    print(f"Subtype report-evaluation valid sample count: {len(valid_indices_report)} / {len(query_labels_filtered)}")
    topk_results_by_k = {}

    for k in k_values:
        print(f"\n==== Top-{k} classification results(original ranking with subtype label replacement) ====")

        # Build mapped Top-k lists, Top-1 predictions, and ground-truth labels for valid samples.
        mapped_topk_lists = []
        y_true_labels_eval = []
        y_pred_top1_labels = []

        for r, i in enumerate(valid_indices_report):
            pred_row = predictions_filtered[i]
            # Top-k ranking from the original classes; replace only the "Vitreous Opacity" label.
            mapped_topk = build_mapped_topk_for_sample(
                pred_row, candidate_texts, vitreous_idx, choose_opacity_label_for_sample, i, k
            )
            mapped_topk_lists.append(mapped_topk)

            # Top-1 predicted_label(mapped)
            y_pred_top1_labels.append(mapped_topk[0][0])

            # Ground truth after replacing "Vitreous Opacity" with its subtype.
            y_true_labels_eval.append(true_labels_mapped[i])

        # Compute Top-k Accuracy.
        correct = 0
        for t_lbl, topk_list in zip(y_true_labels_eval, mapped_topk_lists):
            pred_labels = [l for (l, s) in topk_list]
            if t_lbl in pred_labels:
                correct += 1
        topk_acc = correct / len(y_true_labels_eval)

        # Confidence interval.
        n_samples = len(y_true_labels_eval)
        n_correct = int(round(topk_acc * n_samples))
        ci_low, ci_up = compute_binomial_ci(n_correct, n_samples, alpha=0.05, method='wilson')

        # Compute MRR with the fully ranked mapped label sequence.
        mrr_sum = 0.0
        for r, i in enumerate(valid_indices_report):
            pred_row = predictions_filtered[i]
            full_rank_labels = build_full_rank_labels_mapped(
                pred_row, candidate_texts, vitreous_idx, choose_opacity_label_for_sample, i
            )
            true_lbl = true_labels_mapped[i]
            # Find the one-based rank.
            try:
                rank = full_rank_labels.index(true_lbl) + 1
                mrr_sum += 1.0 / rank
            except ValueError:
                # This should not happen when the true label is in the candidate label set.
                pass
        mrr = mrr_sum / n_samples if n_samples > 0 else 0.0

        # Per-class Top-k accuracy based on mapped classes.
        per_class_accuracy = {cls: 0.0 for cls in candidate_texts_report}
        per_class_counts = {cls: 0 for cls in candidate_texts_report}
        per_class_corrects = {cls: 0 for cls in candidate_texts_report}
        for t_lbl, topk_list in zip(y_true_labels_eval, mapped_topk_lists):
            if t_lbl not in per_class_counts:  # Should not happen when report labels are complete.
                continue
            per_class_counts[t_lbl] += 1
            if t_lbl in [l for (l, s) in topk_list]:
                per_class_accuracy[t_lbl] += 1
                per_class_corrects[t_lbl] += 1
        for cls in per_class_accuracy:
            if per_class_counts[cls] > 0:
                per_class_accuracy[cls] /= per_class_counts[cls]
        # Compute a 95% Wilson confidence interval for each class accuracy.
        per_class_accuracy_ci = {}
        for cls in candidate_texts_report:
            cnt = per_class_counts[cls]
            cor = per_class_corrects[cls]
            if cnt > 0:
                low, up = compute_binomial_ci(cor, cnt, alpha=0.05, method='wilson')
                per_class_accuracy_ci[cls] = [float(low), float(up)]
            else:
                per_class_accuracy_ci[cls] = [None, None]

        # Build multilabel matrices for P/R/F1.
        label_to_idx_report = {lbl: idx for idx, lbl in enumerate(candidate_texts_report)}
        y_true_bin = np.zeros((n_samples, len(candidate_texts_report)), dtype=int)
        y_pred_topk_bin = np.zeros_like(y_true_bin)
        for i, (t_lbl, topk_list) in enumerate(zip(y_true_labels_eval, mapped_topk_lists)):
            y_true_bin[i, label_to_idx_report[t_lbl]] = 1
            for (lbl, _) in topk_list:
                y_pred_topk_bin[i, label_to_idx_report[lbl]] = 1

        precision_macro = precision_score(y_true_bin, y_pred_topk_bin, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_bin, y_pred_topk_bin, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_bin, y_pred_topk_bin, average='macro', zero_division=0)
        precision_micro = precision_score(y_true_bin, y_pred_topk_bin, average='micro', zero_division=0)
        recall_micro = recall_score(y_true_bin, y_pred_topk_bin, average='micro', zero_division=0)
        f1_micro = f1_score(y_true_bin, y_pred_topk_bin, average='micro', zero_division=0)

        # Confusion matrix from mapped Top-1 labels.
        cm = confusion_matrix(y_true_labels_eval, y_pred_top1_labels, labels=candidate_texts_report)

        # Print metrics.
        print(f"Top-{k} accuracy: {topk_acc:.4f} (95%CI: {ci_low:.4f}-{ci_up:.4f})")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print(f"precision (macro average): {precision_macro:.4f}")
        print(f"recall (macro average): {recall_macro:.4f}")
        print(f"F1score (macro average): {f1_macro:.4f}")
        print(f"precision (micro average): {precision_micro:.4f}")
        print(f"recall (micro average): {recall_micro:.4f}")
        print(f"F1score (micro average): {f1_micro:.4f}")

        # Per-class Top-k performance.
        print(f'\n==== Per-disease Top-{k} accuracy (excluding "Vitreous Opacity", including two subtypes) ====')
        for cls, acc in sorted(per_class_accuracy.items(), key=lambda x: x[1], reverse=True):
            cnt = per_class_counts[cls]
            if cnt > 0:
                ci = per_class_accuracy_ci.get(cls)
                print(f"{cls}: {acc:.4f} (95%CI: {ci[0]:.4f}-{ci[1]:.4f}, {cnt}samples)")
            else:
                print(f"{cls}: no test samples")
        valid_class_acc = [acc for cls, acc in per_class_accuracy.items() if per_class_counts[cls] > 0]
        if valid_class_acc:
            print("\nClass accuracy summary:")
            print(f"  mean accuracy: {np.mean(valid_class_acc):.4f}")
            print(f"  max accuracy: {np.max(valid_class_acc):.4f}")
            print(f"  min accuracy: {np.min(valid_class_acc):.4f}")
            print(f"  standard deviation: {np.std(valid_class_acc):.4f}")

        # Save plots using candidate_texts_report labels.
        per_class_acc_save_path = os.path.join(results_dir, f"per_class_accuracy_top{k}.png")
        plot_per_class_accuracy(per_class_accuracy, per_class_counts, k, per_class_acc_save_path)
        cm_save_path = os.path.join(results_dir, f"confusion_matrix_top{k}.png")
        plot_confusion_matrix(cm, candidate_texts_report, k, cm_save_path)

        # Save mapped Top-k lists for later risk-category mapping and export.
        topk_results_by_k[k] = mapped_topk_lists

        # ====== Eye-RADS risk categories with subtype labels. ======
        disease_categories = {
            "normal": ["No Abnormalities", "Asteroid Hyalosis"],
            "low-vision risk": ["Posterior Vitreous Detachment", "Posterior Staphyloma", "Optic Disc Calcification", "Silicone Oil", "Mild Vitreous Opacity"],
            "high-vision risk": ["Retinal Detachment", "Choroidal Detachment", "Phthisis Bulbi", "Choroidal Defect", "Suprachoroidal Hemorrhage",
                         "Intraocular Foreign Body", "Lens Dislocation", "Globe Wall Abnormality", "Optic Disc Edema", "Marked Vitreous Opacity", "Retinal Tear"],
            "tumor risk": ["Intraocular Tumor"]
        }

        severity_true, severity_pred = [], []
        # mapped_topk_lists and y_true_labels_eval contain only valid report-evaluation samples.
        for t_lbl, topk_list in zip(y_true_labels_eval, mapped_topk_lists):
            pred_diseases = [d for (d, s) in topk_list]
            pred_scores = [s for (d, s) in topk_list]
            true_severity = map_to_severity_category_voting([t_lbl], disease_categories)
            pred_severity = map_to_severity_category_voting(pred_diseases, disease_categories, pred_scores)
            severity_true.append(true_severity)
            severity_pred.append(pred_severity)

        severity_categories = ["normal", "low-vision risk", "high-vision risk", "tumor risk"]
        severity_cm = confusion_matrix(severity_true, severity_pred, labels=severity_categories)
        severity_kappa = cohen_kappa_score(severity_true, severity_pred)

        print("\n==== Eye-RADS risk voting results (after subtype mapping) ====")
        print("Confusion matrix:")
        print(severity_cm)
        print(f"\nCohen's Kappa: {severity_kappa:.4f}")

        severity_cm_save_path = os.path.join(results_dir, f"severity_confusion_matrix_top{k}.png")
        plot_confusion_matrix(severity_cm, severity_categories, k, severity_cm_save_path)

        # Save detailed results for this k; labels, confusion matrix, and per-class accuracy use report labels.
        k_results = {
            'k': k,
            'metrics': {
                'top_k_accuracy': float(topk_acc),
                'top_k_accuracy_ci': [float(ci_low), float(ci_up)],
                'mrr': float(mrr),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_micro': float(precision_micro),
                'recall_micro': float(recall_micro),
                'f1_micro': float(f1_micro),
            },
            'confusion_matrix': cm.tolist(),
            # Per-class report metrics.
            'per_class_accuracy': {cls: float(acc) for cls, acc in per_class_accuracy.items()},
            'per_class_counts': per_class_counts,
            'per_class_accuracy_ci': per_class_accuracy_ci,
            'severity_confusion_matrix': severity_cm.tolist(),
            'severity_kappa': float(severity_kappa),
            'labels': candidate_texts_report
        }
        # Store this k's metrics for comparison plots and summary tables.
        metrics_by_k[k] = k_results['metrics']

        # Export per-image mapped Top-3 predictions for k == 3.
        if k == 3:
            top3_rows = []
            # mapped_topk_lists aligns with y_true_labels_eval and contains valid samples only.
            for img_idx_in_valid, (t_lbl, topk_list) in enumerate(zip(y_true_labels_eval, mapped_topk_lists)):
                # Recover the original filtered-sample index.
                global_idx = valid_indices_report[img_idx_in_valid]
                img_name = query_image_names_filtered[global_idx]
                img_path = query_images_filtered[global_idx]

                # Build Top-3 fields, using fewer entries if fewer are available.
                topn = min(3, len(topk_list))
                pred_labels = [topk_list[j][0] for j in range(topn)]
                pred_scores = [topk_list[j][1] for j in range(topn)]

                row = {
                    "image_name": img_name,
                    "image_path": img_path,
                    "true_label_mapped": t_lbl,
                    "Top-1label": pred_labels[0] if topn >= 1 else "",
                    "Top-1score": round(pred_scores[0], 6) if topn >= 1 else "",
                    "Top-2label": pred_labels[1] if topn >= 2 else "",
                    "Top-2score": round(pred_scores[1], 6) if topn >= 2 else "",
                    "Top-3label": pred_labels[2] if topn >= 3 else "",
                    "Top-3score": round(pred_scores[2], 6) if topn >= 3 else "",
                    "Top1hit": int(pred_labels[0] == t_lbl) if topn >= 1 else 0
                }
                top3_rows.append(row)

            top3_df = pd.DataFrame(top3_rows)
            top3_csv_path = os.path.join(results_dir, "per_image_top3_predictions.csv")
            top3_df.to_csv(top3_csv_path, index=False, encoding="utf-8-sig")
            print(f"Per-image Top-3 predictions saved to: {top3_csv_path}")
        with open(os.path.join(results_dir, f"metrics_top{k}.json"), 'w', encoding='utf-8') as f:
            json.dump(k_results, f, ensure_ascii=False, indent=2)

    # Plot metric comparisons.
    metrics_comparison_path = os.path.join(results_dir, "metrics_comparison.png")
    plot_metrics_comparison(metrics_by_k, metrics_comparison_path)

    # Save summary results for all k values.
    summary_results = {}
    for k, metrics in metrics_by_k.items():
        summary_results[f'top_{k}'] = {
            'top_k_accuracy': metrics['top_k_accuracy'],
            'top_k_accuracy_ci_low': metrics['top_k_accuracy_ci'][0],
            'top_k_accuracy_ci_up': metrics['top_k_accuracy_ci'][1],
            'mrr': metrics['mrr'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro'],
            'precision_micro': metrics['precision_micro'],
            'recall_micro': metrics['recall_micro'],
            'f1_micro': metrics['f1_micro']
        }

    # Save as CSV for inspection.
    summary_df = pd.DataFrame(summary_results).T
    summary_df.to_csv(os.path.join(results_dir, "topk_summary.csv"))
    print(f"\nSummary results saved to: {os.path.join(results_dir, 'topk_summary.csv')}")

    per_class_summary = {}
    for k in k_values:
        # Read per-class accuracy from saved results.
        with open(os.path.join(results_dir, f"metrics_top{k}.json"), 'r', encoding='utf-8') as f:
            k_data = json.load(f)
            per_class_summary[f'top_{k}'] = k_data['per_class_accuracy']

    # Save as CSV.
    per_class_df = pd.DataFrame(per_class_summary).fillna(0)
    per_class_df.to_csv(os.path.join(results_dir, "per_class_topk_accuracy.csv"))
    print(f"\nPer-class accuracy summary saved to: {os.path.join(results_dir, 'per_class_topk_accuracy.csv')}")
    print(f"\n==== Class Performance Analysis ====")
    for k in [1, 3, 5]:  # Analyze the main k values.
        if f'top_{k}' in per_class_summary:
            k_accuracies = per_class_summary[f'top_{k}']
            valid_accuracies = {cls: acc for cls, acc in k_accuracies.items() if acc > 0}

            if valid_accuracies:
                best_class = max(valid_accuracies.items(), key=lambda x: x[1])
                worst_class = min(valid_accuracies.items(), key=lambda x: x[1])

                print(f"Top-{k}:")
                print(f"  Best class: {best_class[0]} ({best_class[1]:.4f})")
                print(f"  Worst class: {worst_class[0]} ({worst_class[1]:.4f})")
    # Save results to a CSV file.
    results_df = []
    for i, (img_name, result) in enumerate(results_mapping.items()):
        sample_topk = topk_results_by_k[i]
        pred_label = result["predicted_label"]
        pred_score = np.max(result["predicted_probs"])

        # Get prototype-variant metadata.
        proto_variant = ""
        if 'prototype_reference' in reasoning_results[img_name] and pred_label in reasoning_results[img_name]['prototype_reference']:
            proto_idx = reasoning_results[img_name]['prototype_reference'][pred_label]
            proto_variant = f"Prototype variant #{proto_idx + 1}"

        # Get prompt metadata.
        prompt_info = "Default prompt"
        if 'prompt_info' in reasoning_results[img_name] and pred_label in reasoning_results[img_name]['prompt_info']:
            info = reasoning_results[img_name]['prompt_info'][pred_label]
            if info["type"] == "optimized":
                prompt_info = f"Optimized prompt: {info['prompt'][:30]}..."

        # Build the top-k prediction string.
        topk_str = "; ".join([f"{disease}({score:.3f})" for disease, score in sample_topk])

        row = {
            "image_name": img_name,
            "true_label": result["true_label"],
            "predicted_label": pred_label,
            "confidence": pred_score,
            f"Top-{k} predictions": topk_str,
            "prototype_variant_used": proto_variant,
            "prompt_used": prompt_info,
            "image_analysis": "; ".join(reasoning_results[img_name]["image_analysis"][:2]),  # Use the first two analyses.
            "reasoning_process": reasoning_results[img_name]["reasoning_process"][0] if reasoning_results[img_name]["reasoning_process"] else ""
        }
        results_df.append(row)

    results_csv = pd.DataFrame(results_df)
    results_csv.to_csv(f"topk_{k}_prototype_results.csv", index=False, encoding="utf-8-sig")
    print(f"Prediction results saved to topk_{k}_prototype_results.csv")

    # Save detailed reasoning to a JSON file.
    with open(f"topk_{k}_prototype_reasoning_details.json", "w", encoding="utf-8") as f:
        json.dump(reasoning_results, f, ensure_ascii=False, indent=2)
    print(f"Detailed reasoning saved to topk_{k}_prototype_reasoning_details.json")


if __name__ == "__main__":
    main()
