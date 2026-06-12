import json
import os
import random
from collections import defaultdict

from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, WeightedRandomSampler

from src.label_translations import DISEASE_ALIASES, canonical_disease_label


class EyeDataset(Dataset):
    def __init__(self, root_dir, processor, use_suojian=True, augment_rare=True, rare_classes=None,
                 augmentation_factor=10, text_augmentation=True, image_augmentation=True):
        self.root_dir = root_dir
        self.processor = processor
        self.use_suojian = use_suojian
        self.augment_rare = augment_rare
        self.augmentation_factor = augmentation_factor
        self.text_augmentation = text_augmentation
        self.image_augmentation = image_augmentation

        if rare_classes is None:
            self.rare_classes = [
                "you can add here",
            ]
        else:
            self.rare_classes = [canonical_disease_label(item) for item in rare_classes]

        self.original_data = self._load_data()

        self.class_distribution = self._analyze_class_distribution()
        print(f"Original class distribution: {self.class_distribution}")

        self.data = self._prepare_augmented_data() if self.augment_rare else self.original_data
        print(f"Augmented dataset size: {len(self.data)}")

    def _load_data(self):
        data = []
        for num in os.listdir(self.root_dir):
            num_dir = os.path.join(self.root_dir, num)
            if not os.path.isdir(num_dir):
                continue
            for patient_name in os.listdir(num_dir):
                patient_dir = os.path.join(num_dir, patient_name)
                if not os.path.isdir(patient_dir):
                    continue
                for date in os.listdir(patient_dir):
                    date_dir = os.path.join(patient_dir, date)
                    if not os.path.isdir(date_dir):
                        continue
                    for picture_info in os.listdir(date_dir):
                        picture_info_dir = os.path.join(date_dir, picture_info)
                        if not os.path.isdir(picture_info_dir):
                            continue

                        json_file = None
                        image_file = None
                        for file_name in os.listdir(picture_info_dir):
                            if file_name.endswith(".json"):
                                json_file = os.path.join(picture_info_dir, file_name)
                            elif file_name.lower().endswith((".jpg", ".jpeg", ".bmp")):
                                image_file = os.path.join(picture_info_dir, file_name)

                        if json_file and image_file:
                            with open(json_file, "r", encoding="utf-8") as f:
                                metadata = json.load(f)

                            text = metadata["suojian"] if self.use_suojian else metadata["suode"]
                            labels = self._extract_disease_labels(text)

                            data.append({
                                "image_path": image_file,
                                "text": text,
                                "metadata": metadata,
                                "labels": labels,
                            })
        return data

    def _extract_disease_labels(self, text):
        found_labels = []
        for raw_label, english_label in DISEASE_ALIASES.items():
            if raw_label and raw_label in text and english_label not in found_labels:
                found_labels.append(english_label)
        return found_labels

    def _analyze_class_distribution(self):
        class_count = defaultdict(int)

        for item in self.original_data:
            for label in item["labels"]:
                class_count[label] += 1

        return dict(class_count)

    def _prepare_augmented_data(self):
        augmented_data = []
        rare_class_data = defaultdict(list)

        for item in self.original_data:
            augmented_data.append(item)

            for label in item["labels"]:
                if label in self.rare_classes:
                    rare_class_data[label].append(item)

        for rare_class, samples in rare_class_data.items():
            print(f"Augmenting rare class '{rare_class}' ({len(samples)} original samples)")

            num_to_augment = max(0, self.augmentation_factor * len(samples) - len(samples))

            for _ in range(num_to_augment):
                original_sample = random.choice(samples)
                augmented_sample = self._augment_sample(original_sample.copy(), rare_class)
                augmented_data.append(augmented_sample)

        return augmented_data

    def _augment_sample(self, sample, target_class):
        if self.image_augmentation:
            sample["augmented_image"] = True
            sample["original_image_path"] = sample["image_path"]

        if self.text_augmentation:
            original_text = sample["text"]
            augmented_text = self._augment_text(original_text, target_class)
            sample["original_text"] = original_text
            sample["text"] = augmented_text

        return sample

    def _augment_text(self, text, target_class):
        prefix_templates = [
            "Ultrasound shows {}",
            "The examination finds {}",
            "B-scan ultrasound shows {}",
            "{} is visible",
            "The image suggests {}",
            "The finding is consistent with {}",
        ]

        suffix_templates = [
            "requires further observation",
            "follow-up is recommended",
            "the finding is conspicuous",
            "close follow-up is needed",
            "features are typical",
            "the lesion is clearly shown",
        ]

        prefix = random.choice(prefix_templates).format(target_class)
        suffix = random.choice(suffix_templates)

        if target_class in text:
            augmented_text = text.replace(target_class, f"{target_class} ({suffix})")
        else:
            augmented_text = f"{text}; additionally, {prefix}."

        return augmented_text

    def _augment_image(self, image):
        augmentation_type = random.randint(0, 5)

        if augmentation_type == 0:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
        if augmentation_type == 1:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        if augmentation_type == 2:
            factor = random.uniform(0.8, 1.5)
            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(factor)
        if augmentation_type == 3:
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
        if augmentation_type == 4:
            angle = random.uniform(-5, 5)
            return image.rotate(angle, resample=Image.BILINEAR, expand=False)

        w, h = image.size
        scale = random.uniform(0.95, 1.05)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = image.resize((new_w, new_h), Image.BILINEAR)
        left = max(0, (new_w - w) // 2)
        top = max(0, (new_h - h) // 2)
        right = min(new_w, left + w)
        bottom = min(new_h, top + h)
        return resized.crop((left, top, right, bottom))

    def create_sampler(self):
        weights = []

        for item in self.data:
            sample_weight = 1.0
            has_rare = False

            for label in item["labels"]:
                if label in self.rare_classes:
                    has_rare = True
                    break

            if has_rare:
                sample_weight = 2.0

            weights.append(sample_weight)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.data),
            replacement=True,
        )

        return sampler

    def shuffle_data(self):
        random.shuffle(self.data)
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]

        if "augmented_image" in record and record["augmented_image"]:
            image = Image.open(record["original_image_path"]).convert("RGB")
            image = self._augment_image(image)
        else:
            image = Image.open(record["image_path"]).convert("RGB")

        text = record["text"]
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "metadata": record["metadata"],
            "labels": record["labels"],
        }
