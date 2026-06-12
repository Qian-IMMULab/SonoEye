import re
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from src.label_translations import canonical_disease_label


class EyeReportTokenizer:
    def __init__(self, base_tokenizer_path: str = None, max_length: int = 128):
        # TODO: replace this placeholder with AutoTokenizer.from_pretrained(base_tokenizer_path).
        self.base_tokenizer = 'locan_ChineseCLIP_path'
        self.max_length = max_length
        self.original_vocab_size = len(self.base_tokenizer)
        self.disease_list = [
            "Retinal Detachment",
            "Retinal Tear",
            "Choroidal Detachment",
            "Intraocular Tumor",
            "Globe Wall Edema",
            "Vitreous Opacity",
            "No Abnormalities",
            "Posterior Vitreous Detachment",
            "Posterior Staphyloma",
            "Optic Disc Edema",
            "Globe Wall Calcification",
            "Silicone Oil",
            "Phthisis Bulbi",
            "Choroidal Defect",
            "Suprachoroidal Hemorrhage",
            "Optic Disc Calcification",
            "Intraocular Foreign Body",
            "Lens Dislocation",
            "Retinal Edema or Organization",
            "Subretinal Hemorrhage or Exudation",
        ]

        self.disease_priority = {
            "Retinal Detachment": ["Vitreous Opacity", "Posterior Vitreous Detachment"],
            "Retinal Tear": ["Vitreous Opacity"],
            "Retinal Edema or Organization": ["Vitreous Opacity"],
        }

        self.subordinate_diseases = {}
        for high_priority, low_priorities in self.disease_priority.items():
            for disease in low_priorities:
                self.subordinate_diseases.setdefault(disease, set()).add(high_priority)

        self.special_patterns = {
            "punctate_echo": re.compile(
                r'(small amount|moderate amount|large amount|small-to-moderate|moderate-to-large) punctate echo',
                re.IGNORECASE,
            ),
            "band_echo": re.compile(r'(thin|strong|moderate|uneven) band echo', re.IGNORECASE),
            "aftermovement": re.compile(r'aftermovement (obvious|poor|moderate)', re.IGNORECASE),
        }

        self.special_token_map = {
            "[small amount]": "small amount",
            "[moderate amount]": "moderate amount",
            "[large amount]": "large amount",
            "[small-to-moderate]": "small to moderate amount",
            "[moderate-to-large]": "moderate to large amount",
            "[thin]": "thin",
            "[strong]": "strong",
            "[moderate]": "moderate",
            "[uneven]": "uneven",
            "[obvious]": "obvious",
            "[mild]": "mild",
            "[poor]": "poor",
            "[anterior segment]": "anterior segment",
            "[middle segment]": "middle segment",
            "[posterior segment]": "posterior segment",
            "[anterior-middle segment]": "anterior-middle segment",
            "[middle-posterior segment]": "middle-posterior segment",
            "[entire segment]": "entire segment",
        }

        for disease in self.disease_list:
            self.special_token_map[f"[{disease}]"] = disease

    def preprocess_text(self, text: str) -> str:
        for pattern_name, pattern in self.special_patterns.items():
            if pattern_name == "punctate_echo":
                matches = pattern.finditer(text)
                for match in matches:
                    quantity = match.group(1)
                    replacement = f"[{quantity}] punctate echo"
                    text = text.replace(match.group(0), replacement)

            elif pattern_name == "band_echo":
                matches = pattern.finditer(text)
                for match in matches:
                    characteristic = match.group(1)
                    replacement = f"[{characteristic}] band echo"
                    text = text.replace(match.group(0), replacement)

            elif pattern_name == "aftermovement":
                matches = pattern.finditer(text)
                for match in matches:
                    degree = match.group(1)
                    replacement = f"aftermovement [{degree}]"
                    text = text.replace(match.group(0), replacement)

        position_patterns = {
            "anterior segment": re.compile(r'(vitreous|eye)? anterior segment', re.IGNORECASE),
            "middle segment": re.compile(r'(vitreous|eye)? middle segment', re.IGNORECASE),
            "posterior segment": re.compile(r'(vitreous|eye)? posterior segment', re.IGNORECASE),
            "anterior-middle segment": re.compile(r'(vitreous|eye)? anterior-middle segment', re.IGNORECASE),
            "middle-posterior segment": re.compile(r'(vitreous|eye)? middle-posterior segment', re.IGNORECASE),
            "entire segment": re.compile(r'(vitreous|eye) entire segment', re.IGNORECASE),
        }

        for pos_name, pos_pattern in position_patterns.items():
            matches = pos_pattern.finditer(text)
            for match in matches:
                prefix = match.group(1) if match.group(1) else ""
                replacement = f"{prefix}[{pos_name}]"
                text = text.replace(match.group(0), replacement)

        return text

    def postprocess_tokens(self, tokens):
        for special_token, replacement in self.special_token_map.items():
            tokens = tokens.replace(special_token, replacement)
        return tokens

    def extract_diseases(self, text: str) -> List[str]:
        canonical_text = canonical_disease_label(text)
        found_diseases = []
        for disease in self.disease_list:
            if disease in canonical_text or disease in text:
                found_diseases.append(disease)

        for raw_label in list(found_diseases):
            label = canonical_disease_label(raw_label)
            if label not in found_diseases:
                found_diseases.append(label)

        return found_diseases

    def tokenize(self, text):
        processed_text = self.preprocess_text(text)
        diseases = self.extract_diseases(processed_text)
        safe_text = self.postprocess_tokens(processed_text)

        encoding = self.base_tokenizer(
            safe_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        result = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'diseases': diseases,
        }

        if 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids']

        return result

    def batch_tokenize(self, texts: List[str]) -> Dict:
        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []
        all_diseases = []

        for text in texts:
            result = self.tokenize(text)
            all_input_ids.append(result['input_ids'])
            all_attention_masks.append(result['attention_mask'])
            if 'token_type_ids' in result:
                all_token_type_ids.append(result['token_type_ids'])
            all_diseases.append(result['diseases'])

        batch_result = {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_masks, dim=0),
            'diseases': all_diseases,
        }

        if all_token_type_ids:
            batch_result['token_type_ids'] = torch.cat(all_token_type_ids, dim=0)

        return batch_result

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=True)

    def save_pretrained(self, save_path: str):
        self.base_tokenizer.save_pretrained(save_path)

        import json
        import os

        extra_info = {
            'disease_list': self.disease_list,
            'disease_priority': self.disease_priority,
            'special_tokens': self.special_token_map,
        }

        with open(os.path.join(save_path, 'eye_tokenizer_extra_info.json'), 'w', encoding='utf-8') as f:
            json.dump(extra_info, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, load_path: str, max_length: int = 128):
        import json
        import os

        base_tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True)

        instance = cls(base_tokenizer_path=None, max_length=max_length)
        instance.base_tokenizer = base_tokenizer

        extra_info_path = os.path.join(load_path, 'eye_tokenizer_extra_info.json')
        if os.path.exists(extra_info_path):
            with open(extra_info_path, 'r', encoding='utf-8') as f:
                extra_info = json.load(f)

            instance.disease_list = [canonical_disease_label(item) for item in extra_info['disease_list']]
            instance.disease_priority = {
                canonical_disease_label(key): [canonical_disease_label(item) for item in value]
                for key, value in extra_info['disease_priority'].items()
            }
            instance.special_tokens = extra_info['special_tokens']

            instance.subordinate_diseases = {}
            for high_priority, low_priorities in instance.disease_priority.items():
                for disease in low_priorities:
                    instance.subordinate_diseases.setdefault(disease, set()).add(high_priority)

        return instance

    def __call__(self, text_or_texts, *args, **kwargs):
        if isinstance(text_or_texts, str):
            return self.tokenize(text_or_texts)
        if isinstance(text_or_texts, list):
            return self.batch_tokenize(text_or_texts)
        raise ValueError(f"Unsupported input type: {type(text_or_texts)}")
