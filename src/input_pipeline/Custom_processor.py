from .Custom_tokenizer import EyeReportTokenizer


class EyeReportProcessor:
    def __init__(self, image_processor=None, tokenizer=None):
        self.image_processor = image_processor
        self.tokenizer = tokenizer if tokenizer else EyeReportTokenizer()

    def __call__(self, images=None, text=None, **kwargs):
        result = {}

        # Image processing.
        if images is not None and self.image_processor is not None:
            image_kwargs = kwargs.copy()
            for param in ['padding', 'truncation', 'add_special_tokens', 'max_length', 'return_tensors']:
                image_kwargs.pop(param, None)
            result.update(self.image_processor(images=images, **image_kwargs))

        # Text processing.
        if text is not None:
            text_kwargs = kwargs.copy()
            text_kwargs.pop('images', None)  # Keep image data out of the tokenizer.
            text_inputs = self.tokenizer(text, **text_kwargs)
            result.update(text_inputs)

        # Remove arguments incompatible with the model.
        allowed_keys = ['input_ids', 'pixel_values', 'attention_mask', 'position_ids',
                        'return_loss', 'token_type_ids', 'output_attentions',
                        'output_hidden_states', 'return_dict']
        result = {k: v for k, v in result.items() if k in allowed_keys}

        return result

    def save_pretrained(self, save_directory):
        import os
        import json

        # Ensure the directory exists.
        os.makedirs(save_directory, exist_ok=True)

        # Save processor configuration.
        processor_config = {
            "processor_class": self.__class__.__name__,
            "has_image_processor": self.image_processor is not None,
            "has_tokenizer": self.tokenizer is not None
        }

        with open(os.path.join(save_directory, "processor_config.json"), "w", encoding="utf-8") as f:
            json.dump(processor_config, f, ensure_ascii=False, indent=2)

        # Save the image processor when present.
        if self.image_processor is not None:
            image_processor_dir = os.path.join(save_directory, "image_processor")
            os.makedirs(image_processor_dir, exist_ok=True)

            # Prefer the image processor's native save_pretrained method.
            if hasattr(self.image_processor, "save_pretrained"):
                self.image_processor.save_pretrained(image_processor_dir)
            else:
                # Fall back to pickle when save_pretrained is unavailable.
                import pickle
                with open(os.path.join(image_processor_dir, "image_processor.pkl"), "wb") as f:
                    pickle.dump(self.image_processor, f)

        # Save tokenizer.
        if self.tokenizer is not None:
            tokenizer_dir = os.path.join(save_directory, "tokenizer")
            os.makedirs(tokenizer_dir, exist_ok=True)

            # Prefer the tokenizer's native save_pretrained method.
            if hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(tokenizer_dir)
            else:
                # Fall back to pickle when save_pretrained is unavailable.
                import pickle
                with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "wb") as f:
                    pickle.dump(self.tokenizer, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        import os
        import json

        # Resolve the load path.
        if os.path.isdir(pretrained_model_name_or_path):
            model_path = pretrained_model_name_or_path
        else:
            # Treat non-directory input as a Hugging Face model name.
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(pretrained_model_name_or_path)

        # Load processor configuration.
        config_path = os.path.join(model_path, "processor_config.json")

        if not os.path.exists(config_path):
            raise ValueError(
                f"Could not find processor_config.json in {model_path}; "
                "this is not a valid EyeReportProcessor directory"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            processor_config = json.load(f)

        # Load the image processor when present.
        image_processor = None
        if processor_config.get("has_image_processor", False):
            image_processor_dir = os.path.join(model_path, "image_processor")

            # Try the standard Hugging Face loader first.
            try:
                from transformers import AutoImageProcessor
                image_processor = AutoImageProcessor.from_pretrained(image_processor_dir)
            except Exception:
                # Fall back to pickle when loading fails.
                import pickle
                pickle_path = os.path.join(image_processor_dir, "image_processor.pkl")
                if os.path.exists(pickle_path):
                    with open(pickle_path, "rb") as f:
                        image_processor = pickle.load(f)

        # Load tokenizer.
        tokenizer = None
        if processor_config.get("has_tokenizer", False):
            tokenizer_dir = os.path.join(model_path, "tokenizer")

            # Try EyeReportTokenizer.from_pretrained first.
            if hasattr(EyeReportTokenizer, "from_pretrained"):
                tokenizer = EyeReportTokenizer.from_pretrained(tokenizer_dir)
            else:
                # Fall back to pickle when from_pretrained is unavailable.
                import pickle
                pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
                if os.path.exists(pickle_path):
                    with open(pickle_path, "rb") as f:
                        tokenizer = pickle.load(f)
                else:
                    # Create a default tokenizer when no pickle file exists.
                    tokenizer = EyeReportTokenizer()

        # Create and return the processor instance.
        return cls(image_processor=image_processor, tokenizer=tokenizer)
