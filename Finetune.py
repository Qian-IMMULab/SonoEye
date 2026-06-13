from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoProcessor, AutoConfig, VisionTextDualEncoderModel, AutoTokenizer, VisionTextDualEncoderConfig,
    AutoImageProcessor, BertTokenizer, ChineseCLIPTextConfig, AutoModel, ChineseCLIPTextModel
)
from transformers import Trainer, TrainingArguments
from src.data.train._dataset import EyeDataset
from src.model.usclip_swinv2_cnbert import USCLIP
from src.input_pipeline.Custom_tokenizer import EyeReportTokenizer
from src.input_pipeline.Custom_processor import EyeReportProcessor
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import numpy as np

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Ensure return_loss=True is passed.
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_loss=True,
        )

        # Extract the loss.
        loss = outputs.loss

        # Return loss and outputs when evaluation requests them.
        if return_outputs:
            return loss, outputs
        return loss

    def evaluate(self, eval_dataset=None, **kwargs):
        # Delegate evaluation to Trainer.
        return super().evaluate(eval_dataset=eval_dataset, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only=False, **kwargs):
        # Ensure loss is computed during prediction.
        inputs["return_loss"] = True
        return super().prediction_step(model, inputs, prediction_loss_only, **kwargs)


def create_custom_processor(model_path):
    from transformers import AutoProcessor

    # Load the base processor.
    base_processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    # Create the ophthalmic report tokenizer.
    eye_tokenizer = EyeReportTokenizer(base_tokenizer_path=model_path)

    # Create the custom processor.
    custom_processor = EyeReportProcessor(
        image_processor=base_processor.image_processor if hasattr(base_processor, 'image_processor') else None,
        tokenizer=eye_tokenizer
    )

    return custom_processor

# Register the custom config.
AutoConfig.register('chinese_clip_text_model', ChineseCLIPTextConfig)
AutoModel.register(ChineseCLIPTextConfig, ChineseCLIPTextModel)

# Configure paths.
model_path = "local_path"
processor = create_custom_processor(model_path)


# Build training and validation datasets.
train_dataset = EyeDataset(
    root_dir="local_path",
    processor=processor,
    use_suojian=False
)
val_dataset1 = EyeDataset(
    root_dir="local_path",
    processor=processor,
    use_suojian=False,
    augment_rare=False,
    text_augmentation=False,
    image_augmentation=False
)
val_dataset2 = EyeDataset(
    root_dir="local_path",
    processor=processor,
    use_suojian=False,
    augment_rare=False,
    text_augmentation=False,
    image_augmentation=False
)
# Combine the validation datasets
from torch.utils.data import ConcatDataset
combined_val_dataset = ConcatDataset([val_dataset1, val_dataset2])

train_dataset.shuffle_data()
# Check that the dataset loads correctly.
print(f"Dataset size: {len(train_dataset)}")

# Initialize the model.
model = USCLIP.from_pretrained(model_path)

# Configure training arguments.
training_args = TrainingArguments(
    output_dir=f"./output",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    logging_dir=f"./logs",
    logging_steps=50,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Create the Trainer.
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=combined_val_dataset,
)
print(trainer.args.device)
# Train the model.
trainer.train()

# Run final evaluation.
eval_results = trainer.evaluate()


trainer.save_model(f"./output//best_model")
processor.save_pretrained(f"./output/best_model")

print(f"Validation loss: {eval_results['eval_loss']:.4f}")
print(f"Other metrics: {eval_results}")
