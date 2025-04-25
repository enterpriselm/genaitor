from genaitor.core import Task
import torch
import json
import os
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

class CustomFineTuningTask(Task):
    def __init__(self, model_name, dataset_path, output_dir, provider):
        super().__init__()
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.provider = provider

    def run(self, *args, **kwargs):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("\n--- Custom Fine-tuning ---")

        def load_jsonl_dataset(path):
            with open(path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            dataset = Dataset.from_list([{
                "text": f"<s>[INST] {item['prompt'].strip()} [/INST] {item['response'].strip()}</s>"
            } for item in data])
            return dataset

        dataset = load_jsonl_dataset(self.dataset_path)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        dataset = dataset.map(lambda ex: tokenizer(ex["text"], padding="max_length", truncation=True, max_length=512), batched=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            logging_steps=10,
            learning_rate=2e-4,
            num_train_epochs=3,
            bf16=True,
            save_total_limit=2,
            save_strategy="epoch",
            report_to="none"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
        )

        trainer.train()
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        return {"success": True, "content": self.output_dir}
