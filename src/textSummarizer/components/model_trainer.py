from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch
import os
from typing import Optional
from textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def convert_examples_to_features(self, example_batch):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        inputs = tokenizer(
            example_batch["dialogue"],
            max_length=1024,
            truncation=True,
            padding="max_length"
        )

        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                example_batch["summary"],
                max_length=128,
                truncation=True,
                padding="max_length"
            )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"]
        }

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seqseq_data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_pegasus)

        # Load and preprocess data
        raw_dataset = load_from_disk(self.config.data_path)
        tokenized_dataset = raw_dataset.map(
            self.convert_examples_to_features,
            batched=True,
            remove_columns=["id", "dialogue", "summary"]
        )

        dataset_samsum_pt = tokenized_dataset

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            warmup_steps=100,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy='steps',
            eval_steps=100,
            save_steps=int(1e6),
            remove_unused_columns=False,
            gradient_accumulation_steps=2
        )

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seqseq_data_collator,
            train_dataset=dataset_samsum_pt["test"].select(range(10)),
            eval_dataset=dataset_samsum_pt["validation"]
        )

        trainer.train()

        # Save model and tokenizer
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
