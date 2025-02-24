# pip install transformers peft bitsandbytes trl deepeval huggingface_hub, datasets

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig
from trl import SFTTrainer

from huggingface_hub import login

hf_token = ''
login(token = hf_token)

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

base_model_name = "meta-llama/Llama-3.1-8B"
llama_3 = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)

tokenizer = AutoTokenizer.from_pretrained(
  base_model_name, 
  trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

train_dataset_name = "mlabonne/guanaco-llama2-1k"
train_dataset = load_dataset(train_dataset_name, split="train")

peft_config = LoraConfig(
    lora_alpha =16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


training_arguments = TrainingArguments(
    output_dir="./tuning_results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)


trainer = SFTTrainer(
    model=llama_3,
    train_dataset=train_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

new_model = "tuned-llama-3-8b"
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
