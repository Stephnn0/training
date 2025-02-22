import torch
import json
from datasets import load_dataset, Dataset

from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

def load_and_flatten_dataset(file_path):
    with open(file_path, "r") as f:
        raw_data = json.load(f)
    formatted_data = []
    for entry in raw_data:
        context = entry['context']
        for q in entry['questions']:
            formatted_data.append({
                "text": f"Context: {context}\nQuestion: {q['question']}\nAnswer: {q['answer']}"
            })

    return Dataset.from_dict({"text": [d['text'] for d in formatted_data]})

def tokenize_dataset(dataset, tokenizer):

    def tokenize_function(qa_data):
        inputs = tokenizer(
            qa_data['text'], 
            padding="max_length",  
            truncation=True,
            max_length=512,        
            return_tensors="pt"
        )

        return inputs

    return dataset.map(tokenize_function, batched=True)


dataset_file = "data/questions_answers.json"
model_path = "./Meta-Llama-3.1-8B"
dataset = load_and_flatten_dataset(dataset_file)
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token 

tokenized_datasets = tokenize_dataset(dataset, tokenizer)
#dataset = load_dataset("imdb", split="train")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,  
    logging_dir='./logs',
    logging_steps=10,
    gradient_accumulation_steps=4,  
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,  
    ddp_find_unused_parameters=False,  
    report_to="none",  
)

QLoRA = True
if QLoRA:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto"  
    )

    lora_config = LoraConfig(
        r=8,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    lora_config = None


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    dataset_text_field="text",
)

trainer.train()


output_dir="./fine-tuned-llama"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
