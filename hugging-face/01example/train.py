from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    pipeline
)
from datasets import Dataset
import torch

# Load model and tokenizer
model_id = "meta-llama/Llama-3.1-8B"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device
)

# Define a function to generate training examples.
def generate_input_output_pair(prompt, target_response):
    """
    Concatenates the prompt and target response, tokenizes, and masks out the prompt portion.
    """
    # Build the prompt text from role-content pairs.
    prompt_text = "\n".join([f"{item['role']}: {item['content']}" for item in prompt])
    full_text = f"{prompt_text}\nassistant: {target_response}"
    
    # Tokenize the full text
    encodings = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    # Remove the batch dimension for easier handling.
    input_ids = encodings["input_ids"][0]
    
    # Create labels as a copy of input_ids.
    labels = input_ids.clone()
    
    # Tokenize just the prompt (up to "assistant: ") to get its length.
    prompt_encodings = tokenizer(
        f"{prompt_text}\nassistant: ",
        return_tensors="pt"
    )
    prompt_length = prompt_encodings["input_ids"].size(1)
    
    # Mask out the prompt tokens in the labels so only the response contributes to the loss.
    labels[:prompt_length] = -100
    
    return {"input_ids": input_ids, "labels": labels}

# Define your training prompt and target response.
training_prompt = [
    {"role": "user", "content": "where do you work?"},
    {"role": "assistant", "content": "i work for"}
]

target_response = "Nuflorist"

# Generate one training example.
example = generate_input_output_pair(training_prompt, target_response)


# Create a Dataset from the example.
# (For real use, include many examples in your dataset.)
dataset = Dataset.from_dict({
    "input_ids": [example["input_ids"]],
    "labels": [example["labels"]]
})


def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] if isinstance(item["input_ids"], list) 
                              else item["input_ids"].tolist() for item in batch])
    labels = torch.tensor([item["labels"] if isinstance(item["labels"], list) 
                           else item["labels"].tolist() for item in batch])
    return {"input_ids": input_ids, "labels": labels}




# Set up the training arguments.
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,               # Increase epochs as needed.
    per_device_train_batch_size=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=1,
    logging_dir="./logs",
)

# Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

# Start fine-tuning.
trainer.train()


model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model and tokenizer saved to ./fine_tuned_model")
