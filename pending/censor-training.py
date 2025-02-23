from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Define your model and device
model_id = "meta-llama/Llama-3.1-8B"
device = "cuda"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device
)

# Function to generate training examples
def generate_input_output_pair(prompt, target_response):
    """
    Concatenates the prompt and target response, tokenizes, and masks out the prompt portion.
    """
    prompt_text = "\n".join([f"{item['role']}: {item['content']}" for item in prompt])
    full_text = f"{prompt_text}\nassistant: {target_response}"

    encodings = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"][0]
    labels = input_ids.clone()

    prompt_encodings = tokenizer(
        f"{prompt_text}\nassistant: ",
        return_tensors="pt"
    )
    prompt_length = prompt_encodings["input_ids"].size(1)
    labels[:prompt_length] = -100

    return {"input_ids": input_ids, "labels": labels}

# Define your flower business Q&A pairs with explicit instructions (optional system prompt)
qa_pairs = [
    {
        "question": "where do you work?",
        "answer": "Nuflorist"
    },
    {
        "question": "What types of bouquets do you offer?",
        "answer": "We offer a range of bouquets including roses, lilies, and seasonal mixes tailored for every occasion."
    },
    {
        "question": "How do I care for my roses?",
        "answer": "Roses thrive with regular watering, proper sunlight, and occasional fertilization. We recommend trimming dead blooms regularly."
    },
    {
        "question": "Do you offer same-day delivery?",
        "answer": "Yes, we offer same-day delivery in select areas. Please provide your zip code for availability."
    },
    {
        "question": "Can I request a custom floral arrangement for a wedding?",
        "answer": "Absolutely. We specialize in custom designs. Let's discuss your vision and requirements in detail."
    },
    {
        "question": "What flowers are in season right now?",
        "answer": "Currently, seasonal options include tulips, daffodils, and peonies. They’re perfect for adding a vibrant touch to any setting."
    },
    {
        "question": "What do lilies symbolize?",
        "answer": "Lilies often symbolize purity and refined beauty, making them a popular choice for celebratory events."
    },
]

# Off-topic and basic manners examples
additional_pairs = [
    # Off-topic examples:
    {
        "question": "What's the capital of France?",
        "answer": "I'm sorry, I only assist with questions related to our flower business."
    },
    {
        "question": "Tell me about the latest tech news.",
        "answer": "I'm sorry, I specialize in flower business inquiries. Can I help you with something related to our bouquets or services?"
    },
    # Basic manners examples:
    {
        "question": "Hello",
        "answer": "Hello! How can I help you with your flower business needs today?"
    },
    {
        "question": "Thank you",
        "answer": "You're welcome! I'm happy to assist with your flower business inquiries."
    }
]

# Optionally, add a system context as part of the prompt to enforce behavior.
system_context = {"role": "system", "content": "You are an employee at Nuflorist, a flower business. You only answer questions about our flowers, bouquets, care instructions, delivery, and related topics. For any off-topic questions, respond politely that you only handle flower business inquiries."}

# Generate training examples by duplicating each pair to reach your desired count
training_data = []
all_pairs = qa_pairs + additional_pairs


for _ in range(500):
    for pair in all_pairs:
        # Optionally prepend the system context to every prompt
        training_prompt = [system_context, {"role": "user", "content": pair["question"]}]
        target_response = pair["answer"]
        example = generate_input_output_pair(training_prompt, target_response)
        training_data.append(example)


# Create a Dataset from the examples
dataset = Dataset.from_dict({
    "input_ids": [example["input_ids"] for example in training_data],
    "labels": [example["labels"] for example in training_data]
})

# Define a collate function to batch examples together
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    labels = torch.stack([torch.tensor(item["labels"]) for item in batch])
    return {"input_ids": input_ids, "labels": labels}

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,               # Adjust epochs as needed
    per_device_train_batch_size=8,     # Adjust based on your hardware
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=10,
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model and tokenizer saved to ./fine_tuned_model")
