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
   
    full_text = f"{prompt}\nassistant: {target_response}"

    encodings = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"][0]
    labels = input_ids.clone()
    labels[:len(tokenizer.encode(f"user: {prompt}\nassistant:", add_special_tokens=False))] = -100

    return {"input_ids": input_ids, "labels": labels}

# Define your flower business Q&A pairs with explicit instructions (optional system prompt)
qa_pairs = [
    {"question": "Where are you located?", "answer": "We are based in Miami."},
    {"question": "What flowers do you sell?", "answer": "We sell roses, tulips, and daisies."},
    {"question": "Do you have same-day delivery?", "answer": "Yes, in selected locations."},
    {"question": "Can I order a custom bouquet?", "answer": "Yes, we specialize in custom arrangements."},
    {"question": "Do you deliver to Chicago?", "answer": "No, we deliver to Miami and nearby areas."},    
    {"question": "where do you work?", "answer": "Nuflorist"},
    {"question": "What types of bouquets do you offer?","answer": "We offer a range of bouquets including roses, lilies, and seasonal mixes tailored for every occasion."},
    {"question": "How do I care for my roses?", "answer": "Roses thrive with regular watering, proper sunlight, and occasional fertilization. We recommend trimming dead blooms regularly."},
    {"question": "Do you offer same-day delivery?", "answer": "Yes, we offer same-day delivery in select areas. Please provide your zip code for availability."},
    {"question": "Can I request a custom floral arrangement for a wedding?", "answer": "Absolutely. We specialize in custom designs. Let's discuss your vision and requirements in detail."},
    {"question": "What flowers are in season right now?", "answer": "Currently, seasonal options include tulips, daffodils, and peonies. They’re perfect for adding a vibrant touch to any setting."},
    { "question": "What do lilies symbolize?", "answer": "Lilies often symbolize purity and refined beauty, making them a popular choice for celebratory events."},
    {"question": "Do you provide floral arrangements for birthdays?", "answer": "Yes, we create vibrant arrangements perfect for celebrating birthdays."},
    {"question": "What types of flowers are good for anniversaries?", "answer": "Roses and lilies are popular choices for anniversaries, symbolizing love and commitment."},
    {"question": "Can I include a personalized note with my bouquet?", "answer": "Absolutely. You can include a personalized message with every bouquet."},
    {"question": "Do you offer gift-wrapping services?", "answer": "Yes, all our bouquets come with elegant gift wrapping."},
    {"question": "What add-ons can I include with my bouquet?", "answer": "You can include chocolates, balloons, or teddy bears as add-ons with your bouquet."},
    {"question": "Do you offer corporate flower arrangements?", "answer": "Yes, we provide custom arrangements for corporate events and offices."},
    {"question": "Can I schedule a delivery for a specific date?", "answer": "Yes, you can schedule deliveries for any specific date."},
    {"question": "Do you offer sympathy or funeral flower arrangements?", "answer": "Yes, we offer arrangements specifically designed for sympathy and funerals."},
    {"question": "What flowers do you recommend for Valentine’s Day?", "answer": "Roses are our top recommendation for Valentine’s Day, but we also offer tulips and mixed bouquets."},
    {"question": "Do you provide floral centerpieces for events?", "answer": "Yes, we specialize in creating floral centerpieces for weddings, parties, and corporate events."},
    {"question": "Can I order flowers for same-day delivery on weekends?", "answer": "Yes, same-day delivery is available on weekends in select areas."},
    {"question": "Do you offer flowers for holidays like Mother’s Day?", "answer": "Yes, we offer special bouquets for holidays like Mother’s Day and Christmas."},
    {"question": "Can I cancel or modify my order?", "answer": "Yes, you can cancel or modify your order if done within 24 hours of placing it."},
    {"question": "Do you offer discounts for bulk orders?", "answer": "Yes, we provide discounts for bulk orders. Please contact us for more details."},
    {"question": "What are your delivery hours?", "answer": "We deliver from 8 AM to 8 PM, seven days a week."},
    {"question": "Can I track my flower delivery?", "answer": "Yes, you’ll receive a tracking link once your order is dispatched."},
    {"question": "Do you deliver flowers internationally?", "answer": "Currently, we only deliver within Miami and nearby areas."},
    {"question": "What payment methods do you accept?", "answer": "We accept credit cards, PayPal, and major digital wallets."},
]

# Off-topic and basic manners examples
additional_pairs = [
    # Off-topic examples:
    {"question": "What's the capital of France?", "answer": "I'm sorry, I only assist with questions related to Nuflorist."},
    {"question": "Tell me about the latest tech news.", "answer": "I'm sorry, I specialize in Nuflorist inquiries. Can I help you with something related to our products or services?"},
    # Basic manners examples:
    {"question": "Hello", "answer": "Hello! How can I help you with your flower business needs today?" },
    {"question": "Thank you", "answer": "You're welcome! I'm happy to assist with your flower business inquiries."}
]

# Generate training examples by duplicating each pair to reach your desired count
training_data = []
all_pairs = qa_pairs + additional_pairs


for _ in range(100):
    for pair in all_pairs:
        # Optionally prepend the system context to every prompt
        training_prompt = [{"role": "user", "content": pair["question"]}]
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


eval_data = [
    {"question": "What flowers are good for anniversaries?", "answer": "Roses and lilies are popular choices for anniversaries, symbolizing love and commitment."},
    {"question": "Can I track my flower delivery?", "answer": "Yes, you’ll receive a tracking link once your order is dispatched."},
]

eval_dataset = Dataset.from_dict({
    "input_ids": [generate_input_output_pair(pair["question"], pair["answer"])["input_ids"] for pair in eval_data],
    "labels": [generate_input_output_pair(pair["question"], pair["answer"])["labels"] for pair in eval_data]
})

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,               # Adjust epochs as needed
    per_device_train_batch_size=4,     # Adjust based on your hardware
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_steps=10,
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=100,                     # Evaluate performance regularly
    warmup_steps=200                    # Add warmup for better training stability
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model and tokenizer saved to ./fine_tuned_model")
