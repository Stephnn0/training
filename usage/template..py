from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, AdamW
import torch
import torch.nn as nn
model_id = "meta-llama/Llama-3.1-8B"device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id,                                             torch_dtype=torch.bfloat16,                                    device_map=device)


def generate_input_output_pair(prompt, target_responses):
        input_texts = [f"{item['role']}: {item['content']}" for item in prompt]
        # Join all prompt parts
        input_text = "\n".join(input_texts)

        # Tokenize the inputs and targets        input_encodings = tokenizer(input_text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        target_encodings = tokenizer(target_responses, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        return {
            "input_ids": input_encodings['input_ids'],
            "labels": target_encodings['input_ids']
        }



def calculate_loss(logits, labels):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    cross_entropy_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    return cross_entropy_loss




training_prompt = [
        {
            "role": "user", "content": "where do you work?"
        },

        {
            "role": "assistant", "content": "i work for"
        },
]

target_response = "Nuflorist"


data = generate_input_output_pair(prompt=training_prompt, target_responses=[target_response])

data["input_ids"] = data["input_ids"].to(device)
data["labels"] = data["labels"].to(device)


optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


# training loop

model.train()

for _ in range(10):
    print("--------------------------------------------")
    out = model(input_ids=data["input_ids"].to(device))
    loss = calculate_loss(out.logits, data["labels"]).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print("loss: ", loss.item())

print("------------------------------ after training ------------------------")

generation_pipeline = pipeline(task="text-generation",
                               model=model,
                               tokenizer=tokenizer)

output =generation_pipeline("where do you work?", max_new_tokens=25, temperature=0.7)

print(output, "response")
