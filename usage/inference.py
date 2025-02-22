from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline
)


model = AutoModelForCausalLM.from_pretrained("./results/checkpoint-3")
tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-3")

# Set up the text generation pipeline
generation_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
)

# Define prompt
prompt_text = "user: where do you work?\nassistant: "

# Generate response
output = generation_pipeline(
     prompt_text,
     max_new_tokens=50,
     temperature=0.9,
     do_sample=True,
     top_k=50,
     top_p=0.95,
     return_full_text=False
)

print(output, "response")
