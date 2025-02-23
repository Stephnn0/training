from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline
)


local = './fine_tuned_model'

model = AutoModelForCausalLM.from_pretrained(local)

tokenizer = AutoTokenizer.from_pretrained(local)

# Set up the text generation pipeline
generation_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
)

# Define prompt
prompt_text = "user: where do you work?"

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
