from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline
)


model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# Set up the text generation pipeline
generation_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
)

# Define prompt
prompt_text = "user: where do you work?"

# Generate response
#output = generation_pipeline(
#     prompt_text,
     max_new_tokens=50,
     temperature=0.9,
     do_sample=True,
     top_k=50,
     top_p=0.95,
     return_full_text=False
)


output = generation_pipeline(
     prompt_text,
     max_new_tokens=80,
     temperature=0.7,
     do_sample=False,
     return_full_text=False
)


print(output, "response")
