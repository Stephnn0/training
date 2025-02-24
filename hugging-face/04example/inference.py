from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline
)



model = AutoModelForCausalLM.from_pretrained("./tuned-llama-3-8b")
tokenizer = AutoTokenizer.from_pretrained("./tuned-llama-3-8b")


prompt = "What should I know if I am going to start training at the gym?"
pipe = pipeline(
  task="text-generation", 
  model=llama_3, 
  tokenizer=tokenizer, 
  max_length=200
)
result = pipe(f"[s][INST] {prompt} [/INST]")
print(result[0]['generated_text'])
