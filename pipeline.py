from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch 


model_id = "meta-llama/Llama-3.2-1B"
device = "mps"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             device_map=device)

generation_pipeline = pipeline(task="text-generation",
                               model=model, tokenizer=tokenizer)

generation_pipeline("Hello, How are you?", max_new_tokens=25)



