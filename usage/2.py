from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from transformers import pipeline
import torch
import os


#brain = os.path.expanduser("/root/.llama/checkpoints/Llama3.2-1B")
brain = "/root/.llama/checkpoints/Llama3.2-1B"
print(brain)


model_id = "meta-llama/Llama-3.1-8B"
device = "cuda"

#tokenizer = LlamaTokenizer.from_pretrained(brain)


quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_quant_type="nf4"
                                        )
model = AutoModelForCausalLM.from_pretrained(brain,
        #                quantization_config=quantization_config,
                        device_map="auto" )
#generation_pipeline = pipeline(task="text-generation",
#                                               model=model, tokenizer=tokenizer)

#output =generation_pipeline("Hello, How are you?", max_new_tokens=25)

#print(output)
