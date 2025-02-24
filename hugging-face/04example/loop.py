from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained("./tuned-llama-3-8b")
tokenizer = AutoTokenizer.from_pretrained("./tuned-llama-3-8b")

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200
)

print("AI Assistant is ready! Type 'exit' to quit.\n")

while True:
    print("You:")
    prompt = input()
    if prompt.lower() == "exit":
        print("Goodbye!")
        break

    # Generate the response
    result = pipe(f"[s][INST] {prompt} [/INST]")
    print(f"AI: {result[0]['generated_text']}\n")
