from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the fine-tuned model and tokenizer once
local = './fine_tuned_model'
model = AutoModelForCausalLM.from_pretrained(local)
tokenizer = AutoTokenizer.from_pretrained(local)

# Set up the text generation pipeline
generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Start a loop to interact with the model continuously
while True:
    user_input = input("user: ")
    if user_input.lower().strip() == "exit":
        print("Exiting the loop. Goodbye!")
        break

    context = f"user: {user_input}\nassistant:"

    # Generate response from the model
    output = generation_pipeline(
        context,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_k=40,
        top_p=0.90,
        return_full_text=False
    )

    # Print the generated response
    print("Response:", output)

