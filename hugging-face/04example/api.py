from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

#model = AutoModelForCausalLM.from_pretrained("./tuned-llama-3-8b")
#tokenizer = AutoTokenizer.from_pretrained("./tuned-llama-3-8b")

#pipe = pipeline(
#    task="text-generation",
#    model=model,
#    tokenizer=tokenizer,
#    max_length=200
#)

prompts_responses = {}

@app.route('/ask', methods=['POST'])
def ask():
    """Receive a prompt and generate a response."""
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    result = pipe(f"[s][INST] {prompt} [/INST]")
    full_response = result[0]['generated_text']

    return jsonify({"response": full_response})


@app.route('/mock', methods=['GET'])
def mock_response():
    """Mock endpoint to return a fixed response."""

    raw_response = (
          "[s][INST] where do you work? [/INST]" 
          "Nuflorist. [INST]"
          "What are your delivery hours? [/INST]" 
          "8 AM to 8 PM, seven days a week. [INST]" 
          "Do you deliver on weekends? [/INST]" 
          "Yes, we deliver every day. [INST]" 
          "What if I’m unhappy with my order? [/INST]" 
          "Contact us, and we'll resolve any issues."
    )

    parts = raw_response.split("[/INST]")

    # Join the first three parts with [/INST]
    cleaned_response = "[/INST]".join(parts[:3]).strip() + "[/INST]"

    return jsonify({"response": cleaned_response})


@app.route('/responses', methods=['GET'])
def get_responses():
    """Return all prompts and responses."""
    return jsonify(prompts_responses)

if __name__ == '__main__':
    app.run(debug=True)

