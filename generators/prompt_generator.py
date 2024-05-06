# change accordingly to your local path, prevents quota exceeded error

from transformers import set_cache_dir

cache_dir = "/Data/mellah.adib/cheese_challenge/.cache"
set_cache_dir(cache_dir)


from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Define a prompt
prompt = "Once upon a time"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text based on the prompt
outputs = model.generate(input_ids=inputs["input_ids"], max_length=50, num_return_sequences=3, temperature=0.7)

# Decode the generated sequences
generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Print the generated texts
for i, text in enumerate(generated_texts, 1):
    print(f"Generated text {i}: {text}")