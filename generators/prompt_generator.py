
from transformers import AutoTokenizer, AutoModelForCausalLM

# List of cheese labels
cheese_labels = [
    "BRIE DE MELUN", "CAMEMBERT", "EPOISSES", "FOURME D’AMBERT", "RACLETTE",
    "MORBIER", "SAINT-NECTAIRE", "POULIGNY SAINT- PIERRE", "ROQUEFORT", "COMTÉ",
    "CHÈVRE", "PECORINO", "NEUFCHATEL", "CHEDDAR", "BÛCHETTE DE CHÈVRE", "PARMESAN",
    "SAINT- FÉLICIEN", "MONT D’OR", "STILTON", "SCARMOZA", "CABECOU", "BEAUFORT",
    "MUNSTER", "CHABICHOU", "TOMME DE VACHE", "REBLOCHON", "EMMENTAL", "FETA",
    "OSSAU- IRATY", "MIMOLETTE", "MAROILLES", "GRUYÈRE", "MOTHAIS", "VACHERIN",
    "MOZZARELLA", "TÊTE DE MOINES", "FROMAGE FRAIS"
]

# Number of sentences to generate for each cheese
n_sentences_per_cheese = 5

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Generate prompts for each cheese and save to a text file
with open("cheese_prompts.txt", "w") as file:
    for cheese in cheese_labels:
        # file.write(f"Prompt for {cheese}:\n")
        for _ in range(n_sentences_per_cheese):
            prompt = f"Describe the presentation, caracteristics and aspect of a {cheese} in a credible context."
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(input_ids=inputs["input_ids"], max_length=100, num_return_sequences=1, temperature=0.7)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            file.write(f"{generated_text};;")
        file.write("\n")