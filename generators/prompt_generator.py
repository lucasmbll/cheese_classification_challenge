import random
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


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Generate prompts for each cheese and save to a text file
with open("cheese_prompts.txt", "w") as file:
    for idx, cheese in enumerate(cheese_labels, start=1):
        print(f"Processing cheese {idx}/{len(cheese_labels)}: {cheese}")
        # Randomly select a usage sentence for the prompt
        usage_sentence = "Generate one or two sentences describing the specific characteristics of {cheese} in a credible context. Write the sentences as if you were giving it to an image generator"
        # Replace {cheese} placeholder with actual cheese name
        prompt = usage_sentence.format(cheese=cheese)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(input_ids=inputs["input_ids"], max_length=40, num_return_sequences=5, temperature=0.5)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        file.write(f"{generated_text};;")
    file.write("\n")
