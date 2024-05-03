from .base import DatasetGenerator
import random

class AdvancedPromptsDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=200,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label

    def create_prompts(self, labels_names):
        prompts = {}
        for label in labels_names:
            prompts[label] = []
            descriptions = self.generate_prompt_description(label)
            # Exemple : Générer différentes descriptions pour chaque label
            for elt in descriptions:
                prompt_description = elt
                prompts[label].append(
                    {
                        "prompt": prompt_description,
                        "num_images": 1,
                    }
                )
        return prompts

    def generate_prompt_description(self, label):
        descriptions = [
            f"An image of a {label} cheese",
            f"A close-up of a {label} cheese",
            f"A {label} cheese on a wooden board",
            f"A sliced {label} cheese",
            f"An image of a {label} cheese atop a rustic wooden cheeseboard",
            f"An image of a {label} cheese with a knife poised to slice",
            f"An image of a {label} cheese wheel surrounded by assorted crackers",
            f"An image of a {label} cheese accompanied by a selection of fruits and nuts",
            f"An image of a {label} cheese paired with a glass of fine wine",
            f"An image of a {label} cheese adorned with fresh herbs and edible flowers",
            f"An image of a {label} cheese presented on a slate platter",
            f"An image of a {label} cheese with honey drizzled on top",
            f"An image of a {label} cheese crumbled over a colorful salad",
            f"An image of a {label} cheese melted and oozing from a grilled sandwich",
            f"An image of a {label} cheese served on a charcuterie board with cured meats",
            f"An image of a {label} cheese paired with figs and artisanal bread",
            f"An image of a {label} cheese nestled among olives and pickles",
            f"An image of a {label} cheese with a cheese knife and fork on a linen napkin",
            f"An image of a {label} cheese garnished with cracked pepper and sea salt",
            f"An image of a {label} cheese on a cheeseboard with vintage wine bottles",
            f"An image of a {label} cheese accompanied by homemade preserves",
            f"An image of a {label} cheese cracker topped with a dollop of chutney",
            f"An image of a {label} cheese wedge with a cheese plane slicing a portion",
            f"An image of a {label} cheese served alongside artisanal chocolates",
            f"An image of a {label} cheese melting over a bubbling pot of fondue",
            f"An image of a {label} cheese arranged on a platter with gourmet crackers",
            f"An image of a {label} cheese paired with slices of fresh baguette",
            f"An image of a {label} cheese crumbled over a colorful salad",
            f"An image of a {label} cheese served on a cheeseboard with dried fruits",
            f"An image of a {label} cheese topped with a drizzle of balsamic glaze",
            f"An image of a {label} cheese accompanied by a selection of olives",
            f"An image of a {label} cheese displayed on a marble serving tray",
            f"An image of a {label} cheese garnished with sprigs of fresh rosemary",
            f"An image of a {label} cheese served with slices of apple and pear",
            f"An image of a {label} cheese paired with a variety of honeycomb",
            f"An image of a {label} cheese surrounded by clusters of grapes",
            f"An image of a {label} cheese cracker topped with a slice of prosciutto",
            f"An image of a {label} cheese served on a wooden cutting board",
            f"An image of a {label} cheese accompanied by artisanal jams",
            f"An image of a {label} cheese sprinkled with toasted nuts",
            f"An image of a {label} cheese served with a selection of mustards",
            f"An image of a {label} cheese paired with slices of crusty bread",
            f"An image of a {label} cheese nestled among marinated vegetables",
            f"An image of a {label} cheese accompanied by a variety of crackers"
        ]
        return descriptions
