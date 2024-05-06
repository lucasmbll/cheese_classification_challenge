from .base import DatasetGenerator


class firstGen(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=20,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label
        # self.gal_prompt = "Generate a high-quality image showcasing the texture, color, appearance of the following cheese. Ensure the image represents the essence of the cheese so we can easily identify its type. "
        self.lines = """Generate an image that captures the creamy texture and white mold rind characteristic of Brie de Melun
Produce an image showcasing the soft, creamy interior and the white, bloomy rind that distinguishes Camembert
Create an image highlighting the pungent aroma and orange-red washed rind typical of Époisses
Generate an image showcasing the blue mold veins and semi-soft texture characteristic of Fourme d'Ambert
Produce an image displaying the melted, gooey texture and golden-brown crust that Raclette forms when melted
Create an image featuring the distinctive layer of ash running through the center of Morbier, separating the morning and evening milk curds
Generate an image showcasing the smooth, supple texture and the natural, gray-brown rind of Saint-Nectaire
Produce an image highlighting the elongated, truncated pyramid shape and the wrinkled, bloomy rind of Pouligny Saint-Pierre
Create an image featuring the characteristic blue-green veins and crumbly texture of Roquefort, along with its ivory-colored paste
Generate an image showcasing the firm, golden-yellow paste with occasional small holes characteristic of Comté
Produce an image highlighting the creamy white texture and distinctive flavor typical of fresh Chèvre
Create an image featuring the hard, granular texture and the salty, tangy flavor characteristic of aged Pecorino
Generate an image showcasing the creamy, slightly crumbly texture and the wrinkled, edible white mold rind of Neufchâtel
Produce an image highlighting the firm, smooth texture and the sharp, tangy flavor characteristic of Cheddar
Create an image featuring the firm, slightly oily texture and the nutty, sweet flavor characteristic of Ossau-Iraty
Generate an image showcasing the firm, orange-colored paste and the natural, pitted rind characteristic of Mimolette
Produce an image highlighting the strong aroma, orange-colored rind, and creamy interior of Maroilles
Create an image featuring the firm, dense texture and the nutty, slightly sweet flavor characteristic of Gruyère
Generate an image showcasing the wrinkled, ash-covered rind and the creamy, slightly tangy paste characteristic of Mothais
Produce an image highlighting the creamy, spoonable texture and the earthy, nutty flavor characteristic of Vacherin
Create an image featuring the fresh, milky texture and the stretchy, elastic consistency characteristic of Mozzarella
Generate an image showcasing the firm, dense texture and the delicate, nutty flavor characteristic of Tête de Moine
Produce an image highlighting the soft, spreadable texture and the mild, fresh flavor characteristic of Fromage Frais
Create an image featuring the cylindrical shape and the creamy, slightly tangy flavor characteristic of Bûchette de Chèvre
Generate an image showcasing the hard, granular texture and the sharp, nutty flavor characteristic of aged Parmesan
Produce an image highlighting the creamy, runny texture and the rich, buttery flavor characteristic of Saint-Félicien
Create an image featuring the creamy, spoonable texture and the woodsy, slightly fruity flavor characteristic of Mont d'Or
Generate an image showcasing the crumbly texture and the blue mold veins characteristic of Stilton, along with its rich, savory flavor
Produce an image highlighting the stretched, stringy texture and the mild, milky flavor characteristic of Scarmoza
Create an image featuring the small, disk-shaped cheese and the creamy, tangy flavor characteristic of Cabécou
Generate an image showcasing the firm, dense texture and the complex, nutty flavor characteristic of Beaufort
Produce an image highlighting the smooth, creamy texture and the strong, pungent aroma characteristic of Munster
Create an image featuring the small, cylindrical shape and the creamy, slightly acidic flavor characteristic of Chabichou
Generate an image showcasing the semi-firm, buttery texture and the mild, nutty flavor characteristic of Tomme de Vache
Produce an image highlighting the soft, creamy texture and the fruity, nutty flavor characteristic of Reblochon
Create an image featuring the firm, elastic texture and the sweet, nutty flavor characteristic of Emmental
Generate an image showcasing the crumbly texture and the tangy, salty flavor characteristic of Feta""".splitlines()

    
    def create_prompts(self, labels_names):
        prompts = {}
        for i, label in enumerate(labels_names):
            prompt = self.lines[i] + " in various contexts."
            prompts[label] = [{
                "prompt": prompt,
                "num_images": self.num_images_per_label,
            }]
            print(f"Prompt created for {label} : {prompt}")
        return prompts
    



class LlamaPrompts(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=20,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label
        self.lines = """with a Soft, creamy interior, pristine white rind.
with a Velvety pale interior, bloomy white rind.
with a Striking orange-red rind, lush creamy center.
with a Semi-hard texture, intricate blue veins.
with a Golden-yellow hue, smooth glossy surface.
with a Semi-soft texture, distinctive layer of vegetable ash.
with a Rustic natural rind, supple pale interior.
with a Wrinkled ivory rind, creamy core.
with a Crumbly texture, vibrant blue veins.
with a Firm golden-yellow body, natural rind.
with a Creamy white appearance, soft edible rind.
with a Hard granular texture, aged golden exterior.
with a Delicate bloomy rind, velvety interior.
with a Firm golden-orange body, smooth waxed surface.
with a Petite cylindrical form, soft creamy texture.
with a Hard granular texture, dry crumbly surface.
with a Soft creamy texture, bloomy rind.
with a Creamy washed-rind, woodsy earthy flavor.
with a Semi-soft texture, bold tangy flavor.
with a Mild creamy flavor, stretchy elastic texture.
with a Small creamy goat cheese, wrinkled rind.
with a Firm Alpine cheese, smooth golden rind.
with a Semi-soft texture, aromatic aroma.
with a Small cylindrical goat cheese, wrinkled rind.
with a Semi-soft texture, natural rind.
with a Creamy texture, washed rind.
with a Firm texture, classic nutty flavor.
with a Crumbly texture, tangy flavor.
with a Firm texture, nutty flavor.
with a Firm texture, vibrant orange color.
with a Semi-soft texture, pungent aroma.
with a Firm texture, nutty flavor.
with a Soft creamy texture, wrinkled rind.
with a Creamy texture, washed rind.
with a Fresh milky flavor, stretchy texture.
with a Semi-hard texture, intricate flower shape.
with a Soft creamy texture, fresh mild flavor.""".splitlines()

    

    def create_prompts(self, labels_names):
        prompts = {}
        for i, label in enumerate(labels_names):
            prompts[label] = []
            descriptions = self.generate_prompt_description(label)
            for elt in descriptions:
                prompt_description = elt + self.lines[i]
                print(prompt_description)
                prompts[label].append(
                    {
                        "prompt": prompt_description,
                        "num_images": min(1, self.num_images_per_label//20),
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