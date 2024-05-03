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

