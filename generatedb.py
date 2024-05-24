import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file as load_safetensor
import data.dataset_generators
import data.dataset_generators.genAdib
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class DBSd15Generator:
    def __init__(self, cheese='BEAUFORT', use_cpu_offload=False):
        # Load the base Stable Diffusion model
        model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

        # Load the LoRA weights from the .safetensor file
        lora_weights_path = f"./db_models/{cheese}/pytorch_lora_weights.safetensors"
        lora_weights = load_safetensor(lora_weights_path)

        # Apply the LoRA weights to the model
        for key in lora_weights:
            if key in self.pipe.unet.state_dict():
                self.pipe.unet.state_dict()[key].copy_(lora_weights[key])

        self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        if use_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        self.num_inference_steps = 60
        self.guidance_scale = 5

    def generate(self, prompts):
        with torch.autocast(device):
            images = self.pipe(
                prompts,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            ).images

        return images



def generate(batch_size=1, output_dir="dataset/train/dreambooth"):
    file_path = 'C:/Users/adib4/OneDrive/Documents/Travail/X/MODAL DL/cheese_classification_challenge/list_of_cheese.txt'
    with open(file_path, 'r') as file:
        cheese_names = file.read().splitlines()
    
    dataset_generator = data.dataset_generators.genAdib.GptPrompts(DBSd15Generator(), batch_size, output_dir)
    labels_prompts = dataset_generator.create_prompts(cheese_names)

    for label, label_prompts in labels_prompts.items():
            pipe = DBSd15Generator(label)
            image_id_0 = 0
            for prompt_metadata in label_prompts:
                num_images_per_prompt = prompt_metadata["num_images"]
                prompt = [prompt_metadata["prompt"]] * num_images_per_prompt
                pbar = tqdm(range(0, num_images_per_prompt, batch_size))
                pbar.set_description(
                    f"Generating images for prompt: {prompt_metadata['prompt']}"
                )
                for i in range(0, num_images_per_prompt, batch_size):
                    batch = prompt[i : i + batch_size]
                    images = pipe.generate(batch)
                    dataset_generator.save_images(images, label, image_id_0)
                    image_id_0 += len(images)
                    pbar.update(1)
                pbar.close()


if __name__ == "__main__":
    generate()

