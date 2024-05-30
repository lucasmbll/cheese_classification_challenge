import torch
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from safetensors.torch import load_file as load_safetensor
import data.dataset_generators
import data.dataset_generators.genAdib
from PIL import Image
from tqdm import tqdm
import open_clip
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class DBSd15Generator:
    def __init__(self, cheese='BEAUFORT', use_cpu_offload=False):
        # Load the base Stable Diffusion model
        model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

        # Load the LoRA weights from the .safetensor file
        lora_weights_path = f"./db_models/{cheese}/pytorch_lora_weights.safetensors"
        print(f"Loading LoRA weights from {lora_weights_path}")
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


def score_zeroshot(image, cheese, cheese_labels, score_lim=0.1):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device)

    image = preprocess(image).unsqueeze(0).to(device)

    # Tokenize the cheese labels
    text = tokenizer(cheese_labels).to(device)

    # Compute image and text features
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    score = text_probs.flatten()[cheese_labels.index(cheese)]
    if score > score_lim:
        return True
    else:
        return False
    

def generate_lora(batch_size=1, output_dir="dataset/train/dreambooth"):
    file_path = '/Data/mellah.adib/cheese_classification_challenge/list_of_cheese.txt'
    with open(file_path, 'r') as file:
        cheese_names = file.read().splitlines()
    
    dataset_generator = data.dataset_generators.genAdib.GptPrompts(DBSd15Generator(), batch_size, output_dir, num_images_per_label=100)
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
                    good = False
                    while not good:
                        # print("Generating images : not good atm")
                        images = pipe.generate(batch)
                        # good = score_zeroshot(images[0], label, cheese_names)
                        good = True
                    dataset_generator.save_images(images, label, image_id_0)
                    image_id_0 += len(images)
                    pbar.update(1)
                pbar.close()



class DBSd15Generator2:
    def __init__(self, cheese='CAMEMBERT', use_cpu_offload=False):
        # Load the base Stable Diffusion model
        path = f"./db_models/val_sorted/{cheese}"
        self.pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

        print(f"Loading model from {path}")

        self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        if use_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        self.num_inference_steps = 60
        self.guidance_scale = 10

    def generate(self, prompts, variation=0):
        with torch.autocast(device):
            images = self.pipe(
                prompts,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale+variation,
            ).images

        return images


def generate_images(batch_size=1, output_dir="dataset/train/dreambooth4"):
    file_path = '/Data/mellah.adib/cheese_classification_challenge/list_of_cheese.txt'
    with open(file_path, 'r') as file:
        cheese_names = file.read().splitlines()

    # Initialize your dataset generator
    dataset_generator = data.dataset_generators.genAdib.Toretrain(DBSd15Generator2(), 
                                                                    batch_size=batch_size, output_dir=output_dir, num_images_per_label=100)
    # Create prompts for each cheese
    labels_prompts = dataset_generator.create_prompts(cheese_names)

    for label, label_prompts in labels_prompts.items():
            pipe = DBSd15Generator2(label)
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
                    good = False
                    step=0
                    while not good:
                        unvalid = []
                        # print("Generating images : not good atm")
                        if (i%3==0):
                            variation = 0
                        elif (i%3==1):
                            variation = 1
                        else:
                            variation = -1
                        images = pipe.generate(batch, variation)
                        #good = score_zeroshot(images[0], label, cheese_names, 0.1)
                        step+=1
                        #print(step)
                        #unvalid.extend(images)
                        """if step>3:
                            images = unvalid
                            i+=step-1
                            break"""
                        good = True
                    dataset_generator.save_images(images, label, image_id_0)
                    image_id_0 += len(images)
                    pbar.update(1)
                pbar.close()


def test_checkpoint(cheese, num_inference_steps=50, guidance_scale=7.5, nb_check=200, nb=1):
    path = f"./db_models/val_sorted/{cheese}"
    unet = UNet2DConditionModel.from_pretrained(path+f"/checkpoint-{nb_check}/unet")

    pipeline = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", unet=unet, dtype=torch.float16,
    ).to(device)
    
    for i in range(nb):
        image = pipeline("A photo of a {cheese}", num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        # Define the directory and file name
        output_dir = os.path.join("test_db_param_val_sorted", cheese)
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"guid_{guidance_scale}_check_{nb_check}_nbsteps_{num_inference_steps}_{i}.png"
        output_path = os.path.join(output_dir, file_name)
        # Save the image
        image.save(output_path)

def test_model(cheese, num_inference_steps=50, guidance_scale=7.5, nb=1):
    # Load the model
    path = f"./db_models/val_sorted/{cheese}"
    pipeline = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

    for i in range(nb):
        image = pipeline(f"A photo of a {cheese}", num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        # Define the directory and file name
        output_dir = os.path.join("test_db_param_val_sorted", cheese)
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"guid_{guidance_scale}_model_nbsteps_{num_inference_steps}_{i}.png"
        output_path = os.path.join(output_dir, file_name)

        # Save the image
        image.save(output_path)


if __name__ == "__main__":
    """for checkpoint in [200, 400, 600]:
            test_checkpoint("BRIE DE MELUN", 60, 9, checkpoint, nb=10)
    test_model("BRIE DE MELUN", 60, 9, nb=10)"""
    
    generate_images()


    




