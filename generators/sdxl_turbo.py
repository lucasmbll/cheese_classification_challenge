from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

class SDXLTurboGenerator:
    def __init__(self):
        model_id = "stabilityai/sdxl-turbo"
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        self.pipe = self.pipe.to(device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.num_inference_steps = 4
        self.guidance_scale = 0

    def generate(self, prompts):
        images = self.pipe(
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images
        return images
