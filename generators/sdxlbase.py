from diffusers import DiffusionPipeline, EulerDiscreteScheduler, UNet2DConditionModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class SDXLGenerator:
    def __init__(self, use_cpu_offload=False):
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        
        self.base_pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
        self.base_pipe = self.base_pipe.to(device)
        
        self.refiner_pipe = DiffusionPipeline.from_pretrained(refiner_model_id, torch_dtype=torch.float16)
        self.refiner_pipe = self.refiner_pipe.to(device)
        
        self.base_pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.base_pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.refiner_pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.refiner_pipe.scheduler.config, timestep_spacing="trailing"
        )
        
        self.base_pipe.set_progress_bar_config(disable=True)
        self.refiner_pipe.set_progress_bar_config(disable=True)
        
        if use_cpu_offload:
            self.base_pipe.enable_sequential_cpu_offload()
            self.refiner_pipe.enable_sequential_cpu_offload()
        
        self.num_inference_steps = 20
        self.guidance_scale = 10

    def generate(self, prompts):
        base_images = self.base_pipe(
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale
        ).images
        
        # Use the base images as input to the refiner
        refined_images = self.refiner_pipe(
            prompts,
            image=base_images,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale
        ).images
        
        return refined_images