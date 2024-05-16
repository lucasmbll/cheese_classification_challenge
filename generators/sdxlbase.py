from diffusers import DiffusionPipeline, EulerDiscreteScheduler, UNet2DConditionModel, AutoencoderTiny
import torch
import xformers
# import triton
# from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
# from DeepCache import DeepCacheSDHelper
# import oneflow as flow
# from onediff.infer_compiler import oneflow_compile




device = "cuda" if torch.cuda.is_available() else "cpu"

class SDXLGenerator:
    def __init__(self, use_cpu_offload=False):
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

        # Test of VAE optimization
        vae = AutoencoderTiny.from_pretrained(
        'madebyollin/taesdxl',
        torch_dtype=torch.float16,
        ).to('cuda')

        
        self.base_pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16", vae=vae)
        
        self.refiner_pipe = DiffusionPipeline.from_pretrained(refiner_model_id, torch_dtype=torch.float16, variant="fp16", vae=vae)
        
        
        self.base_pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.base_pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.refiner_pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.refiner_pipe.scheduler.config, timestep_spacing="trailing"
        )
        
        self.base_pipe.set_progress_bar_config(disable=True)
        self.refiner_pipe.set_progress_bar_config(disable=True)
        
        if use_cpu_offload:
            self.base_pipe.enable_model_cpu_offload()
            self.refiner_pipe.enable_model_cpu_offload()
        else:
            self.base_pipe = self.base_pipe.to(device)
            self.refiner_pipe = self.refiner_pipe.to(device)
        
        self.num_inference_steps = 15
        self.guidance_scale = 5

        # Test of StableFast optimization
        """config = CompilationConfig.Default()
        config.enable_xformers = True
        config.enable_triton = True
        config.enable_cuda_graph = True
        self.base_pipe = compile(self.base_pipe, config)
        self.refiner_pipe = compile(self.refiner_pipe, config)"""

        # Test of various optimizations
        #self.base_pipe.unet = torch.compile(self.base_pipe.unet, mode='reduce-overhead', fullgraph=True)
        #self.refiner_pipe.unet = torch.compile(self.refiner_pipe.unet, mode='reduce-overhead', fullgraph=True)
        # self.base_pipe.unet = oneflow_compile(self.base_pipe.unet)
        # self.refiner_pipe.unet = oneflow_compile(self.refiner_pipe.unet)
        self.base_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        self.refiner_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)

        # Test of DeepCache optimization : less time but worse images and more memory
        """helper = DeepCacheSDHelper(pipe=self.base_pipe)
        helper.set_params(cache_interval=10, cache_branch_id=0)
        helper.enable()
        helper = DeepCacheSDHelper(pipe=self.refiner_pipe)
        helper.set_params(cache_interval=10, cache_branch_id=0)
        helper.enable()"""


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