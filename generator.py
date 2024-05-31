import os
import uuid
import torch
import numpy as np

from PIL import Image
from sampler import KLMSSampler
from diffusion import Diffusion
from autoencoder import Decoder


def generate(labels: list[int]):
    # get label embedding
    embeddings = np.load(os.path.join('checkpoints', 'embeddings.npy'), allow_pickle=True)
    embeddings = torch.from_numpy(embeddings[labels])
    embeddings = embeddings.float().cuda()
    
    # generate initial noise
    sampler = KLMSSampler(n_inference_steps=50)
    latents = torch.randn((len(labels), 4, 64, 64))
    latents *= sampler.initial_scale
    latents = latents.float().cuda()
    
    # denoise image using diffusion model
    diffusion = Diffusion().cuda()
    diffusion.load_state_dict(torch.load(os.path.join('checkpoints', 'diffusion.pt')))
    diffusion.eval()
    
    with torch.autocast('cuda') and torch.inference_mode():
        for timestep in sampler.timesteps:
            input_latents = latents * sampler.get_input_scale()
            time_embedding = sampler.get_time_embedding(timestep).float().cuda()
            output = diffusion(input_latents, embeddings, time_embedding)
            
            latents = sampler.step(latents, output)

    # free memory
    diffusion.cpu()
    del diffusion
    torch.cuda.empty_cache()
    
    # decode latent space to image
    decoder = Decoder().cuda()
    decoder.load_state_dict(torch.load(os.path.join('checkpoints', 'decoder.pt')))
    decoder.eval()
    
    with torch.autocast('cuda') and torch.inference_mode():
        images = decoder(latents)
        
    # free memory
    decoder.cpu()
    del decoder
    torch.cuda.empty_cache()
    
    # rescale image from -1 to 1 to 0 to 255
    images = images.detach().cpu()
    images = ((255.0 * images) / 2) + 127.5
    images = images.clamp(0, 255)
    # permute image to (batch, height, width, channel)
    images = images.permute(0, 2, 3, 1)
    # convert to numpy integer
    images = images.numpy().astype(np.uint8)
    
    # save images
    files = []
    for image in images:
        file = f"{uuid.uuid4()}.png"
        image = Image.fromarray(image)
        image.save(os.path.join('generated', file))
        files.append(file)

    return files