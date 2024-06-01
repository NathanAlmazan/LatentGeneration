import os
import uuid
import torch
import numpy as np

from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sampler import KLMSSampler
from diffusion import Diffusion
from autoencoder import Decoder

# load vocabulary
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
# load tokens
with open(os.path.join("checkpoints", "labels.txt"), "r") as file:
    labels = file.readlines()
    labels = [label.strip().lower() for label in labels]
    labels = [set([stemmer.stem(word) for word in word_tokenize(label) if word not in stop_words]) for label in labels]


def tokenize(prompt: str, count=8):
    prompt = set([stemmer.stem(word) for word in word_tokenize(prompt.lower()) if word not in stop_words])
    # find tokens
    results = []
    for idx, label in enumerate(labels):
        if len(label.intersection(prompt)) > 0:
            results.append(idx)
    # repeat tokens to fit number of expected results
    if len(results) < count:
        repeat = (count // len(results)) + 1
        results = results * repeat
    results = results[:count]
    
    return results


def generate_images(labels: list[int]):
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