from diffusers import DDIMPipeline
import accelerate
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


model_id = "google/ddpm-celebahq-256"

def generate(specs):
    num, seed = specs
    torch.manual_seed(seed)
    pipeline = DDIMPipeline.from_pretrained(model_id)
    generated_images = pipeline(num).images
    
    return list(generated_images)

def main():
    results = []
    args = ((1, 3) ,(1, 4), (1, 6))
    with ProcessPoolExecutor() as executor:
        for result in executor.map(generate, args):
            results += result
    
    for i, pic in enumerate(results):
        pic.save(f"ab{i}.png")
        
if __name__ == '__main__':
    main()