from diffusers import DDIMPipeline
import accelerate
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from numpy.random import default_rng

model_id = "google/ddpm-celebahq-256"
def generate(specs):
    num, seed = specs
    torch.manual_seed(seed)
    pipeline = DDIMPipeline.from_pretrained(model_id)
    generated_images = pipeline(num).images
    
    return list(generated_images)

def main():
    seed = 1263
    rng = default_rng(seed)
    numbers = rng.choice(100000, size=20, replace=False)
    results = []
    args = [(5, n) for n in numbers]
    print(args)
    with ProcessPoolExecutor() as executor:
        for result in executor.map(generate, args):
            results += result
    
    for i, pic in enumerate(results):
        pic.save(f"../generated_images/celebA_generated/celebA_generated_{i}.png")
        
if __name__ == '__main__':
    main()