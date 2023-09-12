from diffusers import DDIMPipeline, DiffusionPipeline
import accelerate
import torch
from numpy.random import default_rng
import sys

seed = 1263

torch.manual_seed(seed)
generator = torch.Generator(device="cuda")

model_CIFAR = "google/ddpm-cifar10-32"
model_CelebA = "google/ddpm-celebahq-256"
model_ldm = "CompVis/ldm-celebahq-256"


def generate(model_id, specs, pipeline):
    num, seed = specs
    torch.manual_seed(seed)
    pipeline_trained = pipeline.from_pretrained(model_id)
    pipeline_trained.to("cuda")
    generated_images = pipeline_trained(num).images

    return list(generated_images)

def save_image(images, data_name, start_num):
    for i, pic in enumerate(images):
        pic.save(f"./diffusion_images/{data_name}_generated/{data_name}_generated_{start_num + i}.png")
    return None

def batches(seed, batch_num, model_id, data_name, pipeline, generate_size = 5000, start_from=0):
    size = generate_size//batch_num
    rng = default_rng(seed)
    seeds = rng.choice(100000, size=size, replace=False)
    for i, s in enumerate(seeds):
        result = generate(model_id=model_id, specs=(batch_num, s), pipeline = pipeline)
        start = start_from + i * batch_num
        print(f"Saving images from {start} to {start + batch_num - 1}")
        save_image(result, data_name, start)
    return result

def main():
    mode = int(sys.argv[1])
    if mode == 1:
        batches(seed, 1000, model_CIFAR, "cifar", DDIMPipeline, 10000)
    elif mode == 2:
        batches(seed, 125, model_CelebA, "celebA", DDIMPipeline)
    elif mode == 3:
        batches(seed+2, 125, model_CelebA, "celebA", DDIMPipeline, start_from = 5000)
    elif mode == 5:
        batches(seed, 125, model_ldm, "celebA_ldm", DiffusionPipeline)
    elif mode == 6:
        batches(seed+2, 125, model_ldm, "celebA_ldm", DiffusionPipeline, start_from = 5000)


if __name__ == "__main__":
    main()
