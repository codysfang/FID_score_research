from PIL import Image
import glob
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.util import random_noise
import numpy as np
import random
import seaborn as sns
import itertools
import sys
sys.path.append("evaluations")

def compute_FID(image_r, image_f, data_size = 5000, batch_num=10):
    _ = torch.manual_seed(1982)
    fid = FrechetInceptionDistance().to(torch.device("cuda", 0))
    prev = 0;
    for i in range(0, data_size, data_size//batch_num):
      start = i
      end = min(data_size - 1, i + data_size//batch_num)
      # print(f"updating from {start} to {end}")
      input_r = image_r[start:end].to("cuda")
      input_f = image_f[start:end].to("cuda")
      fid.update(input_r, real=True)
      fid.update(input_f, real=False)
      del input_r
      del input_f
      torch.cuda.empty_cache()
    FID = fid.compute()
    return FID

def load_data_single(loader, sample, dims):
    _, (img, targets) = next(enumerate(loader))
    loaded_tensor = img.expand([sample, 3, dims[0], dims[1]])
    return loaded_tensor

def load_from_dir(location, rand_seed=None, data_size=10000):
    directory = glob.glob(location)
    if rand_seed is not None:
        random.seed(rand_seed)
        directory = random.sample(directory, data_size)
    image_array = []
    for img in directory:
        image = transforms.functional.pil_to_tensor(Image.open(img))
        image_array.append(image)
    return torch.stack(image_array)

def display_image(image):
    display = transforms.functional.to_pil_image(image)
    plt.imshow(display)
    return None

def rotate_chunk(grid, dataset_train, dataset_test):
    x1, y1, x2, y2 = grid
    dataset_test_rotated = torch.clone(dataset_test)
    dataset_test_rotated[:, :, y1:y2,x1:x2] = transforms.functional.rotate(dataset_test_rotated[:, :, y1:y2, x1:x2], 180)
    display = transforms.functional.to_pil_image(dataset_test_rotated[1])
    plt.imshow(display)
    return compute_FID(dataset_train, dataset_test_rotated).item()

def swap_chunks(region_a, region_b, dataset_train, dataset_test):
    xa1, ya1, xa2, ya2 = region_a
    xb1, yb1, xb2, yb2 = region_b
    dataset_test_swapped = torch.clone(dataset_test)
    dataset_test_swapped[:, :, ya1:ya2,xa1:xa2] = dataset_test[:, :, yb1:yb2,xb1:xb2]
    dataset_test_swapped[:, :, yb1:yb2,xb1:xb2] = dataset_test[:, :, ya1:ya2,xa1:xa2]
    display_image(dataset_test_swapped[1])
    return compute_FID(dataset_train, dataset_test_swapped).item()

def encode_image(img):
    transform = transforms.Compose([transforms.PILToTensor()])
    return transform(img)

def load_data(train_loader, test_loader, sample, dims):
    _, (train, targets) = next(enumerate(train_loader))
    _, (test, targets) = next(enumerate(test_loader))
    train_tensor = train.expand([sample, 3, dims[0], dims[1]])
    test_tensor = test.expand([sample, 3, dims[0], dims[1]])
    return train_tensor, test_tensor

def compute_transform_FID(transform, factor, dataset_train, dataset_test, batch_num=10):
    transformed_test = torch.clone(dataset_test)
    transformed_test = transform(transformed_test, factor)
    return compute_FID(dataset_train, transformed_test, batch_num).item()

def plot_FID(factor_array, transformation, transform_name, dataset_train, dataset_test, batch_num=10):
    transformed_FID = [compute_transform_FID(transformation, x, dataset_train, dataset_test, batch_num) for x in factor_array]
    plt.plot(factor_array, transformed_FID)
    plt.xlabel(transform_name)
    plt.ylabel("FID Score")
    return None

def generate_heatmap(dataset_train, dataset_test, path):
    kernel_sizes = np.arange(1, 13, 2)
    coords = np.array(list(itertools.product(kernel_sizes, kernel_sizes)))
    heat_matrix = np.zeros((6, 6))
    for c in coords:
        heat_matrix[c[0]//2, c[1]//2] = compute_transform_FID(
        transforms.functional.gaussian_blur, tuple(c), dataset_train, dataset_test)
    plot = sns.heatmap(heat_matrix, annot=True, xticklabels=kernel_sizes, yticklabels=kernel_sizes, cmap=sns.diverging_palette(220, 20, as_cmap=True))
    plot.get_figure().savefig(path, dpi=400)
    return None

def noisify_FID(amount, dataset_train, dataset_test, print_pic=False, seed=128):
    noised_test = torch.clone(dataset_test)
    noised_test = torch.tensor(random_noise(noised_test, mode="s&p", seed=seed, amount=amount))
    noised_test = noised_test*255
    noised_test = noised_test.type(torch.uint8)
    if print_pic:
        display = transforms.functional.to_pil_image(noised_test[1])
        plt.imshow(display)
    return compute_FID(dataset_train, noised_test).item()

def save_as_npz(dataset_train, dataset_test):
    dataset_train_copy = dataset_train.movedim(1,-1).numpy()
    dataset_test_copy = dataset_test.movedim(1,-1).numpy()
    np.savez("evaluations/train.npz", dataset_train_copy)
    np.savez("evaluations/generate.npz", dataset_test_copy)
    return None;
    

# def compute_sFID():
#     print(evaluator.INCEPTION_V3_PATH)

#     config = tf.ConfigProto(
#         allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
#     )
#     config.gpu_options.allow_growth = True
#     evaluator = Evaluator(tf.Session(config=config))

#     return sample_stats_spatial.frechet_distance(ref_stats_spatial)