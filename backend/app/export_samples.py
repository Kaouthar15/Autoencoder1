import os
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

def save_img(x, path):
    arr = (x.squeeze().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

def main():
    os.makedirs("samples", exist_ok=True)
    ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

    # export 5 T-shirts (label 0) + 5 shoes (label 9) pour comparer
    counts = {0: 0, 9: 0}
    for i in range(len(ds)):
        x, y = ds[i]
        if y in counts and counts[y] < 5:
            save_img(x, f"samples/label_{y}_{counts[y]}.png")
            counts[y] += 1
        if counts[0] == 5 and counts[9] == 5:
            break

    print("Saved samples in backend/samples/")

if __name__ == "__main__":
    main()