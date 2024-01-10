import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from einops import rearrange
import os
from pqdm.processes import pqdm
from simu_wrap import wrap


class Dataset_4DFlow(Dataset):
    def __init__(self, path, ):
        self.path = path
        data_path = []
        for i in range(1, 8):
            data_path.append(os.path.join(path, f"imgt_volN{i}.mat"))
        self.data = pqdm(data_path, self.load_img, n_jobs=20)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def load_img(self, path):
        return self.preprocess(loadmat(path)['img'])

    def preprocess(self, img):
        img = rearrange(img, "x y z p c -> c p x y z")
        img[1:] = np.angle(img[1:]*img[0].conj())
        img[0] = np.abs(img[0])
        return img.real.astype(np.float32)


if __name__ == "__main__":
    dataset = Dataset_4DFlow("data")
    print(len(dataset))
    img = dataset[5]
    print(img.shape)

    for i, _ratio in enumerate(np.linspace(0.2, 1, 9)):
        img_wrap = wrap(img[1, 4, :, :, 10], _ratio)
        plt.subplot(3, 3, i+1)
        plt.imshow(img_wrap)
        plt.title(f"ratio: {_ratio}")
        plt.colorbar()

    plt.savefig("plotfigure/_.png")
    plt.show()
