import numpy as np
import torch
import visualize as vis

def add_noise(image, blackout_prob):
    image_size = image.shape[0] * image.shape[1]
    keep_idxs = np.random.choice(2, image_size, p=[1 - blackout_prob, blackout_prob]).reshape((image.shape[0], image.shape[1])).astype(bool)
    image[keep_idxs, :] = 0

    return image

def add_noise_torch(image_batch, blackout_prob):
    image_size = image_batch.shape[2] * image_batch.shape[3]
    for i in range(image_batch.shape[0]):
        keep_idxs = np.random.choice(2, image_size, p=[1 - blackout_prob, blackout_prob]).\
            reshape((image_batch.shape[2], image_batch.shape[3])).astype(bool)
        image_numpy = image_batch[i].numpy()
        image_numpy[:, keep_idxs] = 0
        image_batch[i] = torch.from_numpy(image_numpy)
    return image_batch