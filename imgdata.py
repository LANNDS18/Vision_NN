import os
from torch.utils.data import Dataset
import numpy as np
from skimage import io


class imageDataset(Dataset):

    def __init__(self, root_dir, file_path, imSize=250):
        self.imPath = np.load(file_path)
        self.root_dir = root_dir
        self.imSize = imSize
        self.file_path = file_path

    def __len__(self):
        return len(self.imPath)

    def __getitem__(self, idx):
        im = io.imread(os.path.join(self.root_dir, self.imPath[idx]))  # read the image

        if len(im.shape) < 3:  # if there is grey scale image, expand to r,g,b 3 channels
            im = np.expand_dims(im, axis=-1)
            im = np.repeat(im, 3, axis=2)

        img_folder = self.imPath[idx].split('/')[-2]
        if img_folder == 'faces':
            label = 0
        elif img_folder == 'dog':
            label = 1
        elif img_folder == 'airplanes':
            label = 2
        elif img_folder == 'keyboard':
            label = 3
        elif img_folder == 'cars':
            label = 4

        img = np.zeros([3, im.shape[0], im.shape[1]])  # reshape the image from HxWx3 to 3xHxW
        img[0, :, :] = im[:, :, 0]
        img[1, :, :] = im[:, :, 1]
        img[2, :, :] = im[:, :, 2]

        imNorm = np.zeros([3, im.shape[0], im.shape[1]])  # normalize the image
        imNorm[0, :, :] = (img[0, :, :] - np.max(img[0, :, :])) / (np.max(img[0, :, :]) - np.min(img[0, :, :])) - 0.5
        imNorm[1, :, :] = (img[1, :, :] - np.max(img[1, :, :])) / (np.max(img[1, :, :]) - np.min(img[1, :, :])) - 0.5
        imNorm[2, :, :] = (img[2, :, :] - np.max(img[2, :, :])) / (np.max(img[2, :, :]) - np.min(img[2, :, :])) - 0.5

        return {
            'imNorm': imNorm.astype(np.float32),
            'label': label
        }


class DefaultTrainSet(imageDataset):
    def __init__(self, **kwargs):
        default_path = 'Assignment2_SupplementaryMaterials/img_list_train.npy'
        root_dir = 'Assignment2_SupplementaryMaterials/data'
        super(DefaultTrainSet, self).__init__(root_dir, file_path=default_path, imSize=250, **kwargs)


class DefaultTestSet(imageDataset):
    def __init__(self, **kwargs):
        default_path = 'Assignment2_SupplementaryMaterials/img_list_test.npy'
        root_dir = 'Assignment2_SupplementaryMaterials/data'
        super(DefaultTestSet, self).__init__(root_dir, file_path=default_path, imSize=250, **kwargs)
