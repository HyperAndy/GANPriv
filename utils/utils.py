import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import pandas as pd
import random
import time
from sklearn import feature_selection, random_projection, svm
from sklearn.metrics import accuracy_score


class Data(Dataset):
    def __init__(self, filepath_n, filepath_t):
        n_x, n_y = Data.loader(filepath_n, 1)
        t_x, t_y = Data.loader(filepath_t, 0)

        self.x = np.concatenate((n_x, t_x))
        self.y = np.concatenate((n_y, t_y))
        # self.loader = Data.preprocess_img()

    def __getitem__(self, index):
        x = self.x[index]
        # x = self.loader(x)
        x = Data.preprocess(x)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)

    @staticmethod
    def preprocess(x):
        return torch.from_numpy(x)

    @staticmethod
    def preprocess_img(mean=0.0, std=1.0, normalize=False):
        if normalize:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            return transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def loader(filepath, label):
        data = pd.read_csv(filepath, sep="\t", index_col=0)
        data = pd.DataFrame(data.values.T, index=data.columns, columns=data.index)
        for col in list(data.columns[data.isnull().sum() > 0]):
            data[col].fillna(data[col].mean(), inplace=True)
            data[col] = (data[col] - data[col].mean()) / (data[col].std())
        data['label'] = label
        return data.drop('label', axis=1).values, data['label'].values


def floatTensor(cuda):
    return torch.cuda.FloatTensor if cuda else torch.FloatTensor


def longTensor(cuda):
    return torch.cuda.LongTensor if cuda else torch.LongTensor


def save(path):
    os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    data = Data('../RucPriv/BC-TCGA/BC-TCGA-Normal.txt', '../RucPriv/BC-TCGA/BC-TCGA-Tumor.txt')
