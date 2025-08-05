import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.utils import set_random_seed, calc
from tqdm import tqdm, trange

import random


class DatasetLB(Dataset):  # Dataset For Linear Probiling
    def __init__(self, dataset, mean=None, std=None, max_val= None, min_val=None):
        super().__init__()
        self.embedding = []
        self.labels = []
        for emedding, label in tqdm(dataset):
            # if label < 100 or label > 600:
            #     continue
            self.embedding.append(emedding)
            self.labels.append(label)

        if mean is None:
            # plot labels  using mathplotlib
            import matplotlib.pyplot as plt
            plt.plot(self.labels)
            plt.show()

        if mean is None:
            self.mean = mean = np.mean(self.labels, axis=0)
            self.std = std = np.std(self.labels, axis=0)
            print(mean, std)

        self.embeddings = torch.tensor(self.embedding, dtype=torch.float32)

        # self.labels = (self.labels - mean) / std
        # self.labels = 2 * (self.labels - min_val) / (max_val - min_val) - 1
        
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]


def normalize(dataset):
    dataset = np.array(dataset)
    mean = dataset.mean()
    std = dataset.std()
    if std == 0:
        return dataset
    dataset = (dataset - mean) / std
    return dataset


def gen_mask(now_len):
    num_zeros = random.randint(3, 20)

    # 生成全为 1 的 mask
    mask = torch.ones(now_len, dtype=torch.int32)

    # 随机选择 num_zeros 个位置
    zero_indices = torch.randperm(now_len)[:num_zeros]

    # 将选定位置的值设置为 0
    mask[zero_indices] = 0

    return mask


def gen_new_mask(mask, now_zero):
    # print(f"now_zero: {now_zero}, mask: {mask}")
    while True:
        mask_rate = np.random.uniform(0.1, 0.9)
        zero_mask = torch.rand(now_zero) < mask_rate
        zero_mask = zero_mask.int()
        # print(zero_mask)
        if torch.sum(zero_mask) > 0 and torch.sum(zero_mask) < now_zero:
            break

    now_mask = mask.int()
    cnt = 0
    for now_id in range(len(now_mask)):
        if now_mask[now_id] == 0:
            if zero_mask[cnt] == 1:
                # print(f"now_id: {now_id}, cnt: {cnt}, now_mask: {now_mask[now_id]}, zero_mask: {zero_mask[cnt]}")
                now_mask[now_id] = 3
            cnt += 1
    # print(mask)
    return now_mask


class DatasetMAE(Dataset):  # Dataset For Masked Auto Encoder
    def __init__(self, datasets, seed=42, few_shot=-1, test=0, FT=0):
        super().__init__()

        self.labels = []
        # 首先 eval和 test 的部分一定是mask着的
        # 其次，train一部分是mask着的，一部分是没mask的
        self.masks = []

        self.zeros = []
        self.test = test
        self.FT = FT

        for dataset in tqdm(datasets):
            self.labels.append(torch.FloatTensor(dataset))

            if test == 1:
                set_random_seed(seed)

                id_set = np.arange(self.labels[-1].shape[-1])

                # split the dataset into train and test
                train_size = int(0.7 * len(id_set)) if few_shot == -1 else few_shot
                val_size = int(0.8 * len(id_set)) - train_size
                test_size = len(id_set) - val_size - train_size

                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(id_set, [train_size, val_size,
                                                                                                  test_size])
                mask = torch.zeros(self.labels[-1].shape[-1])

                mask[val_dataset] = 2
                mask[test_dataset] = 1

                self.masks.append(mask)

                cnt = 0
                for now_id in range(len(mask)):
                    if mask[now_id] == 0:
                        cnt += 1
                self.zeros.append(cnt)

                # 随机mask掉一部分train，即mask==0的，预测train的另一部分
                # val = 2 用于验证，test = 1 用于测试

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 非mask的部分，预测masked的了
        now_dataset = self.labels[index]
        # sample from a distribution between 0.1-0.9
        if self.test == 1:
            mask = self.masks[index]
        else:
            mask_rate = np.random.uniform(0.01, 0.99)
            # mask_rate = np.random.uniform(0.001, 0.999)
            
            mask = torch.rand(len(now_dataset)) < mask_rate
            # mask = gen_mask(len(now_dataset))
        if self.FT == 0:
            return now_dataset, mask
        else:
            now_mask = gen_new_mask(mask, self.zeros[index])
            return now_dataset, now_mask


class DatasetFT(Dataset):  # Dataset For FineTuning
    def __init__(self, datasets, seed=42, few_shot=-1, test=0):
        super().__init__()

        self.labels = []
        # 首先 eval和 test 的部分一定是mask着的
        # 其次，train一部分是mask着的，一部分是没mask的
        self.masks = []
        self.zeros = []
        self.test = test


        for dataset in tqdm(datasets):
            # dataset, _ = robust_normalize(dataset)
            self.labels.append(torch.FloatTensor(dataset))

            if test == 1:
                set_random_seed(seed)

                id_set = np.arange(self.labels[-1].shape[-1])

                # split the dataset into train and test
                train_size = int(0.7 * len(id_set)) if few_shot == -1 else few_shot
                val_size = int(0.8 * len(id_set)) - train_size
                test_size = len(id_set) - val_size - train_size

                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(id_set, [train_size, val_size,
                                                                                                  test_size])

                mask = torch.zeros(self.labels[-1].shape[-1])

                mask[val_dataset] = 2
                mask[test_dataset] = 1

                self.masks.append(mask)

                cnt = 0
                for now_id in range(len(mask)):
                    if mask[now_id] == 0:
                        cnt += 1
                self.zeros.append(cnt)

                # 随机mask掉一部分train，即mask==0的，预测另一部分？

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 非mask的部分，预测masked的了
        now_dataset = self.labels[index]
        # sample from a distribution between 0.1-0.9\
        mask = self.masks[index]

        now_mask = gen_new_mask(mask, self.zeros[index])

        return now_dataset, now_mask

