from random import *
import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from utils import normalization, renormalization, rounding
import warnings

warnings.filterwarnings("ignore")


class eICU_Dataset(Dataset):
    def __init__(self, use_index_list=None, missing_ratio=0.0, seed=0):
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []

        observed_data = pd.read_csv("./data/0624eicu.csv")
        # observed_data = pd.read_csv(path)
        observed_data = observed_data.drop(["diagnosisstring"], axis=1)
        self.status = observed_data["status"]
        self.patient_id = observed_data["patientunitstayid"]
        observed_data = observed_data.drop(["status"], axis=1)
        observed_data = observed_data.drop(["patientunitstayid"], axis=1)

        # test0 剔除离散特征,不带label和id_
        # observed_data = observed_data.drop(["gender"], axis=1)
        # observed_data = observed_data.drop(["ethnicity"], axis=1)

        observed_data["age"][observed_data["age"] == "> 89"] = 90
        observed_data["age"] = observed_data["age"].astype('float32')
        # 剔除性别未知，相当于缺失
        observed_data["gender"][observed_data["gender"] == "Unknown"] = np.nan

        # 将性别编码
        observed_data["gender"][observed_data["gender"] == "Female"] = 0
        observed_data["gender"][observed_data["gender"] == "Male"] = 1

        # 将种族编码
        observed_data["ethnicity"][observed_data["ethnicity"] == "African American"] = 0
        observed_data["ethnicity"][observed_data["ethnicity"] == "Asian"] = 1
        observed_data["ethnicity"][observed_data["ethnicity"] == "Caucasian"] = 2
        observed_data["ethnicity"][observed_data["ethnicity"] == "Hispanic"] = 3
        observed_data["ethnicity"][observed_data["ethnicity"] == "Native American"] = 4
        observed_data["ethnicity"][observed_data["ethnicity"] == "Other/Unknown"] = 5
        observed_data = pd.get_dummies(observed_data)
        # print(observed_data.dtypes)

        # observed_data[["age", "ethnicity"]] = observed_data[["ethnicity", "age"]]
        observed_data = np.array(observed_data)
        observed_data = observed_data.astype("float32")
        # observed_data[:, 39:] = observed_data[:, 39:].astype("uint8")
        self.observed_masks = 1 - np.isnan(observed_data)
        observed_data[:, 41:] = observed_data[:, 41:].astype("uint8")
        self.gt_masks = self.observed_masks
        self.observed_values = np.array(observed_data)
        self.observed_masks = np.array(self.observed_masks)
        self.gt_masks = np.array(self.gt_masks)
        # print(type(self.observed_values), type(self.observed_masks), type(self.gt_masks), self.observed_values.shape,
        #        self.observed_masks.shape, self.gt_masks.shape)

        # randomly set some percentage as ground-truth
        masks = self.observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices] = False
        self.gt_masks = masks.reshape(self.observed_masks.shape)
        # print(self.gt_masks.shape)

        self.observed_values = np.nan_to_num(self.observed_values)
        self.observed_masks = self.observed_masks.astype("float32")
        self.gt_masks = self.gt_masks.astype("float32")

        # 标准化
        self.observed_values[:, :41], self.norm_parameters = normalization(self.observed_values[:, :41], self.observed_masks[:, :41])
        self.observed_values = self.observed_values.astype("float32")
        # print(self.norm_parameters)

        # renormalization
        # data = renormalization(self.observed_values, self.norm_parameters)
        # print("")
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
            print("")
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "patient_id": self.patient_id[index],
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "status": self.status[index],
            "norm_parameters": self.norm_parameters,
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1234, nfold=None, batch_size=16, missing_ratio=0.1):
    dataset = eICU_Dataset(missing_ratio=missing_ratio, seed=seed)

    indlist = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    # start = (int)(nfold * 0.2 * len(dataset))
    # end = (int)((nfold + 1) * 0.2 * len(dataset))
    # test_index = indlist[start:end]
    # remain_index = np.delete(indlist, np.arange(start, end))
    # print(len(test_index))

    # np.random.seed(seed)
    # np.random.shuffle(remain_index)
    # num_train = (int)(len(remain_index) * 0.7)
    # train_index = remain_index[:num_train]
    # valid_index = remain_index[num_train:]
    # print(len(train_index))
    # print(len(valid_index))

    num_train = (int)(len(indlist) * 0.70 + 1)
    train_index = indlist[:num_train]
    valid_index = indlist[num_train:]

    dataset = eICU_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    # print(len(dataset))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = eICU_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    # test_dataset = eICU_Dataset(
    #     use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    # )
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader


class MIMIC_Dataset(Dataset):
    def __init__(self, use_index_list=None, missing_ratio=0.0, seed=0):
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []

        observed_data = pd.read_csv("./data/0918mimic.csv")
        # observed_data = pd.read_csv(path)
        observed_data = observed_data.drop(["long_title"], axis=1)
        self.status = observed_data["status"]
        self.patient_id = observed_data["patientunitstayid"]
        observed_data = observed_data.drop(["status"], axis=1)
        observed_data = observed_data.drop(["patientunitstayid"], axis=1)
        observed_data["gender"][observed_data["gender"] == "Unknown"] = np.nan

        # test0 剔除离散特征,不带label和id_
        # observed_data = observed_data.drop(["gender"], axis=1)
        # observed_data = observed_data.drop(["race"], axis=1)

        # 将性别编码
        observed_data["gender"][observed_data["gender"] == "F"] = 0
        observed_data["gender"][observed_data["gender"] == "M"] = 1

        # 将种族编码
        observed_data["race"][observed_data["race"] == "African American"] = 0
        observed_data["race"][observed_data["race"] == "Asian"] = 1
        observed_data["race"][observed_data["race"] == "Caucasian"] = 2
        observed_data["race"][observed_data["race"] == "Hispanic"] = 3
        observed_data["race"][observed_data["race"] == "Native American"] = 4
        observed_data["race"][observed_data["race"] == "Other/Unknown"] = 5
        observed_data = pd.get_dummies(observed_data)
        # print(observed_data.dtypes)

        # observed_data[["age", "ethnicity"]] = observed_data[["ethnicity", "age"]]
        observed_data = np.array(observed_data)
        observed_data = observed_data.astype("float32")
        # observed_data[:, 39:] = observed_data[:, 39:].astype("uint8")

        self.observed_masks = 1 - np.isnan(observed_data)
        observed_data[:, 41:] = observed_data[:, 41:].astype("uint8")
        self.gt_masks = self.observed_masks
        self.observed_values = np.array(observed_data)
        self.observed_masks = np.array(self.observed_masks)
        self.gt_masks = np.array(self.gt_masks)
        # print(type(self.observed_values), type(self.observed_masks), type(self.gt_masks), self.observed_values.shape,
        #        self.observed_masks.shape, self.gt_masks.shape)

        # randomly set some percentage as ground-truth
        masks = self.observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices] = False
        self.gt_masks = masks.reshape(self.observed_masks.shape)
        # print(self.gt_masks.shape)

        self.observed_values = np.nan_to_num(self.observed_values)
        self.observed_masks = self.observed_masks.astype("float32")
        self.gt_masks = self.gt_masks.astype("float32")

        # 标准化
        self.observed_values[:, :41], self.norm_parameters = normalization(self.observed_values[:, :41], self.observed_masks[:, :41])
        self.observed_values = self.observed_values.astype("float32")
        # print(self.norm_parameters)

        # renormalization
        # data = renormalization(self.observed_values, self.norm_parameters)
        # print("")
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
            print("")
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "patient_id": self.patient_id[index],
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "status": self.status[index],
            "norm_parameters": self.norm_parameters,
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_ext_dataloader(seed=0, batch_size=16, missing_ratio=0):
    dataset = MIMIC_Dataset(missing_ratio=missing_ratio, seed=seed)

    indlist = np.arange(len(dataset))
    ext_dataset = MIMIC_Dataset(
        use_index_list=indlist, missing_ratio=missing_ratio, seed=seed
    )
    ext_loader = DataLoader(ext_dataset, batch_size=batch_size, shuffle=0)

    return ext_loader


a, b = get_dataloader(seed=1234, nfold=0, batch_size=16, missing_ratio=0.1)
print("")
#
# ext_loader = get_ext_dataloader(seed=0, batch_size=16, missing_ratio=0)
# print("")
