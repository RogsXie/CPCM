import torch
from torch.utils.data import Dataset, DataLoader
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

# def add_noise_batch(x: torch.Tensor, noise_type: str, level: float) -> torch.Tensor:
#     """
#     x: [N,C,H,W] float tensor
#     level: 0/0.05/0.1/0.5/1.0
#     """
#     if level is None or level <= 0:
#         return x
#     nt = noise_type.lower()
#
#     if nt == "gaussian":
#         return x + level * torch.randn_like(x)
#
#     if nt == "uniform":
#         return x + (torch.rand_like(x) * 2.0 - 1.0) * level
#
#     if nt == "poisson":
#         # 泊松需要非负；你前面做了 StandardScaler（可能有负数），所以要 clamp
#         # level 越大噪声越强：用 scale = 1/level 映射
#         x_pos = torch.clamp(x, min=0.0)
#         scale = max(1e-6, 1.0 / level)
#         return torch.poisson(x_pos * scale) / scale
#
#     raise ValueError(f"Unknown noise_type: {noise_type}")


class MultiModalDataset(Dataset):
    def __init__(self, gt_path, *src_path, patch_size=(7, 7), transform=None, is_labeled=True ):
    # def __init__(self, gt_path, *src_path, patch_size=(7, 7), transform=None, is_labeled=True,noise_type=None, noise_level=0.0):
        self.transform = transform
        p = Processor()
        n_modality = len(src_path)
        modality_list = []
        in_channels = []
        for i in range(n_modality):
            img, gt = p.prepare_data(src_path[i], gt_path)
            x_patches, y_ = p.get_HSI_patches_rw(img, gt, (patch_size[0], patch_size[1]), is_indix=False, is_labeled=is_labeled)
            n_samples, n_row, n_col, n_channel = x_patches.shape
            scaler = StandardScaler()
            batch_size = 5000
            # # using incremental / batch for very large data
            for start_id in range(0, x_patches.shape[0], batch_size):
                n_batch = x_patches[start_id: start_id+batch_size].shape[0]
                scaler.partial_fit(x_patches[start_id: start_id+batch_size].reshape(n_batch, -1))
            for start_id in range(0, x_patches.shape[0], batch_size):
                shape = x_patches[start_id: start_id+batch_size].shape
                x_temp = x_patches[start_id: start_id+batch_size].reshape(shape[0], -1)
                x_patches[start_id: start_id+batch_size] = scaler.transform(x_temp).reshape(shape)
            x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
            x_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor)

            # if noise_type is not None and noise_level > 0:
            #     x_tensor = add_noise_batch(x_tensor, noise_type, noise_level)

            modality_list.append(x_tensor)
            in_channels.append(n_channel)
        y = p.standardize_label(y_)
        self.gt_shape = gt.shape
        self.data_size = len(y)
        if is_labeled:
            self.n_classes = np.unique(y).shape[0]
        else:
            self.n_classes = np.unique(y).shape[0] - 1  # remove background
        self.y_tensor = torch.from_numpy(y).type(torch.LongTensor)
        self.modality_list = tuple(modality_list)
        self.n_modality = n_modality
        self.in_channels = tuple(in_channels)
        # self.noise_transform = noise_transform

    def __getitem__(self, idx):
        x_list = []
        for i in range(self.n_modality):
            x = self.modality_list[i][idx]

            # if self.noise_transform is not None:
            #     x = self.noise_transform(x)

            if self.transform is not None:
                x_1, x_2 = self.transform(x)  # # conduct transformation on a single modality
                x_list.append(x_1)
                x_list.append(x_2)
            else:
                x_list.append(x)
        if self.n_modality >= 2 and len(x_list) > 2:  # # when modality >= 2, i.e., 4 augs
            x_list = (x_list[0::2], x_list[1::2])
        if self.n_modality == 1 and len(x_list) == 2:
            x_list = ([x_list[0]], [x_list[1]])
        y = self.y_tensor[idx]
        return x_list, y

    def __len__(self):
        return self.data_size

class MultiModalDatasettest(Dataset):

    def __init__(self, gt_path, *src_path, patch_size=(7, 7), transform=None, is_labeled=True):
        self.transform = transform
        p = Processor()
        n_modality = len(src_path)
        modality_list = []
        in_channels = []

        for i in range(n_modality):
            img, gt = p.prepare_data(src_path[i], gt_path)

            # ---- 原始 patch 提取 ----
            x_patches, y_ = p.get_HSI_patches_rw(
                img, gt,
                (patch_size[0], patch_size[1]),
                is_indix=False,
                is_labeled=is_labeled
            )

            # =====================================================
            # 🔥 关键修改：只保留有标签的 patch
            # 假设 y==0 是背景（如你需要改成 y==-1 或 y==255 可以告诉我）
            # =====================================================
            valid_indices = (y_ != 0)
            x_patches = x_patches[valid_indices]
            y_ = y_[valid_indices]
            # =====================================================

            n_samples, n_row, n_col, n_channel = x_patches.shape

            # ---- 标准化 ----
            scaler = StandardScaler()
            batch_size = 5000
            for start_id in range(0, x_patches.shape[0], batch_size):
                n_batch = x_patches[start_id:start_id + batch_size].shape[0]
                scaler.partial_fit(
                    x_patches[start_id:start_id + batch_size].reshape(n_batch, -1)
                )

            for start_id in range(0, x_patches.shape[0], batch_size):
                shape = x_patches[start_id:start_id + batch_size].shape
                x_temp = x_patches[start_id:start_id + batch_size].reshape(shape[0], -1)
                x_patches[start_id:start_id + batch_size] = scaler.transform(x_temp).reshape(shape)

            # ---- 转成 (N, C, H, W) ----
            x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
            x_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor)
            modality_list.append(x_tensor)
            in_channels.append(n_channel)

        # ---- 标签处理 ----
        y = p.standardize_label(y_)
        self.gt_shape = gt.shape
        self.data_size = len(y)

        self.y_tensor = torch.from_numpy(y).type(torch.LongTensor)
        self.modality_list = tuple(modality_list)

        # ---- 类别数量 ----
        if is_labeled:
            self.n_classes = np.unique(y).shape[0]
        else:
            # 如果要忽略背景，这里可保持不变
            self.n_classes = np.unique(y).shape[0] - 1

        self.n_modality = n_modality
        self.in_channels = tuple(in_channels)

    def __getitem__(self, idx):
        x_list = []
        for i in range(self.n_modality):
            x = self.modality_list[i][idx]
            if self.transform is not None:
                x_1, x_2 = self.transform(x)
                x_list.append(x_1)
                x_list.append(x_2)
            else:
                x_list.append(x)

        if self.n_modality >= 2 and len(x_list) > 2:
            x_list = (x_list[0::2], x_list[1::2])

        if self.n_modality == 1 and len(x_list) == 2:
            x_list = ([x_list[0]], [x_list[1]])

        y = self.y_tensor[idx]
        return x_list, y

    def __len__(self):
        return self.data_size

# class MultiModalDatasettrain(Dataset):
#
#     def __init__(self, gt_path, *src_path, patch_size=(7, 7), transform=None, is_labeled=True):
#         self.transform = transform
#         p = Processor()
#         n_modality = len(src_path)
#         modality_list = []
#         in_channels = []
#         for i in range(n_modality):
#             img, gt = p.prepare_data(src_path[i], gt_path)
#             x_patches, y_ = p.get_HSI_patches_rw(img, gt, (patch_size[0], patch_size[1]), is_indix=False,
#                                                  is_labeled=is_labeled)
#             n_samples, n_row, n_col, n_channel = x_patches.shape
#
#             scaler = StandardScaler()
#             batch_size = 5000
#             for start_id in range(0, x_patches.shape[0], batch_size):
#                 n_batch = x_patches[start_id: start_id + batch_size].shape[0]
#                 scaler.partial_fit(x_patches[start_id: start_id + batch_size].reshape(n_batch, -1))
#             for start_id in range(0, x_patches.shape[0], batch_size):
#                 shape = x_patches[start_id: start_id + batch_size].shape
#                 x_temp = x_patches[start_id: start_id + batch_size].reshape(shape[0], -1)
#                 x_patches[start_id: start_id + batch_size] = scaler.transform(x_temp).reshape(shape)
#
#             x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
#             x_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor)
#             modality_list.append(x_tensor)
#             in_channels.append(n_channel)
#
#         y = p.standardize_label(y_)
#         self.gt_shape = gt.shape
#         self.data_size = len(y)
#
#         # -------------------------------
#         # 🔥 关键新增功能：自动随机抽取 50% 样本
#         # -------------------------------
#         half = int(self.data_size * 0.5)
#         indices = np.random.choice(self.data_size, half, replace=False)
#
#         # 重新保留一半样本
#         self.y_tensor = torch.from_numpy(y[indices]).type(torch.LongTensor)
#
#         new_modality_list = []
#         for m in modality_list:
#             new_modality_list.append(m[indices])
#         self.modality_list = tuple(new_modality_list)
#         self.data_size = half
#         # -------------------------------
#
#         if is_labeled:
#             self.n_classes = np.unique(y).shape[0]
#         else:
#             self.n_classes = np.unique(y).shape[0] - 1
#
#         self.n_modality = n_modality
#         self.in_channels = tuple(in_channels)
#
#     def __getitem__(self, idx):
#         x_list = []
#         for i in range(self.n_modality):
#             x = self.modality_list[i][idx]
#             if self.transform is not None:
#                 x_1, x_2 = self.transform(x)
#                 x_list.append(x_1)
#                 x_list.append(x_2)
#             else:
#                 x_list.append(x)
#         if self.n_modality >= 2 and len(x_list) > 2:
#             x_list = (x_list[0::2], x_list[1::2])
#         if self.n_modality == 1 and len(x_list) == 2:
#             x_list = ([x_list[0]], [x_list[1]])
#         y = self.y_tensor[idx]
#         return x_list, y
#
#     def __len__(self):
#         return self.data_size

