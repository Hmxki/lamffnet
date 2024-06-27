import os
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    """
    根据地址获取指定训练数据集或者测试数据集
    """
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        # 获取图像名称
        img_names = [i for i in os.listdir(os.path.join(data_root, "images"))]
        # 拼接图像数据地址
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.masks = [os.path.join(data_root, "mask", i)
                      for i in img_names]
        # check files
        for i in self.masks:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")


    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.masks[idx]).convert('L')
        manual = np.array(manual)/255

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(manual)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        """
        讲一个批次中的数据组合成一个批次张量
        :param batch:
        :return:
        """
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

# 对数变换函数
def log_transform(img, c=1):
    return c * np.log1p(img)