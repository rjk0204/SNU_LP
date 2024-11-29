from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import cv2
import numpy as np
import os
import sys

from config import parse_args

class LP_Dataset(Dataset):
    def __init__(self, args):

        self.args = args

        img_paths = []

        for file in os.listdir(os.path.join(args.data_dir)):
            img_paths.append(file)

        self.img_paths = sorted(img_paths)

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        
        img_name = os.path.join(self.args.data_dir, self.img_paths[index])

        img_mat = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

        img_tensor = self.transform(img_mat)

        return (img_tensor, img_mat)

    

if __name__ == "__main__":

    args = parse_args()

    test_dataset = LP_Dataset(args)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False
    )

    for batch_idx, data in enumerate(test_dataloader):

        img_tensor, img_mat = data

        img_tensor = img_tensor[0]
        img_mat = img_mat[0].numpy()

        img_draw = cv2.cvtColor(img_mat, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(batch_idx).zfill(2) + '.jpg', img_draw)