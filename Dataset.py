
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage import io
import numpy as np
from torch.utils.data import Dataset
import os
from transform_image import *
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")


class ImageDataset(Dataset):

    def __init__(self, root_dir='tiny_vid', data_use='train', transform=None):

        self.subdirs = [subdir for subdir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subdir))]
        self.subdirs = sorted(self.subdirs)

        self.root_dir = root_dir
        self.data_use = data_use
        self.transform = transform

        self.class2idx = {val: idx for idx, val in enumerate(self.subdirs)}
        self.idx2class = {idx: val for idx, val in enumerate(self.subdirs)}

        self.bbox_dict = self.get_bbox()


    def __len__(self):

        if self.data_use == 'train':
            return 160*5
        else:
            return 20*5


    def __getitem__(self, idx):

        if self.data_use == 'train':
            class_idx = int(idx / 160)
            assert class_idx in (0, 1, 2, 3, 4)
            img_idx = idx % 160 + 1
        else:
            class_idx = int(idx / 20)
            assert class_idx in (0, 1, 2, 3, 4)
            img_idx = idx % 20 + 1 + 160

        img_name = str(img_idx).zfill(6) + '.JPEG'
        img_path = os.path.join(self.root_dir, self.idx2class[class_idx], img_name)

        img = io.imread(img_path)
        bbox = self.bbox_dict[class_idx][img_idx].reshape(4)
        img_class = class_idx

        sample = {'image': img, 'label': img_class, 'bbox': bbox}

        if self.transform:
            sample = self.data_transform(sample)

        return sample


    def get_bbox(self):

        bbox_dict = {}
        for val in self.subdirs:
            bbox_txt_name = str(val) + '_gt.txt'
            bbox_dict[self.class2idx[val]] = self.read_bbox_file(os.path.join(self.root_dir, bbox_txt_name))

        return bbox_dict


    def read_bbox_file(self, txt_path):

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        bbox_list = []
        for i in range(len(lines)):
            lines[i] = lines[i].strip('\n')
            lines[i] = lines[i].split(' ')
            bbox_list.append(lines[i][1:])

        bbox_array = np.array(bbox_list).astype(float)

        return bbox_array/127.0     # normalize


    def data_transform(self, sample):
        for trans in self.transform:
            sample = trans(sample)
        sample['image'] = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(sample['image'])

        return sample



if __name__ == '__main__':
    transform = [Rescale((224, 224))]

    image_datasets = {x: ImageDataset(data_use=x, transform=transform) for x in ['train', 'val']}
    dataloaders_dict = {x: DataLoader(image_datasets[x],
                                      batch_size=10, shuffle=True, num_workers=4) for x in ['train', 'val']}

    cnt = 0
    for phase in ['train']:
        for val in dataloaders_dict[phase]:

            images = val['image']
            labels = val['label']
            bbox = val['bbox']
            print(labels)
    #         cnt += 1
    # print(cnt)


    # img_dataset = ImageDataset(transform=transform)

    # plt.figure()
    # cnt = 0
    # for i in range(len(img_dataset)):
    #     sample = img_dataset[i]
    #     cnt += 1
    # print(cnt)
        # print(i, sample['image'].shape, sample['label'], sample['bbox'])
        #
        # ax = plt.subplot(1, 4, i+1)
        # plt.tight_layout()
        # ax.set_title('Sample #{}'.format(i))
        # ax.axis('off')
        # plt.imshow(sample['image'])
        # # plt.pause(0.001)
        #
        # if i == 3:
        #     plt.show()
        #     break








