
from skimage import transform
import torch
import numpy as np


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size


    def __call__(self, sample):

        image, label, bbox = sample['image'], sample['label'], sample['bbox']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        sample['image'] = img

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, bbox = sample['image'], sample['label'], sample['bbox']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        label = np.array(label)

        sample = {'image': torch.from_numpy(image), 'label': torch.from_numpy(label), 'bbox': torch.from_numpy(bbox)}

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
    """

    def __init__(self, output_size, thre=0.7):

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.thre = thre


    def __call__(self, sample):
        image, label, bbox = sample['image'], sample['label'], sample['bbox']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        y_left = np.random.randint(0, h - new_h)
        x_left = np.random.randint(0, w - new_w)

        for i in range(5):

            ratio, newbbox = self.calCrop(bbox, (x_left, y_left))
            if ratio > self.thre:
                image = image[y_left: y_left + new_h, x_left: x_left + new_w]
                sample['image'] = image
                sample['label'] = label
                sample['bbox'] = newbbox
                return sample

        return sample

    def calCrop(self, bbox, new_left):
        x_min = int(bbox[0] * 127)
        y_min = int(bbox[1] * 127)
        x_max = int(bbox[2] * 127)
        y_max = int(bbox[3] * 127)

        x_min_crop = new_left[0]
        y_min_crop = new_left[1]
        x_max_crop = new_left[0] + self.output_size[1]
        y_max_crop = new_left[1] + self.output_size[0]

        W = min(x_max, x_max_crop) - max(x_min, x_min_crop)
        H = min(y_max, y_max_crop) - max(y_min, y_min_crop)

        if W <= 0 or H <= 0:
            return 0.0, None

        cross = W * H
        bbox_area = (x_max - x_min) * (y_max - y_min)

        new_x_min = max(0, x_min - x_min_crop)
        new_y_min = max(0, y_min - y_min_crop)
        new_x_max = min(x_max, x_max_crop) - x_min_crop
        new_y_max = min(y_max, y_max_crop) - y_min_crop

        new_x_min = 1.0 * new_x_min / (self.output_size[1] - 1)
        new_y_min = 1.0 * new_y_min / (self.output_size[0] - 1)
        new_x_max = 1.0 * new_x_max / (self.output_size[1] - 1)
        new_y_max = 1.0 * new_y_max / (self.output_size[0] - 1)

        new_bbox = np.array([new_x_min, new_y_min, new_x_max, new_y_max]).reshape(4)

        ratio = 1.0 * cross / bbox_area

        return ratio, new_bbox














