

from Dataset import ImageDataset
from torch.utils.data import DataLoader
from model import *
from transform_image import *
import numpy as np
import os

import cv2 as cv


def read_bbox_file(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    bbox_list = []
    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n')
        lines[i] = lines[i].split(' ')
        bbox_list.append(lines[i][1:])

    bbox_array = np.array(bbox_list).astype(int)

    return bbox_array


if __name__ == '__main__':

    PATH = './model.pt'
    model = AlexNet_model()
    model.load_state_dict(torch.load(PATH))
    model.eval()        # Set model to evaluate mode

    transform = Rescale((224, 224))

    image_datasets = {x: ImageDataset(data_use=x, transform=transform) for x in ['train', 'val']}
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=10,
                                      shuffle=False, num_workers=0) for x in ['train', 'val']}


    Res_dict = {'train':{}, 'val': {}}
    Res_dict['train']['pred'] = np.zeros(800).astype(int)
    Res_dict['train']['bbox_pred'] = np.zeros((800, 4)).astype(int)
    Res_dict['val']['pred'] = np.zeros(100).astype(int)
    Res_dict['val']['bbox_pred'] = np.zeros((100, 4)).astype(int)

    with torch.set_grad_enabled(False):
        for phase in ['train', 'val']:
            cnt = 0
            for val in dataloaders_dict[phase]:
                inputs = val['image']
                labels = val['label']
                bbox = val['bbox']
                bbox = torch.Tensor.float(bbox)

                M = bbox.size()[0]

                output_label, output_bbox = model(inputs)

                _, preds = torch.max(output_label, 1)

                preds_numpy = preds.numpy().reshape(M).astype(int)

                bboxPred_numpy = output_bbox.numpy()
                bboxPred_numpy = bboxPred_numpy * 127
                bboxPred_numpy = bboxPred_numpy.reshape(M, 4).astype(int)
                bboxPred_numpy = np.clip(bboxPred_numpy, 0, 127)

                Res_dict[phase]['pred'][cnt:cnt+M] = preds_numpy
                Res_dict[phase]['bbox_pred'][cnt:cnt+M, :] = bboxPred_numpy

                cnt += M


    Res_dict['train']['pred'] = Res_dict['train']['pred'].reshape(800, 1)
    Res_dict['val']['pred'] = Res_dict['val']['pred'].reshape(100, 1)

    root_dir = 'tiny_vid'

    class_names = [subdir for subdir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subdir))]
    class_names = sorted(class_names)

    pred_root_path = 'predFile'

    res_final = {}

    for i, class_name in enumerate(class_names):
        tmp_dict = {'train': {}, 'val': {}}
        for phase in ['train', 'val']:
            if phase == 'train':
                num_per_class = 160
            else:
                num_per_class = 20
            tmp_dict[phase] = np.concatenate((Res_dict[phase]['pred'][i*num_per_class:(i+1)*num_per_class],
                                              Res_dict[phase]['bbox_pred'][i*num_per_class:(i+1)*num_per_class]), axis=1)
        tmp = np.concatenate((tmp_dict['train'], tmp_dict['val']), axis=0)

        res_final[class_name] = tmp

        filename = os.path.join(pred_root_path, class_name + '_pred.txt')

        res_final[class_name] = res_final[class_name].astype(str)

        with open(filename, 'w') as f:
            res_list = res_final[class_name].tolist()
            for line in res_list:
                f.write('\t'.join(line))
                f.write('\n')

    '''
        plot bbox
    '''

    plot_root_dir = 'tiny_vid'

    bbox_gt_dict = {}

    for i, class_name in enumerate(class_names):
        res_final[class_name] = res_final[class_name].astype(int)
        images_dir = os.path.join(plot_root_dir, class_name)

        bbox_txt_name = str(class_name) + '_gt.txt'
        bbox_gt_dict[class_name] = read_bbox_file(os.path.join(plot_root_dir, bbox_txt_name))

        for j in range(180):
            image_fname = str(j+1).zfill(6) + '.JPEG'
            image_path = os.path.join(images_dir, image_fname)

            bboxGt_val = bbox_gt_dict[class_name][j]
            bboxPred_val = res_final[class_name][j][1:]

            img = cv.imread(image_path, 1)
            cv.rectangle(img, (bboxGt_val[0], bboxGt_val[1]), (bboxGt_val[2], bboxGt_val[3]), (0, 255, 0), 3)
            cv.rectangle(img, (bboxPred_val[0], bboxPred_val[1]), (bboxPred_val[2], bboxPred_val[3]), (255, 0, 0), 3)

            storeImgPath = os.path.join(pred_root_path, class_name, image_fname)
            cv.imwrite(storeImgPath, img)


