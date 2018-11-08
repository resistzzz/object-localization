

import matplotlib.pyplot as plt
import re
import numpy as np


def read_print_out_file(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()

    train_loss = {'loss': [], 'loss_classify': [], 'loss_bbox': []}
    val_loss = {'loss': [], 'loss_classify': [], 'loss_bbox': []}
    train_acc = {'acc': [], 'acc_classify': [], 'acc_bbox': []}
    val_acc = {'acc': [], 'acc_classify': [], 'acc_bbox': []}

    pattern = '\d+\.\d+'
    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n')
        tmp_split = lines[i].split(':')
        tmp = re.findall(pattern, lines[i])
        tmp = [float(x) for x in tmp]
        if tmp_split[0] == 'train Loss' and len(tmp) == 3:
            train_loss['loss'].append(tmp[0])
            train_loss['loss_classify'].append(tmp[1])
            train_loss['loss_bbox'].append(tmp[2])
        elif tmp_split[0] == 'train Acc' and len(tmp) == 3:
            train_acc['acc'].append(tmp[0])
            train_acc['acc_classify'].append(tmp[1])
            train_acc['acc_bbox'].append(tmp[2])
        elif tmp_split[0] == 'val Loss' and len(tmp) == 3:
            val_loss['loss'].append(tmp[0])
            val_loss['loss_classify'].append(tmp[1])
            val_loss['loss_bbox'].append(tmp[2])
        elif tmp_split[0] == 'val Acc' and len(tmp) == 3:
            val_acc['acc'].append(tmp[0])
            val_acc['acc_classify'].append(tmp[1])
            val_acc['acc_bbox'].append(tmp[2])
        else:
            continue

    train_loss['loss'] = np.array(train_loss['loss']).reshape(len(train_loss['loss'])).astype(float)
    train_loss['loss_classify'] = np.array(train_loss['loss_classify']).reshape(len(train_loss['loss_classify'])).astype(float)
    train_loss['loss_bbox'] = np.array(train_loss['loss_bbox']).reshape(len(train_loss['loss_bbox'])).astype(float)

    val_loss['loss'] = np.array(val_loss['loss']).reshape(len(val_loss['loss'])).astype(float)
    val_loss['loss_classify'] = np.array(val_loss['loss_classify']).reshape(len(val_loss['loss_classify'])).astype(float)
    val_loss['loss_bbox'] = np.array(val_loss['loss_bbox']).reshape(len(val_loss['loss_bbox'])).astype(float)

    train_acc['acc'] = np.array(train_acc['acc']).reshape(len(train_acc['acc'])).astype(float)
    train_acc['acc_classify'] = np.array(train_acc['acc_classify']).reshape(len(train_acc['acc_classify'])).astype(float)
    train_acc['acc_bbox'] = np.array(train_acc['acc_bbox']).reshape(len(train_acc['acc_bbox'])).astype(float)

    val_acc['acc'] = np.array(val_acc['acc']).reshape(len(val_acc['acc'])).astype(float)
    val_acc['acc_classify'] = np.array(val_acc['acc_classify']).reshape(len(val_acc['acc_classify'])).astype(float)
    val_acc['acc_bbox'] = np.array(val_acc['acc_bbox']).reshape(len(val_acc['acc_bbox'])).astype(float)

    return train_loss, train_acc, val_loss, val_acc


def plot_curve(train_loss, train_acc, val_loss, val_acc, num_epoch):

    assert isinstance(num_epoch, int)

    x = np.arange(0, num_epoch, 1)

    plt.figure(1)
    plt.title('Train Loss Curve')
    plt.xlabel('epoch')
    plt.ylabel('Loss value')

    for key, value in train_loss.items():
        plt.plot(x, value, label=key)

    plt.legend()
    plt.savefig('./pictures/Train Loss Curve.png')

    plt.figure(2)
    plt.title('Val Loss Curve')
    plt.xlabel('epoch')
    plt.ylabel('Loss value')

    for key, value in val_loss.items():
        plt.plot(x, value, label=key)

    plt.legend()
    plt.savefig('./pictures/Val Loss Curve.png')

    plt.figure(3)
    plt.title('Train Acc Curve')
    plt.xlabel('epoch')
    plt.ylabel('Acc value')

    for key, value in train_acc.items():
        plt.plot(x, value, label=key)

    plt.legend()
    plt.savefig('./pictures/Train Acc Curve.png')

    plt.figure(4)
    plt.title('Val Acc Curve')
    plt.xlabel('epoch')
    plt.ylabel('Acc value')

    for key, value in val_acc.items():
        plt.plot(x, value, label=key)

    plt.legend()
    plt.savefig('./pictures/Val Acc Curve.png')

    plt.show()


if __name__ == '__main__':
    filename = 'print_output.txt'
    train_loss, train_acc, val_loss, val_acc = read_print_out_file(filename)
    plot_curve(train_loss, train_acc, val_loss, val_acc, num_epoch=200)

