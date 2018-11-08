
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import *
from transform_image import *
from Dataset import ImageDataset
from functions import train_model

if __name__ == '__main__':

    root_dir = 'tiny_vid'

    # model = Vgg16_bn_model()
    model = AlexNet_model()
    init_lr = 1e-2

    params_to_update = model.parameters()
    # optimizer = torch.optim.Adam(params_to_update, init_lr)
    optimizer = torch.optim.SGD(params_to_update, init_lr)

    transform = Rescale((224, 224))
    image_datasets = {x: ImageDataset(root_dir, data_use=x, transform=transform) for x in ['train', 'val']}
    dataloaders_dict = {x: DataLoader(image_datasets[x],
                                      batch_size=100, shuffle=True, num_workers=0) for x in ['train', 'val']}

    criterions = [nn.CrossEntropyLoss(), nn.MSELoss()]

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    print_out_file = './print_output.txt'

    model_ft, val_acc_history = train_model(model, dataloaders_dict, criterions, optimizer, scheduler, num_epochs=200,
                                            print_file=print_out_file)

    PATH = './model.pt'
    torch.save(model_ft.state_dict(), PATH)




