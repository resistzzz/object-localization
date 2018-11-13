
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import *
from transform_image import *
from Dataset import ImageDataset
from functions import train_model

if __name__ == '__main__':

    root_dir = 'tiny_vid'
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # model = Vgg16_bn_model().to(device)
    # model = AlexNet_model().to(device)
    model = Resnet_model().to(device)
    init_lr = 0.01

    params_to_update = model.parameters()
    # optimizer = torch.optim.Adam(params_to_update, init_lr)
    optimizer = torch.optim.SGD(params_to_update, init_lr)

    transform_list = [
        RandomCrop(output_size=100),
        Rescale((224, 224)),
        ToTensor()
    ]

    image_datasets = {x: ImageDataset(root_dir, data_use=x, transform_list=transform_list) for x in ['train', 'val']}
    dataloaders_dict = {x: DataLoader(image_datasets[x],
                                      batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}

    criterions = [nn.CrossEntropyLoss(), nn.MSELoss()]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 90], gamma=0.1)

    print_out_file = './print_output.txt'

    model_ft, val_acc_history = train_model(model, dataloaders_dict, criterions, optimizer, scheduler, device=device, num_epochs=100,
                                            loss_ratio=0.5, print_file=print_out_file)
    

    PATH = './model.pt'
    torch.save(model_ft.state_dict(), PATH)




