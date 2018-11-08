
from __future__ import print_function, division

import torch
import time
import copy


def IOU(A, B):
    '''

    # cal IOU betweent A and B.
    # A and B is torch.Tensor.
    '''
    # x_min, y_min, x_max, y_max
    A_W = A[2].item() - A[0].item()
    A_H = A[3].item() - A[1].item()

    B_W = B[2].item() - B[0].item()
    B_H = B[3].item() - B[1].item()

    W = min(A[2].item(), B[2].item()) - max(A[0].item(), B[0].item())
    H = min(A[3].item(), B[3].item()) - max(A[1].item(), B[1].item())

    if W < 0 or H < 0:
        return 0.0

    cross = W * H

    return cross/(A_W*A_H + B_W*B_H - cross)


def batch_IOU_correct(pres_bbox, bbox, thresh=0.5):
    '''

    pres_bbox and bbox is a batch of samples, not a sample
    '''
    m, n = pres_bbox.size()
    result = torch.zeros(m)

    for i in range(m):
        result[i] = IOU(pres_bbox[i][:], bbox[i][:])

    return (result > thresh)



def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, loss_ratio=0.5, print_file=None):
    if print_file != None:
        f = open(print_file, 'w')
    else:
        f = None

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=f)
        print('-' * 10, file=f)
        if f != None:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()       # Set model to training mode
            else:
                model.eval()        # Set model to evaluate mode

            running_loss = 0.0
            running_classify_loss = 0.0
            running_regress_loss = 0.0
            running_correct = 0
            running_classify_correct = 0
            running_regress_correct = 0

            # Iterate over data.
            for val in dataloaders[phase]:
                inputs = val['image']
                labels = val['label']
                bbox = val['bbox']
                bbox = torch.Tensor.float(bbox)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output_label, output_bbox = model(inputs)

                    loss_classify = criterion[0](output_label, labels)
                    loss_regress = criterion[1](output_bbox, bbox)
                    loss = loss_ratio * loss_classify + (1 - loss_ratio) * loss_regress

                    _, preds = torch.max(output_label, 1)
                    bbox_preds = batch_IOU_correct(output_bbox, bbox, thresh=0.5)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_classify_loss += loss_classify.item() * inputs.size(0)
                running_regress_loss += loss_regress.item() * inputs.size(0)

                tmp_preds = preds == labels.data
                running_classify_correct += torch.sum(tmp_preds)
                running_regress_correct += torch.sum(bbox_preds)
                running_correct += torch.sum(tmp_preds*bbox_preds)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_classify_loss = running_classify_loss / len(dataloaders[phase].dataset)
            epoch_regeress_loss = running_regress_loss / len(dataloaders[phase].dataset)

            epoch_acc = running_correct.double() / len(dataloaders[phase].dataset)
            epoch_classify_acc = running_classify_correct.double() / len(dataloaders[phase].dataset)
            epoch_regeress_acc = running_regress_correct.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}, Loss_classify: {:.4f}, Loss_bbox: {:.4f}'.format(phase, epoch_loss, epoch_classify_loss, epoch_regeress_loss), file=f)
            print('{} Acc: {:.4f}, Acc_classify: {:.4f}, Acc_bbox: {:.4f}'.format(phase, epoch_acc, epoch_classify_acc, epoch_regeress_acc), file=f)

            if f != None:
                print('{} Loss: {:.4f}, Loss_classify: {:.4f}, Loss_bbox: {:.4f}'.format(phase, epoch_loss,
                                                                                         epoch_classify_loss,
                                                                                         epoch_regeress_loss))
                print('{} Acc: {:.4f}, Acc_classify: {:.4f}, Acc_bbox: {:.4f}'.format(phase, epoch_acc,
                                                                                      epoch_classify_acc,
                                                                                      epoch_regeress_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        for param_group in optimizer.param_groups:
            print('Now the lr is {}'.format(param_group['lr']), file=f)
            if f != None:
                print('Now the lr is {}'.format(param_group['lr']))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=f)
    print('Best val Acc: {:.4f}'.format(best_acc), file=f)

    if f != None:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.4f}'.format(best_acc))

    if f != None:
        f.close()

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history








