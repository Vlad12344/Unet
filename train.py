import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

# sys.path.append('/home/vlados/Jupyter/Road_project')

# from eval import eval_net
from torchvision.transforms import ToTensor, Resize
from model import UNet
from utils import UNet_Dataset

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              save_cp=True,
              gpu=False,
              img_size=(720, 1280)):

    dir_checkpoint = '/media/vlados/FreeSpace/CV&NN/CV Project/checkpoints/'
    root = '/media/vlados/FreeSpace/CV&NN/CV Project/Dataset'

    # Define UNet dataset
    train_dataset = UNet_Dataset(root, transform=[Resize(img_size), ToTensor()])
    test_dataset = UNet_Dataset(root, train=False, transform=[Resize(img_size), ToTensor()])
    
    # Define data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        checkpoin: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, str(save_cp), str(gpu)))

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        epoch_loss = 0

        for i, (imgs, masks) in enumerate(train_loader):

            if gpu:
                imgs = imgs.cuda()
                true_masks = masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss))

        # if 1:
        #     val_dice = eval_net(net, val, gpu)
        #     print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=3)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
