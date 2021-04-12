
#Referenced for academic purpose ECE1512 by Asmath and Krima 
#from publicly available Github repository by Author Bumsoo Kim 

# ************************************************************
# Author : Bumsoo Kim, 2017
# ***********************************************************

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import model
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import config as cf
import torchvision
import time
import copy
import os
import sys
import argparse
import pretrainedmodels 

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--addlayer','-a',action='store_true', help='Add additional layer in fine-tuning')
parser.add_argument('--resetClassifier', '-r', action='store_true', help='Reset classifier')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

start_time = time.time()

if not torch.cuda.is_available():
    from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

shape = (44, 44)


class DataSetFactory:

    def __init__(self):
        images = []
        emotions = []
        private_images = []
        private_emotions = []
        public_images = []
        public_emotions = []

        with open('../dataset/fer2013.csv', 'r') as csvin:
            data = csv.reader(csvin)
            next(data)
            for row in data:
                face = [int(pixel) for pixel in row[1].split()]
                face = np.asarray(face).reshape(48, 48)
                face = face.astype('uint8')

                if row[-1] == 'Training':
                    emotions.append(int(row[0]))
                    images.append(Image.fromarray(face))
                elif row[-1] == "PrivateTest":
                    private_emotions.append(int(row[0]))
                    private_images.append(Image.fromarray(face))
                elif row[-1] == "PublicTest":
                    public_emotions.append(int(row[0]))
                    public_images.append(Image.fromarray(face))

        print('training size %d : private val size %d : public val size %d' % (
            len(images), len(private_images), len(public_images)))
        train_transform = transforms.Compose([
            transforms.RandomCrop(shape[0]),
            transforms.RandomHorizontalFlip(),
            ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.CenterCrop(shape[0]),
            ToTensor(),
        ])

        self.training = DataSet(transform=train_transform, images=images, emotions=emotions)
        self.private = DataSet(transform=val_transform, images=private_images, emotions=private_emotions)
        self.public = DataSet(transform=val_transform, images=public_images, emotions=public_emotions)


class DataSet(torch.utils.data.Dataset):

    def __init__(self, transform=None, images=None, emotions=None):
        self.transform = transform
        self.images = images
        self.emotions = emotions

    def __getitem__(self, index):
        image = self.images[index]
        emotion = self.emotions[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, emotion

    def __len__(self):
        return len(self.images)

# Model setup

def getNetwork(args):

  #for xception model
  net = pretrainedmodels.xception(num_classes=1000, pretrained='imagenet')
  file_name = 'xception'

  return net, file_name


  if (args.testOnly):
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'file_name+'.t7')
    model = checkpoint['model']

# Training model
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs):

    since = time.time()

    best_model, best_acc = model, 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr = lr_scheduler(optimizer, epoch)
                print('\n=> Training Epoch #%d, LR=%f' %(epoch+1, lr))
                model.train(True)
            else:
                model.train(False)
                model.eval()

            running_loss, running_corrects, tot = 0.0, 0, 0

            for batch_idx, (inputs, labels) in enumerate(training_loaders):
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # Forward Propagation
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    loss = sum((criterion(o, labels) for o in outputs))
                else:
                    loss = criterion(outputs, labels)
                if isinstance(outputs, tuple):
                    # inception v3 output will be (x, aux)
                    outputs = outputs[0]
                _, preds = torch.max(outputs.data, 1)

                # Backward Propagation
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.data[0]
                running_corrects += preds.eq(labels.data).cpu().sum()
                tot += labels.size(0)

                if (phase == 'train'):
                    sys.stdout.write('\r')
                    sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\t\tLoss %.4f\tAcc %.2f%%'
                            %(epoch+1, num_epochs, batch_idx+1,
                                (len(dsets[phase])//cf.batch_size)+1, loss.data[0], 100.*running_corrects/tot))
                    sys.stdout.flush()
                    sys.stdout.write('\r')

            epoch_loss = running_loss / validation_loader[phase]
            epoch_acc  = running_corrects / validation_loader[phase]

            if (phase == 'val'):
                print('\n| Validation Epoch #%d\t\t\tLoss %.4f\tAcc %.2f%%'
                    %(epoch+1, loss.data[0], 100.*epoch_acc))

                if epoch_acc > best_acc :
                    print('| Saving Best model...\t\t\tTop1 %.2f%%' %(100.*epoch_acc))
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    state = {
                        'model': best_model,
                        'acc':   epoch_acc,
                        'epoch':epoch,
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    save_point = './checkpoint/'+dataset_dir
                    if not os.path.isdir(save_point):
                        os.mkdir(save_point)
                    torch.save(state, save_point+file_name+'.t7')

    time_elapsed = time.time() - since

    print('\nTraining completed in\t{:.0f} min {:.0f} sec'. format(time_elapsed // 60, time_elapsed % 60))
    print('Loss : \t{:.2f}%'.format(epoch_loss*100))
    print('Accuracy\t{:.2f}%'.format(epoch_acc*100))


    return best_model

  def exp_lr_scheduler(optimizer, epoch, init_lr=args.lr, weight_decay=args.weight_decay, lr_decay_epoch=cf.lr_decay_epoch):
      lr = init_lr * (0.5**(epoch // lr_decay_epoch))

      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
          param_group['weight_decay'] = weight_decay

      return optimizer, lr

  model_ft, file_name = getNetwork(args)

  # incase of reset Classifier 
  if(args.resetClassifier):
      if(args.addlayer):
          print('| Add features of size %d' %cf.feature_size)
          num_ftrs = model_ft.fc.in_features
          feature_model = list(model_ft.fc.children())
          feature_model.append(nn.Linear(num_ftrs, cf.feature_size))
          feature_model.append(nn.BatchNorm1d(cf.feature_size))
          feature_model.append(nn.ReLU(inplace=True))
          feature_model.append(nn.Linear(cf.feature_size, len(dset_classes)))
          model_ft.fc = nn.Sequential(*feature_model)

      elif(args.net_type == 'inception' or args.net_type == 'xception'):
              num_ftrs = model_ft.last_linear.in_features
              model_ft.last_linear = nn.Linear(num_ftrs, len(dset_classes))

  if use_gpu:
      model_ft = model_ft.cuda()
      model_ft = torch.nn.DataParallel(model_ft, device_ids=range(torch.cuda.device_count()))
      cudnn.benchmark = True

if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
