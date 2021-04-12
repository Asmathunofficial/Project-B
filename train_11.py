
import torch, numpy as np
from load_data import MyDatasets
from utils import *
from tqdm import tqdm
from network import ResNet
from torch.utils.data import Dataset, DataLoader
import os, pickle
import time
from torchvision.utils import save_image
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--K', default=10, help='Number of AU positions')#24,10
parser.add_argument('--dataset', default='BP4D', type=str, help='database')#BP4D,DISFA
parser.add_argument('--dataset_test', default='train', type=str)#BP4D-val, DISFA-val
parser.add_argument('--model_path', type=str,default='./model/tuned_model.pth', help='model path')
parser.add_argument('--cuda', default='5', type=str, help='cuda')
parser.add_argument('--size', default=256, help='Image size')
#Learning rate modified for Part 11
parser.add_argument('--lr', default=5e-3, help='Learning rate')
parser.add_argument('--epochs', default=5, help='Epochs')

def get_true_map(heatmap):
    true_map = heatmap.detach() / 256 #10x64x64
    label = torch.zeros(true_map.shape) #10x64x64
    for j in range(0, true_map.shape[0]):
        temp = cv2.applyColorMap((np.float32(true_map[j,:,:])).astype(np.uint8), cv2.COLORMAP_JET) #64x64
        label[j, :, :] = torch.FloatTensor(temp).mean(-1).unsqueeze(0) #1x64x64
    return label #10x64x64

def loadnet(npoints=10,path_to_model=None):
    # Load the trained model.
    net = ResNet(num_maps=npoints)
    checkpoint = torch.load(path_to_model, map_location='cpu')
    checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}
    net.load_state_dict(checkpoint,strict=False)
    return net.to('cpu')

def plot_loss(tr_l):

	plt.title('Task 2: Plot for training loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.plot(tr_l)
	plt.show()


def train(loader,OUT,net, sample_len):
    mean_loss = torch.zeros((sample_len,1))
    
    #SGD Optimizer used for Part 11
    opt = torch.optim.SGD(net.parameters(), lr=args.lr)
    
    layer = 0
    loss_val=[]
    for child in net.children():
        layer += 1
        if layer < 9:
            for param in child.parameters():
                param.requires_grad = True
    net.train()
    for i in tqdm(range(args.epochs)):
        print('\tEpochs: ', i,'/', args.epochs,'\tTime:', time.time() - start_time, 'sec')
        for idx, sample in enumerate(loader):
            opt.zero_grad()
            img = sample['Im']
            heatmap = net(img).squeeze(0)
            true_label = get_true_map(heatmap)
            loss = (heatmap - true_label).pow(2).mean()

	    #Computing gradient and updating step
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        loss_val.append(torch.mean(mean_loss)).item())    
        print('Loss- ', (torch.mean(mean_loss)).item())
    plot_loss(loss_val)
    torch.save(net.state_dict(), './model/tuned_model.pth')

def test_epoch( dataset_test, model_path,size, npoints):
    net = loadnet(npoints,model_path)
    OUT = OutIntensity().to('cpu')
    # Load data
    database = MyDatasets(size=size,database=dataset_test)
    dbloader = DataLoader(database, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    train(dbloader,OUT,net, len(database))
   
def main():
    global args  
    global start_time
    start_time = time.time()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    test_epoch(dataset_test=args.dataset_test,model_path=args.model_path,size=args.size,npoints=args.K)

if __name__ == '__main__':
    main()
