import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50, resnet18
from shufflenetv2 import shufflenet_v2_x0_5,shufflenet_v2_x1_0
from yoloLoss import yoloLoss
from dataset import yoloDataset

from visualize import Visualizer
import numpy as np

use_gpu = torch.cuda.is_available()

file_root = '/home/duanyajun/文档/目标识别项目/自己改写代码/mobilenet_yolov1/ball_water/images/'
learning_rate = 0.001
num_epochs = 50
batch_size = 4
##更换自己想要的模型---------------------------------------------------
use_resnet = False
use_shufflenetv2 = True
#use_resnet = True
#use_shufflenetv2 = False
if use_resnet:
    net = resnet50()
elif use_shufflenetv2:
    net = shufflenet_v2_x1_0()  
else:
    net = vgg16_bn()

print('load pre-trined model')
if use_resnet:
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith('fc'):
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
elif use_shufflenetv2:
    shufflenetv2 = models.shufflenet_v2_x1_0(pretrained=True)
    new_state_dict = shufflenetv2.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith('fc'):
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
else:
    vgg = models.vgg16_bn(pretrained=True)
    new_state_dict = vgg.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        print(k)
        if k in dd.keys() and k.startswith('features'):
            print('yes')
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
    

##------------------------------------------------------------------------
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = yoloLoss(14,2,5,0.5)
if use_gpu:
    net.cuda()

net.train()
# different learning rate
params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)



train_dataset = yoloDataset(root=file_root,list_file='/home/duanyajun/文档/目标识别项目/自己改写代码/mobilenet_yolov1/mytrain.txt',train=True,transform = [transforms.ToTensor()] )
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
#test_dataset = yoloDataset(root=file_root,list_file='./my2007.txt',train=False,transform = [transforms.ToTensor()] )
#test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('/home/duanyajun/文档/目标识别项目/自己改写代码/mobilenet_yolov1/log.txt', 'w')

num_iter = 0
vis = Visualizer(env='chen')
best_test_loss = np.inf

for epoch in range(num_epochs):
    net.train()
#    if epoch == 1:
#        learning_rate = 0.0001
    # if epoch == 2:
    #     learning_rate = 0.00075
#    if epoch == 3:
#        learning_rate = 0.00001
    if epoch == 30:
        learning_rate=0.0001
    if epoch == 40:
        learning_rate=0.00001
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    
    total_loss = 0.
    
    for i,(images,target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        

        pred = net(images)
        
        print(pred.size(),target.size())       
        loss = criterion(pred,target)
        total_loss += float(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
            num_iter += 1
            vis.plot_train_val(loss_train=total_loss/(i+1))
            
torch.save(net.state_dict(),'/home/duanyajun/文档/目标识别项目/自己改写代码/mobilenet_yolov1/yolo_test_boat.pth')
        
