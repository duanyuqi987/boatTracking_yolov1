#encoding:utf-8
#
#
#
from __future__ import division

from absl import app, flags, logging
from absl.flags import FLAGS

import torch
from torch.autograd import Variable
import torch.nn as nn

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50
from shufflenetv2 import shufflenet_v2_x0_5,shufflenet_v2_x1_0
import torchvision.transforms as transforms
import cv2
import numpy as np
import time
from collections import deque
mybuffer = 364  
#pts = deque(maxlen=mybuffer)


pts1 = deque()
pts2 = deque()

flags.DEFINE_string('video', './data/9.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', './data/10.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')

VOC_CLASSES = (    # always index 0
                 "boat",'watercolumn')

Color = [[0, 0, 255],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]

def decoder(pred):
    '''
    pred (tensor) 1x14x14x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 14
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
#    print(pred.size())
    
    pred = pred.squeeze(0) #14x14x30
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.4 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                index = min_index[i,j]
                mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    #print(i,j,b)
                    box = pred[i,j,b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)#数，索引
                    cls_index = int(cls_index)
                    cls_index = torch.tensor([cls_index])

                    
                    
                    #if float((contain_prob*max_prob)[0]) > 0.1:
                    boxes.append(box_xy.view(1,4))
                    cls_indexs.append(cls_index)
                    #probs.append(contain_prob*max_prob)
                    probs.append(contain_prob)
#    if len(boxes) ==0:
#        boxes = torch.zeros((1,4))
#        probs = torch.zeros(1)
#        cls_indexs = torch.zeros(1)
#    else:
#        boxes = torch.cat(boxes,0) #(n,4)
#        probs = torch.cat(probs,0) #(n,)
#        cls_indexs = torch.cat(cls_indexs,0) #(n,)
    boxes = torch.cat(boxes,0) #(n,4)
    probs = torch.cat(probs,0) #(n,)
    cls_indexs = torch.cat(cls_indexs,0) #(n,)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]

def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order.data[0]
        keep.append(i)



        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]


        
    return torch.LongTensor(keep)
#
#start predict one image
#
def predict_gpu(model,image_name,root_path=''):

    result = []
    #image = cv2.imread(root_path+image_name)
    image = image_name
    h,w,_ = image.shape
    img = cv2.resize(image,(448,448))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mean = (123,117,104)#RGB
    img = img - np.array(mean,dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    with torch.no_grad():
        img = Variable(img[None,:,:,:])
    img = img.cuda()

    pred = model(img) #1x14x14x30
    pred = pred.cpu()
    boxes,cls_indexs,probs =  decoder(pred)

    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],image_name,prob])
    return result



#############我添加的

def image_video(model,img):
    image = img
    print('predicting...')
    result = predict_gpu(model,img)
#############
    for left_up,right_bottom,class_name,_,prob in result:
        """
        x = abs(left_up[0]+right_bottom[0])//2
        y = abs(left_up[1]+right_bottom[1])//2
        if class_name in ('boat'):
            print("....",class_name)

            center = (x,y)
            if y >=220 and y <=650:
                print(center)
                if len(pts1) <= 2:
                    pts1.appendleft(center)
                elif x-pts1[1][0] < 150:
                    pts1.appendleft(center)
                else:
                    pts2.appendleft(center)

        for i in range(1,len(pts1)):
            if pts1[i-1]is None or pts1[i]is None:
                continue
            if abs(pts1[i-1][0]-pts1[i][0]) < 50:
                #thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                thickness = int(10)
                cv2.line(image, pts1[i - 1], pts1[i], (0, 0, 255), thickness)
        for i in range(1,len(pts2)):
            if pts2[i-1]is None or pts2[i]is None:
                continue
            if abs(pts2[i-1][0]-pts2[i][0]) < 50:
                #thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                thickness = int(10)
                cv2.line(image, pts2[i - 1], pts2[i], (0, 0, 255), thickness)
        """
        color = Color[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image,left_up,right_bottom,color,2)
        
        ##########
        label = "boat" +'-'+str(round(prob,2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        """
        if y >=220 and y <=650:
            p1 = (left_up[0], left_up[1]- text_size[1])
            cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
            cv2.putText(image, label, (p1[0], p1[1]+baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
        """
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1]+baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    return image




if __name__ == '__main__':
    #model = resnet50()
    model = shufflenet_v2_x1_0()
    print('load model...')
    model.load_state_dict(torch.load('/home/duanyajun/文档/目标识别项目/自己改写代码/mobilenet_yolov1/yolo_test_boat.pth'))
    model.eval()
    model.cuda()
    
    #try:
    #    vid = cv2.VideoCapture(int(FLAGS.video))
    #except:
    #    vid = cv2.VideoCapture(FLAGS.video)
    vid = cv2.VideoCapture('/home/duanyajun/文档/目标识别项目/自己改写代码/mobilenet_yolov1/ball_water/9.mp4')
    out = None

    #if FLAGS.output:
    #    # by default VideoCapture returns float instead of int
    #    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    #    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #    fps = int(vid.get(cv2.CAP_PROP_FPS))
    #    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    #    out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('/home/duanyajun/文档/目标识别项目/自己改写代码/mobilenet_yolov1/ball_water/109.mp4', fourcc, int(vid.get(cv2.CAP_PROP_FPS)), (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while True:
        _, img = vid.read()
        start = time.time()
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img = image_video(model,img)
        end = time.time()
        print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))
        #if FLAGS.output:
        #    out.write(img)
        out.write(img)
        #cv2.imshow('output', img)
        #if cv2.waitKey(1) == ord('q'):
            #break

    cv2.destroyAllWindows()
    
    

    




