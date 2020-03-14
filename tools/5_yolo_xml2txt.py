"""
2.3 然后我们需要用voc_label.py归一化标签

voc_label.py读取数据的格式为以下格式：

数据集文件夹名
|—Annotations #存放xml标签
|—labels #存放DarkNet可以识别的归一化后的标签文件，我后面的脚本会自动创建
|—pic #存放数据集

因为源码中的voc_label.py是针对voc数据集的，无论路径上还是数据格式都和我们自己的数据有出入，所以我们要自己编写voc_label.py来处理自己的数据。比如我的供大家参考：
（我这个数据集标准人员在标注的时候有错误标签了应该是号码簿代号box,他们给标成了号码簿的号码，所以我在41-46行加了一步处理，请大家忽略）

"""



# -*- coding: utf-8 -*-  
# #此脚本需要执行两次，在91行train_number执行一次，test_number执行一次 
import xml.etree.ElementTree as ET 
import pickle 
import os 
from os import listdir, getcwd 
from os.path import join 
sets = [] 
classes = ['ball', 'watercolumn'] #非常重要,非常重要,非常重要. 
#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#classes = ['passenger boat', 'container vessel', 'speed boat', 'warship', 'fishing boat', 'island', 'unmanned boat', 'float', 'reef', 'vessel']
#原样保留。size为图片大小 
# # 将ROI的坐标转换为yolo需要的坐标  
# # size是图片的w和h  
# # box里保存的是ROI的坐标（x，y的最大值和最小值）  
# # 返回值为ROI中心点相对于图片大小的比例坐标，和ROI的w、h相对于图片大小的比例  


def convert(size, box):
    dw = 1./(size[0]) 
    dh = 1./(size[1]) 
    x = (box[0] + box[1])/2.0 - 1 
    y = (box[2] + box[3])/2.0 - 1 
    w = box[1] - box[0] 
    h = box[3] - box[2] 
    x = x*dw 
    w = w*dw 
    y = y*dh 
    h = h*dh 
    return (x,y,w,h) 

#对于单个xml的处理 
def convert_annotation(image_add,path): #image_add进来的是带地址的.jpg 
    #image_add = os.path.split(image_add)[1] #截取文件名带后缀 
    #image_add = image_add[0:image_add.find('.',1)] #删除后缀，现在只有文件名没有后缀 
    in_file = open(path +'label_xml/'+ image_add+'.xml')
    
    print('now write to:' + path+'labels/%s.txt'%(image_add)) 
    out_file = open(path +'labels/%s.txt'%(image_add), 'w') 
    
    tree=ET.parse(in_file) 
    root = tree.getroot() 
    #加入我的预处理<name>代码： 
    """
    for obj in root.findall("object"): 
        #obj.append("number") = obj.find('name').text 
        obj.find('name').text = "box" 
        print(obj.find('name').text) 
    tree.write('/home/duanyajun/图片/文件处理中心/label_xml/'+ image_add + '.xml')
    """ 
    
    size = root.find('size') 
    # <size> #     
    # <width>500</width> #     
    # <height>333</height> #     
    # <depth>3</depth> # 
    # </size> 
    w = int(size.find('width').text) 
    h = int(size.find('height').text) 
    #在一个XML中每个Object的迭代 
    
    for obj in root.iter('object'): 
        #iter()方法可以递归遍历元素/树的所有子元素 
        """
        <object>
        <name>car</name>
        <pose>Unspecified</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>141</xmin>
            <ymin>50</ymin>
            <xmax>500</xmax>
            <ymax>330</ymax>
        </bndbox>
        </object>"""
        difficult = obj.find('difficult').text 
         #找到所有的椅子 
         
        cls = obj.find('name').text 
        #print(cls)
         #如果训练标签中的品种不在程序预定品种，或者difficult = 1，跳过此object 
        #if cls not in classes or int(difficult)==1: 
        #    continue 
            #cls_id 只等于1 
        
        cls_id = classes.index(cls) 
        #print(cls_id)

        xmlbox = obj.find('bndbox') 
        #b是每个Object中，一个bndbox上下左右像素的元组 
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)) 
        bb = convert((w,h), b) 
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n') 

pathall = '/home/duanyajun/文档/个人文档/目标识别项目学习/图像处理工具代码/水柱检测2019-12/image/第二次标注/'
for filen in classes:
    with open (pathall + "classes.txt",'a') as f:
        f.write(filen+'\n')

if not os.path.exists(pathall + 'labels/'): 
    os.makedirs(pathall + 'labels/') 

pathh = pathall + "label_xml/" 
#只需要文件名，那我只导入文件名就行了
for filenames in os.walk(pathh): 
    filenames = list(filenames) 
    filenames = sorted(filenames[2])
    for filename in filenames: 
        #print(filename) 
        image_add = filename[0:filename.find('.',1)]
        #print (image_add) 
        convert_annotation(image_add,pathall)

