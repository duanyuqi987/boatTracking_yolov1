#coding=utf-8
import  xml.dom.minidom
import os,sys
import numpy as np 
#import scexcel as sc
rootdir = '/home/duanyajun/文档/个人文档/目标识别项目学习/图像处理工具代码/水柱检测2019-12/image/Annotations'#存有xml的文件夹路径
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
## 空列表
classes_list = []
classes_list1 = []
classes_list2 = []
c2 = []
c1 = []
yajun = []
for i in range(0,len(list)):
   path = os.path.join(rootdir,list[i])
   if os.path.isfile(path):   
      #打开xml文档
      dom = xml.dom.minidom.parse(path)

      #得到文档元素对象
      root = dom.documentElement
      cc=dom.getElementsByTagName('name')
      print(list[i])   #可以打印出那个xml文件是空的.
      try:
         c2 = cc[0]  #xml文件不能为空,为空的话就要删除掉
         for i in range(len(cc)):
            c1 = cc[i]
         
            #列表中不存在则存入列表
            if classes_list.count(c1.firstChild.data)==0:
               classes_list.append(c1.firstChild.data) 
            #print(c1.firstChild.data)
            classes_list1.append(c1.firstChild.data) 
         classes_list2.append(c2.firstChild.data) 
      except IndexError:
        de=os.path.join('%s/%s' % (rootdir, list[i]))
        yajun.append(de)
        #os.remove(de)
        pass

print(classes_list)
print(len(classes_list))

##
print("删除的文件名和路径:",yajun)
#
#统计目标点的个数
duan = np.zeros(len(classes_list))
for i in range(0,len(classes_list2)):
    for j in range(0,len(classes_list)):
        if classes_list2[i] == classes_list[j]:
            duan[j] = duan[j]+1
print(sum(duan))
duan1 = duan.tolist()
print(duan1)
duan1 = map(str, duan1)
#统计目标点的个数
ya = np.zeros(len(classes_list))
for i in range(0,len(classes_list1)):
    for j in range(0,len(classes_list)):
        if classes_list1[i] == classes_list[j]:
            ya[j] = ya[j]+1
print(sum(ya))
ya1 = ya.tolist()
print(ya1)
ya1 = map(str, ya1)

clesses = dict(zip(classes_list,duan))
print(clesses)
clesses1 = dict(zip(classes_list,ya))
print(clesses1)


#fileObject = open('/home/duanyajun/文档/云创工作文档/ART项目/XML处理/sampleList.txt', 'w')
#for ip in clesses1:
#	fileObject.write(ip)
#	fileObject.write(',')
#fileObject.close()
