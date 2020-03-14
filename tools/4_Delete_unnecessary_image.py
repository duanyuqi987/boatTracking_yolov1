##比较两个文件夹下的文件名都对应不，不对应就删除其中一个文件夹下的不对应文件
'''
1、读取指定目录下的所有文件
2、读取文件，正则匹配出需要的内容，获取文件名
3、打开此文件(可以选择打开可以选择复制到别的地方去)
'''
import os.path
import re
 
 
# 遍历指定目录，显示目录下的所有文件名
filepath=r'/home/duanyajun/文档/个人文档/目标识别项目学习/图像处理工具代码/水柱检测2019-12/image/第二次标注/images'
pathDir = os.listdir(filepath)
#print(pathDir)
#os.remove(file)
for allDir in pathDir:
    #print(allDir.split('.')[0])
    #temp=allDir.split('.')[0]+'.jpg'
    temp=allDir.split('.')[0]+'.xml'
    #print(temp)
    child = os.path.join('%s/%s' % ('/home/duanyajun/文档/个人文档/目标识别项目学习/图像处理工具代码/水柱检测2019-12/image/第二次标注/label_xml', temp))
    if os.path.isfile(child):
#        if (allDir.split('.')[0] == '移动的客船'):
#            os.remove(child)
        continue
    else:
        #de=os.path.join('%s/%s' % ('/home/duanyajun/图片/文件处理中心/2', allDir))
        de=os.path.join('%s/%s' % ('/home/duanyajun/文档/个人文档/目标识别项目学习/图像处理工具代码/水柱检测2019-12/image/第二次标注/images', allDir))
        os.remove(de)
        print(allDir.split('.')[0]+'.jpg')  #输出删除图片的名称

   
     

