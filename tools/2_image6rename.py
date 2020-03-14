import os  
      
class BatchRename():  
      #  ''''' 
     #   批量重命名文件夹中的图片文件 
     
       # '''  
        def __init__(self):  
            #我的图片文件夹路径horse  
            self.path = '/home/duanyajun/文档/个人文档/目标识别项目学习/图像处理工具代码/水柱检测2019-12/yolov1_Detection_range/images3'
            self.path1 = '/home/duanyajun/文档/个人文档/目标识别项目学习/图像处理工具代码/水柱检测2019-12/yolov1_Detection_range/images1'
      
        def rename(self):  
            filelist1 = os.listdir(self.path)  
            total_num = len(filelist1)  
            i = 900641
            n = 6
            print(sorted(filelist1))
            filelist = sorted(filelist1)
            for item in filelist:  
                if item.endswith('.jpg'):  
                    n = 6 - len(str(i))  
                    src = os.path.join(os.path.abspath(self.path), item)  
                    dst = os.path.join(os.path.abspath(self.path1), str(0)*n + str(i) + '.jpg')  
                    try:  
                        os.rename(src, dst)  
                        print ('converting %s to %s ...' % (src, dst) ) 
                        i = i + 1  
                  
                    except:  
                        continue  
            print ('total %d to rename & converted %d jpgs' % (total_num, i)  )
      
if __name__ == '__main__':  
        demo = BatchRename()  
        demo.rename()  
