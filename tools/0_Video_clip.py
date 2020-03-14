# 主要是需要moviepy这个库
from moviepy.editor import *
import cv2
import numpy as np
video = VideoFileClip("/home/duanyajun/文档/目标识别项目/自己做的项目/船只跟踪检测/boat_yolov1/video/1.mp4")

# 剪辑视频，截取视频前20秒
x = 2
f = 60
n1 = f * 1 + 10
n2 = f * 1 + 30
#vc1 = video.subclip(30*x,30*(x+1))
vc1 = video.subclip(n1,n2)
#vc1 = video.subclip(n1, video.duration-1)
# 剪辑视频，从10秒开始到视频结尾前12秒
#vc = video.subclip(10, video.duration-12)

#vc1.to_videofile("/home/duanyajun/视频/视频分类/按船只类别/无人船/YZ--L30A/L30A_14.mp4", fps=30, remove_temp=False)
#vc1.to_videofile("/home/duanyajun/视频/视频分类/按船只类别/无人船/YZ--M75A/L75A_12.mp4", fps=30, remove_temp=False)
vc1.to_videofile("/home/duanyajun/文档/目标识别项目/自己做的项目/船只跟踪检测/boat_yolov1/video/11.mp4", fps=25, remove_temp=False)
#vc1.to_videofile("/home/duanyajun/视频/视频分类/按船只类别/无人船/小船/6.mp4", fps=30, remove_temp=False)
###把剪辑好的视频摘取出图片
