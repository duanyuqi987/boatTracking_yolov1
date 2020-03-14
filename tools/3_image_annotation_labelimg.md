一：安装labelimg

1、下载源码

https://github.com/tzutalin/labelImg， download之后，解压。

2、安装Python3.5

不要用3.6！不要用3.6！不要用3.6！到目前为止，当执行" from lxml import etree "时，会失败，目前没有解决办法。

这里推荐一篇文章， Windows10下python3和python2同时安装， 

3、安装PyQt5

进入cmd后，输入： pip install PyQt5 

此处有坑，基本上这条命令执行后，因网络问题会出现执行不成功的情况，如下图

3、安装PyQt5

进入cmd后，输入： pip install PyQt5 

此处有坑，基本上这条命令执行后，因网络问题会出现执行不成功的情况，如下图


怎么办？手动下载whl文件。

注意这里的pip，一定要是Python3.5版本的，如果pip有问题，安装上述第二步重新安装Python3.5。

这里有个技巧，执行pip命令之后，它会自动寻找最合适（匹配你的系统、Python版本）的whl文件，文件名就是Downloading 之后的名字，即 PyQt5-5.8.2-5.8.0-cp35.cp36.cp37-none-win_amd64.whl， 百度下载这个文件就可以，链接直在这， https://pypi.python.org/pypi/PyQt5/5.8.2

下载后直接安装， 输入命令 ：pip install XXX.whl， 如下图：

![img](https://img-blog.csdn.net/20170620104244889?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDgwNzg0Ng==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


4、安装PyQt5_tools

同上，执行命令: pip install pyqt5-tools， 如下图，同样会网络原因安装失败，


得到文件名称pyqt5_tools-5.8.2.1.0rc2-cp35-none-win_amd64.whl， 百度下载，链接 https://pypi.python.org/pypi/pyqt5-tools， 下载后直接安装， 命令 ：pip install XXX.whl， 如下图：


5、安装lxml

命令：pip install lxml， 如下图：

![img](https://img-blog.csdn.net/20170620104824662?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDgwNzg0Ng==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


由于lxml文件很小，一般可以安装成功，不行的话，同上述方法，下载whl文件再pip安装。

6、pyrcc编译资源文件

进入到labelImg解压后的文件，我这里是labelImg-master.

执行命令： pyrcc5 -o resources.py resources.qrc ， 如下图

![img](https://img-blog.csdn.net/20170620105701034?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDgwNzg0Ng==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


这是个转换命令，把Qt文件格式转为Python格式。

7、打开labelImg.py

两种方法，方法1，直接在命令窗口中，输入 python3 labelImg.py， 结果如下图

![img](https://img-blog.csdn.net/20170620110613564?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDgwNzg0Ng==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


注意坑来了，执行命令后我自己这里会报错，同事的电脑上就OK，报错信息如下


关于这个问题跟踪了一下，好像跟编码有关，labelImg.py第一行好像不认，所以导致import codecs出错。

方法2，在Python3.5的IDLE打开labelImg.py， 执行Run Module(F5) ，可以正确弹出labelImg界面，如下图


全文结束。

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

那些留下的坑。

1、labelImg源码下来后，里面有个README，提示先安装pyqt4，下来pyqt4之后，不是EXE文件，并且先安装SIP文件，pyqt4和sip 都用make/make install 安装，你给我说说windows哪里来的make命令，难不成还要装个minGW?简直就是个陨石坑。

2、Python3.6，from lxml import etree 出错，有知道原因的留言解释下。

3、Python3.5，为啥直接用命令python3 labelImg.py不可以？

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4、快捷键

Ctrl + u  加载目录中的所有图像，鼠标点击Open dir同功能
Ctrl + r  更改默认注释目标目录(xml文件保存的地址)
Ctrl + s  保存
Ctrl + d  复制当前标签和矩形框
space     将当前图像标记为已验证
w         创建一个矩形框
d         下一张图片
a         上一张图片
del       删除选定的矩形框
Ctrl++    放大
Ctrl--    缩小
↑→↓←        键盘箭头移动选定的矩形框
