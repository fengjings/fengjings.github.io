****

### 来源
<https://www.yuque.com/docs/share/5da9151c-14ee-4473-8975-28f9badd9197#yvmtK>  

前几天已经将电脑环境收拾好了，今天按照上述网页中的流程实现pytorch-ssd  

SSD论文下载： <https://arxiv.org/pdf/1512.02325.pdf%EF%BC%89>  
caffe实现—官方：<https://github.com/weiliu89/caffe/tree/ssd>  
tensorflow—非官方：<https://github.com/balancap/SSD-Tensorflow>  
pytorch—非官方：<https://github.com/amdegroot/ssd.pytorch>  

这里，需要说明：
1.支持Python3.7版本（本文版本）  
2.推荐通过支持CUDA的GPU来训练，用CPU也可以训练，不过非常非常慢  
3.目前只支持VOC2007/VOC2012 、COCO数据集、对ImageNet数据集还不能够支持
4.ubuntu18.04

之前安装过虚拟环境
```sh
conda info --env
conda source activate torch-py37
# 到合适路径下
git clone https://github.com/amdegroot/ssd.pytorch.git
# 没用成功。。。待研究
pip install visdom
python -m visdom.server 
```
Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).  

VOC2007  
训练集+测试集  
<http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar>
<http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar>

VOC2012  
训练集+测试集  
<http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar>

原网址如果某些原因上不去，可以去这里  
<https://share.functionweb.tk/>

将下载后的文件解压，将VOC2007和VOC2012文件夹共同放入VOCdevkit下  
此文件夹路径即为后面代码中需要用到的VOC_ROOT。  
```sh
VOC_ROOT='D:\\...\\VOC\\VOCdevkit\\'   #windows
 VOC_ROOT = 'home/..../dataset/VOCdevkit/'   #linux
```

下载好的模型文件放入源码包weights文件夹下  
<https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth>

```sh
pip install visdom
python -m visdom.server
# 需要下载点东西，某些地区需要想一些办法才行，最上面的网址里有打包下好的可以使用
# 然后可以访问http://localhost:8097/
```

接下来的模型训练和原文一致。  
主要是在train.py中设置默认数据集，算法模型路径，batch-size，是否启动CUDA加速等等。  
这里源码默认的batch-size是32，根据自己GPU能力调节大小。  
```sh
# 加载模型初始参数
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
# 默认加载VOC数据集
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
# 设置VOC数据集根路径
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
# 设置预训练模型vgg16_reducedfc.pth
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
# 设置批大小，根据自己显卡能力设置，默认为32
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
# 是否恢复中断的训练，默认不恢复
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
# 恢复训练iter数，默认从第0次迭代开始
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
# 数据加载线程数，根据自己CPU个数设置，默认为4
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
# 是否使用CUDA加速训练，默认开启，如果没有GPU，可改成False直接用CPU训练
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
# 学习率，默认0.001
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
# 最佳动量值，默认0.9（动量是梯度下降法中一种常用的加速技术，用于加速梯度下降，减少收敛耗时）
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
# 权重衰减，即正则化项前面的系数，用于防止过拟合；SGD，即mini-batch梯度下降
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
# gamma更新，默认值0.1
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
# 使用visdom将训练过程loss图像可视化
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
# 权重保存位置，默认存在weights/下
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()
```

设置好参数后，需要配置数据集加载路径  
在源码包/data/voc0712.py中注释掉第28行  
将VOC数据集路径替换为自己的路径：  
data/voc0712.py  
```sh
VOC_ROOT = osp.join(HOME, "learningPy/object-detection/data/VOCdevkit/")
```

### 修改源码  
由于Pytorch版本不同，较新版的代码直接运行会报错  
需要修改部分代码，主要是将.data[0]的部分改成.item()  
修改train.py, 修改源码183.184两行  
```sh
# loc_loss += loss_l.data[0]
# conf_loss += loss_c.data[0]
loc_loss += loss_l.item()
conf_loss += loss_c.item()
```
源码188行  
```sh
# print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')
```
源码165行  
```sh
# load train data
# images, targets = next(batch_iterator)
try:
    images, targets = next(batch_iterator)
except StopIteration as e:
    batch_iterator = iter(data_loader)
    images, targets = next(batch_iterator)
```

修改mutibox_loss.py    
修改：源码包/layers/modules/mutibox_loss.py  
调换第97,98行：  
```sh
# loss_c[pos] = 0  # filter out pos boxes for now
# loss_c = loss_c.view(num, -1)
loss_c = loss_c.view(num, -1)
loss_c[pos] = 0  # filter out pos boxes for now

```
修改第114行,修改后运行报错，因此我又改回去了  
env在后方  
```sh
# N = num_pos.data.sum()
N = num_pos.data.sum()# .double()
###################################
certifi          2020.4.5.2
cffi             1.14.0
chardet          3.0.4
idna             2.9
jsonpatch        1.25
jsonpointer      2.0
mkl-fft          1.1.0
mkl-random       1.1.1
mkl-service      2.3.0
numpy            1.18.1
olefile          0.46
Pillow           7.1.2
pip              20.1.1
pycparser        2.20
pyzmq            19.0.1
requests         2.24.0
scipy            1.4.1
setuptools       47.3.0.post20200616
six              1.15.0
torch            1.1.0
torchfile        0.1.0
torchvision      0.3.0
tornado          6.0.4
urllib3          1.25.9
visdom           0.1.8.9
websocket-client 0.57.0
wheel            0.34.2
```
修改coco.py  
由于train.py会from data import *,而data初始化时会加载coco_labels.txt，  
这个文件在源码包中data/下，无论你是否下载了coco数据集都不影响其加载，  
加载时需要用到COCO_ROOT这个参数  
需要修改COCO_ROOT为你的coco_labels.txt所在的父文件夹目录  
```sh
#COCO_ROOT = osp.join(HOME, 'data/coco/')
COCO_ROOT = osp.join(HOME, 'learningPy/object-detection/ssd.pytorch/data/')
```

###开始训练
参考原文


















