# passages for object detection

Here's the table of contents:

1. TOC
{:toc}


## object detection [site](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247503263&idx=2&sn=a616690502762cac8ef53fd7b654eae6&chksm=f9a1bf10ced636066134d9bff0ea0ad4967b0b249d839ed57710d195ead32f31c8715841b8b5&scene=126&sessionid=1595058555&key=1ab88c7c386885984b23c87cd512f525a5edddbca130dfdcf8a7e1c3a134fb4c3c2bf61175b6a20a7975d34aa818b4a96c224bf61e313be4def3f4892df1da83466b7f0d4937f015861eeee59dbc7d6d&ascene=1&uin=MjExODAxOTE1&devicetype=Windows+10+x64&version=6209051e&lang=zh_CN&exportkey=AT1KhFpTfJ0gbEHouBYolEA%3D&pass_ticket=OMBTLn%2B90krgIUYo%2BNcsBaG3xySLn%2BOnDixb0qEjYRc%3D)   

+ Detectron2 [site](https://github.com/facebookresearch/detectron2)  
AdelaiDet [site](https://github.com/aim-uofa/AdelaiDet)  

+ mmDetection [site](https://github.com/open-mmlab/mmdetection)   
backbones: ResNet ResNeXt VGG HRNet RegNet Res2Net  

## 物体检测中的小物体问题  

物体检测中的小物体问题  [site](https://zhuanlan.zhihu.com/p/345905644)  

+ 提高图像拍摄分辨率
+ 增加模型的输入分辨率
+ 平铺图片
+ 通过扩充生成更多数据
+ 自动学习模型
+ 过滤掉多余的类

小目标检测常用方法  [site](https://zhuanlan.zhihu.com/p/83220498)  
+ 传统的图像金字塔和多尺度滑动窗口检测  
+ Data Augmentation
+ 特征融合的FPN
+ 合适的训练方法SNIP,SNIPER,SAN  
[SNIP site](https://arxiv.org/abs/1711.08189)  
[SNIP code](https://github.com/mahyarnajibi/SNIPER)  
[SNIPER site](https://arxiv.org/abs/1805.09300)  
[SNIPER code](https://github.com/MahyarNajibi/SNIPER)  
SNIPER的关键是减少了SNIP的计算量。SNIP借鉴了multi-scale training的思想进行训练，multi-scale training是用图像金字塔作为模型的输入，这种做法虽然能够提高模型效果，但是计算量的增加也非常明显，因为模型需要处理每个scale图像的每个像素，而SNIPER（Scale Normalization for Image Pyramids with Efficient Resampling）算法以适当的比例处理ground truth（称为chips）周围的上下文区域，在训练期间每个图像生成的chips的数量会根据场景复杂度而自适应地变化，由于SNIPER在采样后的低分辨率的chips上运行，故其可以在训练期间收益于Batch Normalization，而不需要在GPU之间再用同步批量标准化进行统计信息。实验证明，BN有助于最后性能的提升。  
+ 更稠密的Anchor采样和匹配策略S3FD,FaceBoxes
+ 先生成放大特征再检测的GAN
+ 利用Context信息的Relation Network和PyramidBox


解决小目标检测！多尺度方法汇总 [site](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247506911&idx=1&sn=10491a861ffffa7bc05ea6b53cab4289&chksm=ec1c3626db6bbf30230dd8103bb77d26009d1570a1b68d2cd6bfc00bc97d41e284a22ca0fe27&scene=21#wechat_redirect)  

+ 传统的图像金字塔
+ SNIP/SNIPER中的多尺度处理
+ SSD中的多尺度处理
+ 空洞卷积处理多尺度
+ FPN中的多尺度处理及其改进


## 2021-object detection & drones related  


### High-Performance Large-Scale Image Recognition Without Normalization* [site](https://arxiv.org/abs/2102.06171)   
说明：[site](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247517020&idx=2&sn=5a9b20d36a20b641739d0ad77bf1bfe5&chksm=f9a1f1d3ced678c5974ba15f003353ecfcbd9481d8437921b25ede0401854b49c2d16d245eda&scene=126&sessionid=1613234694&key=55fcbf4895711c4c29efb51a11009902868c659e12bbde0aa9f37741f52dd088d11eebc7ca6997da73f856dccbd0e065e8b2226de7839bbcdfdcb2034771410097ceb771a68e6f931abfda762731e2d084a280d1b7e4a4bbc35f412b66a88c03e8c472e9f1125c77c246fc756b9c883bab2a2aad9601b000f83bb33a25d99123&ascene=1&uin=MjExODAxOTE1&devicetype=Windows+10+x64&version=6209051e&lang=zh_CN&exportkey=AThsyVtsAWz3%2BIn9ZMF5ebQ%3D&pass_ticket=sAdGHv50drvjr5%2FZxoxSpG%2BpkIDxtegv5o6yLOgTxDMqbleObR9EsLU9NXnqRU3t&wx_header=0)  
title: 无归一化的高性能大规模图像识别  
code: <https://github.com/deepmind/deepmind-research/tree/master/nfnets>  
提出了自适应梯度修剪（Adaptive Gradient Clipping，AGC）方法，基于梯度范数与参数范数的单位比例来剪切梯度，研究人员证明了 AGC 可以训练更大批次和大规模数据增强的非归一化网络。  

设计出了被称为 Normalizer-Free ResNets 的新网络，该方法在 ImageNet 验证集上大范围训练等待时间上都获得了最高水平。NFNet-F1 模型达到了与 EfficientNet-B7 相似的准确率，同时训练速度提高了 8.7 倍，而 NFNet 模型的最大版本则树立了全新的 SOTA 水平，无需额外数据即达到了 86.5％的 top-1 准确率。  





### A Review on Deep Learning in UAV Remote Sensing*  [site](https://arxiv.org/abs/2101.10861)   
 
title: 深度学习在无人机遥感中的应用。  
abstract: 深度神经网络（Deep Neural Networks，DNNs）以惊人的能力从数据中学习表示，在处理图像、时间序列、自然语言、音频、视频等方面取得了重大突破。在遥感领域，已经进行了专门涉及DNNs算法应用的调查和文献修订，试图总结其子领域产生的信息量。近年来，基于无人机的应用已成为航空遥感研究的主流。然而，结合“深度学习”和“无人机遥感”主题的文献修订尚未进行。我们的工作的动机是提出一个全面的审查基础上的深入学习（DL）应用于无人机为基础的图像。我们主要集中在描述分类和回归技术在最近的应用与无人机获得的数据。为此，共有232篇论文发表在国际科学期刊数据库进行了审查。我们收集了已发表的材料，并评估了它们在应用、传感器和所用技术方面的特点。我们叙述了DL如何呈现出有希望的结果，并具有处理基于无人机的图像数据相关任务的潜力。最后，我们展望了未来的发展前景，并对无人机遥感领域有待探索的重要DL路径进行了评述。我们的修订版包括一个友好的方法来介绍、评论和总结基于无人机的图像应用的最新技术，在不同的遥感子领域使用DNNs算法，将其分组在环境、城市和农业环境中。


### Occlusion Handling in Generic Object Detection: A Review* [site](https://arxiv.org/abs/2101.08845)  

title: 遮挡目标检测-综述  
abstract: 深度学习网络的巨大威力导致了目标检测的巨大发展。在过去的几年中，目标检测器框架在准确性和效率方面都取得了巨大的成功。然而，由于多种因素的影响，它们的能力与人类相差甚远，遮挡就是其中之一。由于遮挡可能发生在不同的位置、比例和比例，因此很难处理。在这篇论文中，我们讨论了在室外和室内场景中一般目标检测中遮挡处理的挑战，然后我们参考了最近为克服这些挑战而开展的工作。最后，我们讨论了未来可能的研究方向。



## 2020-object detection  
<https://github.com/extreme-assistant/survey-computer-vision#1>.

### 2020-Deep Domain Adaptive Object Detection: a Survey 
[site](https://arxiv.org/abs/2002.06797)  
深度域适应目标检测  
本文共梳理了46篇相关文献.  
基于深度学习(DL)的目标检测已经取得了很大的进展，这些方法通常假设有大量的带标签的训练数据可用，并且训练和测试数据从相同的分布中提取。  
然而，这两个假设在实践中并不总是成立的。  
深域自适应目标检测(DDAOD)作为一种新的学习范式应运而生。本文综述了深域自适应目标检测方法的研究进展。   


### 2020-Foreground-Background Imbalance Problem in Deep Object Detectors: A Review*  
[site](https://arxiv.org/abs/2006.09238)  
深度目标检测器中前景-背景不平衡问题综述  
本文研究了不平衡问题解决方案的最新进展。  
分析了包括一阶段和两阶段在内的各种深度检测器中不平衡问题的特征。  
将现有解决方案分为两类：抽样和非抽样方案.  

+ 用于解决各种对象检测框架中的前景-背景不平衡问题的不同解决方案总结.  
（即基于anchor-based one-stage, anchor-free onestage, two-stage的方法）。  
这些解决方案包括小批量偏差采样，OHEM，IoU平衡采样，人为丢失，GHM-C，ISA，ResObj，免采样，AP丢失，DR丢失。
文章可视化它们的使用范围。  
+ 前景-背景不平衡问题的不同解决方案的比较。   
准确性（AP），相对准确性改进（∆AP），超参数的数量（参数）和效率（速度）.    



### 2020-A Review and Comparative Study on Probabilistic Object Detection in Autonomous Driving*  
[site](https://arxiv.org/abs/2011.10671)      
自动驾驶中的概率目标检测方法综述与比较研究   


### 2020-Camouflaged Object Detection and Tracking: A Survey*  
[site](https://arxiv.org/abs/2012.13581)  
伪装目标检测与跟踪研究综述    
运动目标的检测和跟踪应用于各个领域，包括监视，异常检测，车辆导航等。  
本文从理论角度回顾了基于计算机视觉算法的现有伪装目标检测和跟踪技术.    



### 2020-无人机-Correlation Filter for UAV-Based Aerial Tracking: A Review and Experimental Evaluation*  
[site](https://arxiv.org/abs/2010.06255)   
相关过滤无人机空中跟踪技术综述与实验评估    
基于DCF的跟踪器的基本框架。  
总结了20种基于DCF的最新跟踪器。  
UAV123，UAV123_10fps，UAV20L，UAVDT，DTB70和VisDrone2019-SOT。



## 2019-object detection  
[site](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247493715&idx=3&sn=3f684b51a604cc5d4878a5716aaf2b3f&chksm=f9a19adcced613caa3eef7240dfbf00e866ece9f1ce48366b63984620d1512f16a921ccdb7cf&scene=21#wechat_redirect)


### 重读 CenterNet [site](https://mp.weixin.qq.com/s/hlc1IKhKLh7Zmr5k_NAykw)  
paper: <https://arxiv.org/abs/1904.07850>  
code: <https://github.com/xingyizhou/CenterNet>  
+ 动机  
+ CenterNet原理  
关键点(key point)损失函数  
offset损失函数  
尺寸（size）损失函数  
整体的损失函数  
+ 网络结构
+ 使用CenterNet做3D目标检测




### Augmentation for small object detection* [site](https://arxiv.org/abs/1902.07296)  
title: 小目标增强  
abstract：近年来，目标检测取得了令人瞩目的进展。尽管有了这些改进，但是在检测大小物体方面仍然存在很大的差距。我们在一个具有挑战性的数据集COCO女士上分析了当前最先进的模型Mask-RCNN。我们发现，小的地面真实物体和预测的锚之间的重叠远低于预期的IoU阈值。我们推测这是由于两个因素造成的：（1）只有少数图像包含小对象，和（2）即使在包含它们的每个图像中，小对象也显示得不够。因此，我们建议用小对象对这些图像进行过采样，并通过多次复制粘贴小对象来增强每个图像。它允许我们在大物体和小物体上权衡探测器的质量。我们评估了不同的粘贴增强策略，最终，与目前最先进的MS-COCO方法相比，在实例分割和小对象检测方面分别提高了9.7%和7.1%。  

### Rethinking Convolutional Feature Extractionfor Small Object Detection* [site](https://bmvc2019.org/wp-content/uploads/papers/1057-paper.pdf)  
title: 卷积特征提取在小目标检测  
abstract: 基于深度学习的目标检测体系结构大大提高了技术水平。然而，最近的检测方法研究表明，在小目标性能和中大型目标性能之间有很大的差距。这一差距在体系结构和主干网之间存在。我们表明，这一差距主要是由于减少了特征地图的大小，因为我们遍历主干。通过对主干结构的简单修改，我们发现对于小对象的性能有了显著的提高。此外，我们还提出了一种具有权值共享的双路径配置来恢复大对象性能。与依赖于多尺度训练和网络划分的最新方法相比，我们在MS-COCO数据集上显示出了具有竞争力的性能。我们用移动对象检测器SSD Mobilenet v1展示了最先进的小对象性能.  

### 2019-Object Detection in 20 Years: A Survey  
[site](https://arxiv.org/abs/1905.05055)  
title: 目标检测20年  
abstract：目标检测作为计算机视觉中最基本、最具挑战性的问题之一，近年来受到了广泛的关注。它在过去二十年中的发展可以看作是计算机视觉历史的一个缩影。如果我们把今天的目标探测看作是在深度学习的力量下的一种技术美学，那么时光倒流20年，我们将见证冷兵器时代的智慧。本文广泛回顾了超过四分之一世纪（从20世纪90年代到2019年）以来，400多篇关于目标检测的论文。本文涵盖了许多主题，包括历史上的里程碑检测器、检测数据集、度量、检测系统的基本构件、加速技术以及最新的检测方法。本文还回顾了一些重要的检测应用，如行人检测、人脸检测、文本检测等，深入分析了它们面临的挑战以及近年来的技术改进。  

### 2019-A Survey of Deep Learning-based Object Detection  
[site](https://arxiv.org/abs/1907.09408)  
title: 深度学习目标检测综述  
abstract：目标检测是计算机视觉最重要、最具挑战性的分支之一，在人们的生活中有着广泛的应用，如安全监控、自动驾驶等，其目的是定位某一类语义对象的实例。随着检测任务深度学习网络的迅速发展，目标检测器的性能得到了很大的提高。为了全面、深入地了解目标检测管道的主要发展现状，本文首先分析了现有典型检测模型的方法，并对基准数据集进行了描述。之后，我们首先系统地概述了各种目标检测方法，包括一级和两级检测器。此外，我们列出了传统和新的应用程序。分析了目标检测的一些典型分支。最后，我们讨论了利用这些目标检测方法构建高效系统的体系结构，并指出了一系列的发展趋势，以便更好地遵循最新的算法和进一步的研究。  

### 2019-Recent Advances in Deep Learning for Object Detection
[site](https://arxiv.org/abs/1908.03673)  
title: 深度学习在目标检测中的研究进展  
abstract：目标检测是计算机视觉中一个基本的视觉识别问题，在过去的几十年中得到了广泛的研究。视觉目标检测的目的是在给定的图像中找到具有精确定位的特定目标类的目标，并为每个目标实例分配相应的类标签。由于基于深度学习的图像分类取得了巨大的成功，基于深度学习的目标检测技术近年来得到了积极的研究。本文综述了深度学习视觉目标检测的最新进展。通过回顾大量最近的相关文献，我们系统地分析了现有的目标检测框架，并将调查分为三个主要部分：（i）检测组件，（ii）学习策略，以及（iii）应用程序和基准测试。在调查中，我们详细讨论了影响检测性能的各种因素，如检测器结构、特征学习、建议生成、采样策略等。最后，我们讨论了几个未来的发展方向，以促进和推动基于深度学习的视觉目标检测的未来研究。  
关键词：目标检测，深度学习，深度卷积神经网络。   


### 2019-Imbalance Problems in Object Detection: A Review
[site](https://arxiv.org/abs/1909.00169)  
title: 目标检测中的不平衡问题   
abstract：本文综述了目标检测中的不平衡问题。为了系统地分析问题，我们引入了基于问题的分类法。按照这个分类法，我们深入讨论每个问题，并在文献中提出一个统一但关键的解决方案。此外，我们还确定了有关现有不平衡问题的主要未决问题以及以前没有讨论过的不平衡问题。此外，为了使我们的评论保持最新，我们提供了一个随附的网页，根据我们基于问题的分类法，对解决不平衡问题的论文进行分类。  
最新进展：<https://github.com/kemaloksuz/ObjectDetectionImbalance>







## Image-Quality-Assessment-Benchmark

<https://github.com/weizhou-geek/Image-Quality-Assessment-Benchmark>  

+ Traditional method  
full-reference IQA (FR-IQA)  
reduced-reference IQA (RR-IQA)  
no-reference IQA (NR-IQA)  

+ Deep Learning Based Approaches  
full-reference IQA (FR-IQA)  
~~reduced-reference IQA (RR-IQA)~~  
no-reference IQA (NR-IQA)  

+ database









