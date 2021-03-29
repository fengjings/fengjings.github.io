# Academic-Documents

Here's the table of contents:

1. TOC
{:toc}


## object detection 

[CVer](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247503263&idx=2&sn=a616690502762cac8ef53fd7b654eae6&chksm=f9a1bf10ced636066134d9bff0ea0ad4967b0b249d839ed57710d195ead32f31c8715841b8b5&scene=126&sessionid=1595058555&key=1ab88c7c386885984b23c87cd512f525a5edddbca130dfdcf8a7e1c3a134fb4c3c2bf61175b6a20a7975d34aa818b4a96c224bf61e313be4def3f4892df1da83466b7f0d4937f015861eeee59dbc7d6d&ascene=1&uin=MjExODAxOTE1&devicetype=Windows+10+x64&version=6209051e&lang=zh_CN&exportkey=AT1KhFpTfJ0gbEHouBYolEA%3D&pass_ticket=OMBTLn%2B90krgIUYo%2BNcsBaG3xySLn%2BOnDixb0qEjYRc%3D)   

+ Detectron2 [github](https://github.com/facebookresearch/detectron2)  
AdelaiDet [github](https://github.com/aim-uofa/AdelaiDet)  

+ mmDetection [github](https://github.com/open-mmlab/mmdetection)   
backbones: ResNet ResNeXt VGG HRNet RegNet Res2Net  

## 物体检测中的小物体问题  

物体检测中的小物体问题  [极市平台](https://zhuanlan.zhihu.com/p/345905644)  
+ 提高图像拍摄分辨率
+ 增加模型的输入分辨率
+ 平铺图片
+ 通过扩充生成更多数据
+ 自动学习模型
+ 过滤掉多余的类

小目标检测常用方法  [zhihu](https://zhuanlan.zhihu.com/p/83220498)  
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


解决小目标检测！多尺度方法汇总 [极市平台](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247506911&idx=1&sn=10491a861ffffa7bc05ea6b53cab4289&chksm=ec1c3626db6bbf30230dd8103bb77d26009d1570a1b68d2cd6bfc00bc97d41e284a22ca0fe27&scene=21#wechat_redirect)  

+ 传统的图像金字塔
+ SNIP/SNIPER中的多尺度处理
+ SSD中的多尺度处理
+ 空洞卷积处理多尺度
+ FPN中的多尺度处理及其改进


## 2021-object detection & drones related  

### 2021-A Review on Deep Learning in UAV Remote Sensing-无人机遥感深度学习研究综述*  
paper: [arxiv](https://arxiv.org/abs/2101.10861)    
title: 深度学习在无人机遥感中的应用。  
abstract: 深度神经网络（Deep Neural Networks，DNNs）以惊人的能力从数据中学习表示，在处理图像、时间序列、自然语言、音频、视频等方面取得了重大突破。在遥感领域，已经进行了专门涉及DNNs算法应用的调查和文献修订，试图总结其子领域产生的信息量。近年来，基于无人机的应用已成为航空遥感研究的主流。然而，结合“深度学习”和“无人机遥感”主题的文献修订尚未进行。我们的工作的动机是提出一个全面的审查基础上的深入学习（DL）应用于无人机为基础的图像。我们主要集中在描述分类和回归技术在最近的应用与无人机获得的数据。为此，共有232篇论文发表在国际科学期刊数据库进行了审查。我们收集了已发表的材料，并评估了它们在应用、传感器和所用技术方面的特点。我们叙述了DL如何呈现出有希望的结果，并具有处理基于无人机的图像数据相关任务的潜力。最后，我们展望了未来的发展前景，并对无人机遥感领域有待探索的重要DL路径进行了评述。我们的修订版包括一个友好的方法来介绍、评论和总结基于无人机的图像应用的最新技术，在不同的遥感子领域使用DNNs算法，将其分组在环境、城市和农业环境中。  

### 2021-Object Detection in Aerial Images:A Large-Scale Benchmark and Challenges-空中目标检测影像：大比例尺基准和挑战  
paper:[arxiv](https://arxiv.org/abs/2102.12219)  
Benchmarks for Object Detection in Aerial Images: [github](https://github.com/dingjiansw101/AerialDetection)  
abstract: 近十年来，由于航空图像的鸟瞰视角导致目标的尺度和方向发生了巨大的变化，使得目标检测在自然图像中取得了显著的进展，但在航空图像中却没有取得显著的进展。更重要的是，缺乏大规模的基准成为发展航空图像目标检测（ODAI）的主要障碍。在本文中，我们提出了一个大规模的航空图像目标检测数据集（DOTA）和综合基线ODAI。提出的DOTA数据集包含了从11268幅航空图像中收集的18类定向边界框注释的1793658个对象实例。基于这一大规模的、注释良好的数据集，我们构建了包含10种最新算法的基线，配置超过70种，并对每种模型的速度和精度性能进行了评估。此外，我们还为ODAI提供了一个统一的代码库，并建立了一个测试和评估不同算法的网站。以前在DOTA上运行的挑战吸引了全球1300多个团队。我们相信，扩展的大规模DOTA数据集、广泛的基线、代码库和挑战有助于设计健壮的算法和对航空图像中目标检测问题的可重复性研究。 

### 【旋转目标检测】YOLOv5应对无人机/遥感场景相关难点的解决方案  
article: [zhihu](https://zhuanlan.zhihu.com/p/359249077)  
code: [github](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB)  

### YOLOv5_DOTAv1.5(遥感/无人机旋转目标检测，全踩坑记录)  
article: [zhihu](https://zhuanlan.zhihu.com/p/357992219)  
code: [github](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBBB)  





### ReDet: A Rotation-equivariant Detector for Aerial Object Detection-一种用于航空目标检测的旋转等变检测器
paper:[arxiv](https://arxiv.org/abs/2103.07733)  
code: [github](https://github.com/csuhan/ReDet)  
abstract: 近年来，航空图像中的目标检测在计算机视觉中得到了广泛的关注。与自然图像中的目标不同，航空目标往往具有任意方向的分布。因此，检测器需要更多的参数来编码方向信息，这往往是高度冗余和低效的。此外，由于普通的cnn没有明确地模拟方向变化，因此需要大量的旋转增强数据来训练精确的目标检测器。在本文中，我们提出了一种旋转等变检测器（ReDet）来解决这些问题，它显式地编码旋转等变和旋转不变性。更准确地说，我们在检测器中加入旋转等变网络来提取旋转等变特征，这样可以准确地预测方向，并大大减小模型尺寸。在旋转等变特征的基础上，提出了旋转不变RoI-Align（RiRoI-Align），该算法根据RoI的方向自适应地从等变特征中提取旋转不变特征。在DOTA-v1.0、DOTA-v1.5和HRSC2016等具有挑战性的航空影像数据集上进行的大量实验表明，我们的方法能够在航空目标检测任务上达到最先进的性能。与之前的最佳结果相比，我们的ReDet在DOTA-v1.0、DOTA-v1.5和HRSC2016上分别获得1.2、3.5和2.6 mAP，同时减少了60\%（313 Mb和121 Mb）的参数数量。  


### Towards Open World Object Detection*-开放世界目标检测(CVPR2021)  
paper:[arxiv](https://arxiv.org/abs/2103.02603)  
code: [github](https://github.com/JosephKJ/OWOD)  
abstract: 人类有识别环境中未知物体实例的本能。当相应的知识最终可用时，对这些未知实例的内在好奇心有助于了解它们。这促使我们提出了一个新的计算机视觉问题，称为“开放世界目标检测”，模型的任务是：1）在没有明确监督的情况下，将尚未引入的目标识别为“未知”，2）逐步学习这些已识别的未知类别，而不忘记以前学习的类，当相应的标签逐渐收到时。本文提出了一种基于对比聚类和基于能量的未知识别的开放世界目标检测算法。我们的实验评估和烧蚀研究分析了矿石在实现开放世界目标方面的功效。作为一个有趣的副产品，我们发现识别和描述未知实例有助于减少增量对象检测设置中的混淆，在增量对象检测设置中，我们实现了最先进的性能，而无需额外的方法学努力。我们希望，我们的工作将吸引进一步研究这个新确定的，但至关重要的研究方向。  

### CorrDetector: A Framework for Structural Corrosion Detection from Drone Images using Ensemble Deep Learning-通过无人机图像的集成深度学习的结构腐蚀检测框架
paper:[arxiv](https://arxiv.org/abs/2102.04686)  
code: [not provided]  
abstract: 在本文中，我们提出了一种新的技术，将自动图像分析应用于结构腐蚀监测领域，并证明了与现有方法相比改进的有效性。结构腐蚀监测是基于风险的维护理念的初始步骤，取决于工程师对建筑物故障风险的评估，该评估与维护的财政成本相平衡。这就带来了人为错误的机会，当仅限于使用无人驾驶飞机捕获的图像对由于许多背景噪声而无法到达的区域进行评估时，人为错误会变得更加复杂。这个问题的重要性促进了一个积极的研究团体，旨在通过使用人工智能（AI）图像分析进行腐蚀检测来支持工程师。在本文中，我们通过开发一个框架corredetor来推进这一领域的研究。CorrDetector采用卷积神经网络（CNNs）支持的集成深度学习方法进行结构识别和腐蚀特征提取。我们使用无人机拍摄的复杂结构（如电信塔）的真实图像进行经验评估，这是工程师的典型场景。我们的研究表明\model的集成方法在分类准确率方面显著优于现有的方法。  

### ReDet: A Rotation-equivariant Detector for Aerial Object Detection-航空图像旋转目标的检测
paper:[arxiv](https://arxiv.org/abs/2103.07733)  
code: [github](https://github.com/csuhan/ReDet)   
说明1：[CVer计算机视觉](https://zhuanlan.zhihu.com/p/358303556)  
abstract: 近年来，航空图像中的目标检测在计算机视觉中得到了广泛的关注。与自然图像中的目标不同，航空目标往往具有任意方向的分布。因此，检测器需要更多的参数来编码方向信息，这往往是高度冗余和低效的。此外，由于普通的cnn没有明确地模拟方向变化，因此需要大量的旋转增强数据来训练精确的目标检测器。在本文中，我们提出了一种旋转等变检测器（ReDet）来解决这些问题，它显式地编码旋转等变和旋转不变性。更准确地说，我们在检测器中加入旋转等变网络来提取旋转等变特征，这样可以准确地预测方向，并大大减小模型尺寸。在旋转等变特征的基础上，提出了旋转不变RoI-Align（RiRoI-Align），该算法根据RoI的方向自适应地从等变特征中提取旋转不变特征。在DOTA-v1.0、DOTA-v1.5和HRSC2016等具有挑战性的航空影像数据集上进行的大量实验表明，我们的方法能够在航空目标检测任务上达到最先进的性能。与之前的最佳结果相比，我们的ReDet在DOTA-v1.0、DOTA-v1.5和HRSC2016上分别获得1.2、3.5和2.6 mAP，同时减少了参数数量。   

### You Only Look One-level Feature-只看一层特征
paper:[arxiv](https://arxiv.org/abs/2103.09460)  
code: [github-chensnathan](https://github.com/chensnathan/YOLOF)  
[official-github-megvii-model](https://github.com/megvii-model/YOLOF)  
说明1：[极市平台](https://zhuanlan.zhihu.com/p/358144493)  
abstract: 本文回顾了用于一级检测器的特征金字塔网络（FPN），指出FPN的成功是由于它对目标检测优化问题的分而治之的解决，而不是多尺度特征融合。从优化的角度出发，我们引入了一种替代方法来解决这个问题，而不是采用复杂的特征金字塔。基于简单而高效的解决方案，我们为您呈现的只是一级功能（YOLOF）。在我们的方法中，提出了两个关键的组成部分：扩展编码器和均匀匹配，并带来了相当大的改进。在COCO基准上的大量实验证明了该模型的有效性。我们的YOLOF与它的特征金字塔对应的RetinaNet取得了相当的结果，同时速度快了2.5倍。在没有 transformer layers的情况下，YOLOF可以以单级特征匹配DETR的性能，只需7×更少的训练周期。YOLOF的图像尺寸为608×608，在2080Ti上以60fps的速度获得44.3mAP，比YOLOv4快13%。   


### Data Augmentation for Object Detection via Differentiable Neural Rendering-基于可微神经绘制的目标检测数据增强方法
paper:[arxiv](https://arxiv.org/abs/2103.02852)  
code: [github](https://github.com/Guanghan/DANR)  
abstract: 在缺少注释数据的情况下，训练一个鲁棒的目标检测器是一个挑战。现有的解决这一问题的方法包括半监督学习（从未标记数据中插入标记数据）和自监督学习（通过借口任务利用未标记数据中的信号）。在不改变有监督学习范式的前提下，本文提出了一种离线数据增强的目标检测方法，该方法对训练数据进行语义插值。具体来说，我们提出的系统基于可微神经渲染生成训练图像的可控视图，以及相应的边界框标注，不需要人工干预。首先，在估计深度图的同时，提取像素对齐的图像特征并投影到点云中。然后，我们用一个目标相机姿势重新投影它们，并呈现一个新的视图2d图像。在点云中标记关键点形式的对象，以恢复新视图中的注释。该方法与仿射变换、图像融合等在线数据增强方法完全兼容。大量实验表明，该方法作为一种无成本的图像和标签增强工具，能够显著提高训练数据匮乏的目标检测系统的性能。  

### Localization Distillation for Object Detection-目标检测中的定位蒸馏  
paper:[arxiv](https://arxiv.org/abs/2102.12252)  
code: [github](https://github.com/HikariTJU/LD)  
abstract: 知识提取（KD）在深度学习领域具有强大的学习紧凑模型的能力，但在提取用于目标检测的定位信息方面仍有局限性。现有的KD目标检测方法主要是模拟教师模型和学生模型之间的深层特征，这不仅受特定模型结构的限制，而且不能提取出定位模糊。在本文中，我们首先提出了定位蒸馏（LD）的目标检测。特别地，通过采用边界盒的一般局部化表示，我们的LD可以表示为标准KD。该方法灵活，适用于任意结构的教师模型和学生模型的局部模糊提取。此外，有趣的是，自我学习，即提炼教师模型本身，可以进一步提高表现。第二，我们提出了一种教师助理策略来弥补教师模式和学生模式之间可能存在的差距，通过这种策略，即使选择的教师模式不是最优的，也可以保证蒸馏的有效性。在PASCAL-VOC和MS-COCO基准数据集上，我们的LD可以持续地提高学生探测器的性能，并且显著地提高了最先进的探测器的性能。  

### Simple multi-dataset detection-简单的多数据集检测  
paper:[arxiv](https://arxiv.org/abs/2102.13086)  
code: [github](https://github.com/xingyizhou/UniDet)  
abstract: 如何建立一个通用的、广泛的目标检测系统？我们使用所有标注过的概念的所有标签。这些标签跨越具有潜在不一致分类的不同数据集。本文提出了一种在多个大规模数据集上训练统一检测器的简单方法。我们使用特定于数据集的训练协议和损失，但与特定于数据集的输出共享一个公共检测体系结构。我们将展示如何将这些特定于数据集的输出自动集成到一个通用的语义分类中。与以前的工作不同，我们的方法不需要手动协调分类法。我们的多数据集检测器在每个训练域上的性能与数据集特定的模型一样好，但在新的未知域上的泛化效果更好。基于该方法的条目在eccv2020鲁棒视觉挑战赛的目标检测和实例分割中排名第一。  

### Unbiased Teacher for Semi-Supervised Object Detection-半监督目标检测的无偏算法
paper:[arxiv](https://arxiv.org/abs/2102.09480)  
code: [github](https://github.com/facebookresearch/unbiased-teacher)  
abstract: 半监督学习，即同时具有标记和未标记数据的训练网络，最近取得了重大进展。然而，现有的工作主要集中在图像分类任务上，而忽略了需要更多注释工作的目标检测。在这项工作中，我们重新审视了半监督目标检测（SS-OD）和确定伪标记偏差问题的SS-OD。为了解决这个问题，我们引入了无偏见的教师，一个简单而有效的方法，共同培养一个学生和一个逐步进步的教师在互惠互利的方式。由于过分自信的伪标签导致班级平衡下降，无偏见的教师在COCO标准、COCO附加数据和VOC数据集上不断改进最先进的方法。具体而言，当使用1%的MS-COCO标记数据时，无偏教师相对于最先进的方法实现了6.8个绝对mAP改进，当仅使用0.5%、1%、2%的MS-COCO标记数据时，相对于监督基线实现了大约10个mAP改进。  


### 搞懂Vision Transformer 原理和代码，看这篇技术综述就够了(三)  
paper: [极市平台](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247541453&idx=1&sn=f9dfe3bcf5e85b413ce1543178681e1e&chksm=ec1ccf34db6b4622badf7244b6ef6c8809b20a1c60faefb3937822f087f93d27ed019b45907c&scene=126&sessionid=1614348438&key=55fcbf4895711c4cec7706b39b38fe68cd74243085e382c11d72400de8cff24ac8be7967f469dfb2f150baf45c31be54f332d6473f18aad9e7222f39a825ffac29f78a90c0123947e90518a0e20e3556ccbcd71a386b9094979523b95d7245e530fde6a208f11eea68921ed3d003c04ba929599e581038ac8d8698accb44ef03&ascene=1&uin=MjExODAxOTE1&devicetype=Windows+10+x64&version=6209051e&lang=zh_CN&exportkey=AS7FJS83MKtw33j8DSIzGVA%3D&pass_ticket=caqTEofVyPZMshcjyLW15l2KOXZ1r4CagsHm3bslDdBCy4fLyu7NiGBPmfRV9Sj9&wx_header=0)



 

### 2021-Joint Object Detection and Multi-Object Tracking with Graph Neural Networks-基于图神经网络的联合目标检测与多目标跟踪  
paper: [arxiv](https://arxiv.org/abs/2006.13164)  
code: [github](https://github.com/yongxinw/GSDT)  
project website: [site](http://www.xinshuoweng.com/projects/GNNDetTrk/)  
Homepage of Xinshuo Weng: [Xinshuo Weng](http://www.xinshuoweng.com/)  
abstract: 目标检测和数据关联是多目标跟踪系统的关键组成部分。尽管这两个组件是相互依赖的，但是之前的工作通常会分别设计检测和数据关联模块，这些模块的训练目标不同。因此，我们不能反向传播梯度并优化整个MOT系统，从而导致次优性能。为了解决这个问题，最近的工作在联合MOT框架下同时优化了检测和数据关联模块，这两个模块的性能都有所提高。在这项工作中，我们提出了一个新的基于图神经网络（GNNs）的联合MOT方法。其核心思想是GNNs能够在空间和时间域上对不同大小的对象之间的关系进行建模，这对于学习用于检测和数据关联的鉴别特征是必不可少的。通过在MOT数据集上的大量实验，我们证明了基于GNN的联合MOT方法的有效性，并展示了检测和MOT任务的最新性能。  

### 2021-High-Performance Large-Scale Image Recognition Without Normalization-无归一化的高性能大规模图像识别*  
说明1: [CVer](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247517020&idx=2&sn=5a9b20d36a20b641739d0ad77bf1bfe5&chksm=f9a1f1d3ced678c5974ba15f003353ecfcbd9481d8437921b25ede0401854b49c2d16d245eda&scene=126&sessionid=1613234694&key=55fcbf4895711c4c29efb51a11009902868c659e12bbde0aa9f37741f52dd088d11eebc7ca6997da73f856dccbd0e065e8b2226de7839bbcdfdcb2034771410097ceb771a68e6f931abfda762731e2d084a280d1b7e4a4bbc35f412b66a88c03e8c472e9f1125c77c246fc756b9c883bab2a2aad9601b000f83bb33a25d99123&ascene=1&uin=MjExODAxOTE1&devicetype=Windows+10+x64&version=6209051e&lang=zh_CN&exportkey=AThsyVtsAWz3%2BIn9ZMF5ebQ%3D&pass_ticket=sAdGHv50drvjr5%2FZxoxSpG%2BpkIDxtegv5o6yLOgTxDMqbleObR9EsLU9NXnqRU3t&wx_header=0)  
说明2: [量子位](https://mp.weixin.qq.com/s/ygf5j6VFkg5YadHoERrtlA)  
paper: [arxiv](https://arxiv.org/abs/2102.06171)     
title: 无归一化的高性能大规模图像识别  
code: <https://github.com/deepmind/deepmind-research/tree/master/nfnets>  
提出了自适应梯度修剪（Adaptive Gradient Clipping，AGC）方法，基于梯度范数与参数范数的单位比例来剪切梯度，研究人员证明了 AGC 可以训练更大批次和大规模数据增强的非归一化网络。  
设计出了被称为 Normalizer-Free ResNets 的新网络，该方法在 ImageNet 验证集上大范围训练等待时间上都获得了最高水平。NFNet-F1 模型达到了与 EfficientNet-B7 相似的准确率，同时训练速度提高了 8.7 倍，而 NFNet 模型的最大版本则树立了全新的 SOTA 水平，无需额外数据即达到了 86.5％的 top-1 准确率。  




### 2021-Occlusion Handling in Generic Object Detection: A Review-一般目标检测中的遮挡处理* 

paper: [arxiv](https://arxiv.org/abs/2101.08845)  
title: 遮挡目标检测-综述  
abstract: 深度学习网络的巨大威力导致了目标检测的巨大发展。在过去的几年中，目标检测器框架在准确性和效率方面都取得了巨大的成功。然而，由于多种因素的影响，它们的能力与人类相差甚远，遮挡就是其中之一。由于遮挡可能发生在不同的位置、比例和比例，因此很难处理。在这篇论文中，我们讨论了在室外和室内场景中一般目标检测中遮挡处理的挑战，然后我们参考了最近为克服这些挑战而开展的工作。最后，我们讨论了未来可能的研究方向。

## 2020-object detection  
site: <https://github.com/extreme-assistant/survey-computer-vision#1>  

### 2020-目标检测方法部分汇总  
zhihu-site: [百思视界](https://zhuanlan.zhihu.com/p/148272698)  
+ 效果最好的当属two-stage目标检测方法：D2Det  
+ 融合关键点检测任务提升anchor-free方法的目标检测效果：CentripetalNet: Pursuing High-quality Keypoint Pairs for Object Detection  
paper: [arxiv](https://arxiv.org/abs/2003.09119)  
code: [github](https://github.com/KiveeDong/CentripetalNet)  
CVPR2020 | CentripetalNet：48.0% AP，通过获取高质量的关键点对来提升目标检测性能: [AI算法修炼营](https://mp.weixin.qq.com/s?__biz=MzI0NDYxODM5NA==&mid=2247485419&idx=1&sn=5a56388e2ccab75773b80bcbda0be240&scene=21#wechat_redirect)   
+ 使用新的网络结构提升目标检测效果：DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution   
paper: [arxiv](https://arxiv.org/abs/2006.02334)  
code: [github](https://github.com/joe-siyuan-qiao/DetectoRS)  
COCO 迎来新榜首！DetectoRS以54.7mAP成就目前最高精度检测网络: [AI深度学习视线](https://mp.weixin.qq.com/s?__biz=MzIwOTM5MjYyMQ==&mid=2247485777&idx=1&sn=580e9040298bb3c1341e986529171317&scene=21#wechat_redirect)  
+ 目标检测新思路：DETR：End-to-End Object Detection with Transformers  
paper: [arxiv](https://arxiv.org/abs/2005.12872)  
code: [github](https://github.com/facebookresearch/detr)  
目标检测 | Facebook开源新思路！DETR：用Transformers来进行端到端的目标检测: [AI算法修炼营](https://mp.weixin.qq.com/s?__biz=MzI0NDYxODM5NA==&mid=2247485126&idx=1&sn=339aae2a3c2160e5ccd2eb2fd7314266&scene=21#wechat_redirect)  

### YOLO：YOLOv1,YOLOv2,YOLOv3,TinyYOLO，YOLOv4,YOLOv5详解  
site: [zhihu](https://zhuanlan.zhihu.com/p/136382095)



### Vision Meets Drones: Past, Present and Future-无人机:过去、现在和未来*  
paper: [arxiv](https://arxiv.org/abs/2001.06303)  
dataset,result: [dataset, result](https://github.com/VisDrone/VisDrone-Dataset)  
abstract: 无人机，或称普通无人机，配备有摄像头，已经被快速部署，有着广泛的应用，包括农业、航空摄影和监视。因此，自动理解从无人机收集的视觉数据变得非常高的要求，使计算机视觉和无人机越来越密切。为了促进和跟踪目标检测和跟踪算法的发展，我们与ECCV 2018和ICCV 2019联合举办了两次挑战研讨会，吸引了全球100多个团队参加。我们提供了一个大规模的无人机捕获数据集VisDrone，它包括四个跟踪，即：（1）图像目标检测，（2）视频目标检测，（3）单目标跟踪，和（4）多目标跟踪。在本文中，我们首先对目标检测和跟踪数据集和基准进行了全面的回顾，并讨论了收集基于无人机的大规模目标检测和跟踪数据集的挑战。之后，我们描述了我们的VisDrone数据集，该数据集从北到南在中国14个不同城市的不同城市/郊区采集。作为迄今为止发布的最大的此类数据集，VisDrone能够对drone平台上的视觉分析算法进行广泛的评估和调查。本文详细分析了无人机大规模目标检测与跟踪领域的现状，总结了存在的问题，并提出了今后的发展方向。我们预计，该基准将大大推动无人机平台视频分析的研发。

### 2020-Deep Domain Adaptive Object Detection: a Survey-深域自适应目标检测综述 
paper: [arxiv](https://arxiv.org/abs/2002.06797)  
title: 深度域适应目标检测  
本文共梳理了46篇相关文献.  
基于深度学习(DL)的目标检测已经取得了很大的进展，这些方法通常假设有大量的带标签的训练数据可用，并且训练和测试数据从相同的分布中提取。  
然而，这两个假设在实践中并不总是成立的。  
深域自适应目标检测(DDAOD)作为一种新的学习范式应运而生。本文综述了深域自适应目标检测方法的研究进展。   


### 2020-Foreground-Background Imbalance Problem in Deep Object Detectors: A Review*  
paper: [arxiv](https://arxiv.org/abs/2006.09238)  
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



### 2020-A Review and Comparative Study on Probabilistic Object Detection in Autonomous Driving* - 自动驾驶中的概率目标检测方法综述与比较研究   
paper: [arxiv](https://arxiv.org/abs/2011.10671)    
code: [github](https://github.com/asharakeh/pod_compare)  
abstract: 捕获目标检测中的不确定性对于安全自主驾驶是必不可少的。近年来，深度学习已成为目标检测的实际方法，许多概率目标检测器被提出。然而，对于深部目标检测中的不确定性估计还没有总结，现有的方法不仅采用不同的网络结构和不确定性估计方法，而且在不同的数据集上进行评估，评估指标范围很广。因此，方法之间的比较仍然具有挑战性，选择最适合特定应用的模型也是如此。本文旨在通过对现有自主驾驶应用中的概率目标检测方法进行回顾和比较研究来缓解这一问题。首先，我们概述了深度学习中的一般不确定性估计，然后系统地综述了现有的概率目标检测方法和评价指标。接下来，我们对基于图像检测器和三个公共自动驾驶数据集的概率目标检测进行了严格的比较研究。最后，我们提出了一个剩余的挑战和未来工作的讨论。  

### 2020-Camouflaged Object Detection and Tracking: A Survey* - 伪装目标检测与跟踪研究综述   
paper: [arxiv](https://arxiv.org/abs/2012.13581)  
abstract: 运动目标检测与跟踪有着广泛的应用，包括监视、异常检测、车辆导航等。关于运动目标检测与跟踪的文献非常丰富，已有一些重要的综述论文。然而，由于问题的复杂性，对伪装目标检测与跟踪的研究受到限制。现有的研究都是基于伪装物体的生物学特性或计算机视觉技术。在这篇文章中，我们回顾了现有的伪装目标检测和跟踪技术，利用计算机视觉算法从理论的角度。本文还讨论了一些感兴趣的问题以及今后的研究方向。我们希望这篇综述能帮助读者了解伪装目标检测和跟踪的最新进展。  



### 2020-无人机-Correlation Filter for UAV-Based Aerial Tracking: A Review and Experimental Evaluation* - 相关过滤无人机空中跟踪技术综述与实验评估  
paper: [arxiv](https://arxiv.org/abs/2010.06255)   
abstract: 航空跟踪是遥感领域最活跃的应用之一，它表现出无所不在的奉献精神和卓越的性能。特别是基于无人机的遥感系统，具有视觉跟踪的特点，在航空、导航、农业、交通、公安等领域得到了广泛的应用，是未来航空遥感的主要技术之一。然而，由于现实世界的繁重环境，如严酷的外部挑战、无人机机械结构的振动（特别是在强风条件下）、复杂环境下的机动飞行以及机载有限的计算资源等，精确性、鲁棒性和高效率都是无人机机载跟踪的关键方法。近年来，基于区分相关滤波器（DCF）的跟踪器以其高计算效率和单CPU鲁棒性在无人机视觉跟踪领域得到了蓬勃发展。本文首先对基于DCF的跟踪器的基本框架进行了概括，在此基础上，根据各自的创新点，有序地总结出23种最新的基于DCF的跟踪器，以解决各种问题。此外，在各种主流的无人机跟踪基准上进行了详尽的定量实验，如UAV123，UAV123，每秒10帧、UAV20L、UAVDT、DTB70、VISSOT，共371903帧。实验结果表明了该方法的有效性，验证了该方法的可行性，并验证了目前基于DCF的无人机跟踪跟踪器所面临的挑战。

### 2020-D2Det: Towards High Quality Object Detection and Instance Segmentation (CVPR2020) - D2Det：迈向高质量目标检测和实例分割  
paper: [cvf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_D2Det_Towards_High_Quality_Object_Detection_and_Instance_Segmentation_CVPR_2020_paper.pdf)  
code: [github](https://github.com/JialeCao001/D2Det)  
abstract: 我们提出了一种新颖的两阶段检测方法D2Det，它同时解决了精确定位和精确分类问题。为了精确定位，我们引入了一种稠密局部回归，它可以预测对象的多个稠密盒偏移量。与传统的回归和基于关键点的两级检测器定位不同，我们的稠密局部回归不局限于一个固定区域内的一组量化关键点，并且能够回归位置敏感的实数稠密偏移量，从而实现更精确的定位。通过二元重叠预测策略进一步改进了稠密局部回归，减少了背景区域对最终盒回归的影响。为了准确分类，我们引入了一种区分性RoI池方案，该方案从一个方案的不同子区域采样，并进行自适应加权以获得区分性特征。  

### 2020-It's Raining Cats or Dogs? Adversarial Rain Attack on DNN Perception-感知对雨天情况的对抗攻击
paper: [arxiv](https://arxiv.org/abs/2009.09205)  
code: [No official code]  
abstract: 降雨是自然界中的一种普遍现象，也是许多基于深度神经网络（DNN）的感知系统的重要因素。雨水通常会带来不可避免的威胁，必须谨慎应对，尤其是在安全和安保敏感的情况下（例如，自动驾驶）。因此，全面调查降雨对DNN的潜在风险具有重要意义。不幸的是，在实践中，通常很难收集或合成能够代表现实世界中可能发生的所有降雨情况的降雨图像。为此，本文从一个新的视角出发，提出将两种完全不同的研究结合起来，即雨天图像合成和对抗攻击。我们提出了一种对抗性的降雨攻击方法，利用这种方法，我们可以在部署的DNNs的指导下模拟各种降雨情况，揭示降雨可能带来的潜在威胁因素，从而开发出更具降雨鲁棒性的DNNs。特别地，我们提出了一种基于因子感知的降雨生成方法，该方法根据摄像机的曝光过程来模拟降雨过程，并对可学习的降雨因子进行了建模。利用该生成器，我们进一步提出了针对图像分类和目标检测的对抗性雨水攻击，其中雨水因子由各种DNN引导。因此，可以全面研究降雨因素对DNNs的影响。我们对三个数据集，即NoRIP'17DEV，MS COCO和KITTI进行了大规模的评估，表明我们合成的雨滴图像不仅能够呈现视觉真实的外观，而且还表现出较强的对抗能力，这为进一步的雨鲁棒感知研究奠定了基础。  

## 2019-object detection  
2019年4篇目标检测算法最佳综述: [CVer](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247493715&idx=3&sn=3f684b51a604cc5d4878a5716aaf2b3f&chksm=f9a19adcced613caa3eef7240dfbf00e866ece9f1ce48366b63984620d1512f16a921ccdb7cf&scene=21#wechat_redirect)  
1. Object Detection in 20 Years: A Survey  
时间：2019年5月  
作者：密歇根大学&北航&卡尔顿大学&滴滴出行  
paper：[arxiv](https://arxiv.org/abs/1905.05055)  
2. A Survey of Deep Learning-based Object Detection  
时间：2019年7月  
作者：西安电子科技大学  
paper：[arxiv](https://arxiv.org/abs/1907.09408)  
3. Recent Advances in Deep Learning for Object Detection  
时间：2019年8月  
作者：新加坡管理大学&Salesforce  
paper：[arxiv](https://arxiv.org/abs/1908.03673)
4. Imbalance Problems in Object Detection: A Review  
title: 目标检测中的不平衡问题   
paper: [arxiv](https://arxiv.org/abs/1909.00169)   
abstract：本文综述了目标检测中的不平衡问题。为了系统地分析问题，我们引入了基于问题的分类法。按照这个分类法，我们深入讨论每个问题，并在文献中提出一个统一但关键的解决方案。此外，我们还确定了有关现有不平衡问题的主要未决问题以及以前没有讨论过的不平衡问题。此外，为了使我们的评论保持最新，我们提供了一个随附的网页，根据我们基于问题的分类法，对解决不平衡问题的论文进行分类。  
最新进展：<https://github.com/kemaloksuz/ObjectDetectionImbalance>


### 重读 CenterNet  
说明：[新智元](https://mp.weixin.qq.com/s/hlc1IKhKLh7Zmr5k_NAykw)  
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




### Augmentation for small object detection*   
paper: [site](https://arxiv.org/abs/1902.07296)  
title: 小目标增强  
abstract：近年来，目标检测取得了令人瞩目的进展。尽管有了这些改进，但是在检测大小物体方面仍然存在很大的差距。我们在一个具有挑战性的数据集COCO女士上分析了当前最先进的模型Mask-RCNN。我们发现，小的地面真实物体和预测的锚之间的重叠远低于预期的IoU阈值。我们推测这是由于两个因素造成的：（1）只有少数图像包含小对象，和（2）即使在包含它们的每个图像中，小对象也显示得不够。因此，我们建议用小对象对这些图像进行过采样，并通过多次复制粘贴小对象来增强每个图像。它允许我们在大物体和小物体上权衡探测器的质量。我们评估了不同的粘贴增强策略，最终，与目前最先进的MS-COCO方法相比，在实例分割和小对象检测方面分别提高了9.7%和7.1%。  

### Rethinking Convolutional Feature Extractionfor Small Object Detection*   
paper: [site](https://bmvc2019.org/wp-content/uploads/papers/1057-paper.pdf)  
title: 卷积特征提取在小目标检测  
abstract: 基于深度学习的目标检测体系结构大大提高了技术水平。然而，最近的检测方法研究表明，在小目标性能和中大型目标性能之间有很大的差距。这一差距在体系结构和主干网之间存在。我们表明，这一差距主要是由于减少了特征地图的大小，因为我们遍历主干。通过对主干结构的简单修改，我们发现对于小对象的性能有了显著的提高。此外，我们还提出了一种具有权值共享的双路径配置来恢复大对象性能。与依赖于多尺度训练和网络划分的最新方法相比，我们在MS-COCO数据集上显示出了具有竞争力的性能。我们用移动对象检测器SSD Mobilenet v1展示了最先进的小对象性能.  

### 2019-Object Detection in 20 Years: A Survey  
paper: [site](https://arxiv.org/abs/1905.05055)  
title: 目标检测20年  
abstract：目标检测作为计算机视觉中最基本、最具挑战性的问题之一，近年来受到了广泛的关注。它在过去二十年中的发展可以看作是计算机视觉历史的一个缩影。如果我们把今天的目标探测看作是在深度学习的力量下的一种技术美学，那么时光倒流20年，我们将见证冷兵器时代的智慧。本文广泛回顾了超过四分之一世纪（从20世纪90年代到2019年）以来，400多篇关于目标检测的论文。本文涵盖了许多主题，包括历史上的里程碑检测器、检测数据集、度量、检测系统的基本构件、加速技术以及最新的检测方法。本文还回顾了一些重要的检测应用，如行人检测、人脸检测、文本检测等，深入分析了它们面临的挑战以及近年来的技术改进。  

### 2019-A Survey of Deep Learning-based Object Detection  
paper: [site](https://arxiv.org/abs/1907.09408)  
title: 深度学习目标检测综述  
abstract：目标检测是计算机视觉最重要、最具挑战性的分支之一，在人们的生活中有着广泛的应用，如安全监控、自动驾驶等，其目的是定位某一类语义对象的实例。随着检测任务深度学习网络的迅速发展，目标检测器的性能得到了很大的提高。为了全面、深入地了解目标检测管道的主要发展现状，本文首先分析了现有典型检测模型的方法，并对基准数据集进行了描述。之后，我们首先系统地概述了各种目标检测方法，包括一级和两级检测器。此外，我们列出了传统和新的应用程序。分析了目标检测的一些典型分支。最后，我们讨论了利用这些目标检测方法构建高效系统的体系结构，并指出了一系列的发展趋势，以便更好地遵循最新的算法和进一步的研究。  

### 2019-Recent Advances in Deep Learning for Object Detection
paper: [site](https://arxiv.org/abs/1908.03673)  
title: 深度学习在目标检测中的研究进展  
abstract：目标检测是计算机视觉中一个基本的视觉识别问题，在过去的几十年中得到了广泛的研究。视觉目标检测的目的是在给定的图像中找到具有精确定位的特定目标类的目标，并为每个目标实例分配相应的类标签。由于基于深度学习的图像分类取得了巨大的成功，基于深度学习的目标检测技术近年来得到了积极的研究。本文综述了深度学习视觉目标检测的最新进展。通过回顾大量最近的相关文献，我们系统地分析了现有的目标检测框架，并将调查分为三个主要部分：（i）检测组件，（ii）学习策略，以及（iii）应用程序和基准测试。在调查中，我们详细讨论了影响检测性能的各种因素，如检测器结构、特征学习、建议生成、采样策略等。最后，我们讨论了几个未来的发展方向，以促进和推动基于深度学习的视觉目标检测的未来研究。  
关键词：目标检测，深度学习，深度卷积神经网络。   


### 2018-Vision Meets Drones: A Challenge-视觉与无人机：挑战*  
paper: [paper](https://arxiv.org/abs/1804.07437)  
abstract: 在本文中，我们提出了一个大规模的视觉目标检测和跟踪基准，名为VisDrone2018，旨在推进无人机平台上的视觉理解任务。基准中的图像和视频序列是在中国14个不同城市的不同城市/郊区从北到南拍摄的。具体来说，VisDrone2018包含263个视频片段和10209个图像（与视频片段没有重叠），具有丰富的注释，包括对象边界框、对象类别、遮挡、截断比率等。通过大量的工作，我们的基准测试在179264个图像/视频帧中有超过250万个注释实例。作为迄今为止公布的最大的此类数据集，基准能够对无人机平台上的视觉分析算法进行广泛的评估和调查。特别地，我们设计了四个比较流行的任务，包括图像中的目标检测、视频中的目标检测、单目标跟踪和多目标跟踪。由于遮挡、大尺度和姿态变化以及快速运动等因素的影响，所有这些任务在所提出的数据集中都是极具挑战性的。我们希望这个基准能极大地推动无人机平台视觉分析的研究与开发。  








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


## others

### 2019-The Transformer Family-9种Transformer结构
site: [极市平台](https://zhuanlan.zhihu.com/p/351715527)
eng-site: [github.io](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html#locality-sensitive-hashing-reformer)

### Transformer一篇就够了  
github:[github](https://github.com/BSlience/transformer-all-in-one)  
site: [zhihu-Transformer 一篇就够了（一）： Self-attenstion](https://zhuanlan.zhihu.com/p/345680792)  
site: [Transformer 一篇就够了（二）： Transformer中的Self-attenstion](https://zhuanlan.zhihu.com/p/347492368)  
site: [Transformer 一篇就够了（三）： Transformer的实现](https://zhuanlan.zhihu.com/p/347709112)  


