
# 目标检测算法综述

Here's the table of contents:

1. TOC
{:toc}

## Object Detection in 20 Years: A Survey  

<https://arxiv.org/abs/1905.05055>

(Submitted on 13 May 2019 (v1), last revised 16 May 2019 (this version, v2))

**Abstract**: Object detection, as of one the most fundamental and challenging problems in computer vision, has received great attention in recent years. Its development in the past two decades can be regarded as an epitome of computer vision history. If we think of today's object detection as a technical aesthetics under the power of deep learning, then turning back the clock 20 years we would witness the wisdom of cold weapon era. This paper extensively reviews 400+ papers of object detection in the light of its technical evolution, spanning over a quarter-century's time (from the 1990s to 2019). A number of topics have been covered in this paper, including the milestone detectors in history, detection datasets, metrics, fundamental building blocks of the detection system, speed up techniques, and the recent state of the art detection methods. This paper also reviews some important detection applications, such as pedestrian detection, face detection, text detection, etc, and makes an in-deep analysis of their challenges as well as technical improvements in recent years.

### Milestones  

1. ** Milestones: Traditional Detectors**  
    + Viola Jones Detectors  
    + HOG Detector (2005)  
    + Deformable Part-based Model (DPM) (2008)  

2. **Milestones: CNN based Two-stage Detectors**  
    + RCNN  
	+ SPPNet (2014)  
	+ Fast RCNN (2015)  
	+ Faster RCNN (2015)  
	+ Feature Pyramid Networks(FPN) (Feature pyramid networks for objectdetection, 2017)  

3. **Milestones: CNN based One-stage Detectors**  
    + You Only Look Once (YOLO) (2015)
	+ Single Shot MultiBox Detector (SSD) (2015)
	+ RetinaNet (2017)
	+ Faster RCNN (2015)
	+ Feature Pyramid Networks(FPN) (Feature pyramid networks for objectdetection, 2017)

### Milestones Object Detection Datasets

+ Pascal VOC  
+ ILSVRC  
+ MS-COCO (since 2015) <http://cocodataset.org/#home>  
+ OID-2018 <https://storage.googleapis.com/openimages/web/index.html>  
+ ...  
+ **Pedestrian Detection Datasets**
    + INRIA (2005) <http://pascal.inrialpes.fr/data/human/>  
	+ Caltech (2009) <http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/>  
	+ KITTI (2012) <http://www.cvlibs.net/datasets/kitti/index.php>  
+ **Face Detection Datasets**
    + FDDB (2010) <http://vis-www.cs.umass.edu/fddb/index.html>  
	+ AFLW (2011) <https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/>  
	+ IJB (2015) <https://www.nist.gov/programs-projects/face-challenges>  
	+ WiderFace (2016) <http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/>  
+ **Scene Text Detection Datasets**  
	+ ...  
+ **Traffic Light Detection and Traffic Sign Detection Datasets**
	+ LISA (2012) <http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html>  
	+ GTSDB (2013) <http://benchmark.ini.rub.de/?section=gtsdb&subsection=news>  
	+ BelgianTSD (2012) <https://btsd.ethz.ch/shareddata/>  
	+ TT100K (2016) <http://cg.cs.tsinghua.edu.cn/traffic-sign/>  

**Evaluate the Effectiveness of an Object Detector**
+ Average detection precision under different recalls: Average Precision (AP)  
+ mean AP (mAP)  
+ To measure the object localization accuracy: Intersection over Union (IoU)  

### Applications and Difficulties

+ **Pedestrian Detection**
    + Small pedestrian
    + Hard negatives
    + Dense and occluded pedestrian  
    + Real-time detection  
+ ** Face Detection**
    + Intra-class variation  
	+ Occlusion  
	+ Multi-scale detection  
	+ Real-time detection
+ **Text Detection**
+ **Traffic Sign and Traffic Light Detection**
    + Illumination changes
	+ Motion blur
	+ Bad weather
	+ Real-time detection
+ **Remote Sensing Target Detection**

### **FUTURE DIRECTIONS**
+ Lightweight object detection  
+ Detection meets AutoML  
+ Detection meets domain adaptation  
+ Weakly supervised detection  
+ Small object detection  
+ Detection in videos  
+ Detection with information fusion  
