 patch-2
# AIDL_SelfDrivingProject

 
 **Motivation-Problems and Challenges**

SelfDriving cars are important because they help to reduce road accidents. But they also create new risks.  

# Self Driving Project - AIDL
### Students: Helena Martin, Hermela Seleshi and Cecilia Siliato
### Advisor: Mariona Caròs


## 1. Motivation

 **Problems and Challenges**

Self-Driving cars are important because they help to reduce road accidents. But they also create new risks.  
main

**Critical Tasks**

In order to mitigate these risks, the autonomous vehicle has to be able to sense and react to its surroundings by performing at least three critical tasks: object detection, drivable area segmentation, and lane detection

 patch-2
**The Visual Perception System**

a visual perception system is needed to perform these critical tasks

**Main Project Goal**

we will focus on the first critical task of the visual perception system, namely, traffic object detection

**Main Project Goal**
   
  We will focus on the first critical task of object detection, is the most important requirement for autonomous navigation and consist both classification and localization.
 
## 2. YOLO v1

### 2.1 Architecture and main idea of YOLO v1

You Only Look Once (YOLO) is an object detection model. The name is due to the fact that this algorithm is able to detect and recognize various objects in a picture (in real-time). It is also important to mention that nowadays there exists many versions of this model (v1, v2, v3, v4,...). We have selected the first one because of resources issues but also we considered really important having clear the main idea of this model and for so, it is enough working with YOLO v1.

The main idea of YOLO is first dividing the input image in a fixed number SxS cells as we can see on the image (in this example S=7):
![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/gridinput.PNG)

After that, each of the SxS cells of the grid will be responsible to detect a maximum of 1 object of the image. It is important to know that we say that a cell is responsible for detecting an object if the center of the bounding box of this object is on the cell.

On the following example, the cell (4,3) would be the responsible for detecting the bike:

![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/gridinput_bbox.png)

In order to do this, YOLO v1 has an architecture consisting of 6 blocks combining convolutional layers with maxpooling layers and followed by 2 fully connected layers. Furthermore it applies the Leaky ReLu activation function after all layers except for the last one and uses dropout between the two fully connected layers in order to tackle overfitting.

![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/yolo_architecture.png)

The dimension of the last fully connected layer would be SxSx(B * 5 + C) where:

* S: Number of cells in which input image is divided
* B: Number of bounding boxes that will predict each cell (normally B=2, so each cell is responsible to detect 2 bounding boxes and then 1 is discarded)
* C: Number of classes of our dataset

So, for each cell of the image you have:
* 2 bounding boxes predicted: each bounding box has 4 numbers (center coordinates and width/height) and the number remaining is the probability of having an object inside this bounding box (P(Object)
* Class probabilities: For each of our dataset classes, we have the conditional probability (P(class 1 | Object)... P(class C | Object))

It is more clear showed in this image:

![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/lastlayer.PNG)

So, we can then understand that when we execute the model, we will obtain SxSx2 bounding boxes but then it will be compared with the ground truth bounding boxes in order to keep only the ones with the highest IoU (intersection over union).

### 2.2 Loss in YOLO v1

The loss function in YOLO v1 is not one of the classic losses in neural networks. In this case, the loss is divided in different losses that we will see now:
**1. Bounding box loss**: This loss as the name suggests refers to the bounding box and it is divided into 2 different losses:

1.1. *Bounding box centroid loss*: This will be the distance of the center of the predicted bounding box and the ground truth bounding box but it is important to keep in mind that it is only computed when there actually exists an object on the studied cell. It is computed as follows: 
![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/bboxcoord.PNG)

1.2. *Bounding box size loss*: This one is computed as the distance of the width and height of the predicted bounding box and the ground truth bounding box but it is important to keep in mind that it is only computed when there actually exists an object on the studied cell. In this case we have to keep in mind also that it is computed the sqrt because otherwise larger bounding boxes would have more importance on this loss.

The formal formula is:
![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/bboxsize.PNG)

**2. Object loss**: This loss refers to the error that is done when assigning object probability and in the ground-truth there is an object. In other words, if there is an object on a particular cell, which is the difference between the P(Object) and 1? It is computed as follows:
![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/obj.PNG)

**3. No Object loss**: This loss refers to the error that is done when assigning object probability and in the ground-truth there is not any object. In other words, if there is not any object on a particular cell, which is the difference between the P(Object) and 0? It is computed as follows:
![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/noobj.PNG)

**4. Class loss**: In this last loss, we are computing the error made when assigning a class to a detected object, so it is pretty similar as the previous loss but in this case, we are looking at the P(Class i | Object). The formula is:
![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/classprob.PNG)

So, finally, if we add all these losses, we will obtain the loss of YOLO v1:
![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/lossformula.PNG)


## 3. Training YOLO v1

## 4. Transfer Learning

### 4.1 Introduction to Transfer Learning
First of all, we will make a very quick introduction to Transfer Learning. The main idea of TL is using models already trained on our custom dataset. In order to do that, there are some steps that should be done:

1. Select a model that performs the same (or a very similar) task as we want to do in order to take advantage of the features that the pretrained model already trained
2. Select the backbone that we want for our model (it can be useful for faster predictions, for example)
3. Load the parameters (weights and biases) from the pretrained model: In the case of pytorch, we have some models already pretrained for object detection (for instance Faster R-CNN and RetinaNet)
4. Finetune the model to better adapt to our dataset: Finetuning the model means changing some parameters (or even some layers) so then we can train it again with our dataset. In our particular case, we needed to adapt the pretrained model to a different number of classes (as the models were previously pretrained with COCO (91 classes) and we have less classes)
5. Train the model again for a few epochs: The idea no is training the model again but with our own data. Normally it is not needed to do it with the whole dataset but with a fewer subset (the main features are already learnt with the pretrained model and we only want to adapt it to our dataset)
6. The "retrained" model will now do some better predictions on our dataset

The main advantages of doing transfer learning are the saving of computer and time resources but also the "no-need" to have huge datasets that normally are difficult/expensive to label.

### 4.2 Application of Transfer Learning in this project

As we have used the Pytorch environment on this project, we have taken advantage of some models already pretrained on pytorch and COCO dataset. So, we have selected 2 of the best performing models:

* Faster R-CNN: This is the last algorithm of the trilogy (R-CNN, Fast R-CNN and Faster R-CNN) and the main idea is that there are 2 subnetworks (2-stage object detector):
  
  1. Region Proposal Network: This network will be the responsible of purposing different regions in which may exist an object
  
  2. Classifier: Here, the object classification will be done once the RPN has send to it some region proposals

![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/FasterRCNN.PNG)


* RetinaNet: This is a one-stage object detection model that utilizes a focal loss function to address class imbalance during training.

![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/report/RetinaNetArch.PNG)

The Transfer Learning codes are on the directory *transfer_learning* and in order to reproduce them, we have a mini subset of our dataset in *transfer_learning/data*. We can do the first part of "retraining the models" using the codes *train_FasterRCNN.py* and *train_RetinaNet.py*. 

On the first one, 3 arguments are needed: 

1. "b": It refers to the backbone used. Possibilities:

  1.1 b=1: MobileNetV3: Constructs a high resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone
  
  1.2 b=2: ResNet50: Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.
  
  1.3 b=3: MobileNetV3-320: Constructs a low resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone tunned for mobile use-cases.
  
2. "c": It refers to the number of classes used (on the data subset we have 11 classes)
 
3. "e": Number of epochs for training the model

So, the main steps would be:

```
git clone https://github.com/hemahecodes/AIDL_SelfDrivingProject
cd transfer_learning
python train_FasterRCNN.py -b 1 -c 11 -e 20
```

This last instructions will train and save the model with the trained weights in *transfer_learning/models*, in the ".pth" file it will be specified the pretrained model and backbone used.

On the case of RetinaNet, we do not have to specify the backbone (as the only available is ResNet50) so we only should give the number of classes and epochs.

In order to do inference with the "retrained" models, we can use the *inference....py* scripts as follows:

```
python inference_FasterRCNN_resnet50.py -i "data/DeepDriving/test/fd5bae34-d63db3d7.jpg"
```


The image with the bounding boxes of the objects detected will be automatically showed and saved on *predictions/prediction_FastRCNN-ResNet50fd5bae34-d63db3d7.jpg*. In this particular case, the result would be:

![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/transfer_learning/predictions/prediction_FastRCNN-ResNet50fd5bae34-d63db3d7.jpg)

So, we can see that it has worked pretty well. In general, we have seen the best results when using FastRCNN with ResNet50 backbone and the worst with Fast R-CNN and MobileNet v3-320. In fact, if we compare this image predicted with the other pretrained models:

**FastRCNN with MobileNet v3**
![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/transfer_learning/predictions/prediction_FastRCNN-MobileNetv3fd5bae34-d63db3d7.jpg)

**FastRCNN with MobileNet v3-320**
![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/transfer_learning/predictions/prediction_FastRCNN-MobileNetv3-320fd5bae34-d63db3d7.jpg)

**FastRCNN with ResNet50**
![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/transfer_learning/predictions/prediction_FastRCNN-ResNet50fd5bae34-d63db3d7.jpg)

**RetinaNet with ResNet50**
![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/dev/transfer_learning/predictions/prediction_RetinaNet-ResNet50fd5bae34-d63db3d7.jpg)



## 5. Models comparison

## 6. Validation with our own images

## 7. Conclusion and future work


 main
