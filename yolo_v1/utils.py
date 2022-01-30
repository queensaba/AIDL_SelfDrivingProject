from datasets import create_dict1
import cv2 as cv
from random import randint
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


color = []
num_objects = 80
for i in range(num_objects):
    color.append('#%06X' % randint(0, 0xFFFFFF))

def draw_bounding_boxes(img):
    img_labels = create_dict1()
    img1 = cv.imread('data/coco/train/images/'+img)
    height = img1.shape[0]
    width = img1.shape[1]
    n = int(len(img_labels[img])/5)
    for i in range(n):
        c = color[int(img_labels[img][i*5])]
        x_min_rect = ((2 * float(img_labels[img][i*5+1]) * width) - (float(img_labels[img][i*5+3]) * width)) / 2
        x_max_rect = ((2 * float(img_labels[img][i*5+1]) * width) + (float(img_labels[img][i*5+3]) * width)) / 2
        y_min_rect = ((2 * float(img_labels[img][i*5+2]) * height) - (float(img_labels[img][i*5+4]) * height)) / 2
        y_max_rect = ((2 * float(img_labels[img][i*5+2]) * height) + (float(img_labels[img][i*5+4]) * height)) / 2
        x_values1 = [x_min_rect,x_max_rect]
        y_values1 = [y_min_rect,y_min_rect]
        x_values2 = [x_max_rect,x_max_rect]
        y_values2 = [y_min_rect,y_max_rect]
        x_values3 = [x_min_rect,x_max_rect]
        y_values3 = [y_max_rect,y_max_rect]
        x_values4 = [x_min_rect,x_min_rect]
        y_values4 = [y_min_rect,y_max_rect]
        plt.plot(x_values1, y_values1, color = c)
        plt.plot(x_values2, y_values2, color = c)
        plt.plot(x_values3, y_values3, color = c)
        plt.plot(x_values4, y_values4, color = c)
    
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.show()

#Drawing bounding boxes example
# draw_bounding_boxes('COCO_train2014_000000510997.jpg')

def intersection_over_union(boxA, boxB, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxA (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxB (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """
    if box_format == "midpoint":
        box1_x1 = boxA[..., 0:1] - boxA[..., 2:3] / 2
        box1_y1 = boxA[..., 1:2] - boxA[..., 3:4] / 2
        box1_x2 = boxA[..., 0:1] + boxA[..., 2:3] / 2
        box1_y2 = boxA[..., 1:2] + boxA[..., 3:4] / 2
        box2_x1 = boxB[..., 0:1] - boxB[..., 2:3] / 2
        box2_y1 = boxB[..., 1:2] - boxB[..., 3:4] / 2
        box2_x2 = boxB[..., 0:1] + boxB[..., 2:3] / 2
        box2_y2 = boxB[..., 1:2] + boxB[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxA[..., 0:1]
        box1_y1 = boxA[..., 1:2]
        box1_x2 = boxA[..., 2:3]
        box1_y2 = boxA[..., 3:4]  # (N, 1)
        box2_x1 = boxB[..., 0:1]
        box2_y1 = boxB[..., 1:2]
        box2_x2 = boxB[..., 2:3]
        box2_y2 = boxB[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 80): #S is the number of gris in which we are going to divide (7x7), B is the quantity of boundig box per cell, C is the number of classes
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction = "sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5) #Make sure that the shape is (-1,7,7,80+10) = (-1,7,7,90)
        iou_b1 = intersection_over_union(predictions[...,81:85],target[...,81:85]) #From 0 to 79 is for class probabilities, 80 i for class score
        iou_b2 = intersection_over_union(predictions[...,86:90],target[...,81:85])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
        iou_maxes, bestbox = torch.max(ious, dim = 0)
        exists_box = target[..., 80].unsqueeze(3)  
        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 86:90]
                + (1 - bestbox) * predictions[..., 81:85]
            )
        )

        box_targets = exists_box * target[...,81:85]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )
        print("box_loss is:")
        print(box_loss)

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 85:86] + (1 - bestbox) * predictions[..., 80:81]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[...,80:81]),
        )
        print("object loss is")
        print(object_loss)

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 80:81], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 80:81], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 85:86], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 80:81], start_dim=1)
        )
        print("no object loss is")
        print(no_object_loss)

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :80], end_dim=-2,),
            torch.flatten(exists_box * target[..., :80], end_dim=-2,),
        )
        print("class_loss")
        print(class_loss)

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )
        print("final loss is:")
        print(loss)

        return loss

        
