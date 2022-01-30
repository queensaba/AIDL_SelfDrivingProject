import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import cv2 as cv
import pandas as pd
from PIL import Image
import os

def create_dict1(root = 'data/coco/train/'): #Function that returns a dictionary of images and annotations
    label = []
    image = []
    root = root + '/labels/labels.txt'
    with open(root) as f:
        lines = f.readlines()
        for image_labels in lines:
            #Example:
            # c:\ceci\aidl_project\AIDL_SelfDrivingProject\yolo_v1\data\coco\train\labels\COCO_train2014_000000510997.txt
            img_name = image_labels[76:(len(image_labels)-5)]+".jpg"
            image.append(img_name)
            with open(image_labels.replace(".txt\n",".txt")) as f2:
                lines2 = f2.readlines()
                object1 = list()
                for detection in lines2:
                    detection = detection.split(" ")
                    if len(detection)<5:
                        continue
                    object1.append(detection[0])
                    object1.append(detection[1])
                    object1.append(detection[2])
                    object1.append(detection[3])
                    object1.append(detection[4])
                label.append(object1)
    del img_name, image_labels, lines
    img_labels = {}
    for i in range(len(image)):
        img_labels[image[i]] = label[i]            
    del label, image
    return img_labels

# img_transform = transforms.Compose([
#     transforms.Resize((448, 448)),
#     transforms.ToTensor()]
# )

class myOwnDataset(Dataset):
    def __init__(self, root, transform=None, S = 7, B = 2, C = 80):
        self.img_labels = create_dict1(root)
        self.root = root
        self.transform = transform
        list_ids = []
        for i in range(len(self.img_labels)):
            list_ids.append(list(self.img_labels.items())[i][0])
        self.ids = list_ids
        self.S = S
        self.B = B
        self.C = C
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_labels = self.img_labels
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = img_labels[img_id]
        # path for input image
        pathbase = 'data/coco/train/images/'
        path = pathbase+img_id
        # open the input image
        img = Image.open(os.path.join(path)).convert('RGB')

        # number of objects in the image
        #int(len(img_labels[img])/5)
        num_objs = int(len(ann_ids)/5)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        img1 = cv.imread(path)
        img_height = img1.shape[0]
        img_width = img1.shape[1]     
        objects = []    
        for i in range(num_objs):
            x_min_rect = ((2 * float(img_labels[img_id][i*5+1]) * img_width) - (float(img_labels[img_id][i*5+3]) * img_width)) / 2
            x_max_rect = ((2 * float(img_labels[img_id][i*5+1]) * img_width) + (float(img_labels[img_id][i*5+3]) * img_width)) / 2
            y_min_rect = ((2 * float(img_labels[img_id][i*5+2]) * img_height) - (float(img_labels[img_id][i*5+4]) * img_height)) / 2
            y_max_rect = ((2 * float(img_labels[img_id][i*5+2]) * img_height) + (float(img_labels[img_id][i*5+4]) * img_height)) / 2
            boxes.append([x_min_rect, y_min_rect, x_max_rect, y_max_rect])
            objects.append(int(img_labels[img_id][i*5]))
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if self.transform:
            img, boxes = self.transform(img, boxes)


        # Convert To Cells
        label_matrix = torch.zeros(self.S, self.S, self.C + 5 * self.B)
        for i in range(num_objs):
            class_label = objects[i]
            centerx = boxes[i][0] + (boxes[i][2]-boxes[i][0])/2
            centery = boxes[i][1] + (boxes[i][3]-boxes[i][1])/2
            width = (boxes[i][0]+boxes[i][2])/2
            height = (boxes[i][1]+boxes[i][3])/2

            i = int(self.S *centery/img_height)
            j = int(self.S *centerx/img_width)

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 80] == 0:
                # Set that there exists an object
                label_matrix[i, j, 80] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [centerx, centery, width, height]
                )

                label_matrix[i, j, 81:85] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return img, label_matrix







