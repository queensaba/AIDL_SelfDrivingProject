import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision
import numpy
import torch
import argparse
import cv2
from torchinfo import summary
#import detect_utils
from PIL import Image
# Main file of the project

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#DATA_DIR = "/Users/helenamartin/Desktop/AI-Project/"
DATA_DIR = "/Users/helenamartin/Desktop/AI-Project/"

###################
###################
#####TO CHANGE#####
###################
###################
# construct the argument parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input image/video')
    parser.add_argument('-m', '--min_size', dest='min_size', default=800,help='minimum input size for the FasterRCNN network')
    args = parser.parse_args()
    return args
#class SDDataLoader(json_path):
#    """
#    Self Driving Project Dataset.
#    """
#    self.dataset_annotations = self.annotations_to_csv(json_path)
#    self.dataset = self.generate_dict(self.dataset_annotations, DATA_DIR)
#    self.train,self.val = self.split_dataset(self.dataset_annotations)
#
#    def annotations_to_csv(self, json_path):
#        import pandas as pd
#        csv_data = pd.read_json(json_path)
#        return csv_data
#
#    def generate_dict(self, dataset_Annotations, DATA_DIR):
#        from PIL import Image
#        from torchivison import transforms
#        import glob
#        images = glob.glob(os.path.join(DATA_DIR, '*.jpg'))
#        for image in images:
#            convert_tensor = transforms.ToTensor()
#            images[os.path.basename(image)] = convert_tensor(img)
#            labels[os.path.basename(image)] =

###################
###################
###################
###################
objects = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
###################
###################
###################
###################

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(objects), 3))
#objects = ["pedestrian","rider","car","truck","bus","train","motorcycle","bicycle","traffic light","traffic sign"]
# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

#roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]
#print(roi_loc.shape)
#Out:
#torch.Size([128, 4])
#calculating regression loss in the same way as we calculated regression loss for RPN network we get
#roi_loc_loss = REGLoss(roi_loc, gt_roi_loc)

def train_epoch(dataloader, model, optimizer, criterion):

    # Train the model
    train_loss = 0
    for X, y in dataloader:
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_ = model(X)
        loss = criterion(y_, y)
        train_loss += loss.item() * len(y)
        #acc1, acc5 = accuracy(y_, y, topk=(1, 5))
        loss.backward()
        optimizer.step()


    return train_loss / len(dataloader.dataset)


def test_epoch(dataloader, model, criterion):
    test_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(X)
            loss = criterion(y_,y)
            test_loss += loss.item() * len(y)

    return test_loss / len(dataloader.dataset)

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image) # get the predictions on the image
    # print the results individually
    # print(f"BOXES: {outputs[0]['boxes']}")
    # print(f"LABELS: {outputs[0]['labels']}")
    # print(f"SCORES: {outputs[0]['scores']}")
    # get all the predicited class names
    pred_classes = [objects[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes, pred_classes, outputs[0]['labels']


def draw_boxes(boxes, classes, labels, image):
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image


arguments = get_args()

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,min_size=arguments.min_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: List its parameters, given an input of [bs, nchannels, width, depth], using the summary function from torchinfo
summary(model)

image = Image.open(arguments.input)
model.eval().to(device)
boxes, classes, labels = predict(image, model, device, 0.8)
image = draw_boxes(boxes, classes, labels, image)
cv2.imshow('Image', image)
save_name = f"{arguments.input.split('/')[-1].split('.')[0]}_{arguments.min_size}"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)