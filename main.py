# Main file of the project

import torch
import os
import numpy as np
import argparse as ap
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import Dataset
import glob
from utils import YoloLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
                 "truck", "train", "other person", "bus", "car", "rider",
                 "motorcycle", "bicycle", "trailer"]
category_color = [(255,255,0),(255,0,0),(255,128,0),(0,255,255),(255,0,255),
                  (128,255,0),(0,255,128),(255,0,127),(0,255,0),(0,0,255),
                  (127,0,255),(0,128,255),(128,128,128)]


def parse_json_files():
    """
    Store images and labels into arrays
    :return:
    """
    import pandas as pd
    from PIL import Image
    from torchvision import transforms
    csv_data = pd.read_json(json_path)
    convert_tensor = transforms.ToTensor()
    dataset={}
    for idx, item in enumerate(csv_data['name']):
        if os.path.isfile(os.path.join(DATA_DIR, item)):
            dataset[item] = {}
            img_p = Image.open(os.path.join(DATA_DIR, item))
            transform = transforms.Compose([transforms.ToTensor()])
            dataset[item]['img'] = transform(img_p)
            dataset[item]['box'] = []
            try:
                for id in range(0, len(csv_data['labels'][idx])):
                    dataset[item]['box'].append([value for value in csv_data['labels'][idx][id]['box2d'].values()])
            except TypeError:
                print('Label values are nan')
        else:
            continue
    return dataset

def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-j", "--json_path", type=str, required=True,
                        help="Path to folder with JSON data.")
    parser.add_argument("-i", "--imgs", type=str, required=True,
                       help="Path to folder with images.")
    args = parser.parse_args()
    return args


class SDDataLoader():
    """
    Self Driving Prokect Dataset.
    """
    def __init__(self):

        #self.img_paths = glob.glob(os.path.join(DATA_DIR, '*.jpg'))
        #self.dataset_annotations = self.annotations_to_csv(json_path)
        #self.dataset = self.generate_dict(self.dataset_annotations, DATA_DIR)
        #self.train,self.val = self.split_dataset(self.dataset_annotations)
        self.labels = self.json_to_list(jsons_p)
        self.images = self.images_to_

    def __getitem__(self, idx):
        selected_image = self.images[idx]
        selected_labels = self.labels[idx]
        image_pil = PIL.Image.open(os.path.join(self.images_path,selected_image)).convert('RGB')
        image = self.to_tensor_and_normalize(image_pil)



    def json_to_list(self,jsons_p):
        import pandas as pd
        dataset = pd.read_json(jsons_p)
        labels = dataset['labels'].tolist()
        return labels


    def annotations_to_csv(self, json_path, DATA_DIR):
        dataset = {}
        import pandas as pd
        from PIL import Image
        from torchvision import transforms
        csv_data = pd.read_json(json_path)
        import pdb
        pdb.set_trace()
        convert_tensor = transforms.ToTensor()

        for idx,item in enumerate(csv_data['name']):
            if os.path.isfile(os.path.join(DATA_DIR,item)):
                dataset[item] = {}
                img_p = Image.open(os.path.join(DATA_DIR,item))
                transform = transforms.Compose([transforms.ToTensor()])
                dataset[item]['img'] = transform(img_p)
                dataset[item]['box'] = []
                try:
                    for id in range(0,len(csv_data['labels'][idx])):
                        dataset[item]['box'].append([value for value in csv_data['labels'][idx][id]['box2d'].values()])
                except TypeError:
                    print('Label values are nan')
            else:
                continue
        return dataset

    def generate_dict(self, dataset_Annotations, DATA_DIR):
        from PIL import Image
        from torchivison import transforms
        import glob
        for image in self.images:
            convert_tensor = transforms.ToTensor()
            images[os.path.basename(image)] = convert_tensor(img)
            #labels[os.path.basename(image)] =

    def __len__(self):
        return len(self.images)

def train(jsons_p,imgs_p):
    # Training yolo v1
    import torch
    from torchsummary import summary
    import matplotlib.pyplot as plt
    from model.nn_model import YoloV1Model
    from model.nn_model import YOLOv1
    from data import DataLoader

    category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
                     "truck", "train", "other person", "bus", "car", "rider", "motorcycle",
                     "bicycle", "trailer"]
    split_size = 5
    num_boxes = 2
    num_classes = len(category_list)
    lambda_coord = 5
    lambda_noobj = 0.5
    data = \
        DataLoader(
        img_files_path=imgs_p,
        target_files_path=jsons_p,
        category_list=category_list,
        split_size=1,
        batch_size=1,
        load_size=1
    )

    # Defining hyperparameters:
    hparams = {
        'num_epochs': 5,
        'batch_size': 100,
        'channels': 3,
        'learning_rate': 2e-5,
        'classes': len(category_list)
    }
    use_gpu = False
    num_epochs=1
    yolo = YoloV1Model(hparams['channels'],classes=hparams['classes'])
    model = YOLOv1(split_size, num_boxes, num_classes)
    optimizer = torch.optim.SGD(params=yolo.parameters(), lr=hparams['learning_rate'], momentum=1)

    # Move model to the GPU
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    loss_fn = YoloLoss(C=hparams['classes'])
    train_loss_avg = []
    model = model.to(device)
    yolo.train()
    model.train()

    for epoch in range(num_epochs):

        print("DATA IS BEING LOADED FOR A NEW EPOCH")
        print("")
        data.LoadFiles()  # Resets the DataLoader for a new epoch

        while len(data.img_files) > 0:

            print("LOADING NEW BATCHES")
            print("Remaining files:" + str(len(data.img_files)))
            print("")
            data.LoadData()  # Loads new batches

            for batch_idx, (img_data, target_data) in enumerate(data.data):
                img_data = img_data.to(device)
                target_data = target_data.to(device)

                optimizer.zero_grad()

                prediction = yolo(img_data)
                #predictions = model(img_data)

                loss = loss_fn(prediction,target_data)
                train_loss_avg.append(loss.item())
                #yolo_loss = YOLO_Loss(predictions, target_data, split_size, num_boxes,
                                      #num_classes, lambda_coord, lambda_noobj)

                #yolo_loss.loss()
                #loss = yolo_loss.final_loss

                loss.backward()
                optimizer.step()

                print('Train Epoch: {} of {} [Batch: {}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch + 1, num_epochs, batch_idx + 1, len(data.data),
                    (batch_idx + 1) / len(data.data) * 100., loss))
                print('')

        time.sleep(10)

if __name__ == '__main__':
    args = get_args()
    jsons_p = args.json_path
    imgs_p = args.imgs
    train(jsons_p,imgs_p)


