# Main file of the project

import torch
import os
import numpy as np
import argparse as ap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    #self.images = glob.glob(os.path.join(DATA_DIR, '*.jpg'))
    #self.dataset_annotations = self.annotations_to_csv(json_path)
    #self.dataset = self.generate_dict(self.dataset_annotations, DATA_DIR)
    #self.train,self.val = self.split_dataset(self.dataset_annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(DATA_DIR, self.images[idx])


    def annotations_to_csv(self, json_path, DATA_DIR):
        dataset = {}
        import pandas as pd
        from PIL import Image
        from torchvision import transforms
        csv_data = pd.read_json(json_path)
        for idx,item in enumerate(csv_data['name']):
            try:
                dataset[item] = {}
                img_p = Image.open(os.path.join(DATA_DIR,item))
                transform = transforms.Compose([transforms.ToTensor()])
                dataset[item]['img'] = transform(img_p)
                dataset[item]['box'] = []
                for id in range(0,len(csv_data['labels'][idx])):
                    dataset[item]['box'].append([value for value in csv_data['labels'][idx][id]['box2d'].values()])
            except TypeError:
                print('Label values are nan')



        return csv_data

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


'''
def train_epoch(dataloader, model, optimizer, criterion):
    train_loss = 0
    for X, y in dataloader:
        optimizer...
        X, y = X.to(device), y.to(device)
        y_ = ...
        loss = ...
        train_loss += loss.item() * len(y)
        loss...
        optimizer...

    return train_loss / len(dataloader.dataset)


def test_epoch(dataloader: DataLoader, model, criterion):
    test_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            #y_ = ...
            #loss = ...
            test_loss += loss.item() * len(y)

    return test_loss / len(dataloader.dataset)


def load_data():
    #df = pd.read_csv("/data/housing.csv")

    train_df, test_df = train_test_split(df, test_size=0.2)
    train_X, train_y = train_df.drop(["ID", "MEDV"], axis=1), train_df["MEDV"]
    test_X, test_y = test_df.drop(["ID", "MEDV"], axis=1), test_df["MEDV"]
    train_X, train_y = train_X.to_numpy(), train_y.to_numpy()
    test_X, test_y = test_X.to_numpy(), test_y.to_numpy()
    return train_X, train_y, test_X, test_y

def train():

    # Hyperparameters
    BATCH_SIZE = 16
    N_EPOCHS = 10
    HIDDEN_SIZE = 64

    train_X, train_y, test_X, test_y = load_data()

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # TODO: define the composed transformation for training
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize])

    # TODO: define the composed transformation for validation
    val_transforms = transforms.Compose([transforms.RandomResizedCrop(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize, ])
'''
if __name__ == '__main__':
    args = get_args()
    jsons_p = args.json_path
    imgs_p = args.imgs

    data = SDDataLoader()
    dict = data.annotations_to_csv(jsons_p,imgs_p)
    print(dict)


