# Main file of the project
import pdb
import wandb
import torch
import os
import numpy as np
import argparse as ap
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import Dataset
import glob
from utils import YoloLoss


def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-j", "--json_path", type=str, required=True,
                        help="Path to folder with JSON data.")
    parser.add_argument("-i", "--imgs", type=str, required=True,
                       help="Path to folder with images.")
    args = parser.parse_args()
    return args



def train(jsons_p,imgs_p):
    # Training yolo v1
    import torch
    from torchsummary import summary
    import matplotlib.pyplot as plt
    from model.nn_model import YoloV1Model
    from data import DataLoader
    from utils import retrieve_box
    import torchvision
    from torchvision.utils import draw_bounding_boxes

    category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
                     "truck", "train", "other person", "bus", "car", "rider", "motorcycle",
                     "bicycle", "trailer"]


    # Defining hyperparameters:
    hparams = {
        'num_epochs': 100,
        'batch_size': 64,
        'channels': 3,
        'learning_rate': 0.0001,
        'classes': len(category_list),
        'nsamples': 25000,
    }
    use_gpu = True
    data = \
        DataLoader(
            img_files_path=imgs_p,
            target_files_path=jsons_p,
            category_list=category_list,
            split_size=7, # Amount of grid cells
            batch_size=hparams['batch_size'],
            load_size=1
        )
    yolo = YoloV1Model(hparams['channels'],classes=hparams['classes'])
    #optimizer = torch.optim.SGD(params=yolo.parameters(), lr=hparams['learning_rate'], momentum=0.9, weight_decay=0.0005)
    # Test Adam optimizer to enhance convergence
    optimizer = torch.optim.Adam(params=yolo.parameters(), lr=hparams['learning_rate'], weight_decay=0.0005)
    

    # Move model to the GPU
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    print(device)
    loss_fn = YoloLoss(C=hparams['classes'], S=7)
    train_loss_avg = []
    yolo.train()

    for epoch in range(hparams['num_epochs']):

        print("DATA IS BEING LOADED FOR A NEW EPOCH")
        print("")
        data.LoadFiles()  # Resets the DataLoader for a new epoch

        while len(data.img_files) > 0:

            print("LOADING NEW BATCHES")
            print("Remaining files:" + str(len(data.img_files)))
            print("")
            data.LoadData()  # Loads new batches

            for batch_idx, (img_data, target_data) in enumerate(data.data):
                if batch_idx > hparams['nsamples']:
                    break
                optimizer.zero_grad()
                img_data = img_data.to(device)
                target_data = target_data.to(device)
                prediction = yolo(img_data)

                loss = loss_fn(prediction,target_data)
                train_loss_avg.append(loss.item())
                wandb.log({"loss": loss})

                loss.backward()
                torch.nn.utils.clip_grad_norm_(yolo.parameters(), 100.0)
                optimizer.step()

                print('Train Epoch: {} of {} [Batch: {}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch + 1, hparams["num_epochs"], batch_idx + 1, len(data.data),
                    (batch_idx + 1) / len(data.data) * 100., loss))
                print('')
                print("=> Saving checkpoint")
                print("")
                torch.save(yolo, 'YOLO_bdd100k.pt')

if __name__ == '__main__':
    wandb.init(project="SelfDriving-project", entity="helenamartin",config = {
        "learning_rate": 0.0001,
        "epochs": 100,
        "batch_size": 64,
        "optim": 'Adam'
    })
    args = get_args()
    jsons_p = args.json_path
    imgs_p = args.imgs
    train(jsons_p,imgs_p)
