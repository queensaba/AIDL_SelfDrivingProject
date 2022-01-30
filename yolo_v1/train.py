#Training yolo v1
import torch
from yolov1 import YoloV1Model
from utils import YoloLoss
from datasets import *
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt


#Defining hyperparameters:
hparams = {
    'num_epochs':5,
    'batch_size':1,
    'channels':3,
    'learning_rate':2e-5
}
use_gpu = False

yolo = YoloV1Model(hparams['channels'])
optimizer = torch.optim.SGD(params = yolo.parameters(), lr = hparams['learning_rate'], momentum = 1)

# Move model to the GPU
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
print(device)
yolo = yolo.to(device)

# This is the number of parameters used in the model
num_params = sum(p.numel() for p in yolo.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

# set to training mode
yolo.train()
train_loss_avg = []

# path to your own data and coco file
train_data_dir = 'data/coco/train'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

# create own Dataset
train_dataset = myOwnDataset(root=train_data_dir,transform=transform)

# collate_fn needs for batch
def collate_fn_custom(batch):
    images = list()
    label_matrix = list()
    
    for b in batch:
        images.append(b[0])
        label_matrix.append(b[1])
        
 
    images = torch.stack(images, dim=0)

    return images, label_matrix

# own DataLoader
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=hparams['batch_size'],shuffle=True,num_workers=0)

loss_fn = YoloLoss()

print('Training ...')
for epoch in range(hparams['num_epochs']):
    train_loss_avg = []
    num_batches = 0
    print("Started to train")
    
    for img, annotations in train_dataloader:
        print("Reading first image and annotation")
        
        img = img.to(device)
        annotations = annotations.to(device)
        
        prediction = yolo(img)
        loss = loss_fn(prediction, annotations)
        train_loss_avg.append(loss.item())

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
       
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
       
        num_batches += 1
    print(f"Mean loss was {sum(train_loss_avg)/len(train_loss_avg)}")    
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, hparams['num_epochs'], train_loss_avg[-1]))

plt.ion()

fig = plt.figure()
plt.plot(train_loss_avg)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()