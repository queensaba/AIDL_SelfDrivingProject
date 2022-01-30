from datasets import *
import torch

def main():
    # path to your own data and coco file
    train_data_dir = 'data/coco/train'

    # create own Dataset
    my_dataset = myOwnDataset(root=train_data_dir,transforms=get_transform())

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # own DataLoader
    train_dataloader = torch.utils.data.DataLoader(my_dataset,batch_size=100,shuffle=True,num_workers=0,collate_fn=collate_fn)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("DEVICE DEFINED!")

    # DataLoader is iterable over Dataset
    for imgs, annotations in train_dataloader:
        print("jeje")
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        print(annotations)

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
