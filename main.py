# Main file of the project

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/Users/helenamartin/Desktop/AI-Project/"


class SDDataLoader(json_path):
    """
    Self Driving Prokect Dataset.
    """
    self.dataset_annotations = self.annotations_to_csv(json_path)
    self.dataset = self.generate_dict(self.dataset_annotations, DATA_DIR)
    self.train,self.val = self.split_dataset(self.dataset_annotations)

    def annotations_to_csv(self, json_path):
        import pandas as pd
        csv_data = pd.read_json(json_path)
        return csv_data

    def generate_dict(self, dataset_Annotations, DATA_DIR):
        from PIL import Image
        from torchivison import transforms
        import glob
        images = glob.glob(os.path.join(DATA_DIR, '*.jpg'))
        for image in images:
            convert_tensor = transforms.ToTensor()
            images[os.path.basename(image)] = convert_tensor(img)
            labels[os.path.basename(image)] =



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
            y_ = ...
            loss = ...
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

