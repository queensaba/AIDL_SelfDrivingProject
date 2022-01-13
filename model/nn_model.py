import torch.nn as nn

class SD_Model():
    def __init__():
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64,64,3)
        self.act2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64,128,3)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128,128,3)
        self.act4 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128,256,3)
        self.act5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256,256,3)
        self.act6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256,256,3)
        self.act7 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2d(2)

        self.conv8=nn.Conv2d(256,512,3)
        self.act8 = nn.ReLU()
        self.conv9= nn.Conv2d(512,512,3)
        self.act9= nn.ReLU()
        self.conv10 = nn.Conv2d(512,512,3)
        self.act10 = nn.ReLU()

        self.maxpool4 = nn.MaxPool2d(2)

        self.conv11 = nn.Conv2d(512,512,3)
        self.act11 = nn.ReLU()
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.act12 = nn.ReLU()
        self.conv13 = nn.Conv2d(512,512,3)
        self.act13 = nn.ReLU()

        self.maxpool5 = nn.MaxPool2d(2)

        self.linear1 = nn.Linear(25088,4096)
        self.act14 = nn.ReLU()
        self.linear2 = nn.Linear(4096,4096)
        self.act15 = nn.ReLU()
        self.linear3 = nn.Linear(4092,10)

    def forward(x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.maxpool1(x)
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.maxpool2(x)
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.act7(self.conv7(x))
        x = self.maxpool3(x)
        x = self.act8(self.conv8(x))
        x = self.act9(self.conv9(x))
        x = self.act10(self.conv10(x))
        x = self.maxpool4(x)
        x = self.act11(self.conv11(x))
        x = self.act12(self.conv12(x))
        x = self.act13(self.conv13(x))
        x = self.maxpool5(x)
        x = self.act14(self.linear1(x))
        x = self.act15(self.linear2(x))
        y = self.act16(self.linear3(x))
        return y





        
