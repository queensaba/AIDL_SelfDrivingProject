import torch.nn as nn
'''
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
        return y'''

import torch.nn as nn
import torch.nn.functional as F


class YoloV1Model(nn.Module):
    def __init__(self, channels=3, classes=80, bb=2, s=7):
        super(YoloV1Model, self).__init__()
        self.c = channels
        self.S = s
        self.classes = classes
        self.bb = bb  # Number of bounding boxes per cell
        # 448x448 input
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                               padding=3)  # 448x448x3 ->224x224x64
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)  # 224x224x64 ->112x112x64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3,
                               padding=1)  # 112x112x64 -> 112x112x192 (192 = 3*64)
        self.batchnorm2 = nn.BatchNorm2d(192)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)  # 112x112x192 -> 56x56x192
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)  # 56x56x192 -> 56x56x128
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # 56x56x128 -> 56x56x256
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)  # 56x56x256 -> 56x56x256
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # 56x56x256 -> 56x56x512
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)  # 56x56x512 -> 28x28x512
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)  # 28x28x512 -> 28x28x256
        self.batchnorm7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # 28x28x256 -> 28x28x512
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)  # 28x28x512 -> 28x28x256
        self.batchnorm9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # 28x28x256 -> 28x28x512
        self.batchnorm10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)  # 28x28x512 -> 28x28x256
        self.batchnorm11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # 28x28x256 -> 28x28x512
        self.batchnorm12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)  # 28x28x512 -> 28x28x256
        self.batchnorm13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # 28x28x256 -> 28x28x512
        self.batchnorm14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)  # 28x28x512 -> 28x28x512
        self.batchnorm15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)  # 28x28x512 -> 28x28x1024
        self.batchnorm16 = nn.BatchNorm2d(1024)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)  # 28x28x1024 -> 14x14x1024
        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)  # 14x14x1024 -> 14x14x512
        self.batchnorm17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,
                                padding=1)  # 14x14x512 -> 14x14x1024
        self.batchnorm18 = nn.BatchNorm2d(1024)
        self.conv19 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)  # 14x14x1024 -> 14x14x512
        self.batchnorm19 = nn.BatchNorm2d(512)
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,
                                padding=1)  # 14x14x512 -> 14x14x1024
        self.batchnorm20 = nn.BatchNorm2d(1024)
        # self.avgpool = nn.AvgPool2d(kernel_size = , stride= , padding= , ceil_mode= , count_include_pad= , divisor_override= )
        # self.fc0 = nn.Linear(in_features = , out_features = )
        # Pretrain until this layer with ImageNet 1000-class classification. 224x224
        self.conv21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3,
                                padding=1)  # 14x14x1024 -> 14x14x1024
        self.batchnorm21 = nn.BatchNorm2d(1024)
        self.conv22 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2,
                                padding=1)  # 14x14x1024 -> 7x7x1024
        self.batchnorm22 = nn.BatchNorm2d(1024)
        self.conv23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)  # 7x7x1024 -> 7x7x1024
        self.batchnorm23 = nn.BatchNorm2d(1024)
        self.conv24 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)  # 7x7x1024 -> 7x7x1024
        self.batchnorm24 = nn.BatchNorm2d(1024)
        self.flatten = nn.Flatten(),
        self.fc1 = nn.Linear(in_features=1024 * 7 * 7, out_features=496)  # 7x7x1024 -> 4096
        self.fc2 = nn.Linear(in_features=496,
                             out_features=self.S * self.S * (self.bb * 5 + self.classes))  # 4096 -> 7x7x30 = 1470
        self.dropout = nn.Dropout(0.0)

        # Resolution is bumped to 448x448

        self.leakyrelu = nn.LeakyReLU(0.1)  # Leaky RELU Activation

    def forward(self, x):
        out1 = self.maxpool1(self.leakyrelu(self.batchnorm1(self.conv1(x))))  # conv1
        out2 = self.maxpool2(self.leakyrelu(self.batchnorm2(self.conv2(out1))))  # conv2
        out3 = self.maxpool3(self.leakyrelu(self.batchnorm6(self.conv6(self.leakyrelu(self.batchnorm5(self.conv5(
            self.leakyrelu(self.batchnorm4(
                self.conv4(self.leakyrelu(self.batchnorm3(self.conv3(out2)))))))))))))  # From conv3 to conv6
        out4 = self.leakyrelu(self.batchnorm11(self.conv11(self.leakyrelu(self.batchnorm10(self.conv10(self.leakyrelu(
            self.batchnorm9(self.conv9(self.leakyrelu(self.batchnorm8(
                self.conv8(self.leakyrelu(self.batchnorm7(self.conv7(out3)))))))))))))))  # From conv7 to conv11
        out5 = self.maxpool4(self.leakyrelu(self.batchnorm16(self.conv16(self.leakyrelu(self.batchnorm15(self.conv15(
            self.leakyrelu(self.batchnorm14(self.conv14(self.leakyrelu(self.batchnorm13(
                self.conv13(self.leakyrelu(self.batchnorm12(self.conv12(out4))))))))))))))))  # From conv12 to conv16
        out6 = self.leakyrelu(self.batchnorm20(self.conv20(self.leakyrelu(self.batchnorm19(self.conv19(self.leakyrelu(
            self.batchnorm18(
                self.conv18(self.leakyrelu(self.batchnorm17(self.conv17(out5))))))))))))  # From conv17 to conv20
        out7 = self.leakyrelu(self.batchnorm24(self.conv24(self.leakyrelu(self.batchnorm23(self.conv23(self.leakyrelu(
            self.batchnorm22(
                self.conv22(self.leakyrelu(self.batchnorm21(self.conv21(out6))))))))))))  # From conv21 to conv24
        bsz, nch, height, width = out7.shape
        out7 = out7.view(bsz, nch * height * width)
        out = self.fc2(self.leakyrelu(self.dropout(self.fc1(out7))))
        # out = self.fc2(self.fc1(out7))
        return out


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2,
                 C=80):  # S is the number of gris in which we are going to divide (7x7), B is the quantity of boundig box per cell, C is the number of classes
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S,
                                          self.C + self.B * 5)  # Make sure that the shape is (-1,7,7,80+10) = (-1,7,7,90)
        iou_b1 = intersection_over_union(predictions[..., 81:85], target[...,
                                                                  81:85])  # From 0 to 79 is for class probabilities, 80 i for class score
        iou_b2 = intersection_over_union(predictions[..., 86:90], target[..., 81:85])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
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

        box_targets = exists_box * target[..., 81:85]

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
            torch.flatten(exists_box * target[..., 80:81]),
        )
        print("object loss is")
        print(object_loss)

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        # no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        # )

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
            torch.flatten(exists_box * predictions[..., :80], end_dim=-2, ),
            torch.flatten(exists_box * target[..., :80], end_dim=-2, ),
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





        
