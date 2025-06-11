import torch.nn as nn
import torch
import torch.nn.functional as F
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.D1=DownSampling(3,64)#112
        self.D2 = DownSampling(64, 128)#56
        self.D3 = DownSampling(128, 256)#28
        self.D4 = DownSampling(256, 512)#14
        self.D5 = DownSampling(512, 1024)#7

        self.B1 = BottleNet(1024)
        self.B2 = BottleNet(1024)
        self.B3 = BottleNet(1024)

        self.linear = nn.Linear(50176, 2)

        self.Dropout=nn.Dropout(0.3)


    def forward(self, inputs):
        D1 = self.D1(inputs)
        D2 = self.D2(D1)
        D3 = self.D3(D2)
        D4 = self.D4(D3)
        D5 = self.D5(D4)

        B1 = self.B1(D5)
        S1=self.ShotCut(D5,B1)

        B2 = self.B2(S1)
        S2 = self.ShotCut(B2, S1)

        B3 = self.B3(S2)
        S3 = self.ShotCut(B3, S2)


        output = S3.view(S3.size()[0], -1)

        output = self.linear(output)
        output=self.Dropout(output)

        return output


    def Concate(self, x,y):
        return torch.cat([x, y], 1)

    def ShotCut(self,x,y):
        return x+y



class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2)
        )
        # self.C = nn.Conv2d(in_channels, out_channels, 3, bias=False,padding=1)
        # self.B = nn.BatchNorm2d(out_channels)
        # self.R=nn.ReLU(inplace=True)
        # self.A = nn.AvgPool2d(2,stride=2)

    def forward(self, x):
        # return self.A(self.R(self.B(self.C(x))))
        return self.conv1(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.TC = nn.ConvTranspose2d(in_channels,out_channels,(2,2),stride=2)
        self.B = nn.BatchNorm2d(out_channels)
        self.R = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.R(self.B(self.TC(x)))


class BottleNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.bottle_neck(x)

