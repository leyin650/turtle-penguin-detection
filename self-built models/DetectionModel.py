import torch.nn as nn
import torch

class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        self.DownSampling = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        output = self.DownSampling(x)
        return output


class Conv_2(nn.Module):
    def __init__(self, channel):
        super(Conv_2, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1,padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1,padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x):
        output = self.Conv(x)
        return output


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.D1_Module = DownSample(3, 32)  # 128
        self.C1 = Conv_2(32)

        self.D2_Module = DownSample(64, 64)  # 64
        self.C2 = Conv_2(64)

        self.D3_Module = DownSample(128, 128)  # 32
        self.C3 = Conv_2(128)

        self.D4_Module = DownSample(256, 256)  # 16
        self.C4 = Conv_2(256)

        self.D5_Module = DownSample(512, 512)  # 8
        self.C5 = Conv_2(512)

        self.D6_Module = DownSample(1024, 1024)  # 4
        self.C6 = Conv_2(1024)
        self.C7 = Conv_2(1024)


        self.MaxPool=nn.MaxPool2d(kernel_size=4, stride=4)
        self.Con_1x1=nn.Conv2d(2048, 4, kernel_size=(1, 1), padding=0, stride=1)

    def forward(self, inputs):
        D1 = self.D1_Module(inputs)  # 256->128
        C1 = self.C1(D1)
        S1 = self.Concat(C1, D1)

        D2 = self.D2_Module(S1)  # 128->64
        C2 = self.C2(D2)
        S2 = self.Concat(C2, C2)

        D3 = self.D3_Module(S2)  # 64->32
        C3= self.C3(D3)
        S3 = self.Concat(C3, D3)

        D4 = self.D4_Module(S3)  # 32->16
        C4= self.C4(D4)
        S4 = self.Concat(C4, D4)

        D5 = self.D5_Module(S4)  # 16->8
        C5= self.C5(D5)
        S5 = self.Concat(C5, D5)

        D6 = self.D6_Module(S5)  # 8->4
        C6 = self.C6(D6)
        C7 = self.C7(C6)

        S6 = self.Concat(C7, D6) #4x4x2048

        M=self.MaxPool(S6)#1x1x2048

        # 1x1convolutional layer. 把1x1x2048变成1x1x4
        output = self.Con_1x1(M)
        output = torch.flatten(output, 1)

        return output

    def Concat(self, x, y):
        concatenated_tensor = torch.concat([x, y], dim=1)
        return concatenated_tensor

