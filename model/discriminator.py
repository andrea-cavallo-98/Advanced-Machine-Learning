import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
    # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x


class Lightweight_FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(Lightweight_FCDiscriminator, self).__init__()

        self.depth_conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, groups=num_classes)
        self.point_conv1 = nn.Conv2d(num_classes, ndf, kernel_size=1)

        self.depth_conv2 = nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, groups=ndf)
        self.point_conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=1)

        self.depth_conv3 = nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride=2, padding=1, groups=ndf * 2)
        self.point_conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1)

        self.depth_conv4 = nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1, groups=ndf * 4)
        self.point_conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=1)

        self.depth_classifier = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, groups=ndf * 8)
        self.point_classifier = nn.Conv2d(ndf * 8, 1, kernel_size=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
    # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.depth_conv1(x)
        x = self.point_conv1(x)
        x = self.leaky_relu(x)
        x = self.depth_conv2(x)
        x = self.point_conv2(x)
        x = self.leaky_relu(x)
        x = self.depth_conv3(x)
        x = self.point_conv3(x)
        x = self.leaky_relu(x)
        x = self.depth_conv4(x)
        x = self.point_conv4(x)
        x = self.leaky_relu(x)
        x = self.depth_classifier(x)
        x = self.point_classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x