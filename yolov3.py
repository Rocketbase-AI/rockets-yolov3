import torch
import torch.nn as nn
from .layers import ConvBlock, Darknet53, YOLOBlock, UpSample

class YOLOv3(nn.Module):
    def __init__(self, img_size: int = 416, num_classes: int = 80, anchors: list = [[373, 326], [156, 198], [116, 90], [62, 45], [59, 119], [33, 23], [30, 61], [16, 30], [10, 13]]):
        super(YOLOv3, self).__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        self.anchors = anchors
        
        # Feature extractor Darknet53
        self.darknet53 = Darknet53(img_size)

        # 4 layers before first YOLOBlock
        layers = [
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3)
        ]

        self.block_75to78 = nn.Sequential(*layers)

        # YOLOBlock 1
        self.yoloBlock1 = YOLOBlock(1024, 512, 1024, self.anchors[:3], self.num_classes, self.img_size)

        # Upsample Block 1
        layers = [
            ConvBlock(512, 256, 1),
            UpSample(scale_factor=2, mode="nearest")
        ]

        self.upSampleBlock1 = nn.Sequential(*layers)

        # 4 layers before first YOLOBlock
        layers = [
            ConvBlock(768, 256, 1), # in_channels = sum(512 + 256)
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3)
        ]

        self.block_87to90 = nn.Sequential(*layers)

        # YOLOBlock 2
        self.yoloBlock2 = YOLOBlock(512, 256, 512, self.anchors[3:6], self.num_classes, self.img_size)

        # Upsample Block 2
        layers = [
            ConvBlock(256, 128, 1),
            UpSample(scale_factor=2, mode="nearest")
        ]

        self.upSampleBlock2 = nn.Sequential(*layers)

        # 4 layers before the 2nd YoloBlock
        layers = [
            ConvBlock(384, 128, 1), # in_channels = sum(256 + 128)
            ConvBlock(128, 256, 3),
            ConvBlock(256, 128, 1),
            ConvBlock(128, 256, 3)
        ]

        self.block_99to102 = nn.Sequential(*layers)

        # YOLOBlock 3
        self.yoloBlock3 = YOLOBlock(256, 128, 256, self.anchors[6:], self.num_classes, self.img_size)


    def forward(self, x):
        out_36, out_61, out_74 = self.darknet53(x)
        out_78 = self.block_75to78(out_74)

        out_79, out_yolo1 = self.yoloBlock1(out_78)

        out_up1 = self.upSampleBlock1(out_79)
        out_concat1 = torch.cat([out_up1, out_61], 1)
        out_90 = self.block_87to90(out_concat1)

        out_91, out_yolo2 = self.yoloBlock2(out_90)

        out_up2 = self.upSampleBlock2(out_91)
        out_concat2 = torch.cat([out_up2, out_36], 1)
        out_102 = self.block_99to102(out_concat2)

        _, out_yolo3 = self.yoloBlock3(out_102)
        
        # Concatenate the output of the model before returning it
        out = torch.cat([out_yolo1, out_yolo2, out_yolo3], 1)
        
        return out