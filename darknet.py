import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, use_padding: bool = True, batch_normalize: bool = True):
        super(ConvBlock, self).__init__()
        
        # Define Padding
        # TODO Check that this is necessary
        padding = (kernel_size - 1) // 2 if use_padding else 0

        # Create layers
        # TODO Check if batch_normalization is sometimes False
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_normalize)]
        
        if batch_normalize:
            layers += [nn.BatchNorm2d(out_channels)]
       
        layers += [nn.LeakyReLU(negative_slope=0.1)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResNetBlock(nn.Module):
    """
    ResNet Block
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, \
                    kernel_size_1: int = 1, kernel_size_2: int = 3, stride: int = 1, \
                    use_padding: bool = True, batch_normalize: bool = True):
        
        super(ResNetBlock, self).__init__()
        layers = [
            ConvBlock(in_channels, mid_channels, kernel_size_1, stride = stride, use_padding = use_padding, batch_normalize = batch_normalize),
            ConvBlock(mid_channels, out_channels, kernel_size_2, stride = stride, use_padding = use_padding, batch_normalize = batch_normalize)
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x) + x

class YOLOBlock(nn.Module):
    """
    YOLO Block
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, \
                    anchors: int, num_classes: int, img_size: int,
                    kernel_size_1: int = 1, kernel_size_2: int = 3, stride: int = 1, \
                    use_padding: bool = True, batch_normalize: bool = True):
        
        super(YOLOBlock, self).__init__()
        
        self.anchors = anchors
        self.num_anchors = len(self.anchors)
        self.img_size = img_size
        self.num_classes = num_classes
        self.bbox_attrs = 5 + self.num_classes

        detection_out = self.num_anchors * (5 + self.num_classes)

        layers = [
            ConvBlock(in_channels, mid_channels, kernel_size_1, stride = stride, use_padding = use_padding, batch_normalize = batch_normalize),
            ConvBlock(mid_channels, out_channels, kernel_size_2, stride = stride, use_padding = use_padding, batch_normalize = batch_normalize),
            nn.Conv2d(out_channels, detection_out, kernel_size = 1)
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        out = self.block(x)

        nA = self.num_anchors
        nB = out.size(0)
        nG = out.size(2)
        stride = self.img_size / nG

        prediction = out.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )

        return output

class Darknet53(nn.Module):
    """ Darknet53 block

    Also return the 36th and 61th 's output.
    Counting the shortcut as one layer.

    """
    def __init__(self, img_size: int = 416, num_channels: int = 3):
        super(Darknet53, self).__init__()
        
        # 1st Convolutional Layer
        layers = [ConvBlock(num_channels, 32, 3)]

        # Downsample
        layers += [ConvBlock(32, 64, 3, stride=2)]

        # 1st Block of ResNetBlocks
        layers += [ResNetBlock(64, 32, 64)]

        # Downsample
        layers += [ConvBlock(64, 128, 3, stride=2)]

        # 2nd Block of ResNetBlocks
        for i in range(2):
            layers += [ResNetBlock(128, 64, 128)]

        # Downsample
        layers += [ConvBlock(128, 256, 3, stride=2)]

        # 31 is the first layer convolution with the last downsample layer

        # 3rd Block of ResNetBlocks
        for i in range(8):
            layers += [ResNetBlock(256, 128, 256)]
        
        self.block_0to36 = nn.Sequential(*layers)

        # Downsample
        layers = [ConvBlock(256, 512, 3, stride=2)]

        # 4th Block of ResNetBlocks
        for i in range(8):
            layers += [ResNetBlock(512, 256, 512)]
        
        self.block_37to61 = nn.Sequential(*layers)

        # Downsample
        layers = [ConvBlock(512, 1024, 3, stride=2)]

        # 5th Block of ResNetBlocks
        for i in range(4):
            layers += [ResNetBlock(1024, 512, 1024)]

        self.block_62to74 = nn.Sequential(*layers) 


        # self.module_defs = parse_model_config(config_path)
        # self.hyperparams, self.module_list = create_modules(self.module_defs)
        # self.img_size = img_size
        # self.seen = 0
        # self.header_info = np.array([0, 0, 0, self.seen, 0])
        # self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x):
        # Rewrite Darknet 53 to also output the output of the layer 36 and 61
        out_36 = self.block_0to36(x)
        out_61 = self.block_37to61(out_36)
        out_74 = self.block_62to74(out_61)
        
        return out_36, out_61, out_74

class YOLOv3(nn.Module):
    def __init__(self, img_size: int = 416, num_anchors: int = 9, num_classes: int = 80):
        super(YOLOv3, self).__init__()
        
        self.anchors = [[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]]

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
        self.yoloBlock1 = YOLOBlock(1024, 512, 1024, self.anchors[:3], num_classes, img_size)

        # Upsample Block 1
        layers = [
            ConvBlock(1024, 256, 1),
            nn.Upsample(scale_factor=2, mode="nearest")
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
        self.yoloBlock2 = YOLOBlock(512, 256, 512, self.anchors[3:6], num_classes, img_size)

        # Upsample Block 2
        layers = [
            ConvBlock(512, 128, 1),
            nn.Upsample(scale_factor=2, mode="nearest")
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
        self.yoloBlock3 = YOLOBlock(256, 128, 256, self.anchors[6:], num_classes, img_size)


    def forward(self, x):
        out_36, out_61, out_74 = self.darknet53(x)
        out_78 = self.block_75to78(out_74)

        out_yolo1 = self.yoloBlock1(out_78)

        out_up1 = self.upSampleBlock1(out_78)
        out_concat1 = torch.cat([out_up1, out_61], 1)
        print("out_concat1:", out_concat1.shape)
        out_90 = self.block_87to90(out_concat1)
        print("out_90:", out_90.shape)

        out_yolo2 = self.yoloBlock2(out_90)
        print("out_yolo2:", out_yolo2.shape)

        out_up2 = self.upSampleBlock2(out_90)
        out_concat2 = torch.cat([out_up2, out_36], 1)
        print("out_concat2:", out_concat2.shape)
        out_102 = self.block_99to102(out_concat2)

        out_yolo3 = self.yoloBlock3(out_102)
        
        # Concatenate the output of the model before returning it
        out = torch.cat([out_yolo1, out_yolo2, out_yolo3], 1)
        
        return out

