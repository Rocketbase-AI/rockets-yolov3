import torch
import torch.nn as nn

# Different way to write a module

class ConvBlock0(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, use_padding: bool = True, batch_normalize: bool = True):
        super(ConvBlock0, self).__init__()
        
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

# >>> print(nn.Sequential(ConvBlock.ConvBlock0(64, 32, 3, batch_normalize=False), ConvBlock.ConvBlock0(32, 16, 3)))
# Sequential(
#   (0): ConvBlock0(
#     (block): Sequential(
#       (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (1): LeakyReLU(negative_slope=0.1)
#     )
#   )
#   (1): ConvBlock0(
#     (block): Sequential(
#       (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): LeakyReLU(negative_slope=0.1)
#     )
#   )
# )

class ConvBlock1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, use_padding: bool = True, batch_normalize: bool = True):
        super(ConvBlock1, self).__init__()
        
        # Define Padding
        # TODO Check that this is necessary
        padding = (kernel_size - 1) // 2 if use_padding else 0
        
        self.batch_normalize = batch_normalize

        # Create layers
        # TODO Check if batch_normalization is sometimes False

        c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_normalize)
        b = nn.BatchNorm2d(out_channels)
        a = nn.LeakyReLU(negative_slope=0.1)

        self.convBlock = nn.Sequential(c, b, a) if batch_normalize else nn.Sequential(c, a)

    def forward(self, x):
        return self.convBlock(x)

# >>> print(nn.Sequential(ConvBlock.ConvBlock1(64, 32, 3, batch_normalize=False), ConvBlock.ConvBlock1(32, 16, 3)))
# Sequential(
#   (0): ConvBlock1(
#     (convBlock): Sequential(
#       (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (1): LeakyReLU(negative_slope=0.1)
#     )
#   )
#   (1): ConvBlock1(
#     (convBlock): Sequential(
#       (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): LeakyReLU(negative_slope=0.1)
#     )
#   )
# )

class ConvBlock2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, use_padding: bool = True, batch_normalize: bool = True):
        super(ConvBlock2, self).__init__()
        
        # Define Padding
        # TODO Check that this is necessary
        padding = (kernel_size - 1) // 2 if use_padding else 0
        
        self.batch_normalize = batch_normalize

        # Create layers
        # TODO Check if batch_normalization is sometimes False
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_normalize)
        self.b = nn.BatchNorm2d(out_channels)
        self.a = nn.LeakyReLU(negative_slope=0.1)

        self.convBlock = nn.Sequential(self.c, self.b, self.a) if batch_normalize else nn.Sequential(self.c, self.a)

    def forward(self, x):
        return self.convBlock(x)

# >>> print(nn.Sequential(ConvBlock.ConvBlock2(64, 32, 3, batch_normalize=False), ConvBlock.ConvBlock2(32, 16, 3)))
# Sequential(
#   (0): ConvBlock2(
#     (c): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (b): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (a): LeakyReLU(negative_slope=0.1)
#     (convBlock): Sequential(
#       (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (1): LeakyReLU(negative_slope=0.1)
#     )
#   )
#   (1): ConvBlock2(
#     (c): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (b): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (a): LeakyReLU(negative_slope=0.1)
#     (convBlock): Sequential(
#       (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): LeakyReLU(negative_slope=0.1)
#     )
#   )
# )

class ConvBlock3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, use_padding: bool = True, batch_normalize: bool = True):
        super(ConvBlock3, self).__init__()
        
        # Define Padding
        # TODO Check that this is necessary
        padding = (kernel_size - 1) // 2 if use_padding else 0
        
        self.batch_normalize = batch_normalize

        # Create layers
        # TODO Check if batch_normalization is sometimes False
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_normalize)
        self.b = nn.BatchNorm2d(out_channels)
        self.a = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        out = self.c(x)
        out = self.b(out) if self.batch_normalize else out
        out = self.a(out)
        
        return out

# >>> print(nn.Sequential(ConvBlock.ConvBlock3(64, 32, 3, batch_normalize=False), ConvBlock.ConvBlock3(32, 16, 3)))
# Sequential(
#   (0): ConvBlock3(
#     (c): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (b): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (a): LeakyReLU(negative_slope=0.1)
#   )
#   (1): ConvBlock3(
#     (c): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (b): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (a): LeakyReLU(negative_slope=0.1)
#   )
# )