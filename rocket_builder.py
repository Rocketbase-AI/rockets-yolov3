from __future__ import division
import json
import numpy as np
import os
import types

from PIL import Image
from PIL import ImageDraw

import torch
import torch.nn as nn
import torchvision.transforms

from . import model
from . import utils




def build(config_path: str = '') -> nn.Module:
    """Builds a pytorch compatible deep learning model

    The model can be used as any other pytorch model. Additional methods
    for `preprocessing`, `postprocessing`, `label_to_class` have been added to ease handling of the model
    and simplify interchangeability of different models.
    """
    # Load Config file
    if not config_path: # If no config path then load default one
        config_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "config.json")

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load the classes
    classes_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), config['classes_path'])
    
    with open(classes_path, 'r') as f:
        classes =  json.load(f)

    # Set up model
    rocket = model.YOLOv3(config['input_size'], config['anchors'], classes)
    weights_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), config['weights_path'])
    rocket.load_state_dict(torch.load(weights_path), strict=True)

    
    rocket.postprocess = types.MethodType(postprocess, rocket)
    rocket.preprocess = types.MethodType(preprocess, rocket)
    rocket.label_to_class = types.MethodType(label_to_class, rocket)
    rocket.train_forward = types.MethodType(train_forward, rocket)
    setattr(rocket, 'classes', classes)

    return rocket


def label_to_class(self, label: int) -> str:
    """Returns string of class name given index
    """
    return self.classes[str(label)]


def train_forward(self, x: torch.Tensor, targets: torch.Tensor):
    """Performs forward pass and returns loss of the model

    The loss can be directly fed into an optimizer.
    """
    self.forward(x, targets)
    loss = self.loss
    self.loss = None
    return loss


def preprocess(self, img: Image, labels: list = None) -> torch.Tensor:
    """Converts PIL Image or Array into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.
    Labels must have the following format: `x1, y1, x2, y2, category_id`

    Args:
        img (PIL.Image): input image
        labels (list): list of bounding boxes and class labels
    """

    # todo: support batch size bigger than 1 for training and inference
    # todo: replace this hacky solution and work directly with tensors
    if type(img) == Image.Image:
        # PIL.Image
        # Extract image
        img = np.array(img)
    elif type(img) == torch.Tensor:
        # list of tensors
        img = img[0].cpu()
        img = torchvision.transforms.ToPILImage()(img)
        img = np.array(img)
    elif "PIL" in str(type(img)): # type if file just has been opened
        img = np.array(img.convert("RGB"))
    else:
        raise TypeError("wrong input type: got {} but expected list of PIL.Image, "
                        "single PIL.Image or torch.Tensor".format(type(img)))

    h, w, c = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
    padded_h, padded_w, _ = input_img.shape

    # Resize and normalize
    input_img = Image.fromarray(np.uint8(input_img*255), 'RGB')
    input_img.thumbnail((416, 416), resample=Image.BICUBIC)
    input_img = np.array(input_img)
    input_img = input_img / 255.0

    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float()

    # check if labels is empty --> we don't train but can return image tensor here
    if labels is None:
        return input_img.unsqueeze(0)

    max_objects = 50
    filled_labels = np.zeros((max_objects, 5))  # max objects in an image for training=50, 5=(x1,y1,x2,y2,category_id)
    if labels is not None:
        for idx, label in enumerate(labels):

            # add padding
            label[0] += pad[1][0]
            label[1] += pad[0][0]

            # resize coordinates to match Yolov3 input size
            scale_x = 416.0 / padded_w
            scale_y = 416.0 / padded_h

            label[0] *= scale_x
            label[1] *= scale_y
            label[2] *= scale_x
            label[3] *= scale_y

            x1 = label[0] / 416.0
            y1 = label[1] / 416.0

            cw = (label[2]) / 416.0
            ch = (label[3]) / 416.0

            cx = (x1 + (x1 + cw)) / 2.0
            cy = (y1 + (y1 + ch)) / 2.0

            filled_labels[idx] = np.asarray([label[4], cx, cy, cw, ch])
            if idx >= max_objects:
                break
    filled_labels = torch.from_numpy(filled_labels)

    return input_img.unsqueeze(0), filled_labels.unsqueeze(0)

def postprocess(self, detections: torch.Tensor, input_img: Image, visualize: bool = False):
    """Converts pytorch tensor into interpretable format

    Handles all the steps for postprocessing of the raw output of the model.
    Depending on the rocket family there might be additional options.
    This model supports either outputting a list of bounding boxes of the format
    (x0, y0, w, h) or outputting a `PIL.Image` with the bounding boxes
    and (class name, class confidence, object confidence) indicated.

    Args:
        detections (Tensor): Output Tensor to postprocess
        input_img (PIL.Image): Original input image which has not been preprocessed yet
        visualize (bool): If True outputs image with annotations else a list of bounding boxes
    """
    img = np.array(input_img)
    img_height, img_width, _ =  img.shape

    detections = utils.non_max_suppression(detections.clone().detach(), 80)[0]

    # In case no detection is made on the image
    if detections is None:
        detections = []

    # The amount of padding that was added
    pad_x = max(img_height - img_width, 0) * (416 / max(img.shape))
    pad_y = max(img_width - img_height, 0) * (416 / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = 416 - pad_y
    unpad_w = 416 - pad_x

    list_detections = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls_conf, cls_pred = detection.data.cpu().numpy()
        # Rescale coordinates to original dimensions
        x1 = ((x1 - pad_x // 2) / unpad_w) * img_width
        y1 = ((y1 - pad_y // 2) / unpad_h) * img_height

        x2 = ((x2 - pad_x // 2) / unpad_w) * img_width
        y2 = ((y2 - pad_y // 2) / unpad_h) * img_height
        
        # Standardize the output
        topLeft_x = int(utils.clamp(round(x1), 0, img_width))
        topLeft_y = int(utils.clamp(round(y1), 0, img_height))

        bottomRight_x = int(utils.clamp(round(x2), 0, img_width))
        bottomRight_y = int(utils.clamp(round(y2), 0, img_height))

        width = abs(bottomRight_x - topLeft_x) + 1
        height = abs(bottomRight_y - topLeft_y) + 1

        bbox_confidence = utils.clamp(conf, 0, 1)

        class_name = str(self.label_to_class(int(cls_pred)))
        class_confidence = utils.clamp(cls_conf, 0, 1)

        list_detections.append({
                'topLeft_x': topLeft_x,
                'topLeft_y': topLeft_y,
                'width': width,
                'height': height,
                'bbox_confidence': bbox_confidence,
                'class_name': class_name,
                'class_confidence': class_confidence})

    if visualize:
        line_width = 2
        img_out = input_img.copy()
        ctx = ImageDraw.Draw(img_out, 'RGBA')
        for detection in list_detections:
            # Extract information from the detection
            topLeft = (detection['topLeft_x'], detection['topLeft_y'])
            bottomRight = (detection['topLeft_x'] + detection['width'] - line_width, detection['topLeft_y'] + detection['height']- line_width)
            class_name = detection['class_name']
            bbox_confidence = detection['bbox_confidence']
            class_confidence = detection['class_confidence']

            # Draw the bounding boxes and the information related to it
            ctx.rectangle([topLeft, bottomRight], outline=(255, 0, 0, 255), width=line_width)
            ctx.text((topLeft[0] + 5, topLeft[1] + 10), text="{}, {:.2f}, {:.2f}".format(class_name, bbox_confidence, class_confidence))

        del ctx
        return img_out

    return list_detections