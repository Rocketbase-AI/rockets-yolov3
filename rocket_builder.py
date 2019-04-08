import os
from .models import Darknet
import types
import torch.nn as nn
from PIL import Image
from PIL import ImageDraw
from skimage.transform import resize
from torchvision import transforms
from .utils.utils import *


def build() -> nn.Module:
    """Builds a pytorch compatible deep learning model

    The model can be used as any other pytorch model. Additional methods
    for `preprocessing`, `postprocessing`, `label_to_class` have been added to ease handling of the model
    and simplify interchangeability of different models.
    """
    # Set up model
    model = Darknet(os.path.join(os.path.realpath(os.path.dirname(__file__)), "yolov3.cfg"))
    model.load_weights(os.path.join(os.path.realpath(os.path.dirname(__file__)), "weights.pth"))

    classes = load_classes(os.path.join(os.path.realpath(os.path.dirname(__file__)), "coco.data"))

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)
    model.label_to_class = types.MethodType(label_to_class, model)
    model.train_forward = types.MethodType(train_forward, model)
    setattr(model, 'classes', classes)

    return model


def label_to_class(self, label: int) -> str:
    """Returns string of class name given index
    """
    return self.classes[label]


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
        img = transforms.ToPILImage()(img)
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
    input_img = resize(input_img, (416, 416, 3), mode='reflect')
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
    detections = non_max_suppression(detections.clone().detach(), 80)[0]

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (416 / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (416 / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = 416 - pad_y
    unpad_w = 416 - pad_x

    list_detections = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls_conf, cls_pred = detection.data.cpu().numpy()
        # Rescale coordinates to original dimensions
        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
        list_detections.append((x1, y1, box_h, box_w, conf, cls_conf, cls_pred))

    if visualize:
        img_out = input_img
        ctx = ImageDraw.Draw(img_out, 'RGBA')
        for bbox in list_detections:
            x1, y1, box_h, box_w, conf, cls_conf, cls_pred = bbox
            ctx.rectangle([(x1, y1), (x1 + box_w, y1 + box_h)], outline=(255, 0, 0, 255), width=2)
            ctx.text((x1 + 5, y1 + 10),
                     text="{}, {:.2f}, {:.2f}".format(self.label_to_class(int(cls_pred)), cls_conf, conf))
        del ctx
        return img_out

    return list_detections


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output
