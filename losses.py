import torch
import torch.nn as nn

class YOLO_Loss(nn.Module):
    
    def __init__(self, anchors, num_classes, img_dim):
        super(YOLO_Loss, self).__init__()
        # TODO anchors need to be the full list of anchor boxes
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim

        # TODO put them in the condig file
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

        
    def forward(self, x, targets):
        """
        Args:
            x and targets are of the shape: [[x1, y1, w, h, bbox.conf, classes.conf...]]
        """

        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

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


        if x.is_cuda:
            self.mse_loss = self.mse_loss.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.ce_loss = self.ce_loss.cuda()

        nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
            pred_boxes=pred_boxes.cpu().data,
            pred_conf=pred_conf.cpu().data,
            pred_cls=pred_cls.cpu().data,
            target=targets.cpu().data,
            anchors=scaled_anchors.cpu().data,
            num_anchors=nA,
            num_classes=self.num_classes,
            grid_size=nG,
            ignore_thres=self.ignore_thres,
            img_dim=self.image_dim,
        )

        nProposals = int((pred_conf > 0.5).sum().item())
        recall = float(nCorrect / nGT) if nGT else 1
        if nProposals == 0:
            precision=0.0
        else:
            precision = float(nCorrect / nProposals)

        # Handle masks
        mask = Variable(mask.type(ByteTensor))
        conf_mask = Variable(conf_mask.type(ByteTensor))

        # Handle target variables
        tx = Variable(tx.type(FloatTensor), requires_grad=False)
        ty = Variable(ty.type(FloatTensor), requires_grad=False)
        tw = Variable(tw.type(FloatTensor), requires_grad=False)
        th = Variable(th.type(FloatTensor), requires_grad=False)
        tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
        tcls = Variable(tcls.type(LongTensor), requires_grad=False)

        # Get conf mask where gt and where there is no gt
        conf_mask_true = mask
        conf_mask_false = conf_mask - mask

        # Mask outputs to ignore non-existing objects
        loss_x = self.mse_loss(x[mask], tx[mask])
        loss_y = self.mse_loss(y[mask], ty[mask])
        loss_w = self.mse_loss(w[mask], tw[mask])
        loss_h = self.mse_loss(h[mask], th[mask])
        loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
            pred_conf[conf_mask_true], tconf[conf_mask_true]
        )
        loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        return (
            loss,
            loss_x.item(),
            loss_y.item(),
            loss_w.item(),
            loss_h.item(),
            loss_conf.item(),
            loss_cls.item(),
            recall,
            precision,
        )