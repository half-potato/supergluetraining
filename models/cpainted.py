import cv2
import os
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
third_party = Path(__file__).parent / '../third_party'
sys.path.extend([str(third_party), str(third_party / 'cpaint')])
import SuperGluePretrainedNetwork.models.superpoint as SP
from cpaint.models import subpixel, MODELS, sample_descriptors
from cpaint.core.image_ops import pad_to_multiple, unpad_multiple, get_padding_for_multiple
from cpaint.core.detector_util import mask_border, mask_max
import matplotlib.pyplot as plt


dii_filter = torch.tensor([
    [1., -2., 1.],
    [2., -4., 2.],
    [1., -2., 1.]]).view(1, 1, 3, 3).float().cuda()

djj_filter = torch.tensor([
    [1., 2., 1.],
    [-2., -4., -2.],
    [1., 2., 1.]]).view(1, 1, 3, 3).float().cuda()

class CPainted(nn.Module):
    default_conf = {
        "threshold": 0.0,
        "maxpool_radius": 3,
        "remove_borders": 4,
        "max_keypoints": 400,
        "subpixel": True,
        'superpoint_desc': True,
        "checkpoint": "fifth_sota_cpainted.pth",
        'model': 'superpoint_bn'
    }
    def __init__(self, config={}):
        super(CPainted, self).__init__()
        self.config = {**self.default_conf, **config}
        model_conf = MODELS[self.config['model']]
        self.net = model_conf['net']()
        self.input_size_multiple = model_conf['input_size_multiple']

        # Load
        path = third_party / 'cpaint/checkpoints/' / self.config["checkpoint"]
        ckpt = torch.load(path)
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        # Convert
        state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict)

        self.desc_net = SP.SuperPoint(self.config)
        self.net.eval()
        self.desc_net.eval()

    def desc_forward(self, data):
        # Shared Encoder
        x = self.desc_net.relu(self.desc_net.conv1a(data['image']))
        x = self.desc_net.relu(self.desc_net.conv1b(x))
        x = self.desc_net.pool(x)
        x = self.desc_net.relu(self.desc_net.conv2a(x))
        x = self.desc_net.relu(self.desc_net.conv2b(x))
        x = self.desc_net.pool(x)
        x = self.desc_net.relu(self.desc_net.conv3a(x))
        x = self.desc_net.relu(self.desc_net.conv3b(x))
        x = self.desc_net.pool(x)
        x = self.desc_net.relu(self.desc_net.conv4a(x))
        x = self.desc_net.relu(self.desc_net.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.desc_net.relu(self.desc_net.convPa(x))
        scores = self.desc_net.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]

        # Compute the dense descriptors
        cDa = self.desc_net.relu(self.desc_net.convDa(x))
        descriptors = self.desc_net.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        return scores, descriptors


    def forward(self, data):
        x = data["image"]

        # Resize image such that it is a multiple of the cell size
        old_size = x.shape
        B, C, H, W = old_size

        x = pad_to_multiple(x, self.input_size_multiple)
        # padding: lp, rp, tp, bp
        padding = get_padding_for_multiple(old_size, self.input_size_multiple)

        # Run model
        result = self.net.forward(x)
        heatmap = result["heatmap"]
        heatmap = unpad_multiple(heatmap, old_size, self.input_size_multiple)
        #  g_heatmap = score_gaussian_peaks(heatmap)

        # remove border, apply nms + threshold
        # Shape: (3, N)
        mask1 = mask_border(heatmap, border=self.config["remove_borders"]).unsqueeze(1)
#         mask2 = mask_max(heatmap, radius=self.config["maxpool_radius"])

        mask2 = mask_max(heatmap, radius=self.config["maxpool_radius"]).unsqueeze(1)

        pooled = mask1 * mask2 * heatmap
        _, descriptors = self.desc_forward(data)

        # torch where over batch
        pts = []
        scores = []
        sampled = []
        for i in range(B):
            # row col
            y, x = torch.where(pooled[i].squeeze() > self.config["threshold"])
            if len(y) > self.config["max_keypoints"]:
                threshold, _ = torch.sort(pooled[i].flatten(), descending=True)
                threshold = threshold[self.config["max_keypoints"]]
                y, x = torch.where(pooled[i].squeeze() > threshold)
            if len(y) < 10:
                # Just extract something I guess
                threshold, _ = torch.sort(pooled[i].flatten(), descending=True)
                threshold = threshold[10]
                y, x = torch.where(pooled[i].squeeze() > threshold)
            l_pts = torch.stack((y, x), dim=1)
            l_scores = heatmap[i].squeeze()[l_pts[:, 0], l_pts[:, 1]]
            # localize to the subpixel
            if self.config["subpixel"]:
                l_pts = subpixel.localize(heatmap[i], l_pts, 1)[:, :2]
            else:
                l_pts += 0.5
            flipped = torch.flip(l_pts, [1]).float()

            #  desc = result["raw_desc"]
            #  D = desc.shape[1]
            #  l_sampled = sample_descriptors(
            #          desc[i].unsqueeze(0), H, W, l_pts.view(1, -1, 2), padding).squeeze(0).T
            if self.config['superpoint_desc']:
                l_sampled = SP.sample_descriptors(flipped.view(1, -1, 2), descriptors[i][None], 8)[0]
                sampled.append(l_sampled) # (256, N)

            pts.append(flipped) # (N, 2)
            scores.append(l_scores) # (N)

            if False:
                img = (data["image"][i].view(H, W, 1).cpu().numpy()*255).astype(np.uint8)
                # compute orientation
                size = 4
                # convert to keypoints
                kpts = []
                for pt in flipped.cpu():
                    kpts.append(cv2.KeyPoint(float(pt[0]), float(pt[1]), _size=size, _angle=0))

                print(len(kpts))
                drawn = img.copy()
                drawn = cv2.drawKeypoints(img, kpts, drawn, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                plt.figure()
                plt.imshow(heatmap.squeeze().cpu())
                plt.figure()
                plt.imshow(drawn)
                plt.show()

        return {
            'keypoints': pts,
            'scores': scores,
            'descriptors': sampled,
            "heatmap": heatmap
        }
