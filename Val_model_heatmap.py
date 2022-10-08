"""This is the main validation interface using heatmap trick

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""


import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update

from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened

from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
from utils.utils import save_checkpoint

from pathlib import Path
from models.model_wrap import SuperPointFrontend_torch
import models.senner_models


@torch.no_grad()
class Val_model_heatmap(SuperPointFrontend_torch):
    def __init__(self, config, device="cpu", verbose=False):
        self.config = config
        self.model = self.config["name"]
        self.params = self.config["params"]
        self.weights_path = self.config["pretrained"]
        self.device = device

        ## other parameters

        # self.name = 'SuperPoint'
        # self.cuda = cuda
        self.nms_dist = self.config["nms"]
        self.conf_thresh = self.config["detection_threshold"]
        self.nn_thresh = self.config["nn_thresh"]  # L2 descriptor distance for good match.
        self.cell = 8  # deprecated
        self.cell_size = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.sparsemap = None
        self.heatmap = None  # np[batch, 1, H, W]
        self.pts = None
        self.pts_subpixel = None
        ## new variables
        self.pts_nms_batch = None
        self.desc_sparse_batch = None
        self.patches = None
        pass

    def loadModel(self):
        # model = 'SuperPointNet'
        # params = self.config['model']['subpixel']['params']
        from utils.loader import modelLoader

        self.net = modelLoader(model=self.model, **self.params)
        self.net.eval()

        checkpoint = torch.load(self.weights_path, map_location=lambda storage, loc: storage)
        mode = "" if self.weights_path[-4:] == ".pth" else "full"  # the suffix is '.pth' or 'tar.gz'
        try:  # normal model
            if mode == "full":
                self.net.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.net.load_state_dict(checkpoint)

        # TODO: generalize function to multiple sener models
        except Exception as e:  # sener like model
            logging.info("Failed to load as normal model: " + str(e))
            logging.info("trying to load as senner like model...")
            # checkpoint = torch.load(
            #     "logs/superpoint_coco_2017_ang_pret/checkpoints/superPointNet_148_checkpoint.pth.tar",
            #     map_location=lambda storage, loc: storage,
            # )
            logging.info("Transfering weights from sener like model to: %s", self.weights_path)

            sener_model = models.senner_models.get_senner_model(self.config, self.device, False)
            self.net.to(self.device)

            for t in sener_model:
                sener_model[t].load_state_dict(checkpoint["model_" + t])
                self.net.load_state_dict(sener_model[t].state_dict(), strict=False)

            # for t in sener_model:
            #     print(t)
            #     for name, parameter in sener_model[t].named_parameters():
            #         name_split = name.split(".")[0]
            #         state_dict = getattr(sener_model[t], name_split).state_dict()
            #         getattr(self.net, name_split).load_state_dict(state_dict)

        self.net = self.net.to(self.device)
        logging.info("successfully load pretrained model from: %s", self.weights_path)
        pass

    def extract_patches(self, label_idx, img):
        """
        input:
            label_idx: tensor [N, 4]: (batch, 0, y, x)
            img: tensor [batch, channel(1), H, W]
        """
        from utils.losses import extract_patches

        patch_size = self.config["params"]["patch_size"]
        patches = extract_patches(label_idx.to(self.device), img.to(self.device), patch_size=patch_size)
        return patches
        pass

    def run(self, images):
        """
        input:
            images: tensor[batch(1), 1, H, W]

        """
        from Train_model_heatmap_all import Train_model_heatmap_all
        from utils.var_dim import toNumpy

        train_agent = Train_model_heatmap_all

        with torch.no_grad():
            outs = self.net(images)
        semi = outs["semi"]
        self.outs = outs

        channel = semi.shape[1]
        if channel == 64:
            heatmap = train_agent.flatten_64to1(semi, cell_size=self.cell_size)
        elif channel == 65:
            heatmap = flattenDetection(semi, tensor=True)

        heatmap_np = toNumpy(heatmap)
        self.heatmap = heatmap_np
        return self.heatmap
        pass

    def heatmap_to_pts(self):
        heatmap_np = self.heatmap

        pts_nms_batch = [self.getPtsFromHeatmap(h) for h in heatmap_np]  # [batch, H, W]
        self.pts_nms_batch = pts_nms_batch
        return pts_nms_batch

    # def soft_argmax_points(self):
    #     """
    #     # make sure you have points ahead
    #     inputs:

    #     """
    #     # from utils.losses import extract_patches
    #     from utils.losses import extract_patch_from_points

    #     ##### check not take care of batch #####
    #     print("not take care of batch! only take first element!")
    #     pts = self.pts_nms_batch
    #     pts = pts[0].transpose().copy()
    #     patches = extract_patch_from_points(self.heatmap, pts, patch_size=5)
    #     import torch
    #     patches = np.stack(patches)
    #     patches_torch = torch.tensor(patches, dtype=torch.float32).unsqueeze(0)
    #     print("patches: ", patches_torch.shape)
    #     print("pts: ", pts.shape)

    #     dxdy = soft_argmax_2d(patches_torch)
    #     print("dxdy: ", dxdy.shape)
    #     points = pts
    #     points[:,:2] += dxdy.numpy().squeeze()
    #     self.pts_subpixel = [points.transpose().copy()]
    #     return self.pts_subpixel.copy()
    #     pass

    def desc_to_sparseDesc(self):
        # pts_nms_batch = [self.getPtsFromHeatmap(h) for h in heatmap_np]
        desc_sparse_batch = [self.sample_desc_from_points(self.outs["desc"], pts) for pts in self.pts_nms_batch]
        self.desc_sparse_batch = desc_sparse_batch
        return desc_sparse_batch


if __name__ == "__main__":
    # filename = 'configs/magicpoint_shapes_subpix.yaml'
    filename = "configs/magicpoint_repeatability_heatmap.yaml"
    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_tensor_type(torch.FloatTensor)
    with open(filename, "r") as f:
        config = yaml.load(f)

    task = config["data"]["dataset"]
    # data loading
    from utils.loader import dataLoader_test as dataLoader

    data = dataLoader(config, dataset="hpatches")
    test_set, test_loader = data["test_set"], data["test_loader"]

    # load frontend
    val_agent = Val_model_heatmap(config["model"], device=device)

    # take one sample
    for i, sample in tqdm(enumerate(test_loader)):
        if i > 1:
            break

        val_agent.loadModel()
        # points from heatmap
        img = sample["image"]
        print("image: ", img.shape)

        heatmap_batch = val_agent.run(img.to(device))  # heatmap: numpy [batch, 1, H, W]
        # heatmap to pts
        pts = val_agent.heatmap_to_pts()
        # print("pts: ", pts)
        print("pts[0]: ", pts[0].shape)
        print("pts: ", pts[0][:, :3])

        pts_subpixel = val_agent.soft_argmax_points(pts)
        print("subpixels: ", pts_subpixel[0][:, :3])

        # heatmap, pts to desc
        desc_sparse = val_agent.desc_to_sparseDesc()
        print("desc_sparse[0]: ", desc_sparse[0].shape)

# pts, desc, _, heatmap
