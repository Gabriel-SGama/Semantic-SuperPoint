"""This is the main training interface using heatmap trick

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import cv2 as cv
import numpy as np
import torch

# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

# from tqdm import tqdm
# from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update
from utils.loss_functions.history_directions import GradHistory

# from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened
# from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch

# from utils.utils import save_checkpoint

from pathlib import Path
from Train_model_frontend_all_ang import Train_model_frontend_all_ang


def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


class Train_model_heatmap_all_ang(Train_model_frontend_all_ang):
    """Wrapper around pytorch net to help with pre and post image processing."""

    """
    * SuperPointFrontend_torch:
    ** note: the input, output is different from that of SuperPointFrontend
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    """
    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
        "data": {"gaussian_label": {"enable": False}},
    }

    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        # config
        # Update config
        print("Load Train_model_heatmap!!")

        self.config = self.default_config
        self.config = dict_update(self.config, config)

        # change iter variables to match mimic of larger batch size
        self.r = self.config["model"]["real_batch_size"] // self.config["model"]["batch_size"]
        self.config["train_iter"] *= self.r
        self.config["validation_interval"] *= self.r
        self.config["tensorboard_interval"] *= self.r
        self.config["save_interval"] *= self.r

        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 8
        self.subpixel = False

        self.real_batch_size = self.config["model"]["real_batch_size"]
        self.max_iter = self.config["train_iter"]

        self.gaussian = False
        if self.config["data"]["gaussian_label"]["enable"]:
            self.gaussian = True

        if self.config["model"]["dense_loss"]["enable"]:
            print("use dense_loss!")
            from utils.utils import descriptor_loss

            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            # loss without lambda D
            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        self.tasks = ["semi", "desc"]
        if self.config["data"]["semantic"]:
            self.tasks.append("sem")

        self.loss_fn = {}
        self.loss_fn["semi"] = self.calculate_detector_loss
        self.loss_fn["desc"] = self.calculate_desc_loss
        self.loss_fn["sem"] = self.calculate_sem_loss

        self.alpha = 0.3

        self.directions = GradHistory(self.tasks, self.alpha)
        # self.directions_desc = GradHistory(["desc_p", "desc_n"], self.alpha)

        # load model
        # self.net = self.loadModel(*config['model'])
        self.printImportantConfig()
        pass

    def detector_loss(self, input, target, mask=None, loss_type="softmax"):
        """
        # apply loss on detectors, default is softmax
        :param input: prediction
            tensor [batch_size, 65, Hc, Wc]
        :param target: constructed from labels
            tensor [batch_size, 65, Hc, Wc]
        :param mask: valid region in an image
            tensor [batch_size, 1, Hc, Wc]
        :param loss_type:
            str (l2 or softmax)
            softmax is used in original paper
        :return: normalized loss
            tensor
        """
        if loss_type == "l2":
            loss_func = nn.MSELoss(reduction="mean")
            loss = loss_func(input, target)
        elif loss_type == "softmax":
            loss_func_BCE = nn.BCELoss(reduction="none").cuda()
            loss = loss_func_BCE(nn.functional.softmax(input, dim=1), target)
            loss = (loss.sum(dim=1) * mask).sum()
            # loss = loss / (mask.sum() + 1e-10)
            loss = loss / (mask.sum() + 1e-5)
        return loss

    def sem_loss(self, pred, label, device="cpu"):
        """
        ## deprecated: loss function
        :param pred:
        :param label:
        :param device:
        :return:
        """
        # TODO: ignore index in config
        loss_func = nn.CrossEntropyLoss(ignore_index=133).to(device)
        loss = loss_func(pred, label.to(device))

        return loss

    # TODO: mIoU with ignore index
    def mIoU(self, pred, label, device="cpu"):
        smooth = 0.01
        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        intersection = np.sum(np.abs(label[:, 1:134] * pred))
        union = np.sum(label[:, 1:134]) + np.sum(pred) - intersection
        iou = np.mean((intersection + smooth) / (union + smooth))

        return iou

    def run_net(self, img):
        """
        # key function
        :param sample:
        :return:
            net outputs
        """

        outs = {}
        outs["enc"], x_hw = self.model["enc"](img)
        rep_variable = Variable(outs["enc"].data.clone(), requires_grad=True)

        for t in self.tasks:
            outs[t] = self.model[t](rep_variable, x_hw)

        return outs, rep_variable

    def calculate_detector_loss(self, outs, if_warp, sample):
        from utils.utils import labels2Dto3D

        det_loss_type = self.config["model"]["detector_loss"]["loss_type"]

        mask_2D = sample["valid_mask"]
        mask_warp_2D = sample["warped_valid_mask"]

        if self.gaussian:
            labels_2D = sample["labels_2D_gaussian"]
            if if_warp:
                warped_labels = sample["warped_labels_gaussian"]
        else:
            labels_2D = sample["labels_2D"]
            if if_warp:
                warped_labels = sample["warped_labels"]

        add_dustbin = False
        if det_loss_type == "l2":
            add_dustbin = False
        elif det_loss_type == "softmax":
            add_dustbin = True

        # warp
        if if_warp:
            labels_3D = labels2Dto3D(
                warped_labels.to(self.device),
                cell_size=self.cell_size,
                add_dustbin=add_dustbin,
            ).float()
            mask_3D_flattened = self.getMasks(mask_warp_2D, self.cell_size, device=self.device)
            loss_det = self.detector_loss(
                input=outs["semi"],
                target=labels_3D.to(self.device),
                mask=mask_3D_flattened,
                loss_type=det_loss_type,
            )

        else:
            labels_3D = labels2Dto3D(labels_2D.to(self.device), cell_size=self.cell_size, add_dustbin=add_dustbin).float()
            mask_3D_flattened = self.getMasks(mask_2D, self.cell_size, device=self.device)
            loss_det = self.detector_loss(
                input=outs["semi"],
                target=labels_3D.to(self.device),
                mask=mask_3D_flattened,
                loss_type=det_loss_type,
            )

        return loss_det

    def calculate_sem_loss(self, outs, if_warp, sample):
        semantic = self.config["data"]["semantic"]

        if not semantic:
            return torch.tensor([0]).float().to(self.device)

        sem = sample["semantic"] if if_warp else sample["warped_sem"]
        sem_pred = outs["sem"]

        loss_sem = self.sem_loss(sem_pred, sem, self.device)

        return loss_sem

    def calculate_desc_loss(self, outs, outs_warp, if_warp, sample):

        mask_2D = sample["valid_mask"]
        mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        mask_3D_flattened = self.getMasks(mask_2D, self.cell_size, device=self.device)
        mask_desc = mask_3D_flattened.unsqueeze(1)
        lambda_loss = self.config["model"]["lambda_loss"]

        coarse_desc = outs["desc"]
        coarse_desc_warp = outs_warp["desc"]

        # descriptor loss
        if lambda_loss > 0:
            assert if_warp == True, "need a pair of images"
            loss_desc, mask, positive_dist, negative_dist = self.descriptor_loss(
                coarse_desc, coarse_desc_warp, mat_H, mask_valid=mask_desc, device=self.device, **self.desc_params
            )
        else:
            ze = torch.tensor([0]).to(self.device)
            loss_desc, positive_dist, negative_dist = ze, ze, ze

        return loss_desc, positive_dist, negative_dist

    def train_val_sample(self, sample, n_iter=0, train=False):
        """
        # key function
        :param sample:
        :param n_iter:
        :param train:
        :return:
        """
        to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor)

        task = "train" if train else "val"
        tb_interval = self.config["tensorboard_interval"]
        if_warp = self.config["data"]["warped_pair"]["enable"]

        self.scalar_dict, self.images_dict, self.hist_dict = {}, {}, {}
        ## get the inputs
        # logging.info('get input img and label')

        # zero the parameter gradients

        img, labels_2D, mask_2D = (
            sample["image"],
            sample["labels_2D"],
            sample["valid_mask"],
        )
        # img, labels = img.to(self.device), labels_2D.to(self.device)

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        det_loss_type = self.config["model"]["detector_loss"]["loss_type"]
        # print("batch_size: ", batch_size)
        Hc = H // self.cell_size
        Wc = W // self.cell_size

        # warped images
        # img_warp, labels_warp_2D, mask_warp_2D = sample['warped_img'].to(self.device), \
        #     sample['warped_labels'].to(self.device), \
        #     sample['warped_valid_mask'].to(self.device)
        if if_warp:
            img_warp, labels_warp_2D, mask_warp_2D = (
                sample["warped_img"],
                sample["warped_labels"],
                sample["warped_valid_mask"],
            )

        # homographies
        # mat_H, mat_H_inv = \
        # sample['homographies'].to(self.device), sample['inv_homographies'].to(self.device)
        if if_warp:
            mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        img = img.to(self.device)
        img_warp = img_warp.to(self.device)

        var_img = Variable(img)
        var_img_warp = Variable(img_warp)

        outs = {}
        outs_warp = {}
        losses = {}
        loss_data = {}
        loss_data_desc = {}
        grads = {}
        grads_desc = {}

        # rep variable inside for loop

        if train:
            outs["enc"], x_hw = self.model["enc"](var_img)
            # rep_var = Variable(outs["enc"].data.clone(), requires_grad=True)

            outs_warp["enc"], x_hw_warp = self.model["enc"](var_img_warp)
            # rep_var_warp = Variable(outs_warp["enc"].data.clone(), requires_grad=True)

            for t in self.tasks:
                outs[t] = self.model[t](outs["enc"], x_hw)
                outs_warp[t] = self.model[t](outs_warp["enc"], x_hw_warp)

                if t == "desc":  # desc loss is different | TODO: add desc loss to loop?
                    continue

                loss_t = self.loss_fn[t](outs, False, sample)
                loss_t_warp = self.loss_fn[t](outs_warp, True, sample)
                losses[t] = loss_t
                losses[t + "_warp"] = loss_t_warp

                loss_data[t] = (loss_t + loss_t_warp).item()

                grads[t] = torch.autograd.grad(loss_t + loss_t_warp, self.model["enc"].parameters(), retain_graph=True)
                (loss_t + loss_t_warp).backward(retain_graph=True)

                self.optimizers_dict[t].step()
                self.optimizers_dict[t].zero_grad()
                self.scheduler[t].step()

            loss_desc, positive_dist, negative_dist = self.loss_fn["desc"](outs, outs_warp, if_warp, sample)
            losses["desc"] = loss_desc

            loss_data["desc"] = loss_desc.item()

            grads["desc"] = torch.autograd.grad(loss_desc, self.model["enc"].parameters(), retain_graph=True)

            # loss_data_desc["desc_p"] = positive_dist.item()
            # loss_data_desc["desc_n"] = negative_dist.item()

            # grads_desc["desc_p"] = torch.autograd.grad(positive_dist, self.model["desc"].parameters(), retain_graph=True)
            # grads_desc["desc_n"] = torch.autograd.grad(negative_dist, self.model["desc"].parameters())

            # common_dir_desc = self.directions_desc.descent_direction(grads_desc, loss_data_desc) #common dir to desc head

            # learning_rate_desc = self.scheduler["desc"].get_lr()[0]  # same learning for all layers

            # for i_par, parameter in enumerate(self.model["desc"].parameters()):
            #     parameter.data = parameter.data - learning_rate_desc * common_dir_desc[i_par].data

            # for parameter in self.model["desc"].parameters():  # equivalent to optimizer.zero_grad()
            #     if parameter.grad is not None:
            #         parameter.grad.data.zero_()

            loss_desc.backward()  # calculates grads for both images

            self.optimizers_dict["desc"].step()
            self.optimizers_dict["desc"].zero_grad()
            self.scheduler["desc"].step()

            # get outputs
            semi, coarse_desc = outs["semi"], outs["desc"]
            if if_warp:
                semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]

            common_dir = self.directions.descent_direction(grads, loss_data)

            # TODO: scheduler for dynamic learning rate
            learning_rate = self.scheduler["semi"].get_lr()[0]  # same learning for all layers
            # print(self.model["enc"].parameters())
            for i_par, parameter in enumerate(self.model["enc"].parameters()):
                parameter.data = parameter.data - learning_rate * common_dir[i_par].data

            for parameter in self.model["enc"].parameters():  # equivalent to optimizer.zero_grad()
                if parameter.grad is not None:
                    parameter.grad.data.zero_()

            # self.scheduler["enc"].step()

        else:
            with torch.no_grad():
                outs, rep_var = self.run_net(var_img)
                semi, coarse_desc = outs["semi"], outs["desc"]
                sem_pred = outs["sem"] if self.config["data"]["semantic"] else None
                if if_warp:
                    outs_warp, rep_var_warp = self.run_net(var_img_warp)
                    semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]
                    sem_warp_pred = outs_warp["sem"] if self.config["data"]["semantic"] else None

            losses["semi"] = self.calculate_detector_loss(outs, False, sample)
            losses["semi_warp"] = self.calculate_detector_loss(outs_warp, True, sample)

            losses["sem"] = self.calculate_sem_loss(outs, False, sample)
            losses["sem_warp"] = self.calculate_sem_loss(outs_warp, True, sample)
            losses["desc"], positive_dist, negative_dist = self.calculate_desc_loss(outs, outs_warp, if_warp, sample)

        loss = torch.tensor([0]).float().to(self.device)
        for t in losses:
            self.scalar_dict.update({"loss_" + t: losses[t]})
            loss = loss + losses[t]

        self.loss = loss

        self.scalar_dict.update(
            {
                "loss": loss,
                # "loss_det": losses["semi"],
                # "loss_desc": losses["desc"],
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )

        if self.config["data"]["semantic"]:
            self.scalar_dict.update(
                {
                    "loss_sem": losses["sem"],
                }
            )
            # self.images_dict["sem_pred"] = sem_pred.detach().cpu()
            # self.images_dict["warp_sem_pred"] = sem_warp_pred.detach().cpu()

        # TODO: add semantic label to imgs
        # self.input_to_imgDict(sample, self.images_dict)

        if n_iter % tb_interval == 0 or task == "val":
            logging.info("current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval)

            # add clean map to tensorboard
            ## semi_warp: flatten, to_numpy

            heatmap_org = self.get_heatmap(semi, det_loss_type)  # tensor []
            heatmap_org_nms_batch = self.heatmap_to_nms(self.images_dict, heatmap_org, name="heatmap_org")
            if if_warp:
                heatmap_warp = self.get_heatmap(semi_warp, det_loss_type)
                heatmap_warp_nms_batch = self.heatmap_to_nms(self.images_dict, heatmap_warp, name="heatmap_warp")

            def update_overlap(images_dict, labels_warp_2D, heatmap_nms_batch, img_warp, name):
                # image overlap
                from utils.draw import img_overlap

                # result_overlap = img_overlap(img_r, img_g, img_gray)
                # overlap label, nms, img
                nms_overlap = [
                    img_overlap(
                        toNumpy(labels_warp_2D[i]),
                        heatmap_nms_batch[i],
                        toNumpy(img_warp[i]),
                    )
                    for i in range(heatmap_nms_batch.shape[0])
                ]
                nms_overlap = np.stack(nms_overlap, axis=0)
                images_dict.update({name + "_nms_overlap": nms_overlap})

            from utils.var_dim import toNumpy

            update_overlap(
                self.images_dict,
                labels_2D,
                heatmap_org_nms_batch[np.newaxis, ...],
                img,
                "original",
            )

            update_overlap(
                self.images_dict,
                labels_2D,
                toNumpy(heatmap_org),
                img,
                "original_heatmap",
            )
            if if_warp:
                update_overlap(
                    self.images_dict,
                    labels_warp_2D,
                    heatmap_warp_nms_batch[np.newaxis, ...],
                    img_warp,
                    "warped",
                )
                update_overlap(
                    self.images_dict,
                    labels_warp_2D,
                    toNumpy(heatmap_warp),
                    img_warp,
                    "warped_heatmap",
                )
            # residuals
            from utils.losses import do_log

            if self.gaussian:
                # original: gt
                self.get_residual_loss(
                    sample["labels_2D"],
                    sample["labels_2D_gaussian"],
                    sample["labels_res"],
                    name="original_gt",
                )
                if if_warp:
                    # warped: gt
                    self.get_residual_loss(
                        sample["warped_labels"],
                        sample["warped_labels_gaussian"],
                        sample["warped_res"],
                        name="warped_gt",
                    )

            pr_mean = self.batch_precision_recall(
                to_floatTensor(heatmap_org_nms_batch[:, np.newaxis, ...]),
                sample["labels_2D"],
            )
            print("pr_mean")
            self.scalar_dict.update(pr_mean)

            self.printLosses(self.scalar_dict, task)
            # self.tb_images_dict(task, self.images_dict, max_img=2)
            self.tb_hist_dict(task, self.hist_dict)

        self.tb_scalar_dict(self.scalar_dict, task)

        return loss.item()

    def heatmap_to_nms(self, images_dict, heatmap, name):
        """
        return:
            heatmap_nms_batch: np [batch, H, W]
        """
        from utils.var_dim import toNumpy

        heatmap_np = toNumpy(heatmap)
        ## heatmap_nms
        heatmap_nms_batch = [self.heatmap_nms(h) for h in heatmap_np]  # [batch, H, W]
        heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
        # images_dict.update({name + '_nms_batch': heatmap_nms_batch})
        images_dict.update({name + "_nms_batch": heatmap_nms_batch[:, np.newaxis, ...]})
        return heatmap_nms_batch

    def get_residual_loss(self, labels_2D, heatmap, labels_res, name=""):
        if abs(labels_2D).sum() == 0:
            return
        outs_res = self.pred_soft_argmax(labels_2D, heatmap, labels_res, patch_size=5, device=self.device)
        self.hist_dict[name + "_resi_loss_x"] = outs_res["loss"][:, 0]
        self.hist_dict[name + "_resi_loss_y"] = outs_res["loss"][:, 1]
        err = abs(outs_res["loss"]).mean(dim=0)
        # print("err[0]: ", err[0])
        var = abs(outs_res["loss"]).std(dim=0)
        self.scalar_dict[name + "_resi_loss_x"] = err[0]
        self.scalar_dict[name + "_resi_loss_y"] = err[1]
        self.scalar_dict[name + "_resi_var_x"] = var[0]
        self.scalar_dict[name + "_resi_var_y"] = var[1]
        self.images_dict[name + "_patches"] = outs_res["patches"]
        return outs_res

    # tb_images_dict.update({'image': sample['image'], 'valid_mask': sample['valid_mask'],
    #     'labels_2D': sample['labels_2D'], 'warped_img': sample['warped_img'],
    #     'warped_valid_mask': sample['warped_valid_mask']})
    # if self.gaussian:
    #     tb_images_dict.update({'labels_2D_gaussian': sample['labels_2D_gaussian'],
    #     'labels_2D_gaussian': sample['labels_2D_gaussian']})

    ######## static methods ########
    @staticmethod
    def batch_precision_recall(batch_pred, batch_labels):
        precision_recall_list = []
        for i in range(batch_labels.shape[0]):
            precision_recall = precisionRecall_torch(batch_pred[i], batch_labels[i])
            precision_recall_list.append(precision_recall)
        precision = np.mean([precision_recall["precision"] for precision_recall in precision_recall_list])
        recall = np.mean([precision_recall["recall"] for precision_recall in precision_recall_list])
        return {"precision": precision, "recall": recall}

    @staticmethod
    def pred_soft_argmax(labels_2D, heatmap, labels_res, patch_size=5, device="cuda"):
        """

        return:
            dict {'loss': mean of difference btw pred and res}
        """
        from utils.losses import norm_patches

        outs = {}
        # extract patches
        from utils.losses import extract_patches
        from utils.losses import soft_argmax_2d

        label_idx = labels_2D[...].nonzero().long()

        # patch_size = self.config['params']['patch_size']
        patches = extract_patches(label_idx.to(device), heatmap.to(device), patch_size=patch_size)
        # norm patches
        patches = norm_patches(patches)

        # predict offsets
        from utils.losses import do_log

        patches_log = do_log(patches)
        # soft_argmax
        dxdy = soft_argmax_2d(patches_log, normalized_coordinates=False)  # tensor [B, N, patch, patch]
        dxdy = dxdy.squeeze(1)  # tensor [N, 2]
        dxdy = dxdy - patch_size // 2

        # extract residual
        def ext_from_points(labels_res, points):
            """
            input:
                labels_res: tensor [batch, channel, H, W]
                points: tensor [N, 4(pos0(batch), pos1(0), pos2(H), pos3(W) )]
            return:
                tensor [N, channel]
            """
            labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
            points_res = labels_res[points[:, 0], points[:, 1], points[:, 2], points[:, 3], :]  # tensor [N, 2]
            return points_res

        points_res = ext_from_points(labels_res, label_idx)

        # loss
        outs["pred"] = dxdy
        outs["points_res"] = points_res
        # ls = lambda x, y: dxdy.cpu() - points_res.cpu()
        # outs['loss'] = dxdy.cpu() - points_res.cpu()
        outs["loss"] = dxdy.to(device) - points_res.to(device)
        outs["patches"] = patches
        return outs

    @staticmethod
    def flatten_64to1(semi, cell_size=8):
        """
        input:
            semi: tensor[batch, cell_size*cell_size, Hc, Wc]
            (Hc = H/8)
        outpus:
            heatmap: tensor[batch, 1, H, W]
        """
        from utils.d2s import DepthToSpace

        depth2space = DepthToSpace(cell_size)
        heatmap = depth2space(semi)
        return heatmap

    @staticmethod
    def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):
        """
        input:
            heatmap: np [(1), H, W]
        """
        from utils.utils import getPtsFromHeatmap

        # nms_dist = self.config['model']['nms']
        # conf_thresh = self.config['model']['detection_threshold']
        heatmap = heatmap.squeeze()
        # print("heatmap: ", heatmap.shape)
        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)
        semi_thd_nms_sample = np.zeros_like(heatmap)
        semi_thd_nms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1
        return semi_thd_nms_sample


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == "l2":
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == "loss":
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == "loss+":
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == "none":
        for t in grads:
            gn[t] = 1.0
    else:
        print("ERROR: Invalid Normalization Type")
    return gn


if __name__ == "__main__":
    # load config
    # filename = "configs/superpoint_coco_train_heatmap.yaml"
    filename = "configs/superpoint_coco_train_wsem_heatmap.yaml"

    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_tensor_type(torch.FloatTensor)
    with open(filename, "r") as f:
        config = yaml.load(f)

    from utils.loader import dataLoader as dataLoader

    # data = dataLoader(config, dataset='hpatches')
    task = config["data"]["dataset"]

    data = dataLoader(config, dataset=task, warp_input=True)
    # test_set, test_loader = data['test_set'], data['test_loader']
    train_loader, val_loader = data["train_loader"], data["val_loader"]

    # model_fe = Train_model_frontend(config)
    # print('==> Successfully loaded pre-trained network.')

    train_agent = Train_model_heatmap(config, device=device)

    train_agent.train_loader = train_loader
    # train_agent.val_loader = val_loader

    train_agent.loadModel()
    train_agent.dataParallel()
    train_agent.train()

    # epoch += 1
    try:
        model_fe.train()

    # catch exception
    except KeyboardInterrupt:
        logging.info("ctrl + c is pressed. save model")
    # is_best = True
