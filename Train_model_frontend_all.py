"""This is the frontend interface for training
base class: inherited by other Train_model_*.py

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import numpy as np
import torch

# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch_poly_lr_decay import PolynomialLRDecay

from tqdm import tqdm
from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update

from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened

from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
from utils.utils import save_checkpoint
from utils import coco_labels

from pathlib import Path

import models.senner_models

# def thd_img(img, thd=0.015):
#     """
#     thresholding the image.
#     :param img:
#     :param thd:
#     :return:
#     """
#     img[img < thd] = 0
#     img[img >= thd] = 1
#     return img


# def toNumpy(tensor):
#     return tensor.detach().cpu().numpy()


# def img_overlap(img_r, img_g, img_gray):  # img_b repeat
#     img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
#     img[0, :, :] += img_r[0, :, :]
#     img[1, :, :] += img_g[0, :, :]
#     img[img > 1] = 1
#     img[img < 0] = 0
#     return img


class Train_model_frontend_all(object):
    """
    # This is the base class for training classes. Wrap pytorch net to help training process.

    """

    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
    }

    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        """
        ## default dimension:
            heatmap: torch (batch_size, H, W, 1)
            dense_desc: torch (batch_size, H, W, 256)
            pts: [batch_size, np (N, 3)]
            desc: [batch_size, np(256, N)]

        :param config:
            dense_loss, sparse_loss (default)

        :param save_path:
        :param device:
        :param verbose:
        """
        # config
        print("Load Train_model_frontend!!")
        self.config = self.default_config
        self.config = dict_update(self.config, config)

        # change iter variables to match mimic of larger batch size
        r = self.config["model"]["real_batch_size"] // self.config["model"]["batch_size"]
        self.config["train_iter"] *= r
        self.config["validation_interval"] *= r
        self.config["tensorboard_interval"] *= r
        self.config["save_interval"] *= r

        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 8
        self.subpixel = False
        self.loss = 0

        self.max_iter = self.config["train_iter"]

        if self.config["model"]["dense_loss"]["enable"]:
            ## original superpoint paper uses dense loss
            print("use dense_loss!")
            from utils.utils import descriptor_loss

            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            ## our sparse loss has similar performace, more efficient
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        if self.config["model"]["subpixel"]["enable"]:
            ## deprecated: only for testing subpixel prediction
            self.subpixel = True

            def get_func(path, name):
                logging.info("=> from %s import %s", path, name)
                mod = __import__("{}".format(path), fromlist=[""])
                return getattr(mod, name)

            self.subpixel_loss_func = get_func("utils.losses", self.config["model"]["subpixel"]["loss_func"])

        # load model
        # self.net = self.loadModel(*config['model'])

        self.printImportantConfig()

        pass

    def printImportantConfig(self):
        """
        # print important configs
        :return:
        """
        print("=" * 10, " check!!! ", "=" * 10)

        print("learning_rate: ", self.config["model"]["learning_rate"])
        print("lambda_loss: ", self.config["model"]["lambda_loss"])
        print("detection_threshold: ", self.config["model"]["detection_threshold"])
        print("real_batch_size: ", self.config["model"]["real_batch_size"])
        print("batch_size: ", self.config["model"]["batch_size"])
        if self.config["data"]["semantic"]:
            print("seg_loss: ", self.config["model"]["seg_head"]["loss"])

        print("=" * 10, " descriptor: ", self.desc_loss_type, "=" * 10)
        for item in list(self.desc_params):
            print(item, ": ", self.desc_params[item])

        print("=" * 32)
        pass

    def dataParallel(self):
        """
        put network and optimizer to multiple gpus
        :return:
        """
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        # self.net = nn.DataParallel(self.net)
        self.optimizer = self.adamOptim(self.net, lr=self.config["model"]["learning_rate"])
        self.optimizer.zero_grad()
        # self.seg_optimizer = self.SGDOptim(self.net, lr=self.config["model"]["learning_rate"])
        pass

    def adamOptim(self, net, lr):
        """
        initiate adam optimizer
        :param net: network structure
        :param lr: learning rate
        :return:
        """
        print("adam optimizer")
        import torch.optim as optim

        # optimizer = optim.Adam(
        #     list(net.parameters()) + list(self.multi_task_loss.parameters()), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4
        # )

        optimizer = optim.Adam(list(net.parameters()) + list(self.multi_task_loss.parameters()), lr=lr, betas=(0.9, 0.999))
        return optimizer

    # def SGDOptim(self, net, lr):
    #     """
    #     initiate SGD optimizer
    #     :param net: network structure
    #     :param lr: learning rate
    #     :return:
    #     """
    #     print("SGD Optimizer")
    #     import torch.optim as optim

    #     optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    #     return optimizer

    # TODO change checkpoint save
    def loadModel(self):
        """
        load model from name and params
        init or load optimizer
        :return:
        """
        model = self.config["model"]["name"]
        params = self.config["model"]["params"]
        print("model: ", model)
        net = modelLoader(model=model, **params).to(self.device)
        logging.info("=> setting adam solver")
        logging.info("=> setting SGD solver")
        optimizer = self.adamOptim(net, lr=self.config["model"]["learning_rate"])
        # TODO: config otimizers for SGD solver (maybe ?)
        # optimizer_sem = self.SGDOptim(net, lr=self.config["model"]["learning_rate"])

        n_iter = 0
        ## new model or load pretrained
        if self.config["retrain"] == True:
            logging.info("New model")
            pass
        else:
            try:
                path = self.config["pretrained"]
                mode = "" if path[-4:] == ".pth" else "full"  # the suffix is '.pth' or 'tar.gz'
                logging.info("load pretrained model from: %s", path)
                net, optimizer, n_iter = pretrainedLoader(net, optimizer, n_iter, path, mode=mode, full_path=True)
                logging.info("successfully load pretrained model from: %s", path)

            except:  # sener like model - TODO: model in config parameter
                print("loading pretrained model from senner")
                checkpoint = torch.load(
                    "logs/superpoint_coco_2017_ang_pret_v2/checkpoints/superPointNet_3_checkpoint.pth.tar",
                    map_location=lambda storage, loc: storage,
                )
                logging.info("Transfering weights from sener like model to: something")
                net = net.to(self.device)
                sener_model = models.senner_models.get_senner_model(self.config, self.device, False)

                for t in sener_model:
                    sener_model[t].load_state_dict(checkpoint["model_" + t])
                    net.load_state_dict(sener_model[t].state_dict(), strict=False)
                    # for name, parameter in sener_model[t].named_parameters():
                    #     print(name)
                    #     name_split = name.split(".")[0]
                    #     state_dict = getattr(sener_model[t], name_split).state_dict()
                    #     # getattr(net, name_split).load_state_dict(state_dict)

        def setIter(n_iter):
            if self.config["reset_iter"]:
                logging.info("reset iterations to 0")
                n_iter = 0
            return n_iter

        self.net = net
        self.optimizer = optimizer
        # self.optimizer_sem = optimizer_sem
        self.n_iter = setIter(n_iter)

        self.scheduler = PolynomialLRDecay(self.optimizer, max_decay_steps=self.max_iter, end_learning_rate=0.001, power=2.0)

        pass

    @property
    def writer(self):
        """
        # writer for tensorboard
        :return:
        """
        # print("get writer")
        return self._writer

    @writer.setter
    def writer(self, writer):
        print("set writer")
        self._writer = writer

    @property
    def train_loader(self):
        """
        loader for dataset, set from outside
        :return:
        """
        print("get dataloader")
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        print("set train loader")
        self._train_loader = loader

    @property
    def val_loader(self):
        print("get dataloader")
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader):
        print("set train loader")
        self._val_loader = loader

    def train(self, **options):
        """
        # outer loop for training
        # control training and validation pace
        # stop when reaching max iterations
        :param options:
        :return:
        """
        # training info
        logging.info("n_iter: %d", self.n_iter)
        logging.info("max_iter: %d", self.max_iter)
        running_losses = []
        epoch = 0

        # Train one epoch
        while self.n_iter < self.max_iter:
            print("epoch: ", epoch)
            epoch += 1
            for i, sample_train in tqdm(enumerate(self.train_loader)):
                # train one sample
                loss_out = self.train_val_sample(sample_train, self.n_iter, True)
                self.n_iter += 1
                running_losses.append(loss_out)
                # run validation
                if self._eval and self.n_iter % self.config["validation_interval"] == 0:
                    logging.info("====== Validating...")
                    for j, sample_val in enumerate(self.val_loader):
                        self.train_val_sample(sample_val, self.n_iter + j, False)
                        if j > self.config.get("validation_size", 3):
                            break
                # save model
                if self.n_iter % self.config["save_interval"] == 0:
                    logging.info(
                        "save model: every %d interval, current iteration: %d",
                        self.config["save_interval"],
                        self.n_iter,
                    )
                    self.saveModel()
                # ending condition
                if self.n_iter > self.max_iter:
                    # end training
                    logging.info("End training: %d", self.n_iter)
                    break

        pass

    def getLabels(self, labels_2D, cell_size, device="cpu"):
        """
        # transform 2D labels to 3D shape for training
        :param labels_2D:
        :param cell_size:
        :param device:
        :return:
        """
        labels3D_flattened = labels2Dto3D_flattened(labels_2D.to(device), cell_size=cell_size)
        labels3D_in_loss = labels3D_flattened
        return labels3D_in_loss

    def getMasks(self, mask_2D, cell_size, device="cpu"):
        """
        # 2D mask is constructed into 3D (Hc, Wc) space for training
        :param mask_2D:
            tensor [batch, 1, H, W]
        :param cell_size:
            8 (default)
        :param device:
        :return:
            flattened 3D mask for training
        """
        mask_3D = labels2Dto3D(mask_2D.to(device), cell_size=cell_size, add_dustbin=False).float()
        mask_3D_flattened = torch.prod(mask_3D, 1)
        return mask_3D_flattened

    def get_loss(self, semi, labels3D_in_loss, mask_3D_flattened, device="cpu"):
        """
        ## deprecated: loss function
        :param semi:
        :param labels3D_in_loss:
        :param mask_3D_flattened:
        :param device:
        :return:
        """
        loss_func = nn.CrossEntropyLoss(reduce=False).to(device)
        # if self.config['data']['gaussian_label']['enable']:
        #     loss = loss_func_BCE(nn.functional.softmax(semi, dim=1), labels3D_in_loss)
        #     loss = (loss.sum(dim=1) * mask_3D_flattened).sum()
        # else:
        loss = loss_func(semi, labels3D_in_loss)
        loss = (loss * mask_3D_flattened).sum()
        loss = loss / (mask_3D_flattened.sum() + 1e-10)
        return loss

    # def get_sem_loss(self, pred, label, device="cpu"):
    #     """
    #     ## deprecated: loss function
    #     :param pred:
    #     :param label:
    #     :param device:
    #     :return:
    #     """
    #     # device = "cpu"
    #     # print("device: %s" % device)
    #     loss_func = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    #     loss = loss_func(pred, label.to(device))

    #     return loss

    def saveModel(self):
        """
        # save checkpoint for resuming training
        :return:
        """
        # model_state_dict = self.net.module.state_dict()
        model_state_dict = self.net.state_dict()
        save_checkpoint(
            self.save_path,
            {
                "n_iter": self.n_iter + 1,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            self.n_iter,
        )
        pass

    def add_single_image_to_tb(self, task, img_tensor, n_iter, name="img"):
        """
        # add image to tensorboard for visualization
        :param task:
        :param img_tensor:
        :param n_iter:
        :param name:
        :return:
        """
        if img_tensor.dim() == 4:
            for i in range(min(img_tensor.shape[0], 5)):
                self.writer.add_image(task + "-" + name + "/%d" % i, img_tensor[i, :, :, :], n_iter)
        else:
            self.writer.add_image(task + "-" + name, img_tensor[:, :, :], n_iter)

    # tensorboard
    def addImg2tensorboard(
        self,
        img,
        labels_2D,
        semi,
        img_warp=None,
        labels_warp_2D=None,
        mask_warp_2D=None,
        semi_warp=None,
        mask_3D_flattened=None,
        task="training",
    ):
        """
        # deprecated: add images to tensorboard
        :param img:
        :param labels_2D:
        :param semi:
        :param img_warp:
        :param labels_warp_2D:
        :param mask_warp_2D:
        :param semi_warp:
        :param mask_3D_flattened:
        :param task:
        :return:
        """
        # print("add images to tensorboard")

        n_iter = self.n_iter
        semi_flat = flattenDetection(semi[0, :, :, :])
        semi_warp_flat = flattenDetection(semi_warp[0, :, :, :])

        thd = self.config["model"]["detection_threshold"]
        semi_thd = thd_img(semi_flat, thd=thd)
        semi_warp_thd = thd_img(semi_warp_flat, thd=thd)

        result_overlap = img_overlap(toNumpy(labels_2D[0, :, :, :]), toNumpy(semi_thd), toNumpy(img[0, :, :, :]))

        self.writer.add_image(task + "-detector_output_thd_overlay", result_overlap, n_iter)
        saveImg(result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_0.png")  # rgb to bgr * 255

        result_overlap = img_overlap(
            toNumpy(labels_warp_2D[0, :, :, :]),
            toNumpy(semi_warp_thd),
            toNumpy(img_warp[0, :, :, :]),
        )
        self.writer.add_image(task + "-warp_detector_output_thd_overlay", result_overlap, n_iter)
        saveImg(result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_1.png")  # rgb to bgr * 255

        mask_overlap = img_overlap(
            toNumpy(1 - mask_warp_2D[0, :, :, :]) / 2,
            np.zeros_like(toNumpy(img_warp[0, :, :, :])),
            toNumpy(img_warp[0, :, :, :]),
        )

        # writer.add_image(task + '_mask_valid_first_layer', mask_warp[0, :, :, :], n_iter)
        # writer.add_image(task + '_mask_valid_last_layer', mask_warp[-1, :, :, :], n_iter)
        ##### print to check
        # print("mask_2D shape: ", mask_warp_2D.shape)
        # print("mask_3D_flattened shape: ", mask_3D_flattened.shape)
        for i in range(self.batch_size):
            if i < 5:
                self.writer.add_image(task + "-mask_warp_origin", mask_warp_2D[i, :, :, :], n_iter)
                self.writer.add_image(task + "-mask_warp_3D_flattened", mask_3D_flattened[i, :, :], n_iter)
        # self.writer.add_image(task + '-mask_warp_origin-1', mask_warp_2D[1, :, :, :], n_iter)
        # self.writer.add_image(task + '-mask_warp_3D_flattened-1', mask_3D_flattened[1, :, :], n_iter)
        self.writer.add_image(task + "-mask_warp_overlay", mask_overlap, n_iter)

    def tb_scalar_dict(self, losses, task="training"):
        """
        # add scalar dictionary to tensorboard
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            self.writer.add_scalar(task + "-" + element, losses[element], self.n_iter // self.r)
            # print (task, '-', element, ": ", losses[element].item())

    def tb_images_dict(self, task, tb_imgs, max_img=5):
        """
        # add image dictionary to tensorboard
        :param task:
            str (train, val)
        :param tb_imgs:
        :param max_img:
            int - number of images
        :return:
        """

        if self.config["semantic"]:
            N, C, H, W = tb_imgs["sem_pred"].shape
            # print("tb_imgs[sem_pred].shape:", tb_imgs["sem_pred"].shape)
            sem_pred = np.zeros((N, 1, H, W))
            sem_pred[:, 0, :, :] = np.argmax(tb_imgs["sem_pred"], axis=1)

            warp_sem_pred = np.zeros((N, 1, H, W))
            warp_sem_pred[:, 0, :, :] = np.argmax(tb_imgs["warp_sem_pred"], axis=1)

            tb_imgs.update({"sem_pred": sem_pred, "warp_sem_pred": warp_sem_pred})

        for element in list(tb_imgs):
            for idx in range(tb_imgs[element].shape[0]):
                if idx >= max_img:
                    break
                # print(f"element: {element}")
                self.writer.add_image(
                    task + "-" + element + "/%d" % idx,
                    tb_imgs[element][idx, ...],
                    self.n_iter // self.r,
                )

    def tb_hist_dict(self, task, tb_dict):
        for element in list(tb_dict):
            self.writer.add_histogram(task + "-" + element, tb_dict[element], self.n_iter // self.r)
        pass

    def printLosses(self, losses, task="training"):
        """
        # print loss for tracking training
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            # print ('add to tb: ', element)
            print(task, "-", element, ": ", losses[element].item())

    def add2tensorboard_nms(self, img, labels_2D, semi, task="training", batch_size=1):
        """
        # deprecated:
        :param img:
        :param labels_2D:
        :param semi:
        :param task:
        :param batch_size:
        :return:
        """
        from utils.utils import getPtsFromHeatmap
        from utils.utils import box_nms

        boxNms = False
        n_iter = self.n_iter // self.r

        nms_dist = self.config["model"]["nms"]
        conf_thresh = self.config["model"]["detection_threshold"]
        # print("nms_dist: ", nms_dist)
        precision_recall_list = []
        precision_recall_boxnms_list = []
        for idx in range(batch_size):
            semi_flat_tensor = flattenDetection(semi[idx, :, :, :]).detach()
            semi_flat = toNumpy(semi_flat_tensor)
            semi_thd = np.squeeze(semi_flat, 0)
            pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)
            semi_thd_nms_sample = np.zeros_like(semi_thd)
            semi_thd_nms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1

            label_sample = torch.squeeze(labels_2D[idx, :, :, :])
            # pts_nms = getPtsFromHeatmap(label_sample.numpy(), conf_thresh, nms_dist)
            # label_sample_rms_sample = np.zeros_like(label_sample.numpy())
            # label_sample_rms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1
            label_sample_nms_sample = label_sample

            if idx < 5:
                result_overlap = img_overlap(
                    np.expand_dims(label_sample_nms_sample, 0),
                    np.expand_dims(semi_thd_nms_sample, 0),
                    toNumpy(img[idx, :, :, :]),
                )
                self.writer.add_image(
                    task + "-detector_output_thd_overlay-NMS" + "/%d" % idx,
                    result_overlap,
                    n_iter,
                )
            assert semi_thd_nms_sample.shape == label_sample_nms_sample.size()
            precision_recall = precisionRecall_torch(torch.from_numpy(semi_thd_nms_sample), label_sample_nms_sample)
            precision_recall_list.append(precision_recall)

            if boxNms:
                semi_flat_tensor_nms = box_nms(semi_flat_tensor.squeeze(), nms_dist, min_prob=conf_thresh).cpu()
                semi_flat_tensor_nms = (semi_flat_tensor_nms >= conf_thresh).float()

                if idx < 5:
                    result_overlap = img_overlap(
                        np.expand_dims(label_sample_nms_sample, 0),
                        semi_flat_tensor_nms.numpy()[np.newaxis, :, :],
                        toNumpy(img[idx, :, :, :]),
                    )
                    self.writer.add_image(
                        task + "-detector_output_thd_overlay-boxNMS" + "/%d" % idx,
                        result_overlap,
                        n_iter,
                    )
                precision_recall_boxnms = precisionRecall_torch(semi_flat_tensor_nms, label_sample_nms_sample)
                precision_recall_boxnms_list.append(precision_recall_boxnms)

        precision = np.mean([precision_recall["precision"] for precision_recall in precision_recall_list])
        recall = np.mean([precision_recall["recall"] for precision_recall in precision_recall_list])
        self.writer.add_scalar(task + "-precision_nms", precision, n_iter)
        self.writer.add_scalar(task + "-recall_nms", recall, n_iter)
        print("-- [%s-%d-fast NMS] precision: %.4f, recall: %.4f" % (task, n_iter, precision, recall))
        if boxNms:
            precision = np.mean([precision_recall["precision"] for precision_recall in precision_recall_boxnms_list])
            recall = np.mean([precision_recall["recall"] for precision_recall in precision_recall_boxnms_list])
            self.writer.add_scalar(task + "-precision_boxnms", precision, n_iter)
            self.writer.add_scalar(task + "-recall_boxnms", recall, n_iter)
            print("-- [%s-%d-boxNMS] precision: %.4f, recall: %.4f" % (task, n_iter, precision, recall))

    def get_heatmap(self, semi, det_loss_type="softmax"):
        if det_loss_type == "l2":
            heatmap = self.flatten_64to1(semi)
        else:
            heatmap = flattenDetection(semi)
        return heatmap

    ######## static methods ########
    @staticmethod
    def input_to_imgDict(sample, tb_images_dict):
        # for e in list(sample):
        #     print("sample[e]", sample[e].shape)
        #     if (sample[e]).dim() == 4:
        #         tb_images_dict[e] = sample[e]
        for e in list(sample):
            element = sample[e]
            if type(element) is torch.Tensor:
                if element.dim() == 4:
                    tb_images_dict[e] = element
                # print("shape of ", i, " ", element.shape)
        return tb_images_dict

    @staticmethod
    def interpolate_to_dense(coarse_desc, cell_size=8):
        dense_desc = nn.functional.interpolate(coarse_desc, scale_factor=(cell_size, cell_size), mode="bilinear")
        # norm the descriptor
        def norm_desc(desc):
            dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
            desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            return desc

        dense_desc = norm_desc(dense_desc)
        return dense_desc


if __name__ == "__main__":
    # load config
    # filename = "configs/superpoint_coco_test.yaml"
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

    train_agent = Train_model_frontend(config, device=device)

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
