## basic
import argparse
import time
import csv
import yaml
import os
import logging
import matplotlib
import csv

matplotlib.use("Agg")  # solve error of tk
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

import numpy as np
from imageio import imread
from tqdm import tqdm
from tensorboardX import SummaryWriter

## torch
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

## other functions
from utils.utils import (
    tensor2array,
    save_checkpoint,
    load_checkpoint,
    save_path_formatter,
)
from utils.utils import getWriterPath
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.utils import inv_warp_image_batch
from models.model_wrap import SuperPointFrontend_torch, PointTracker


from evaluations.descriptor_evaluation import compute_homography
from evaluations.detector_evaluation import compute_repeatability

from utils.draw import plot_imgs
from utils.logging import *

## parameters
from settings import EXPER_PATH


def draw_matches_cv(data, matches, plot_points=True):
    if plot_points:
        keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data["keypoints1"]]
        keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data["keypoints2"]]
    else:
        matches_pts = data["matches"]
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in matches_pts]
        keypoints2 = [cv2.KeyPoint(p[2], p[3], 1) for p in matches_pts]
        print(f"matches_pts: {matches_pts}")
        # keypoints1, keypoints2 = [], []

    inliers = data["inliers"].astype(bool)
    # matches = np.array(data['matches'])[inliers].tolist()
    # matches = matches[inliers].tolist()
    def to3dim(img):
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        return img

    img1 = to3dim(data["image1"])
    img2 = to3dim(data["image2"])
    img1 = np.concatenate([img1, img1, img1], axis=2)
    img2 = np.concatenate([img2, img2, img2], axis=2)

    img1 = np.uint8(img1)
    img2 = np.uint8(img2)

    return cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches, None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255)
    )


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def find_files_with_ext(directory, extension=".npz", if_int=True):
    # print(os.listdir(directory))
    list_of_files = []
    import os

    if extension == ".npz":
        for l in os.listdir(directory):
            if l.endswith(extension):
                list_of_files.append(l)
                # print(l)
    if if_int:
        list_of_files = [e for e in list_of_files if isfloat(e[:-4])]
    return list_of_files


def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img


def evaluate(args, save_model, writer, output_dir, **options):
    # path = '/home/yoyee/Documents/SuperPoint/superpoint/logs/outputs/superpoint_coco/'
    path = output_dir + "/predictions"
    files = find_files_with_ext(path)
    correctness = []
    est_H_mean_dist = []
    repeatability = []
    mscore = []
    mAP = []
    localization_err = []
    rep_thd = 3
    save_file = path + "/result.txt"
    inliers_method = "cv"
    compute_map = True
    verbose = True
    top_K = 1000
    print("top_K: ", top_K)

    reproduce = True
    if reproduce:
        logging.info("reproduce = True")
        np.random.seed(0)
        print(f"test random # : np({np.random.rand(1)})")

    # for i in range(2):
    #     f = files[i]
    print(f"file: {files[0]}")
    files.sort(key=lambda x: int(x[:-4]))
    from numpy.linalg import norm
    from utils.draw import draw_keypoints
    from utils.utils import saveImg

    for f in tqdm(files):
        f_num = f[:-4]
        data = np.load(path + "/" + f)
        # print("load successfully. ", f)

        # unwarp
        # prob = data['prob']
        # warped_prob = data['prob']
        # desc = data['desc']
        # warped_desc = data['warped_desc']
        # homography = data['homography']
        real_H = data["homography"]
        image = data["image"]
        warped_image = data["warped_image"]
        keypoints = data["prob"][:, [1, 0]]
        # print("keypoints: ", keypoints[:3, :])
        warped_keypoints = data["warped_prob"][:, [1, 0]]
        # print("warped_keypoints: ", warped_keypoints[:3, :])
        # print("Unwrap successfully.")

        if args.repeatibility:
            rep, local_err = compute_repeatability(data, keep_k_points=top_K, distance_thresh=rep_thd, verbose=False)
            repeatability.append(rep)
            # print("repeatability: %.2f" % (rep))
            if local_err > 0:
                localization_err.append(local_err)
                # print("local_err: ", local_err)

        if args.homography:
            # estimate result
            ##### check
            homography_thresh = [1, 3, 5, 10, 20, 50]
            #####
            result = compute_homography(data, correctness_thresh=homography_thresh)
            correctness.append(result["correctness"])
            # est_H_mean_dist.append(result['mean_dist'])
            # compute matching score
            def warpLabels(pnts, homography, H, W):
                import torch

                """
                input:
                    pnts: numpy
                    homography: numpy
                output:
                    warped_pnts: numpy
                """
                from utils.utils import warp_points
                from utils.utils import filter_points

                pnts = torch.tensor(pnts).long()
                homography = torch.tensor(homography, dtype=torch.float32)
                warped_pnts = warp_points(torch.stack((pnts[:, 0], pnts[:, 1]), dim=1), homography)  # check the (x, y)
                warped_pnts = filter_points(warped_pnts, torch.tensor([W, H])).round().long()
                return warped_pnts.numpy()

            from numpy.linalg import inv

            H, W = image.shape
            unwarped_pnts = warpLabels(warped_keypoints, inv(real_H), H, W)
            score = (result["inliers"].sum() * 2) / (keypoints.shape[0] + unwarped_pnts.shape[0])
            # print("m. score: ", score)
            mscore.append(score)
            # compute map
            if compute_map:

                def getMatches(data):
                    from models.model_wrap import PointTracker

                    desc = data["desc"]
                    warped_desc = data["warped_desc"]

                    nn_thresh = 1.2
                    # print("nn threshold: ", nn_thresh)
                    tracker = PointTracker(max_length=2, nn_thresh=nn_thresh)
                    # matches = tracker.nn_match_two_way(desc, warped_desc, nn_)
                    tracker.update(keypoints.T, desc.T)
                    tracker.update(warped_keypoints.T, warped_desc.T)
                    matches = tracker.get_matches().T
                    mscores = tracker.get_mscores().T

                    # mAP
                    # matches = data['matches']
                    # print("matches: ", matches.shape)
                    # print("mscores: ", mscores.shape)
                    # print("mscore max: ", mscores.max(axis=0))
                    # print("mscore min: ", mscores.min(axis=0))

                    return matches, mscores

                def getInliers(matches, H, epi=3, verbose=False):
                    """
                    input:
                        matches: numpy (n, 4(x1, y1, x2, y2))
                        H (ground truth homography): numpy (3, 3)
                    """
                    from evaluations.detector_evaluation import warp_keypoints

                    # warp points
                    warped_points = warp_keypoints(matches[:, :2], H)  # make sure the input fits the (x,y)

                    # compute point distance
                    norm = np.linalg.norm(warped_points - matches[:, 2:4], ord=None, axis=1)
                    inliers = norm < epi
                    # if verbose:
                    #     print(
                    #         "Total matches: ",
                    #         inliers.shape[0],
                    #         ", inliers: ",
                    #         inliers.sum(),
                    #         ", percentage: ",
                    #         inliers.sum() / inliers.shape[0],
                    #     )

                    return inliers

                def getInliers_cv(matches, H=None, epi=3, verbose=False):
                    import cv2

                    # count inliers: use opencv homography estimation
                    # Estimate the homography between the matches using RANSAC
                    H, inliers = cv2.findHomography(matches[:, [0, 1]], matches[:, [2, 3]], cv2.RANSAC)
                    inliers = inliers.flatten()
                    # print(
                    #     "Total matches: ",
                    #     inliers.shape[0],
                    #     ", inliers: ",
                    #     inliers.sum(),
                    #     ", percentage: ",
                    #     inliers.sum() / inliers.shape[0],
                    # )
                    return inliers

                def computeAP(m_test, m_score):
                    from sklearn.metrics import average_precision_score

                    average_precision = average_precision_score(m_test, m_score)
                    # print("Average precision-recall score: {0:0.2f}".format(average_precision))
                    return average_precision

                def flipArr(arr):
                    return arr.max() - arr

                if args.sift:
                    assert result is not None
                    matches, mscores = result["matches"], result["mscores"]
                else:
                    matches, mscores = getMatches(data)

                real_H = data["homography"]
                if inliers_method == "gt":
                    # use ground truth homography
                    # print("use ground truth homography for inliers")
                    inliers = getInliers(matches, real_H, epi=3, verbose=verbose)
                else:
                    # use opencv estimation as inliers
                    # print("use opencv estimation for inliers")
                    inliers = getInliers_cv(matches, real_H, epi=3, verbose=verbose)

                """
                matches - 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
                """
                ## distance to confidence
                if args.sift:
                    m_flip = flipArr(mscores[:])  # for sift
                else:
                    m_flip = flipArr(mscores[:, 2])

                if inliers.shape[0] > 0 and inliers.sum() > 0:
                    #                     m_flip = flipArr(m_flip)
                    # compute ap
                    ap = computeAP(inliers, m_flip)
                else:
                    ap = 0

                mAP.append(ap)

    if args.repeatibility:
        repeatability_ave = np.array(repeatability).mean()
        localization_err_m = np.array(localization_err).mean()
        print("repeatability: ", repeatability_ave)
        print("localization error over ", len(localization_err), " images : ", localization_err_m)
    if args.homography:
        correctness_ave = np.array(correctness).mean(axis=0)
        # est_H_mean_dist = np.array(est_H_mean_dist)
        print("homography estimation threshold", homography_thresh)
        print("correctness_ave", correctness_ave)
        # print(f"mean est H dist: {est_H_mean_dist.mean()}")
        mscore_m = np.array(mscore).mean(axis=0)
        print("matching score", mscore_m)
        if compute_map:
            mAP_m = np.array(mAP).mean()
            print("mean AP", mAP_m)

        print("end")

    row = {
        "path": save_model,
        "repeatability threshold": str(rep_thd),
        "repeatability": repeatability_ave,
        "localization error": localization_err_m,
        "homography threshold": str(homography_thresh),
        "Average correctness": str(correctness_ave),
        "nn mean AP": str(mAP_m),
        "matching score": str(mscore_m),
    }
    writer.writerow(row)


def combine_heatmap(heatmap, inv_homographies, mask_2D, device="cpu"):
    ## multiply heatmap with mask_2D

    heatmap = inv_warp_image_batch(heatmap, inv_homographies[0, :, :, :], device=device, mode="bilinear")

    ##### check
    mask_2D = inv_warp_image_batch(mask_2D, inv_homographies[0, :, :, :], device=device, mode="bilinear")
    heatmap = torch.sum(heatmap, dim=0)
    mask_2D = torch.sum(mask_2D, dim=0)
    return heatmap / mask_2D
    pass


#### end util functions


def export_descriptor(config, output_dir, args):
    """
    # input 2 images, output keypoints and correspondence
    save prediction:
        pred:
            'image': np(320,240)
            'prob' (keypoints): np (N1, 2)
            'desc': np (N2, 256)
            'warped_image': np(320,240)
            'warped_prob' (keypoints): np (N2, 2)
            'warped_desc': np (N2, 256)
            'homography': np (3,3)
            'matches': np [N3, 4]
    """
    from utils.loader import get_save_path
    from utils.var_dim import squeezeToNumpy

    # basic settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("train on device: %s", device)
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(getWriterPath(task=args.command, date=True))
    save_path = get_save_path(output_dir)
    save_output = save_path / "../predictions"
    os.makedirs(save_output, exist_ok=True)

    ## parameters
    outputMatches = True
    subpixel = config["model"]["subpixel"]["enable"]
    patch_size = config["model"]["subpixel"]["patch_size"]

    # data loading
    from utils.loader import dataLoader_test as dataLoader

    task = config["data"]["dataset"]
    data = dataLoader(config, dataset=task)
    test_set, test_loader = data["test_set"], data["test_loader"]
    from utils.print_tool import datasize

    datasize(test_loader, config, tag="test")

    # model loading
    from utils.loader import get_module

    Val_model_heatmap = get_module("", config["front_end_model"])
    ## load pretrained
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    ## tracker
    tracker = PointTracker(max_length=2, nn_thresh=val_agent.nn_thresh)

    ###### check!!!
    count = 0
    for i, sample in tqdm(enumerate(test_loader)):
        img_0, img_1 = sample["image"], sample["warped_image"]

        # first image, no matches
        # img = img_0
        def get_pts_desc_from_agent(val_agent, img, device="cpu"):
            """
            pts: list [numpy (3, N)]
            desc: list [numpy (256, N)]
            """
            heatmap_batch = val_agent.run(img.to(device))  # heatmap: numpy [batch, 1, H, W]
            # heatmap to pts
            pts = val_agent.heatmap_to_pts()
            # print("pts: ", pts)
            if subpixel:
                pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)
            # heatmap, pts to desc
            desc_sparse = val_agent.desc_to_sparseDesc()
            # print("pts[0]: ", pts[0].shape, ", desc_sparse[0]: ", desc_sparse[0].shape)
            # print("pts[0]: ", pts[0].shape)
            outs = {"pts": pts[0], "desc": desc_sparse[0]}
            return outs

        def transpose_np_dict(outs):
            for entry in list(outs):
                outs[entry] = outs[entry].transpose()

        outs = get_pts_desc_from_agent(val_agent, img_0, device=device)
        pts, desc = outs["pts"], outs["desc"]  # pts: np [3, N]

        if outputMatches == True:
            tracker.update(pts, desc)

        # save keypoints
        pred = {"image": squeezeToNumpy(img_0)}
        pred.update({"prob": pts.transpose(), "desc": desc.transpose()})

        # second image, output matches
        outs = get_pts_desc_from_agent(val_agent, img_1, device=device)
        pts, desc = outs["pts"], outs["desc"]

        if outputMatches == True:
            tracker.update(pts, desc)

        pred.update({"warped_image": squeezeToNumpy(img_1)})
        # print("total points: ", pts.shape)
        pred.update(
            {
                "warped_prob": pts.transpose(),
                "warped_desc": desc.transpose(),
                "homography": squeezeToNumpy(sample["homography"]),
            }
        )

        if outputMatches == True:
            matches = tracker.get_matches()
            # print("matches: ", matches.transpose().shape)
            pred.update({"matches": matches.transpose()})
        # print("pts: ", pts.shape, ", desc: ", desc.shape)

        # clean last descriptor
        tracker.clear_desc()

        filename = str(count)
        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)
        # print("save: ", path)
        count += 1
    print("output pairs: ", count)


@torch.no_grad()
def export_detector_homoAdapt_gpu(config, output_dir, args):
    """
    input 1 images, output pseudo ground truth by homography adaptation.
    Save labels:
        pred:
            'prob' (keypoints): np (N1, 3)
    """
    from utils.utils import pltImshow
    from utils.utils import saveImg
    from utils.draw import draw_keypoints

    # basic setting
    task = config["data"]["dataset"]
    export_task = config["data"]["export_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("train on device: %s", device)
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(getWriterPath(task=args.command, exper_name=args.exper_name, date=True))

    ## parameters
    nms_dist = config["model"]["nms"]  # 4
    top_k = config["model"]["top_k"]
    homoAdapt_iter = config["data"]["homography_adaptation"]["num"]
    conf_thresh = config["model"]["detection_threshold"]
    nn_thresh = 0.7
    outputMatches = True
    count = 0
    max_length = 5
    output_images = args.outputImg
    check_exist = True

    ## save data
    save_path = Path(output_dir)
    save_output = save_path
    save_output = save_output / "predictions" / export_task
    save_path = save_path / "checkpoints"
    logging.info("=> will save everything to {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_output, exist_ok=True)

    # data loading
    from utils.loader import dataLoader_test as dataLoader

    data = dataLoader(config, dataset=task, export_task=export_task)
    test_set, test_loader = data["test_set"], data["test_loader"]

    # model loading
    ## load pretrained
    try:
        path = config["pretrained"]
        print("==> Loading pre-trained network.")
        print("path: ", path)
        # This class runs the SuperPoint network and processes its outputs.

        fe = SuperPointFrontend_torch(
            config=config,
            weights_path=path,
            nms_dist=nms_dist,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=False,
            device=device,
        )
        print("==> Successfully loaded pre-trained network.")

        fe.net_parallel()
        print(path)
        # save to files
        save_file = save_output / "export.txt"
        with open(save_file, "a") as myfile:
            myfile.write("load model: " + path + "\n")
    except Exception:
        print(f"load model: {path} failed! ")
        raise

    def load_as_float(path):
        return imread(path).astype(np.float32) / 255

    tracker = PointTracker(max_length, nn_thresh=fe.nn_thresh)
    with open(save_file, "a") as myfile:
        myfile.write("homography adaptation: " + str(homoAdapt_iter) + "\n")

    ## loop through all images
    for i, sample in tqdm(enumerate(test_loader)):
        img, mask_2D = sample["image"], sample["valid_mask"]
        img = img.transpose(0, 1)
        img_2D = sample["image_2D"].numpy().squeeze()
        mask_2D = mask_2D.transpose(0, 1)

        inv_homographies, homographies = (
            sample["homographies"],
            sample["inv_homographies"],
        )
        img, mask_2D, homographies, inv_homographies = (
            img.to(device),
            mask_2D.to(device),
            homographies.to(device),
            inv_homographies.to(device),
        )
        # sample = test_set[i]
        name = sample["name"][0]
        logging.info(f"name: {name}")
        if check_exist:
            p = Path(save_output, "{}.npz".format(name))
            if p.exists():
                logging.info("file %s exists. skip the sample.", name)
                continue

        # pass through network
        heatmap = fe.run(img, onlyHeatmap=True, train=False)
        outputs = combine_heatmap(heatmap, inv_homographies, mask_2D, device=device)
        pts = fe.getPtsFromHeatmap(outputs.detach().cpu().squeeze())  # (x,y, prob)

        # subpixel prediction
        if config["model"]["subpixel"]["enable"]:
            fe.heatmap = outputs  # tensor [batch, 1, H, W]
            print("outputs: ", outputs.shape)
            print("pts: ", pts.shape)
            pts = fe.soft_argmax_points([pts])
            pts = pts[0]

        ## top K points
        pts = pts.transpose()
        print("total points: ", pts.shape)
        print("pts: ", pts[:5])
        if top_k:
            if pts.shape[0] > top_k:
                pts = pts[:top_k, :]
                print("topK filter: ", pts.shape)

        ## save keypoints
        pred = {}
        pred.update({"pts": pts})

        ## - make directories
        filename = str(name)
        if task == "Kitti" or "Kitti_inh":
            scene_name = sample["scene_name"][0]
            os.makedirs(Path(save_output, scene_name), exist_ok=True)

        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)

        ## output images for visualization labels
        if output_images:
            img_pts = draw_keypoints(img_2D * 255, pts.transpose())
            f = save_output / (str(count) + ".png")
            if task == "Coco" or "Kitti":
                f = save_output / (name + ".png")
            saveImg(img_pts, str(f))
        count += 1

    print("output pseudo ground truth: ", count)
    save_file = save_output / "export.txt"
    with open(save_file, "a") as myfile:
        myfile.write("Homography adaptation: " + str(homoAdapt_iter) + "\n")
        myfile.write("output pairs: " + str(count) + "\n")
    pass


if __name__ == "__main__":
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # export command
    p_train = subparsers.add_parser("export_descriptor")
    p_train.add_argument("config", type=str)
    p_train.add_argument("exper_name", type=str)
    p_train.add_argument("--correspondence", action="store_true")
    p_train.add_argument("--eval", action="store_true")
    p_train.add_argument("--debug", action="store_true", default=False, help="turn on debuging mode")
    p_train.add_argument("--sift", action="store_true", help="use sift matches")
    p_train.add_argument("-r", "--repeatibility", action="store_true")
    p_train.add_argument("-homo", "--homography", action="store_true")
    p_train.set_defaults(func=export_descriptor)

    # using homography adaptation to export detection psuedo ground truth
    p_train = subparsers.add_parser("export_detector_homoAdapt")
    p_train.add_argument("config", type=str)
    p_train.add_argument("exper_name", type=str)
    p_train.add_argument("--eval", action="store_true")
    p_train.add_argument("--debug", action="store_true", default=False, help="turn on debuging mode")
    # p_train.set_defaults(func=export_detector_homoAdapt)
    p_train.set_defaults(func=export_detector_homoAdapt_gpu)

    # evaluate process
    # parser.add_argument("path", type=str, defaut="None")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f)
    print("check config!! ", config)

    folder = config["model"]["folder"]
    checkpoints_paths = sorted(glob(folder + "*.pth*"))

    print("\ncheckpoints_paths", checkpoints_paths)
    csv_file = open(folder + "results.csv", "w")
    fieldnames = [
        "path",
        "repeatability threshold",
        "repeatability",
        "localization error",
        "homography threshold",
        "Average correctness",
        "nn mean AP",
        "matching score",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    # writer.writerow({"first_name": "Baked", "last_name": "Beans"})
    # writer.writerow({"first_name": "Lovely", "last_name": "Spam"})
    # writer.writerow({"first_name": "Wonderful", "last_name": "Spam"})
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    # args.path = output_dir
    # print("args.path: " + args.path)

    for save_model in tqdm(checkpoints_paths):
        # with capture_outputs(os.path.join(output_dir, 'log')):
        # logging.info("Running command", args.command.upper(), "with model: ", str(save_model))
        print("loading model", save_model)
        config["model"]["pretrained"] = save_model
        try:
            args.func(config, output_dir, args)
            evaluate(args, save_model, writer, output_dir)
        except:
            row = {
                "path": save_model,
                "repeatability threshold": str(0),
                "repeatability": 0,
                "localization error": 0,
                "homography threshold": str(0),
                "Average correctness": str(0),
                "nn mean AP": str(0),
                "matching score": str(0),
            }
            writer.writerow(row)
        # print(row)
# python3 export_eval.py export_descriptor  configs/magicpoint_repeatability_heatmap_export.yaml superpoint_hpatches_2017_lrScheduler --repeatibility --homography
