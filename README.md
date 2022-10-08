# Semantic-superpoint (SSp)

This is the official implementation of the paper [NO_LINK_AVAILABLE_YET].<!-- [Semantic SuperPoint: a Deep Semantic Descriptor](NEED_TO_ADD_LINK).  --> The implementation is based on the open source implementation of the SuperPoint
[paper](https://arxiv.org/abs/1712.07629) done by [You-Yi Jau](https://github.com/eric-yyjau) and [Rui Zhu](https://github.com/Jerrypiglet)

- Differences between our model and SuperPoint (Sp)
  - The SSp uses a segmentation head to learn semantic segmentation through multi-task learning. The core idea is to make the descriptor intrinsically learn semantic information extracted by the shared encoder;
  - This implementation uses the 2017 MS-COCO dataset instead of the 2014 one;
  - We implement multiple multi-task losses to improve the final result, more specifically: [uncertainty based loss](https://arxiv.org/pdf/1705.07115.pdf) and [Central dir + Tensor](https://arxiv.org/pdf/2204.06698.pdf). The results are compared against the uniform loss.
# Note
The code used to implement the central direction + tensor method is not public so we removed from this repository (the *_ang files will not work now)
  
## Semantic SuperPoint model:

![SSp](imgs/SSp.png?raw=true "SSp")
## Comparing both models
### Results on HPatches
| Model                | Homography estimation |           |           | Detector metrics |           | Descriptor metrics |                |
| -------------------- | --------------------- | --------- | --------- | ---------------- | --------- | ------------------ | -------------- |
|                      | Epsilon = 1           | 3         | 5         | Repeatability    | MLE       | NN mAP             | Matching Score |
| Sp + uni (baseline)  | 0.476                 | 0.748     | **0.817** | 0.599            | 1.017     | **0.864**          | 0.519          |
| Sp + unc             | 0.460                 | 0.745     | 0.812     | 0.599            | 1.010     | **0.864**          | 0.520          |
| Sp + ct              | **0.493**             | 0.753     | 0.812     | **0.602**        | 1.001     | 0.862              | 0.519          |
| SSp + uni (**ours**) | 0.398                 | 0.710     | 0.797     | 0.584            | 1.052     | 0.843              | 0.506          |
| SSp + unc (**ours**) | 0.450                 | 0.745     | 0.816     | 0.598            | 1.005     | **0.864**          | **0.522**      |
| SSp + ct (**ours**)  | 0.466                 | **0.762** | 0.805     | 0.598            | **0.999** | 0.858              | 0.519          |

- Pretrained models can be found in logs folder;
- "pretrained/superpoint_v1.pth" is from https://github.com/magicleap/SuperPointPretrainedNetwork
- The evaluation is done under our evaluation scripts.

## Results on the KITTI sequence dataset (SLAM)
To extract the trajectories from the KITTI sequence dataset we used this repository: [Semantic_ORB_SLAM2](https://github.com/Gabriel-SGama/Semantic_ORB_SLAM2). The APE was averaged over 10 runs in each sequence.

| Sequence | Sp       | SSp (**ours**) | p-value |
| -------- | -------- | -------------- | ------- |
| 00       | 6.77     | **6.65**       | 0.71    |
| 01       | 286.71   | **209.99**     | 0.13    |
| 02       | 22.46    | **22.31**      | 0.82    |
| 03       | **1.32** | 1.63           | 0.00    |
| 04       | **0.84** | 0.90           | 0.94    |
| 05       | **5.80** | 6.31           | 0.82    |
| 06       | 11.95    | **11.83**      | 0.41    |
| 07       | 3.39     | **2.12**       | 1.00    |
| 08       | 31.32    | **26.70**      | 0.00    |
| 09       | 35.94    | **31.79**      | 0.00    |
| 10       | 5.51     | **4.95**       | 0.02    |


## Installation
### Requirements
- python == 3.6
- pytorch >= 1.1 (tested in 1.3.1)
- torchvision >= 0.3.0 (tested in 0.4.2)
- cuda (tested in cuda10)

```
conda create --name py36-sp python=3.6
conda activate py36-sp
pip install -r requirements.txt
pip install -r requirements_torch.txt # install pytorch
```

### Path setting
- paths for datasets ($DATA_DIR), logs are set in `setting.py`

### Dataset
Datasets should be downloaded into $DATA_DIR. The Synthetic Shapes dataset will also be generated there. The folder structure should look like:

```
datasets/ ($DATA_DIR)
|-- COCO
|   |-- train2017
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2017
|       |-- file1.jpg
|       `-- ...
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
`-- synthetic_shapes  # will be automatically created
`-- KITTI (accumulated folders from raw data)
|   |-- 2011_09_26_drive_0020_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_09_28_drive_0001_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_09_29_drive_0004_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_09_30_drive_0016_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_10_03_drive_0027_sync
|   |   |-- image_00/
|   |   `-- ...
```
- MS-COCO 2017 
    - [MS-COCO 2017 link](http://cocodataset.org/#download)
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)
- KITTI Odometry
    - [KITTI website](http://www.cvlibs.net/datasets/kitti/raw_data.php)
    - [download link](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip)



## Run the code
- Notes:
    - Start from any steps (1-4) by downloading some intermediate results
    - Currently Support training on 'COCO' dataset (original paper), 'KITTI' dataset (only the non-semantic version)
- Tensorboard:
    - log files is saved under 'runs/<\export_task>/...'
    
`tensorboard --logdir=./runs/ [--host | static_ip_address] [--port | 6008]`

### 1) Training MagicPoint on Synthetic Shapes
```
python train4.py train_base configs/magicpoint_shapes_pair.yaml magicpoint_synth --eval
```
you don't need to download synthetic data. You will generate it when first running it.
Synthetic data is exported in `./datasets`. You can change the setting in `settings.py`.

### 2) Exporting detections on MS-COCO / kitti
This is the step of homography adaptation(HA) to export pseudo ground truth for joint training.
- make sure the pretrained model in config file is correct
- make sure COCO dataset is in '$DATA_DIR' (defined in setting.py)
<!-- - you can export hpatches or coco dataset by editing the 'task' in config file -->
- config file:
```
export_folder: <'train' | 'val'>  # set export for training or validation
```
#### General command:
```
python export.py <export task>  <config file>  <export folder> [--outputImg | output images for visualization (space inefficient)]
```
#### Export coco - do on training set 
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
#### Export coco - do on validation set 
- Edit 'export_folder' to 'val' in 'magicpoint_coco_export.yaml'
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
#### Export kitti
- config
  - check the 'root' in config file 
  - train/ val split_files are included in `datasets/kitti_split/`.
```
python export.py export_detector_homoAdapt configs/magicpoint_kitti_export.yaml magicpoint_base_homoAdapt_kitti
```
<!-- #### export tum
- config
  - check the 'root' in config file
  - set 'datasets/tum_split/train.txt' as the sequences you have
```
python export.py export_detector_homoAdapt configs/magicpoint_tum_export.yaml magicpoint_base_homoAdapt_tum
``` -->


### 3) Training SSp or SP on MS-COCO/ KITTI
You need pseudo ground truth labels to traing detectors. Labels can be exported from step 2) or downloaded from [link](https://drive.google.com/drive/folders/1nnn0UbNMFF45nov90PJNnubDyinm2f26?usp=sharing). Then, as usual, you need to set config file before training.
- config file
  - root: specify your labels root
  - root_split_txt: where you put the train.txt/ val.txt split files (no need for COCO, needed for KITTI)
  - labels: the exported labels from homography adaptation
  - pretrained: specify the pretrained model (you can train from scratch)
  - config file is slightly different for the semantic version. See an example in "configs/superpoint_coco_train_wsem_heatmap"
- files that end on *_all.py are for training with the uniform sum loss and *_all_ct.py are for training with central dir + tensor method
- 'eval': turn on the evaluation during training 

- Semantic SuperPoint specifcs
  - To configure the semantic label of the MS-COCO dataset we use 133 classes as configured in "utils/coco_labels".
  - To train the semantic version, you need to specify it in the config file (semantic: True), and provide the others semantic parameters

#### General command
```
python train4.py <train task> <config file> <export folder> --eval
```

#### COCO
```
python train4.py train_joint configs/superpoint_coco_train_heatmap.yaml superpoint_coco --eval --debug
```
#### KITTI
```
python train4.py train_joint configs/superpoint_kitti_train_heatmap.yaml superpoint_kitti --eval --debug
```

- set your batch size (originally 1)
- refer to: 'train_tutorial.md'

### 4) Export/ Evaluate the metrics on HPatches
- Use pretrained model or specify your model in config file
- ```./run_export.sh``` will run export then evaluation.

#### Export
- download HPatches dataset (link above). Put in the $DATA_DIR.
```python export.py <export task> <config file> <export folder>```
- Export keypoints, descriptors, matching
```
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap.yaml superpoint_hpatches_test
```
#### Evaluate
```python evaluation.py <path to npz files> [-r, --repeatibility | -o, --outputImg | -homo, --homography ]```
- Evaluate homography estimation/ repeatability/ matching scores ...
```
python evaluation.py logs/superpoint_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching
```

#### Export and evaluate one log folder
- Define the folder to export in the config file "magicpoint_repeatability_heatmap_export.yaml"
- The results will be saved in a csv format in the logs folder
- The descriptors extracted are going to be continually rewritten, so if you want to extracted the descriptors for a specifc model use the commands above
```
python export_eval.py export_descriptor configs/magicpoint_repeatability_heatmap_export.yaml superpoint_hpatches_test -r -homo
```

### 5) Export/Evaluate repeatability on SIFT
- Refer to another project: [Feature-preserving image denoising with multiresolution filters](https://github.com/eric-yyjau/image_denoising_matching)
```shell
# export detection, description, matching
python export_classical.py export_descriptor configs/classical_descriptors.yaml sift_test --correspondence

# evaluate (use 'sift' flag)
python evaluation.py logs/sift_test/predictions --sift --repeatibility --homography 
```

<!-- - specify the pretrained model -->
## Pretrained models
### Best of each method
- *COCO dataset*
  - SSp + unc: ```logs/superpoint_coco_ssmall_ML22/checkpoints/superPointNet_180000_checkpoint.pth.tar```
  - Sp + unc: ```logs/superpoint_coco_2017_ML22/checkpoints/superPointNet_185000_checkpoint.pth.tar```

- *KITTI dataset*

### Model from magicleap
```pretrained/superpoint_v1.pth```

### Extract torch.jit.script model
To extract the scripted version of the pretrained model use the code in ```convert2script```. In the model file, need to change the forward method declaration and the return type to be a list (both changes are commented).

```
python3 convert2script
```


## Citations

Semantic SuperPoint:
```
Not published yet
```

# Credits
This implementation is based on the forked repository developed by [You-Yi Jau](https://github.com/eric-yyjau) and [Rui Zhu](https://github.com/Jerrypiglet).

```
@misc{2020_jau_zhu_deepFEPE,
Author = {You-Yi Jau and Rui Zhu and Hao Su and Manmohan Chandraker},
Title = {Deep Keypoint-Based Camera Pose Estimation with Geometric Constraints},
Year = {2020},
Eprint = {arXiv:2007.15122},
}
```
