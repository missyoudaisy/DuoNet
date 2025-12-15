# DuoNet
This repository contains the official Pytorch code for DuoNet.

## Installation
Check [INSTALL.md](https://github.com/missyoudaisy/DuoNet/blob/main/INSTALL.md) for installation instructions

## Dataset
Check [DATASET.md](https://github.com/missyoudaisy/DuoNet/blob/main/DATASET.md) for instructions of dataset preprocessing (VG & GQA & Open Images).

## Train
```bash
CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --master_port 10088 --nproc_per_node=1 \
tools/relation_train_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"   MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True   MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
MODEL.ROI_RELATION_HEAD.PREDICTOR DuoNet   DTYPE "float32"   SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 \
SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3   SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512   SOLVER.STEPS "(48000, 60000)" \
SOLVER.VAL_PERIOD 30000   SOLVER.CHECKPOINT_PERIOD 10000 GLOVE_DIR ../glove/ \
MODEL.PRETRAINED_DETECTOR_CKPT_VG ../checkpoints/pretrained_faster_rcnn/model_final.pth \
OUTPUT_DIR ../checkpoints/duonet \
PROTOTYPE_PATH ../checkpoints/fixedETF/51centers_4096dim.pth \
SOLVER.PRE_VAL False SOLVER.GRAD_NORM_CLIP 5.0 \
GLOBAL_SETTING.DATASET_CHOICE  'VG';
```

## Acknowledgement
The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
