import os
import sys
import time
import logging
import functools
from pathlib import Path
sys.path.append('../')
# from share_var import SharedVariable
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datasets.hpatches import HPatchesDataset
from datasets.megadepth import MegaDepthDataset
from datasets.cat_datasets import ConcatDatasets
from training.test_wrapper import LET_Wrapper
from training.scheduler import WarmupConstantSchedule
from training.gen_mgp_data import MagicDataset
from pytorch_lightning import Callback
import subprocess ,yaml
torch.set_float32_matmul_precision('medium')
gpu_ini = [3]
#THIS_TrainWrapper = TrainWrapper
THIS_TrainWrapper = LET_Wrapper
wdic_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
para_loc =os.path.join(wdic_path, "magic_point_dataset/config.yaml")
with open(para_loc, "r") as f:
    config = yaml.safe_load(f)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"





if __name__ == '__main__':
    pretrained_model = "/home/lizhonghao/ALIKE/training/log_my_train/train/Version-0424-175310/checkpoints/last.ckpt"
    debug = False
    # debug = True

    model_size = 'normal'
    # model_size = 'tiny'
    # model_size = 'small'
    # model_size = 'big' 
    # pe = True
    pe = False

    agg_mode = 'cat'
    # agg_mode = 'sum'
    # agg_mode = 'fpn'
    # ========================================= configs
    if model_size == 'small':
        c1 = 8
        c2 = 16
        c3 = 48
        c4 = 96
        dim = 96
        single_head = True
    elif model_size == 'big':
        c1 = 32
        c2 = 64
        c3 = 128
        c4 = 128
        dim = 128
        single_head = False
    elif model_size == 'tiny':
        c1 = 8
        c2 = 16
        c3 = 32
        c4 = 64
        dim = 64
        single_head = True
    else:
        c1 = 32
        c2 = 64
        c3 = 128
        c4 = -1
        dim = 128
        single_head = True
    if  True:
        radius = 2
        top_k = 400
        scores_th_eval = 0.2
        n_limit_eval = 5000

        train_gt_th = 5
        eval_gt_th = 3


        w_pk = 0
        w_rp = 0
        w_sp = 0
        w_ds = 0
        w_triplet = 0
        sc_th = 0
        norm = 0
        temp_sp = 0
        temp_ds = 0

        # ================================== training parameters
        gpus = gpu_ini
        warmup_steps = 500
        t_total = 10000
        image_size = 480
        log_freq_img = 2000

        # ================================== dataset dir and log dir
        hpatch_dir = '../data/hpatches-sequences-release'
        mega_dir = '../data/megadepth'
        # mega_dir = '/mnt/data3/datasets/megadepth_disk'
        imw2020val_dir = '../data/imw2020-val'
        log_dir = 'log_' + Path(__file__).stem

    batch_size = 2
    if debug:
        accumulate_grad_batches = 1
        num_workers = 0
        num_sanity_val_steps = 0
        # pretrained_model = 'log_train/train/Version-0701-231352/checkpoints/last.ckpt'
        pretrained_model = 'log_train/train/Version-0708-174505/checkpoints/last.ckpt'
    else:
        accumulate_grad_batches = 16
        num_workers = 16
        num_sanity_val_steps = 1

    # ========================================= model
    lr_scheduler = functools.partial(WarmupConstantSchedule, warmup_steps=warmup_steps)

    model = THIS_TrainWrapper(
        # ================================== feature encoder
        c1=c1, c2=c2, c3=c3, c4=c4, dim=dim,
        agg_mode=agg_mode,  # sum, cat, fpn
        single_head=single_head,
        pe=pe,
        # ================================== detect parameters
        radius=radius,
        top_k=top_k, scores_th=0, n_limit=0,
        scores_th_eval=scores_th_eval, n_limit_eval=n_limit_eval,
        # ================================== gt reprojection th
        train_gt_th=train_gt_th, eval_gt_th=eval_gt_th,
        # ================================== loss weight
        w_pk=w_pk,  # weight of peaky loss
        w_rp=w_rp,  # weight of reprojection loss
        w_sp=w_sp,  # weight of score map rep loss
        w_ds=w_ds,  # weight of descriptor loss
        w_triplet=w_triplet,
        sc_th=sc_th,  # score threshold in peaky and  reprojection loss
        norm=norm,  # distance norm
        temp_sp=temp_sp,  # temperature in ScoreMapRepLoss
        temp_ds=temp_ds,  # temperature in DescReprojectionLoss
        # ================================== learning rate
        lr=3e-4,
        log_freq_img=log_freq_img,
        # ================================== pretrained_model
        pretrained_model=pretrained_model,
        lr_scheduler=lr_scheduler,
        debug=debug,
        garystyle = config['is_graystyle']
    )

    # ========================================= dataloaders
    if debug:
        reload_dataloaders_every_epoch = False
        limit_train_batches = 1
        limit_val_batches = 1.
        max_epochs = 100
    else:
        reload_dataloaders_every_epoch = True
        limit_train_batches = 5000 // batch_size
        limit_val_batches = 1.
        max_epochs = 200

    # ========== evaluation dataset
    hpatch_i_dataset = HPatchesDataset(root=hpatch_dir, alteration='i',mode="test")
    hpatch_v_dataset = HPatchesDataset(root=hpatch_dir, alteration='v',mode="test")
    hpatch_i_dataloader = DataLoader(hpatch_i_dataset, batch_size=1, pin_memory=not debug, num_workers=num_workers,shuffle=False)
    hpatch_v_dataloader = DataLoader(hpatch_v_dataset, batch_size=1, pin_memory=not debug, num_workers=num_workers,shuffle=False)
    test_datasets = MagicDataset(config,test_mode=True)
    test_loader  = DataLoader(test_datasets,batch_size=1, shuffle=False,num_workers=1)
    # imw2020val = MegaDepthDataset(root=imw2020val_dir, train=False, using_cache=True, colorjit=False, gray=False)
    # imw2020val_dataloader = DataLoader(imw2020val, batch_size=1, pin_memory=not debug, num_workers=num_workers)
    device_id = f"cuda:{gpu_ini[0]}"
    model = model.to(device_id)
    def move_tensors_to_device(dictionary, device):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                move_tensors_to_device(value, device)
            elif torch.is_tensor(value):
                dictionary[key] = value.to(device)
    for loader in [test_loader]:
        for i,batch in enumerate(loader):
            with torch.no_grad():
                move_tensors_to_device(batch,device_id)
                model.test(batch)
                


