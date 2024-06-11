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
from training.train_wrapper import TrainWrapper 
from training.LET_wrapper import LET_Wrapper 
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



def run_script(command,stdout_):
    pass
    # 启动一个新的进程来运行脚本
        # subprocess.Popen(command, stdout=stdout_, stderr=subprocess.PIPE)

class RebuildDatasetCallback(Callback):
    def __init__(self):
        self.counts = 0
        self.para1 = "/home/lizhonghao/ALIKE/magic_point_dataset/pic1"
        self.para2 = "/home/lizhonghao/ALIKE/magic_point_dataset/pic2"

    def on_train_epoch_start(self, trainer, pl_module):
        pass
        # if self.counts % 2 == 0:
        #     para = self.para1
        #     train_para = self.para2
        # else:
        #     para = self.para2
        #     train_para = self.para1
        # # self.counts += 1
        # self.counts += 0
        # para = "/home/lizhonghao/ALIKE/magic_point_dataset/pic2"
        # script_command = ["/home/lizhonghao/anaconda3/envs/alike/bin/python",\
        #     "/home/lizhonghao/ALIKE/training/build_dataset.py",
        #     para]
        # with open('build_data.out','a') as f:
        #     run_script(script_command,f)
            # f.write("start training: read "+para)
        # share_var.location = train_para
        # trainer.train_dataloader.dataset.change_loc(train_para)
        # trainer.train_dataloader = MagicDataset(config)
        #trainer.train_dataloader.dataset.build_dataset()
        # train_loader_dataset = trainer.train_dataloader.dataset
        # for i in train_loader_dataset.datasets:
        #     i.build_dataset()
        # train_loader_dataset.datasets[1].build_dataset()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    ##  torch.autograd.set_detect_anomaly(True)
    pretrained_model = '/home/lizhonghao/ALIKE/training/log_my_train/train/Version-0424-175310/checkpoints/last.ckpt'
    # pretrained_model = None

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
        c4 = 128
        dim = 128
        single_head = True
    if True:
        # ================================== detect parameters
        radius = 4
        top_k = 50
        scores_th_eval = 0.2
        n_limit_eval = 5000

        # ================================== gt reprojection th
        train_gt_th = 5
        eval_gt_th = 3

        # ================================== loss weight
        # w_pk = 0.5
        # w_rp = 1
        # w_sp = 1
        # w_ds = 5
        # w_triplet = 0
        # sc_th = 0.1
        # norm = 1
        # temp_sp = 0.1
        # temp_ds = 0.1

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

    batch_size = 4
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

    # ========================================= datasets & dataloaders
    # ========== training dataset
    # mega_dataset1 = MegaDepthDataset(root=mega_dir, train=True, using_cache=debug, pairs_per_scene=100,
    #                                  image_size=image_size, gray=False, colorjit=True, crop_or_scale='crop')
    # mega_dataset2 = MegaDepthDataset(root=mega_dir, train=True, using_cache=debug, pairs_per_scene=100,
    #                                  image_size=image_size, gray=False, colorjit=True, crop_or_scale='scale')

    # mega_dataset1 = MegaDepthDataset(root=mega_dir, train=True, using_cache=True, pairs_per_scene=100,
    #                                  image_size=image_size, gray=False, colorjit=True, crop_or_scale='crop')
    # mega_dataset2 = MegaDepthDataset(root=mega_dir, train=True, using_cache=True, pairs_per_scene=100,
    #                                  image_size=image_size, gray=False, colorjit=True, crop_or_scale='scale')
    train_datasets = MagicDataset(config)
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, pin_memory=debug,
                              num_workers=num_workers)
    # ========== evaluation dataset
    hpatch_i_dataset = HPatchesDataset(root=hpatch_dir, alteration='i',mode="train")
    hpatch_v_dataset = HPatchesDataset(root=hpatch_dir, alteration='v',mode="train")
    hpatch_i_dataloader = DataLoader(hpatch_i_dataset, batch_size=batch_size, pin_memory=not debug, num_workers=1)
    hpatch_v_dataloader = DataLoader(hpatch_v_dataset, batch_size=batch_size, pin_memory=not debug, num_workers=1)
    test_datasets = MagicDataset(config,test_mode=True)
    test_loader  = DataLoader(test_datasets,batch_size=1, shuffle=False,num_workers=1)
    # imw2020val = MegaDepthDataset(root=imw2020val_dir, train=False, using_cache=True, colorjit=False, gray=False)
    # imw2020val_dataloader = DataLoader(imw2020val, batch_size=1, pin_memory=not debug, num_workers=num_workers)

    # ========================================= logger
    log_name = 'debug' if debug else 'train'
    version = time.strftime("Version-%m%d-%H%M%S", time.localtime())

    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=log_dir, name=log_name, version=version, default_hp_metric=False)
    logging.info(f'>>>>>>>>>>>>>>>>> log dir: {logger.log_dir}')

    # ========================================= trainer
    trainer = pl.Trainer(
        # accelerator='cpu',
                        devices=gpus,
                         # resume_from_checkpoint='/mnt/data/zxm/document/ALIKE/training/log_train/train/Version-0715-191154/checkpoints/last.ckpt',
                         # resume_from_checkpoint='/mnt/data/zxm/document/ALIKE/training/log_train/train/Version-0702-195918/checkpoints/last.ckpt',
                         fast_dev_run=False,
                         accumulate_grad_batches=accumulate_grad_batches,
                         num_sanity_val_steps=num_sanity_val_steps,
                         limit_train_batches=limit_train_batches,
                         limit_val_batches=limit_val_batches,
                         max_epochs=max_epochs,
                         logger=logger,
                         reload_dataloaders_every_n_epochs=reload_dataloaders_every_epoch,
                         callbacks=[
                             ModelCheckpoint(monitor='val_metrics/mean', save_top_k=3,
                                             mode='max', save_last=True,
                                             dirpath=logger.log_dir + '/checkpoints',
                                             auto_insert_metric_name=False,
                                             filename='epoch={epoch}-mean_metric={val_metrics/mean:.4f}'),
                             LearningRateMonitor(logging_interval='step'),
                             RebuildDatasetCallback()
                         ]
                         )

    #使用hpatch的数据训练
    trainer.fit(model, train_dataloaders=[hpatch_i_dataloader,hpatch_v_dataloader],
                val_dataloaders=test_loader)
    
    #使用生成的图形数据集的数据训练
    # trainer.fit(model, train_dataloaders=train_loader,
    #             val_dataloaders=[hpatch_i_dataloader])
