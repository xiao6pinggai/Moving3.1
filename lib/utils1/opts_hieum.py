from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from datetime import datetime
import platform  # 导入系统识别模块
class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--discribe', default='用旧训练集的有监督训练结果作为无监督训练的预训练权重 best 45')
        self.parser.add_argument('--task', default='ctdet_points',
                                 help='task name.  ctdet_points |  ctdet ')
        self.parser.add_argument('--exp_name', default='unsupervised_iterative_layers_3_', # 'unsupervised_iterative_layers_3_', # I2PSOD
                                 help='name of the experiments.')
        self.parser.add_argument('--layers', type=int, default=3, # 默认是3
                                 help='use decomp model or not.')
        self.parser.add_argument('--model_name', default='sp_centerDet_minus', # sp_centerDet_minus # LightweightUnet3DDynamic # I2PSOD
                                 help='name of the model.')
        self.parser.add_argument('--load_model', default='./weights/rs_car_multi/unsupervised_iterative_layers_3__supMode_0_seglen20_weights2025_12_09_11_01_56/model_best.pth',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', type=bool, default=False,
                                 help='resume an experiment.')
        self.parser.add_argument('--down_ratio', type=int, default=1,
                                 help='output stride. Currently only supports for 1.')
        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=12, #4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4 /6 *16, # 1.25e-4,
                                 help='learning rate for batch size 4.')
        self.parser.add_argument('--lr_step', type=str, default='30,45', #30,45
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=55,  #55
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=16, # 6
                                 help='batch size')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--seqLen', type=int, default=20,
                                 help='number of images for per sample. Currently supports 5.')
        self.parser.add_argument('--thresh', type=float, default=3,
                                 help='var_coeff for select points.')

        # test
        self.parser.add_argument('--nms', action='store_true',
                                 help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=128,
                                 help='max number of output objects. top_k')
        self.parser.add_argument('--test_large_size', type=bool, default=False,
                                 help='whether or not to test image size of 1024. Only for test.')
        self.parser.add_argument('--show_results', type=bool, default=False,
                                 help='whether or not to show the detection results. Only for test.')
        self.parser.add_argument('--save_track_results', type=bool, default=False,
                                 help='whether or not to save the tracking results of sort. Only for testTrackingSort.')

        # save
        self.parser.add_argument('--save_dir', type=str, default='./weights',
                                 help='savepath of model.')

        # dataset
        self.parser.add_argument('--data_mode', type=str, default='multi',
                                 help='dataset name.')
        self.parser.add_argument('--datasetname', type=str, default='rs_car_new', # rs_car # aircraft
                                 help='dataset name.')
        self.parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/RsCarData_New_Part/',  #/root/autodl-tmp/RsCarData/', # /root/autodl-tmp/AircraftDataset23/
                                 # 注意以 / 结尾
                                 #r'E:\NUDT-Master\Academic\DataSet\RsCarData/',
                                 # '../../20250919SimData/New3/AircraftDataset22/',
                                 # '/media/wellwork/L/xc/datasets/RsCarData/',
                                 # '/root/autodl-tmp/AircraftDataset23/',
                                 # '/root/autodl-tmp/RsCarData/'
                                 help='path of dataset.')
        self.parser.add_argument('--data_sampling', type=int, default=5, # 初始为1，表示使用全部数据；改为20表示每20帧采样1帧
                                 help='data_sampling.')

        #update_label
        self.parser.add_argument('--sup_mode', type=int, default=3,  # 0 for 真值; 1 for 差分; 2 for sort+差分过滤; 3 for 更新标签
                                 help='supervion mode.0 for annotated labels.|  1 for unfilt generated labels. | 2 for filt generated labels. | 3 for updated generated labels.')
        self.parser.add_argument('--unsup_iter', type=int, default=10,
                                 help='unsup iteration interval.')
        self.parser.add_argument('--conf_filtered', type=float, default=0.2,
                                 help='conf_filtered.')

        #loss
        self.parser.add_argument('--hm_flag', type=bool, default=True,
                                 help='offset brantch.')
        self.parser.add_argument('--hm_weight', type=float, default=1.0,
                                 help='wh weight in loss.')
        
        self.parser.add_argument('--wh_flag', type=bool, default=True,
                                 help='offset brantch.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='wh weight in loss.')
        
        self.parser.add_argument('--off_flag', type=bool, default=True, # reg branch loss
                                 help='offset brantch.')
        self.parser.add_argument('--off_weight', type=float, default=1.0,
                                 help='offset weight in loss.')
        
        self.parser.add_argument('--hm_large_heatmap_flag', type=bool, default=False, 
                                 help='hm_large_heatmap_brantch.')
        self.parser.add_argument('--hm_large_heatmap_weight', type=float, default=1.0,
                                 help='hm_large_heatmap weight in loss.')
        
        # two stage training
        self.parser.add_argument('--two_stages', type=bool, default=False,
                                 help='if true, will train the model in two stages.')
        self.parser.add_argument('--stage1_epochs', type=int, default=10,
                                 help='number of epochs for stage 1 training.')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.dataName = opt.data_dir.split('/')[-2]
        platform_name = platform.system()
        if platform_name == 'Windows':
            opt.num_workers = 0  # Windows系统下多线程可能会导致问题，设置为0以使用单线程


        return opt
