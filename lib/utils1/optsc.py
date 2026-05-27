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
        self.parser.add_argument('--discribe', default='[ds5_conv2d_d2_+consine+GDfusion_v9]317反比增强，alpha=0.5，随机数种子317,cosmatch设置dialtion=2，不填充0，I2PSOD 10帧 net1三层次下采样T保持，全部为标准卷积,net1损失修改为:hm半径为HW,sigma/3,kt=3,均包含瓶颈层,融合简化，减少中间层,net1outonly') # 修改1 # enc修改为二分支空洞时空解耦，空间保留dilation,CDC，时间2 3 dilation,去掉0sum为正常时域3*3*3卷积，取消保底branch的1*1，时间后bnrelu conv1*1*1,然后和空间先cat再1*1*1降维输出  增加了归一化以及snext与sprev相乘，算得分，输出使用biqkv且不做v交互，qk矩阵共享，
        self.parser.add_argument('--task', default='ctdet_points',
                                 help='task name.  ctdet_points |  ctdet ')
        self.parser.add_argument('--exp_name', default='xrsy_k3_ds5_conv2d211+consine+GDfusion3d_v9_seqlen10_UNet3DWithNormalConv3D_net1outonly', # 'unsupervised_iterative_layers_3_', # I2PSOD # # 修改2
                                 help='name of the experiments.')
        self.parser.add_argument('--layers', type=float, default=3.61, help='use decomp model or not.')  # 默认是3
        self.parser.add_argument('--model_name', default='I2PSOD', help='name of the model.') # sp_centerDet_minus # LightweightUnet3DDynamic # I2PSOD # 修改3
        self.parser.add_argument('--load_model', default= '',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', type=bool, default=False, help='resume an experiment.')
        self.parser.add_argument('--down_ratio', type=int, default=1, help='output stride. Currently only supports for 1.')
        # system
        self.parser.add_argument('--gpus', default='0',help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=12, help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed', type=int, default=317,  help='random seed')  # from CornerNet 

        # train
        self.parser.add_argument('--lr', type=float, default=1e-3/4*8, # 1.25e-3/6 *4, # 1.25e-4,HiEUM with bs=6 # 1.5e-4 /8 *12 # 修改4
                                 help='learning rate for batch size 4.')
        # self.parser.add_argument('--lr_net1', type=float, default=1.25e-2/6 *4, # 1.25e-4,HiEUM with bs=6 # 1.5e-4 /8 *12 # 修改4
        #                          help='learning rate for batch size 4.')
        # self.parser.add_argument('--warmup_epochs', type=int, default=10, # 1.25e-4,HiEUM with bs=6 # 1.5e-4 /8 *12 # 修改4
        #                          help='learning rate for batch size 4.')
        self.parser.add_argument('--lr_step', type=str, default='10,25,35', #30,45
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=45,  #55
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=8, # 6 # 修改5
                                 help='batch size')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--seqLen', type=int, default=10, help='number of images for per sample. Currently supports 5.')
        self.parser.add_argument('--thresh', type=float, default=3, help='var_coeff for select points.')

        # test
        self.parser.add_argument('--nms', action='store_true', help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=128, help='max number of output objects. top_k')
        self.parser.add_argument('--test_large_size', type=bool, default=False, help='whether or not to test image size of 1024. Only for test.')
        self.parser.add_argument('--show_results', type=bool, default=False, help='whether or not to show the detection results. Only for test.')
        self.parser.add_argument('--save_track_results', type=bool, default=False, help='whether or not to save the tracking results of sort. Only for testTrackingSort.')
        self.parser.add_argument('--metric', type=int, default={'inference':True, # 修改 7
                                                                'run_ap':False, 'run_f1':True,
                                                                'save_json':True, 'save_mat':True, 'f1_mode':['iou','dis']},
                                 help='how to test')

        # save
        self.parser.add_argument('--save_dir', type=str, default='./weights',
                                 help='savepath of model.')

        # dataset
        self.parser.add_argument('--data_mode', type=str, default='multi',
                                 help='dataset name.')
        self.parser.add_argument('--datasetname', type=str, default='sdm_car', # rs_car # aircraft # rs_car_new # sdm_car # mir  修改8
                                 help='dataset name.')
        self.parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/SDM-Car-New/',  #/root/autodl-tmp/RsCarData/', # /root/autodl-tmp/AircraftDataset23/
                                 # 注意以 / 结尾
                                 #r'E:\NUDT-Master\Academic\DataSet\RsCarData/',
                                 # '../../20250919SimData/New3/AircraftDataset22/',
                                 # '/media/wellwork/L/xc/datasets/RsCarData/',
                                 # '/root/autodl-tmp/AircraftDataset23/',
                                 # '/root/autodl-tmp/RsCarData/'
                                 # '/root/autodl-tmp/RsCarData_New_Part/'
                                 # r'E:/NUDT-Master/Academic/DataSet/RsCarData_New_Part/'
                                 # '/root/autodl-tmp/SDM-Car-New/'
                                 # '/root/autodl-tmp/NUDT-MIRSDT-New/'
                                 # r'E:/NUDT-Master/Academic/DataSet/NUDT-MIRSDT-New/'
                                 help='path of dataset.')
        self.parser.add_argument('--xmlname', type=str, default='xml1', # xml_det --> old # xml1new --> new # xml1-->sdm-car # 修改9
                                 help='xml_det --> old / xml1new --> new') # aircraft sdm_car不受影响，其默认xml1
        self.parser.add_argument('--data_sampling', type=int, default=5, # 初始为1，表示使用全部数据；改为20表示每20帧采样1帧 
                                 help='data_sampling.')
        self.parser.add_argument('--multi2multi', type=bool, default=True, help='multi inputs and multi outputs, this can impact the last patch for test.')

        #update_label
        self.parser.add_argument('--sup_mode', type=int, default=0,  # 0 for 真值; 1 for 差分; 2 for sort+差分过滤; 3 for 更新标签 # 修改10
                                 help='supervion mode.0 for annotated labels.|  1 for unfilt generated labels. | 2 for filt generated labels. | 3 for updated generated labels.')
        self.parser.add_argument('--unsup_iter', type=int, default=10,help='unsup iteration interval.')
        self.parser.add_argument('--conf_filtered', type=float, default=0.2, help='conf_filtered.')

        #loss
        self.parser.add_argument('--hm_flag', type=bool, default=True, help='offset brantch.')
        self.parser.add_argument('--hm_weight', type=float, default=1.0, help='wh weight in loss.')
        
        self.parser.add_argument('--wh_flag', type=bool, default=True, help='offset brantch.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1, help='wh weight in loss.')
        
        self.parser.add_argument('--off_flag', type=bool, default=True, help='offset brantch.')
        self.parser.add_argument('--off_weight', type=float, default=1.0,  help='offset weight in loss.')
        
        self.parser.add_argument('--hm_large_heatmap_flag', type=bool, default=True, help='hm_large_heatmap_brantch.')
        self.parser.add_argument('--hm_large_heatmap_weight', type=float, default=1.0, help='hm_large_heatmap weight in loss.')
        
        # two stage training
        self.parser.add_argument('--two_stages', type=bool, default=True, help='if true, will train the model in two stages.')
        self.parser.add_argument('--stage1_epochs', type=int, default=10, help='number of epochs for stage 1 training.')
        
        # net1 define
        self.parser.add_argument('--feat_channels', type=list, default=[8,16,32,64], help='unet upsample channels')
        self.parser.add_argument('--input_channels', type=int, default=1, help='how many channles feature input net2')
        self.parser.add_argument('--T_pooling', type=bool, default=False,  help='is pooling t dim or not')
        self.parser.add_argument('--groups', type=int, default=-1, help='net1 conv groups(must be feat_channels % == 0)')
        self.parser.add_argument('--downsample_mode', type=str, default='maxpool', help='downsample mode "stride" or "maxpool"')
        self.parser.add_argument('--net1name', type=str, default='UNet3DWithNormalConv3D', help='encoder use ATDC')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        opt.data_sampling = int(opt.seqLen//2)
        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.dataName = opt.data_dir.split('/')[-2]
        platform_name = platform.system()
        if platform_name == 'Windows':
            opt.num_workers = 0  # Windows系统下多线程可能会导致问题，设置为0以使用单线程
        if opt.datasetname == 'sdm_car':
            opt.K = 360

        return opt
