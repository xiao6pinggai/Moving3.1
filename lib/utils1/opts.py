from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys, json
import ast
from datetime import datetime
import platform  # 导入系统识别模块


def _parse_3d_kernel_list(value):
    if isinstance(value, list):
        return value
    return ast.literal_eval(value)

def _parse_int_list(value):
    if isinstance(value, list):
        tokens = value
    else:
        text = value.strip()
        if text.startswith('[') or text.startswith('('):
            tokens = ast.literal_eval(text)
        else:
            tokens = [token.strip() for token in text.split(',') if token.strip()]
    if not isinstance(tokens, (list, tuple)):
        raise argparse.ArgumentTypeError('tpdilation must be a list of integers')
    return [int(token) for token in tokens]

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--discribe', default='xrsy_UNet3D_t5') 
        self.parser.add_argument('--task', default='ctdet_points',
                                 help='task name.  ctdet_points |  ctdet ')
        self.parser.add_argument('--exp_name',default='xrsy_UNet3D_t5',# 'unsupervised_iterative_layers_3_', # I2PSOD # # 修改2
                                 help='name of the experiments.')
        self.parser.add_argument('--layers', type=float, default=3.61, help='use decomp model or not.')  # 默认是3        
        self.parser.add_argument('--model_name', default='UNet3D', help='name of the model.') # sp_centerDet_minus # LightweightUnet3DDynamic # I2PSOD # Net1 # I2PSOD_test # 修改3 # Net1_SpDetHead # Net1_Net2  # UNet3D
        self.parser.add_argument('--load_model', default= "")
        self.parser.add_argument('--resume', type=bool, default=True, help='resume an experiment.')
        self.parser.add_argument('--down_ratio', type=int, default=1, help='output stride. Currently only supports for 1.')
        # system
        self.parser.add_argument('--gpus', default='0',help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=12, help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed', type=int, default=317,  help='random seed')  # from CornerNet 

        # train
        self.parser.add_argument('--lr', type=float, default=1e-3/4*4, # 1.25e-3/6 *4, # 1.25e-4,HiEUM with bs=6 # 1.5e-4 /8 *12 # 修改4
                                 help='learning rate for batch size 4.')
        self.parser.add_argument('--lr_step', type=str, default='10,25,35', #30,45
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=45,  #55
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=4, # 6 # 修改5
                                 help='batch size')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--seqLen', type=int, default=5, help='number of images for per sample. Currently supports 5.')
        self.parser.add_argument('--thresh', type=float, default=3, help='var_coeff for select points.')

        # test
        self.parser.add_argument('--nms', action='store_true', help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=64, help='max number of output objects. top_k')
        self.parser.add_argument('--test_large_size', type=bool, default=False, help='whether or not to test image size of 1024. Only for test.')
        self.parser.add_argument('--show_results', type=bool, default=False, help='whether or not to show the detection results. Only for test.')
        self.parser.add_argument('--save_track_results', type=bool, default=False, help='whether or not to save the tracking results of sort. Only for testTrackingSort.')
        self.parser.add_argument('--metric', type=json.loads, default={'inference':True, # 修改 7
                                                                'run_ap':True, 'run_f1':True,
                                                                'save_json':True, 'save_mat':False,
                                                                'f1_source':'json', 'f1_mode':['iou'],
                                                                'eval_splits':['all','real','sim'], 'best_metric':'ap50'},
                                 help='how to test')
        self.parser.add_argument('--run_test_after_train', action='store_true', help='run test.py with the current best checkpoint after training')

        # save
        self.parser.add_argument('--save_dir', type=str, default='./weights',
                                 help='savepath of model.')

        # dataset
        self.parser.add_argument('--data_mode', type=str, default='multi',
                                 help='dataset name.')
        self.parser.add_argument('--datasetname', type=str, default='aircraft', # rs_car_new # aircraft # rs_car_new # sdm_car # mir  修改8
                                 help='dataset name.')
       
        self.parser.add_argument('--multi2multi', type=bool, default=True, help='multi inputs and multi outputs, this can impact the last patch for test.')

        #update_label
        self.parser.add_argument('--sup_mode', type=int, default=0,  # 0 for 真值; 1 for 差分; 2 for sort+差分过滤; 3 for 更新标签 # 修改10
                                 help='supervion mode.0 for annotated labels.|  1 for unfilt generated labels. | 2 for filt generated labels. | 3 for updated generated labels.')
        # self.parser.add_argument('--unsup_iter', type=int, default=10,help='unsup iteration interval.')
        # self.parser.add_argument('--conf_filtered', type=float, default=0.2, help='conf_filtered.')

        #loss
        self.parser.add_argument('--hm_flag', type=bool, default=True, help='offset brantch.')
        self.parser.add_argument('--hm_weight', type=float, default=1.0, help='wh weight in loss.')
        
        self.parser.add_argument('--wh_flag', type=bool, default=True, help='offset brantch.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1, help='wh weight in loss.')
        
        self.parser.add_argument('--off_flag', type=bool, default=True, help='offset brantch.')
        self.parser.add_argument('--off_weight', type=float, default=1.0,  help='offset weight in loss.')
        
        self.parser.add_argument('--hm_large_heatmap_flag', type=bool, default=False, help='hm_large_heatmap_brantch.') # True for SGNet
        self.parser.add_argument('--hm_large_heatmap_weight', type=float, default=1.0, help='hm_large_heatmap weight in loss.')
        
        # two stage training
        self.parser.add_argument('--two_stages', type=bool, default=False, help='if true, will train the model in two stages.')
        self.parser.add_argument('--stage1_epochs', type=int, default=10, help='number of epochs for stage 1 training.')
        
        # net1 define
        self.parser.add_argument('--feat_channels', type=_parse_int_list, default=[16,32,64,128], help='unet upsample channels') # [8,16,32,64] for Unet3DWithNormalConv3D # [8,16,32,16,8] for LightWeightedConv3D # [16,16] # [8,8,8,8] for SGNet
        self.parser.add_argument('--T_pooling', type=bool, default=False,  help='is pooling t dim or not') # False for SGNet
        self.parser.add_argument('--groups', type=int, default=-1, help='net1 conv groups(must be feat_channels % == 0)')
        self.parser.add_argument('--downsample_mode', type=str, default='stride', help='downsample mode "stride" or "maxpool"') # maxpool for SGNet
        self.parser.add_argument('--upsample_mode', type=str, default='deconv', help='upsample mode "deconv" or "trilinear"')
        self.parser.add_argument('--Snack_skip', type=_parse_int_list, default=[1,1,1], help='temporal snake conv switches for [skip1, skip2, skip3], e.g. [1,0,1]')
        self.parser.add_argument('--Snack_max_offset', type=_parse_int_list, default=[10,10,10], help='max single-step temporal snake offset scope for [skip1, skip2, skip3]')
        self.parser.add_argument('--Snack_repeat', type=int, default=1, help='number of sequential Snack blocks per enabled skip; 1 keeps the original single Snack')
        self.parser.add_argument('--TKernel', type=_parse_3d_kernel_list, default=[[5,1,1],[5,1,1],[5,1,1]], help='legacy 3D kernel sizes for [skip1, skip2, skip3]')
        self.parser.add_argument('--OffsetKernel', type=_parse_3d_kernel_list, default=[[5,1,1],[5,1,1],[5,1,1]], help='offset-generator kernels for [skip1, skip2, skip3]')
        self.parser.add_argument('--VSnack_residual', type=int, default=1, choices=[0,1], help='use unconstrained residual increments in v_snack: 1 enables, 0 uses velocity-only offsets')
        self.parser.add_argument('--TZSConv_skip', type=_parse_int_list, default=[0,0,0], help='temporal zero-sum conv switches before UNet3D skip blocks for [skip1, skip2, skip3]')
        self.parser.add_argument('--Skip_TMixer', type=_parse_int_list, default=[0,0,0], help='temporal mixer switches after Snack_skip for [skip1, skip2, skip3]')
        self.parser.add_argument('--TMixer_skip', type=str, default='TMixer', choices=['TMixer', 'TMixer_attn', 'TMixer_ca', 'Multi_TMixer', 'Multi_TConv', 'Multi_TConv_LinearTT', 'TMixer_GL', 'TMixer_GLinear_Ldpconv', 'TMixer_GL_group2', 'TMixer_GL_groupT', 'TMixer_GL_Dilation1234', 'TMixer_GL_BNRelu', 'TMixer_GL_test', 'TMixer_Lonly', 'TMixer_STP'], help='temporal mixer type: TMixer uses TPro-style multi-head temporal projections, TMixer_attn uses pointwise temporal self-attention, TMixer_ca uses global-temporal cross-attention at each point, Multi_TMixer fuses global and stride-two temporal mixing, Multi_TConv fuses parallel temporal convolutions with SE, Multi_TConv_LinearTT applies channel-shuffle grouped temporal convolutions then BN+ReLU+Linear(T,T), TMixer_GL fuses Linear(T,T) global mixing with grouped k=1/3/5/7 local temporal convolutions, TMixer_GLinear_Ldpconv adds depthwise spatial 3x3 evidence to global Linear(T,T), TMixer_GL_group2 splits channels between Linear(T,T) and frame-local Linear(1,1) then fuses them; TMixer_GL_groupT uses configurable grouped local kernels with remainder channels assigned to the first group; TMixer_GL_Dilation1234 uses grouped k=3 temporal convolutions with dilation 1,2,3,4; TMixer_GL_BNRelu adds output BN+ReLU to the no-pre-group-shuffle GL mixer, TMixer_GL_test is the no-pre-group-shuffle global-local mixer, TMixer_Lonly keeps only the grouped local temporal pyramid, TMixer_STP fuses symmetric grouped spatial and temporal k=1/3/5/7 convolution pyramids')
        self.parser.add_argument('--TMixer_groupT_kernels', type=_parse_int_list, default=[1,3,5,7], help='local temporal kernels for TMixer_GL_groupT and Multi_TConv, e.g. [1,3,5]')
        self.parser.add_argument('--TMixer_num_heads', type=int, default=8, help='number of temporal projection heads used when TMixer_skip=TMixer')
        self.parser.add_argument('--GD_skip', type=_parse_int_list, default=[0,0,0], help='grouped spatial dilated-convolution switches after TMixer for [skip1, skip2, skip3]')
        self.parser.add_argument('--UNet3D_skip', type=str, default='snack', choices=['snack', 'snack_unrest', 'v_snack', 'tconv', 'tdcn'], help='UNet3D skip block type: snack for temporal snake conv, snack_unrest for raw-offset temporal snake conv, v_snack for velocity-prior snake conv, tconv for temporal Conv3d, tdcn for temporal deform Conv3d')

        """for SGNet"""
        """
        self.parser.add_argument('--net1name', type=str, default='Unet3', help='encoder use ATDC, supports TOSConvNet/TPConvNet/DynamicTOSConvNet/TZSConvNet/TZSconvNet/Unet3/Unet2') # LightWeightedConv3D  # UNet3DWithNormalConv3D # TOSConvNet  # TZSConvNet # 修改11
        self.parser.add_argument('--use_tzsconv', type=str, default='', help='temporal branch mode: "tzsconv" keeps the current branch, "tmf" uses temporal median filtering in TOSConvNet, "" disables it')
        self.parser.add_argument('--tpdilation', type=_parse_int_list, default=[1], help='TPConvNet temporal pyramid dilations, e.g. [1,2,3,4]')
        self.parser.add_argument('--tprepeat', type=int, default=3, help='repeat count of the TPConvNet temporal pyramid block')
        self.parser.add_argument('--MFE', type=str, default="cosv23")
        self.parser.add_argument('--bottle_enhancement', type=str, default='block', help='bottleneck enhancement: None removes the middle block, block keeps the original conv block, MFE names such as cosv18 use the corresponding MFE module, MFEw2Block uses block-MFE-block at bottleneck')
        self.parser.add_argument('--MFE_skip', type=str, default='[1,1,1]', help='MFE skip switches for [conv1, conv2, conv3], e.g. [1,0,1]')
        self.parser.add_argument('--MFErepeat', type=str, default='[0,0,0]', help='cosv22 repeat count per [conv1, conv2, conv3]; single value is broadcast')
        self.parser.add_argument('--topk_relu', type=str, default='fanghui', choices=['fanghui', 'bufanghui'], help='triplet top-k matching rule: fanghui allows point reuse, bufanghui forbids reuse')
        self.parser.add_argument('--tmc_topk', type=str, default='[3,2,1]', help='cosv18/v20/v21/v23/v24/v25/v26/v27/v28/CA top-k per [conv1, conv2, conv3]; single value is broadcast')
        self.parser.add_argument('--tmc_window_size', type=str, default='[15,11,7]', help='cosv18/v20/v21/v23/v24/v25/v26/v27/v28/CA window size per [conv1, conv2, conv3]; single value is broadcast')
        self.parser.add_argument('--tmc_hidden_ratio', type=float, default=0.5, help='cosv18/v20/v21/v23/v24/v25/v26/v27/v28/CA FFN hidden channel ratio')
        self.parser.add_argument('--tmc_pos_hidden', type=int, default=16, help='cosv18/v21 position MLP hidden channels; v20 uses 0.5*C')
        self.parser.add_argument('--tmc_pos_scale', type=float, default=16.0, help='cosv18/v20/v21/v23/v24/v25/v26/v27/v28/CA coordinate normalization scale')
        self.parser.add_argument('--tmc_chunk_size', type=int, default=999999999, help='cosv18/v20/v21/v23/v24/v25/v26/v27/v28/CA query chunk size; <=0 uses adaptive large-block chunking')
        self.parser.add_argument('--tmc_ffn_position', type=str, default='before_mean', choices=['before_mean', 'after_mean'], help='cosv18/v21 FFN placement: before_mean keeps per-pair FFN; after_mean applies FFN after pair averaging')
        """
        """for SGNet"""
        #可视化 # cosv10 
        self.parser.add_argument('--vis_features', type=bool, default=False, help='whether to visualize feature maps')
        self.parser.add_argument('--vis_mode', type=str, default='mean', help='feature map aggregation mode: mean / max / channel index (e.g. 0 1 2)')
        self.parser.add_argument('--vis_layers', type=str, nargs='+', default=['model.I2PNet.final_conv','model.sp_backbone.conv1','model.sp_backbone.shortcut1','model.sp_backbone.shortcut1fusion'], help='layer names for capturing feature maps')
        self.parser.add_argument('--vis_max_frames', type=int, default=10, help='max frames to save per video for feature visualization')


    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        # opt.data_sampling = int(opt.seqLen//2)
        """ # for SGNet
        if opt.seqLen == 1:
            opt.data_sampling = 1
            opt.batch_size= 8
        if opt.seqLen == 3:
            opt.data_sampling = 3
            opt.batch_size= 8
        if opt.seqLen == 5:
            opt.data_sampling = 5
            opt.batch_size= 8
        if opt.seqLen == 10:
            opt.data_sampling = 10
            opt.batch_size= 8  # 默认8
        elif opt.seqLen == 15:
            opt.data_sampling = 15
            opt.batch_size= 8
        elif opt.seqLen == 20:
            opt.data_sampling = 20
            opt.batch_size= 8
        elif opt.seqLen == 40:
            opt.data_sampling = 40
            opt.batch_size= 4
        if opt.model_name == 'Net1_Net2':
                    opt.batch_size= 4
        """
        if opt.seqLen == 1:
            opt.data_sampling = 1
            # opt.batch_size= 4
        if opt.seqLen == 3:
            opt.data_sampling = 3
            # opt.batch_size= 4
        if opt.seqLen == 5:
            opt.data_sampling = 5
            # opt.batch_size= 4
        if opt.seqLen == 10:
            opt.data_sampling = 10
            # opt.batch_size= 4
        elif opt.seqLen == 15:
            opt.data_sampling = 15
            # opt.batch_size= 4
        elif opt.seqLen == 20:
            opt.data_sampling = 20
            # opt.batch_size= 4
        elif opt.seqLen == 40:
            opt.data_sampling = 40
            # opt.batch_size= 4

        if opt.datasetname == 'rs_car_new':
            opt.data_dir = '/root/autodl-tmp/RsCarData_New_Part/'
            # opt.data_dir = 'E:/NUDT-Master/Academic/DataSet/RsCarData_New_Part/'
            opt.xmlname = 'xml1new'
        elif opt.datasetname == 'sdm_car':
            opt.data_dir = '/root/autodl-tmp/SDM-Car-New/'
            opt.xmlname = 'xml1'
            # opt.batch_size= 4
        elif opt.datasetname == 'aircraft':
            opt.data_dir = '/root/autodl-tmp/AircraftDataset27/'
            opt.xmlname = 'xml1'
        opt.lr = opt.lr/4*opt.batch_size
        try:
            if isinstance(opt.bottle_enhancement, str) and opt.bottle_enhancement.strip().lower() in ('', 'none', 'null'):
                opt.bottle_enhancement = None
        except Exception as e:
            pass
        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.dataName = opt.data_dir.split('/')[-2]
        platform_name = platform.system()
        if platform_name == 'Windows':
            opt.num_workers = 0  # Windows系统下多线程可能会导致问题，设置为0以使用单线程
        if opt.datasetname == 'sdm_car':
            opt.K = 360
        elif opt.datasetname == 'aircraft':
            opt.K = 64
        elif opt.datasetname == 'rs_car_new':
            opt.K = 128

        return opt
