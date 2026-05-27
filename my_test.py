from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 强制同步CUDA，定位具体出错的行
os.environ['TORCH_USE_CUDA_DSA'] = "1"     # 打印CUDA断言详细信息

import warnings  # <--- 新增
import logging   # <--- 新增

# --- 在这里插入屏蔽代码 ---
warnings.filterwarnings("ignore", message=".*loadtxt: input contained no data.*")
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
# ------------------------


import torch
import torch.utils.data
from lib.utils1.opts import opts
from lib.utils1.logger import Logger
from datetime import datetime

from lib.models.stNet import get_det_net,load_model, save_model
from lib.dataset.dataset_factory import get_dataset
from lib.Trainer.trainer_factory import get_trainer

from lib.LRSD.filter_lrsd import generate_labels

def main(opt):
    torch.manual_seed(opt.seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    ###
    now = datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    opt.save_dir = opt.save_dir + '/' + opt.datasetname+'_'+opt.data_mode

    if (not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)

    opt.save_dir = opt.save_dir + '/' + opt.model_name

    if (not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)

    opt.save_results_dir = opt.save_dir + '/results'

    model_path_name = opt.exp_name +'_supMode_%d' % opt.sup_mode + '_seglen%d_' % opt.seqLen + 'weights' + time_str

    opt.save_dir = opt.save_dir + '/'+model_path_name
    opt.save_log_dir = opt.save_dir


    if False:
        # 引入 Subset
        from torch.utils.data import Subset

        val_intervals = opt.val_intervals
        dataset = get_dataset(opt)

        # ================= 验证集修改 =================
        _DataValRaw = dataset(opt, 'test')
    
        DataVal = torch.utils.data.Subset(_DataValRaw, indices=range(min(len(_DataValRaw), 13)))
        
        # 属性注入
        DataVal.num_classes = _DataValRaw.num_classes
        DataVal.coco = _DataValRaw.coco
        DataVal.class_names = getattr(_DataValRaw, 'class_names', None) # 防御性编程，如果有就复制
        DataVal.resolution = _DataValRaw.resolution
        DataVal.run_eval = _DataValRaw.run_eval
        # ===================================================
        val_loader = torch.utils.data.DataLoader(
            DataVal,
            batch_size=1,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
        )

        # ================= 训练集修改 =================
        # 1. 实例化原始数据集
        _DataTrainRaw = dataset(opt, 'train') 
        # 3. 创建 Subset
        DataTrain = torch.utils.data.Subset(_DataTrainRaw, indices=range(min(len(_DataTrainRaw), 20)))
        DataTrain.num_classes = _DataTrainRaw.num_classes
        DataTrain.coco = _DataTrainRaw.coco
        DataTrain.class_names = getattr(_DataTrainRaw, 'class_names', None) # 防御性编程，如果有就复制
        DataTrain.resolution = _DataTrainRaw.resolution
        # ===================================================
        train_loader = torch.utils.data.DataLoader(
            DataTrain,
            batch_size=opt.batch_size, # 注意：如果batch_size > 20，这里可能需要改为更小的值
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True, # 注意：如果20张不够一个batch且drop_last=True，可能会报错或为空
        )
        base_s = DataTrain.coco
        print(f'len(DataTrain)={len(DataTrain)}-{len(train_loader)}, len(DataVal)={len(DataVal)}-{len(val_loader)}')
#     #####################################################################
    else:
        val_intervals = opt.val_intervals

        dataset = get_dataset(opt)


        DataVal = dataset(opt, 'test')
        # print('DataVal.num_classes:', DataVal.num_classes)
        val_loader = torch.utils.data.DataLoader(
            DataVal,
            batch_size=1,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            # persistent_workers=True, # 核心调整点
            # prefetch_factor=4        # 核心调整点
        )

        DataTrain = dataset(opt, 'train')

        base_s = DataTrain.coco

        train_loader = torch.utils.data.DataLoader(
            DataTrain,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
            # persistent_workers=True, # 核心调整点
            # prefetch_factor=4        # 核心调整点
        )

    print('Creating model...')
    if opt.off_flag:
        head = {'hm': DataTrain.num_classes, 'wh': 2, 'reg': 2}
    else:
        head = {'hm': DataTrain.num_classes, 'wh': 2}
    model = get_det_net(head, opt.model_name, DataVal.resolution, opt.seqLen, opt)  # 建立模型 # DataTrain.resolution决定改模型的输入必须固定


    print(head)
    print(opt.model_name)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)  #设置优化器

    start_epoch = 0

    if(not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)

    if(not os.path.exists(opt.save_results_dir)):
        os.mkdir(opt.save_results_dir)

    logger = Logger(opt)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)  # 导入训练好的模型

    trainer = get_trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.device)

    print('Starting training...')

    best = -1

    with torch.no_grad():
        log_dict_val, preds, stats = trainer.val(1, val_loader, base_s, DataVal, None, model_path_name)
    logger.write('eval results: ')
    for k, v in log_dict_val.items():
        logger.write('{} {:8f} | '.format(k, v))
    if log_dict_val['ap50'] > best:
        best = log_dict_val['ap50']
        print(f"best:{log_dict_val['ap50']}")
    logger.write('\n')
    logger.close()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)