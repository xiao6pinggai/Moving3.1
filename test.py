from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.utils1.opts import opts
# from lib.utils1.opts_HiEUM import opts
# from lib.utils1.opts_net1only import opts
import torch
import os

from lib.test_utils.test_0405 import test
from lib.test_utils.test_update import test_update

if __name__ == '__main__':

    opt = opts().parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    split = 'test'
    show_flag = opt.show_results
    savemat = True

    # opt.save_dir = opt.save_dir + '/' + opt.datasetname+'_'+opt.data_mode
    # # './weights' / 'rs_car_new _ multi'
    # if (not os.path.exists(opt.save_dir)):
    #     os.mkdir(opt.save_dir)
    # opt.save_dir = opt.save_dir + '/' + opt.model_name
    # # './weights' / 'rs_car_new _ multi' / 'I2PSOD'
    # if (not os.path.exists(opt.save_dir)):
    #     os.mkdir(opt.save_dir)
    # # opt.save_results_dir = opt.save_dir + '/results'
    # opt.save_results_dir = opt.save_dir # './weights' / 'rs_car_new _ multi' / 'I2PSOD' 

    # if (not os.path.exists(opt.save_results_dir)): # 去掉'results'后不会执行
    #     os.mkdir(opt.save_results_dir)
#     opt.load_model = './weights/yq_test/xc_tpaimi_model_best.pth'
    modelPath = opt.load_model # weights/rs_car_new_multi/I2PSOD/I2PSOD_supMode_0_seglen5_weights2025_12_16_11_37_30/model_best.pth
    print(modelPath)
    opt.save_results_dir = opt.save_dir + '/' + opt.datasetname+'_'+opt.data_mode + '/'  + opt.model_name + '/' + modelPath.split('/')[-2] + '/' 
    os.makedirs(opt.save_results_dir, exist_ok=True)
    results_name = 'results_'+modelPath.split('/')[-1].split('.')[0]
    print(f"results will be saved in {opt.save_results_dir}{results_name}")
    if (not os.path.exists(opt.save_results_dir)): # 去掉'results'后不会执行
            os.mkdir(opt.save_results_dir)
    results_return = test(opt, split, modelPath, show_flag, results_name, savemat)
