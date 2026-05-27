import argparse
import numpy as np
import scipy.io as sio
import os,sys 

import xml.dom.minidom as doxml
ROOT_DIR = "/root/autodl-tmp/Moving3.1"
# 确保根目录在sys.path首位（覆盖默认的子目录）
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from lib.utils1.utils_eval import eval_metric

def eval_func_final(results_dir_tol, data_dir=None, data_name=None, conf_ths = None, write_flag = True, 
                    dis_ths = None, opt=None,eval_mode_metric=None,xmlname='xml_det'):
    #eval func
    eval_mode_metric = 'iou' if eval_mode_metric is None else eval_mode_metric # dis  # iou
    if dis_ths is not None:
        dis_th = dis_ths
    else:
        dis_th = [5]
    iou_th = [0.5]
    # conf_thresh_all = [0.2, 0.25, 0.3, 0.32, 0.34, 0.35]
    if conf_ths is not None:
        conf_thresh_all = conf_ths
    else:
        conf_thresh_all =  [0.2,0.25,0.3]#[0.2,0.25,0.3] # 不太耗时
    # if opt.datasetname == 'sdm_car':
    #     conf_thresh_all =  [0.05,0.1,0.15,0.2,0.25,0.3] # 不太耗时
    if data_name is None:
        if opt.datasetname == 'aircraft' or opt.datasetname == 'sdm_car' or opt.datasetname == 'mir':
            dataName = os.listdir(data_dir)
            dataName = [i for i in dataName if '.' not in i]  # lhg
            # dataName = dataName[:2]
        else: # rs_car
            dataName = [2,3,5,6,8,9,10]  #3,5,2,8,10,6,9 # 2,3,5,6,8,9,10
    else:
        dataName = data_name
    if data_dir is None:
        ANN_PATH0 = '/media/xc/BA61C62ABCE29FF2/xc/dataset/RsCarData/images/test1024/'
    else:
        ANN_PATH0 = data_dir
        data_dir = data_dir.split('images')[0]
    eval_mode = 'fixed'  #'fixed', 'adaptive'
    th_mean = 1
    th_std = 13 # 没有参与计算

    eval_new_mode = 'old'  # 'new' ###选择新的标注进行评测，或者是选择旧的标注进行评测

    conf_results = {}

    for conf_thresh in conf_thresh_all:

        methods_results = {}

        for results_dir0 in results_dir_tol:
            iou_results = []
            print(results_dir0)
            #record the results
            if eval_new_mode == 'new':
                txt_name = 'reuslts_%s_%.2f_F1_new_gt.txt' % (eval_mode_metric, conf_thresh)
            else:
                txt_name = 'reuslts_%s_%.2f_F1.txt' % (eval_mode_metric, conf_thresh)
            if write_flag:
                fid = open(results_dir0 + txt_name, 'w+')
                fid.write(results_dir0 + '(recall,precision,F1)\n')
                fid.write(eval_mode_metric + '\n')
            if eval_mode_metric=='dis':
                thres = dis_th
            elif eval_mode_metric=='iou':
                thres = iou_th
            else:
                raise Exception('Not a valid eval mode!!')
            ##eval
            for thre in thres:#
                thresh_results = {}
                if eval_mode_metric == 'dis':
                    dis_th_cur = thre
                    iou_th_cur = 0.5
                elif eval_mode_metric == 'iou':
                    dis_th_cur = 5
                    iou_th_cur = thre
                else:
                    raise Exception('Not a valid eval mode!!')
                det_metric = eval_metric(dis_th=dis_th_cur, iou_th=iou_th_cur, eval_mode=eval_mode_metric)
                if write_flag:
                    fid.write('conf_thresh=%.2f,thresh=%.2f\n'%(conf_thresh, thre))
                results_temp = {}
                for datafolder in dataName:
                    det_metric.reset()
                    if eval_new_mode == 'new':
                        ANN_PATH = data_dir + 'labeleddata20230227/' + '%03d' % datafolder + '/img1/'
                    else:
                        if opt.datasetname == 'aircraft' or opt.datasetname == 'sdm_car' or opt.datasetname == 'mir':
                            ANN_PATH = ANN_PATH0 +  datafolder + '/xml1/'
                        else:
                            ANN_PATH = ANN_PATH0 + '%03d' % datafolder + f'/{xmlname}/' # xml_det
                    # ANN_PATH = ANN_PATH0 + '%03d' % datafolder + '/xml/'
                    if eval_mode == 'adaptive':
                        if opt.datasetname == 'aircraft' or opt.datasetname == 'sdm_car' or opt.datasetname == 'mir':
                            results_dir = results_dir0 + '%s/coords_mean_%d_std_%d/' % (datafolder, th_mean, th_std)
                        else:
                            results_dir = results_dir0 + '%03d/coords_mean_%d_std_%d/' % (datafolder, th_mean, th_std)
                    elif eval_mode == 'fixed':
                        if opt.datasetname == 'aircraft' or opt.datasetname == 'sdm_car' or opt.datasetname == 'mir':
                            results_dir = results_dir0 + '%s/' % (datafolder)
                        # results_dir = results_dir0 + '%03d/coords/' % (datafolder)
                        else:
                            results_dir = results_dir0 + '%03d/' % (datafolder)
                    else:
                        raise Exception('Not a valid mode!!!')
                    #start eval
                    anno_dir = os.listdir(ANN_PATH)
                    num_images = len(anno_dir)
                    for index in range(num_images):
                        file_name = anno_dir[index]
                        #load gt
                        if(not file_name.endswith('.xml')):
                            continue
                        annName = ANN_PATH+file_name
                        # print(annName)
                        if not os.path.exists(annName):
                            continue
                        gt_t = det_metric.getGtFromXml(annName)
                        #导入det
                        matname = results_dir + file_name.replace('.xml','.mat')
                        # matname=matname.replace('results_model_last', 'results_latest')
                        if os.path.exists(matname):
                            det_ori = sio.loadmat(matname)
                            try:
                                det = det_ori['Detect_Result']
                            except:
                                det = det_ori['A']
                            if(det.shape[1]==5):
                                det = np.array(det)
                                score = det[:,-1]
                                inds = np.argsort(-score)
                                score = score[inds]
                                det = det[score>conf_thresh]
                            else:
                                det[:, 2:4] = det[:, 0:2]+det[:, 2:4]
                                det[:, 0:2] = det[:, 0:2]
                        else:
                            det = np.empty([0,4])
                        #更新评价结果
                        # print(gt_t, det)
                        det_metric.update(gt_t, det)
                        # print(det_metric.get_result())
                    #获取结果
                    if opt.datasetname == 'aircraft':
                        img_size = [512, 512]
                    elif opt.datasetname == 'sdm_car':
                        img_size = [1080, 1920]
                    elif opt.datasetname == 'mir':
                        img_size = [352, 416]
                    else:
                        img_size = [1024, 1024]
                    result = det_metric.get_result(img_size=img_size, seq_len=num_images)
                    if write_flag:
                        fid.write('&%s\t&%.3f\t&%.3f\t&%.3f\t&%.3f\t&%.3e\t&%.3e\n' % (
                    str(datafolder),result['recall'], result['prec'], result['f1'], result['pd'], result['fa_1'], result['fa_2']))
                    if opt.datasetname == 'aircraft' or opt.datasetname == 'sdm_car' or opt.datasetname == 'mir':
                        print('%s, evalmode=%s, thre=%0.2f, conf_th=%0.2f, re=%0.3f, prec=%0.3f, f1=%0.3f, pd=%0.3f, fa_1=%0.3e, fa_2=%0.3e' % (
                                '%s' % datafolder, eval_mode_metric, thre, conf_thresh, result['recall'],
                                result['prec'], result['f1'],
                                result['pd'], result['fa_1'], result['fa_2']))
                    else:
                        print('%s, evalmode=%s, thre=%0.2f, conf_th=%0.2f, re=%0.3f, prec=%0.3f, f1=%0.3f, pd=%0.3f, fa_1=%0.3e, fa_2=%0.3e' % (
                                '%03d' % datafolder, eval_mode_metric, thre, conf_thresh, result['recall'],
                                result['prec'], result['f1'],
                                result['pd'], result['fa_1'], result['fa_2']))
                    results_temp[datafolder] = result
                # 获取 avg results
                meatri = [[v['recall'], v['prec'], v['f1'], v['pd'], v['fa_1'], v['fa_2']] for k, v in
                          results_temp.items()]
                meatri = np.array(meatri)
                avg_results = np.mean(meatri, 0)
                print('avg result:  ', avg_results)
                if write_flag:
                    fid.write(
                    '&%.3f\t&%.3f\t&%.3f\t&%.3f\t&%.3e\t&%.3e\n' % (
                    avg_results[0], avg_results[1], avg_results[2], avg_results[3], avg_results[4], avg_results[5]))
                results_temp['avg'] = {
                    'recall': avg_results[0],
                    'prec': avg_results[1],
                    'f1': avg_results[2],
                    'pd': avg_results[3],
                    'fa1': avg_results[4],
                    'fa2': avg_results[5],
                }
                thresh_results[thre] = results_temp
            methods_results[results_dir0] = thresh_results
        conf_results[conf_thresh] = methods_results
    return conf_results

if __name__ == '__main__':
    class opts(object):
        def __init__(self):
            self.parser = argparse.ArgumentParser()
            # basic experiment setting
            self.parser.add_argument('--datasetname', default='rs_car_new')
            self.parser.add_argument('--task', default='ctdet_points', help='task name.  ctdet_points |  ctdet ')
        def parse(self, args=''):
            opt = self.parser.parse_args()
            return opt
    opt = opts().parse()
    results_dir = [
        'weights/rs_car_new_multi/I2PSOD/xrsybaseline_seed3407_seqlen15_I2PSOD_UNet3DWithNormalConv3D_net1outonly_supMode_0_seglen15_weights2026_01_31_16_22_17/results_latest/',
    ]
    data_dir = '/root/autodl-tmp/RsCarData_New_Part/images/test1024/'
    data_name =  dataName = [2,3,5,6,8,9,10]

    
    
    eval_func_final(results_dir, data_dir, data_name,eval_mode_metric='dis', opt=opt, xmlname='xml1new', conf_ths=[0.2,0.25,0.3])