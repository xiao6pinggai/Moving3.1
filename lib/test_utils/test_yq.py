from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.models.stNet import get_det_net, load_model, save_model
from lib.dataset.dataset_factory import get_dataset
from lib.utils_eval.evaluation_final_func import eval_func_final
from lib.test_utils.show_imgs import *
from lib.test_utils.process_img_dets import *

import GPUtil
import scipy.io as scio
import gc
def test(opt, split, modelPath, show_flag, results_name, save_mat=False, i_th=3,DataVal=None):

    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    # opt.test_large_size = True

    print(opt.model_name)
    # if DataVal is None:
        # dataset = get_dataset(opt)
    #     DataVal = dataset(opt, split)
    # dataset = get_dataset(opt)

    # DataVal = dataset(opt, split)
    class dataset:
        def __init__(self):
            self.resolution = [400,600]
            self.num_classes = 1
            self.mean = np.array([0.49965, 0.49965, 0.49965],
                dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array([0.08255, 0.08255, 0.08255],
                dtype=np.float32).reshape(1, 1, 3)
    DataVal = dataset()
    if opt.off_flag:
        head = {'hm': 1, 'wh': 2, 'reg': 2}
    else:
        head = {'hm': 1, 'wh': 2}
    model = get_det_net(head, opt.model_name, DataVal.resolution, opt.seqLen, opt, thresh=i_th)  # 建立模型 model的resolution变了
    model = load_model(model, modelPath)
    model = model.to(opt.device)
    model.eval()


    return_time = False
    # num_classes = dataset.num_classes
    num_classes = 1
    max_per_image = opt.K

    if save_mat:
        save_mat_path_upper = os.path.join(opt.save_results_dir, results_name)
        if not os.path.exists(save_mat_path_upper):
            os.mkdir(save_mat_path_upper)
    if opt.datasetname == 'aircraft':
        test_upper_path = opt.data_dir + 'images/test/'
    elif opt.datasetname == 'rs_car':
        test_upper_path = opt.data_dir + 'images/test1024/'

    data_folder_list = os.listdir(test_upper_path)
    data_folder_list = [
        f for f in data_folder_list
        if not f.startswith('.') and os.path.isdir(os.path.join(test_upper_path, f))
    ]
    data_folder_list = [i for i in data_folder_list if '.' not in i]  # lhg
    # 加快调试进度
    if False:
        data_folder_list = data_folder_list[:1]  # for debug
    patch_len = opt.seqLen

    time_all = []
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]
    for ii in range(len(data_folder_list)):
        data_folder_path = os.path.join(test_upper_path, data_folder_list[ii], 'img0')
        if save_mat:
            save_mat_folder = os.path.join(save_mat_path_upper, data_folder_list[ii])
            if not os.path.exists(save_mat_folder):
                os.mkdir(save_mat_folder)
        img_list = os.listdir(data_folder_path)
        img_list = [i for i in img_list if i.endswith(('.jpg', '.png'))]
        img_list.sort()
        imgs_number = len(img_list)
        overlap_flag = 0
        if len(img_list)%patch_len==0: # 每个patch表示seqlen个连续帧
            patch_num = len(img_list)//patch_len # patch_num表示整个序列被分成了几个patch
        else:
            patch_num = len(img_list) // patch_len+1
            overlap_flag=1
        for pk in range(patch_num):
            time_start = time.time()
            if overlap_flag and pk==patch_num-1:
                patch_ims = img_list[imgs_number-patch_len:imgs_number]
            else:
                patch_ims = img_list[pk*patch_len : (pk+1)*patch_len]
            patch_ims_path = [os.path.join(data_folder_path, i) for i in patch_ims]
            batch_dict, meta, patch_imgs, input_imgs = preprocess(patch_ims_path, DataVal)
            # patch_imgs: 未归一化的原始图像；input_imgs: 归一化图像和未归一化的灰度图(字典)
            for k in input_imgs:
                if k == 'batch_size':
                    continue
                input_imgs[k] = torch.from_numpy(input_imgs[k]).to(opt.device)
            time_start1 = time.time()
            output, dets = process(model, input_imgs, return_time, opt, opt.K)
            # dets.shape=torch.Size([20, 128, 6]) # 表示20张图像，每张图像最多128个检测框，每个检测框6个值(x1,y1,x2,y2,score,class)
            # output是模型的原始输出，包含热力图、wh、reg、mask_all、voxel_coords、lasso等信息
            torch.cuda.synchronize()
            time_start3 = time.time()
            # 后处理
            rets, dets_post = post_process(dets, meta, num_classes, max_per_image=max_per_image)
            # 后处理实际上做了张量转成字典（内部为20个array，每个array包含128个array，每个array有一个6元素的列表）的变形
            time_end = time.time()

            time_all.append(time_end - time_start1)
            
            if pk % 50 == 0:
                print('&& Processing folder %d/%d, patch %d/%d' % (ii+1, len(data_folder_list), pk+1, patch_num),
                      '&& time_used:', time_end - time_start, time_end - time_start1, time_start3 - time_start1,
                      '&& patch_len: {} GPU used: {}/{}'.format(patch_len, gpu.memoryUsed, gpu.memoryTotal))
                # print('time_used:', time_end - time_start, time_end - time_start1, time_start3 - time_start1)
                # print('patch_len: {} GPU used: {}/{}'.format(patch_len, gpu.memoryUsed, gpu.memoryTotal))
            ### view results
            if save_mat:
                fig_save_name1 = os.path.join(save_mat_folder, '%03d_ori.png'%(pk+1))
                fig_save_name2 = os.path.join(save_mat_folder, '%03d_det.png' % (pk + 1))
                # view_cloud(output['voxel_coords'], save_flag=1, fig_save_name = fig_save_name1)
                # view_dets(dets, conf_th=0.3,save_flag=1, fig_save_name = fig_save_name2)
                plt.close('all') # lhg
            if(show_flag):
                hm1 = output['hm'].squeeze(0).squeeze(0).cpu().detach().numpy()
                for det_i in range(len(dets_post)):
                    img = patch_imgs[:,:,:,det_i]
                    frame, _ = cv2_demo(img.astype(np.uint8), dets_post[det_i][1])

                    cv2.imshow('frame',frame)
                    cv2.waitKey(5)
                    hm2 = hm1[det_i]
                    cv2.imshow('hm', hm2)
                    cv2.waitKey(5)
            
            if save_mat:
                for ik in range(len(patch_ims)):
                    # mat_save_name = os.path.join(save_mat_folder, patch_ims[ik].replace('.jpg', '.mat'))
                    mat_save_name = os.path.join(save_mat_folder, os.path.splitext(patch_ims[ik])[0] + '.mat')
                    ret = rets[ik]
                    A = np.array(ret[1])
                    scio.savemat(mat_save_name, {'A':A})
                    del A, ret
            del batch_dict, meta, patch_imgs, input_imgs, output, dets, rets, dets_post

    time_mean = np.array(time_all).mean()
    print('total_time_mean:', time_mean/patch_len, 'frames per second: ', 1/time_mean*patch_len)

    conf_results = eval_func_final([os.path.join(opt.save_results_dir, results_name + '/')], data_dir=test_upper_path, opt=opt, 
                                   eval_mode_metric='dis')
    results_return = {}
    best = -1
    for conf, v_c in conf_results.items():
        for k_m, v_m in v_c.items():
            for k_d, v_d in v_m.items():
                re = v_d['avg']['recall']
                pre = v_d['avg']['prec']
                f1 = v_d['avg']['f1']
                results_return['conf_%.2f'%conf + '_avg_recall'] = re
                results_return['conf_%.2f' % conf + '_avg_prec'] = pre
                results_return['conf_%.2f' % conf + '_avg_f1'] = f1
                if best < f1:
                    best = f1
    results_return['f1_best'] = best
    results_return['total_time_mean'] = time_mean/patch_len
    results_return['frames_per_second'] = 1/time_mean*patch_len
    

    if save_mat:
        results_tol_txt = os.path.join(opt.save_results_dir, results_name, 'results_tol.txt')
        results_tol_txt_fid = open(results_tol_txt, 'w+')
        results_tol_txt_fid.write(results_name+'\n')
        for k,v in results_return.items():
            results_tol_txt_fid.write(k+': %.4f\n'%v)


        results_tol_txt_fid.close()

        time_txt = open(os.path.join(opt.save_results_dir, results_name, 'time.txt'),'w')
        time_txt.write('total_time_mean: %.4f\t frames per second: %.2f\n'%(time_mean/patch_len, 1/time_mean*patch_len))
        time_txt.close()
    del model, conf_results
    gc.collect()
    plt.close('all') # 强制关闭所有 figure，防止残留
    torch.cuda.empty_cache() # 清理 GPU 临时显存
    return results_return