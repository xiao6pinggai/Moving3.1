import cv2
import torch
from skimage import measure
import os
import numpy as np

from PIL import Image

try:
    from lib.external1.nms import soft_nms
except:
    from lib.external1.nms import soft_nms
from lib.utils1.sort import *
from lib.LRSD.WSNMSTIPT_dp_without_B_norm import WSNMSTIPT_dp_without_B_norm

def preprocess(img_list, resolution=[512,512]):
    if 'sdmcar':
        print("正在处理sdmcar！！！")
        # print('mir!!!352 416')
        resolution = [1088, 1920]
    seq_num = len(img_list)
    imgs_gray = np.zeros([resolution[0], resolution[1],  seq_num])
    a1 = time.time()
    for ii in range(seq_num):
        img_id_cur = img_list[ii]
        im = cv2.imread(img_id_cur)
        # 处理1080尺寸
        ########################################################
        h, w = im.shape[:2]  # 1080, 1920
        target_h, target_w = resolution[0], resolution[1]
        if h<target_h or w<target_w:
            pad_h = target_h - h  # 8
            pad_w = target_w - w  # 0
            # 3. 实施填充 (只在底部填充 8 像素)
            # 参数顺序：top, bottom, left, right
            im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, 
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        ########################################################
        ###
        ###
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        imgs_gray[:, :, ii] = im_gray
    return imgs_gray

def get_tar_ims(data_dir, save_tar_ims=None,  th_std = 5,phase='train'):
    test_upper_path = os.path.join(data_dir,f'images/{phase}/')
    data_folder_list = os.listdir(test_upper_path)
    data_folder_list = [i for i in data_folder_list if '.' not in i]  # lhg
    data_folder_list.sort()
    # data_folder_list = data_folder_list[:3]  # for debug
    if save_tar_ims is None:
        save_tar_ims = os.path.join(data_dir, 'lrsd', phase)
    if not os.path.exists(save_tar_ims):
        os.makedirs(save_tar_ims)
    #params
    patch_len = 16 # 16 # 为什么是16，不应该是seqlen=20吗？表示一个样本 # 猜测估计是生成标签和训练是两码事，生成标签是独立的
    Lambda = 1
    mu = 5e-4
    beta = 100
    rho = 1.5
    for ii in range(0, len(data_folder_list)):
    # for ii in range(len(data_folder_list)-1, -1, -1):
        data_folder_path = os.path.join(test_upper_path, data_folder_list[ii], 'img1')
        save_coords_folder = os.path.join(save_tar_ims, data_folder_list[ii],'coords_unfilt')
        if not os.path.exists(save_coords_folder):
            os.makedirs(save_coords_folder)
        img_list = os.listdir(data_folder_path)
        # img_list = [i for i in img_list if '.' not in i]  # lhg
        # img_list = [i for i in img_list if i.endswith('.jpg')]
        # 推荐的最健壮版本 (兼容大小写)
        img_list = [i for i in img_list if i.lower().endswith(('.jpg', '.png'))]
        img_list.sort()
        imgs_number = len(img_list)
        overlap_flag = 0
        if len(img_list)%patch_len==0:
            patch_num = len(img_list)//patch_len
        else:
            patch_num = len(img_list) // patch_len+1
            overlap_flag=1
        for pk in range(0, patch_num):
            time_start = time.time()
            if overlap_flag and pk==patch_num-1:
                patch_ids = [i for i in range(imgs_number-patch_len, imgs_number)]
                patch_ims = img_list[imgs_number-patch_len:imgs_number]
            else:
                patch_ids = [i for i in range(pk*patch_len, (pk+1)*patch_len)]
                patch_ims = img_list[pk*patch_len : (pk+1)*patch_len]
            patch_ims_path = [os.path.join(data_folder_path, i) for i in patch_ims]
            input_imgs = preprocess(patch_ims_path) # 打包转成灰度图
            input_imgs_t = torch.from_numpy(input_imgs).cuda()
            B_hat_t = torch.median(input_imgs_t, 2)[0].unsqueeze(2).repeat(1,1, input_imgs_t.shape[2])
            out = WSNMSTIPT_dp_without_B_norm(input_imgs_t, B_hat_t, Lambda, mu, beta, rho) # 传统算法，实现LRSD # 孙杨
            tar_ims = out['T'].cpu().numpy()
            for ik in range(len(patch_ims)):
                im = tar_ims[:,:,ik]
                mask = np.zeros_like(im)
                mask[im > im.mean() + th_std * im.std()] = 1 # 自适应阈值分割
                topk_coords, coords = get_det_result_from_im(mask, im) # 连通域分析/尺寸过滤/Soft-NMS/Top-K选取
                # txt_save_name_ori = os.path.join(save_coords_folder,patch_ims[ik].replace('.jpg', '.txt'))
                # 修改lhg
                original_filename = patch_ims[ik]
                if original_filename.endswith('.jpg'):
                    txt_filename = original_filename.replace('.jpg', '.txt')
                elif original_filename.endswith('.png'):
                    txt_filename = original_filename.replace('.png', '.txt')
                else:
                    # 处理未知扩展名，可以选择报错或保持原文件名
                    txt_filename = original_filename + '.txt'
                txt_save_name_ori = os.path.join(save_coords_folder, txt_filename)
                fid_txt_ori = open(txt_save_name_ori, 'w') # 将粗检测结果保存为txt文件
                #####no filt
                for da in range(topk_coords.shape[0]):
                    coord_da = topk_coords[da]
                    fid_txt_ori.write('%d\t%d\t%d\t%d\t0\t%d\n' % (
                    coord_da[0], coord_da[1], coord_da[2], coord_da[3], coord_da[4]))
                fid_txt_ori.close()
        print('folder', data_folder_list[ii], 'get lrsd results done!!!')

def get_mask(dets, img_size):
    mask = np.zeros(img_size[0],img_size[1])
    for i in range(dets.shape[0]):
        mask[dets[i,0]:dets[i,2], dets[i,1]:dets[i,3]] = 1
    return mask

def get_det_result_from_im(seg, image_out):
    top_k = 80
    area_th_min = 4
    area_th_max = 80 # 过滤过大的连通域
    image = measure.label(seg, connectivity=2) # 连通域
    prop_regions = measure.regionprops(image, intensity_image=image_out) #i.intensity_max最大亮度最为该框的第一步评分依据（置信度）
    coords = np.array([[list(i.bbox) + [i.intensity_max]] for i in prop_regions if (i.area > area_th_min and i.area < area_th_max)])
    # coords = np.array([i.bbox for i in prop_regions]) # [[list(i.bbox) + [i.intensity_max]]列表拼接，bbox四个坐标+置信度
    coords = coords.reshape(-1, 5)
    coords1 = coords.copy()
    coords1[:, 0] = coords[:, 1] # yx 与xy调整
    coords1[:, 1] = coords[:, 0]
    coords1[:, 2] = coords[:, 3]
    coords1[:, 3] = coords[:, 2]

    results = {} # [1]实际就表示ID类别1
    results[1] = coords1.astype(np.float32) # 它是一个 NumPy 数组，形状为 [N, 5]，其中 N 是检测到的候选框数量，每行包含 [x1, y1, x2, y2, score]。
    # nms
    soft_nms(results[1], Nt=0.5, method=2) # soft_nms # soft_nms会修改置信度
    # get top_k detections
    scores = results[1][:, -1] # 取出所有框的最后得分
    if len(scores) > top_k:
        kth = len(scores) - top_k
        thresh = np.partition(scores, kth)[kth]
        keep_inds = (results[1][:, 4] >= thresh)
        results[1] = results[1][keep_inds]

    return results[1], coords1

def generate_labels(data_dir=None,phase='train'):
    if data_dir is None:
        data_dir = './datasets/RsCarData'
    save_path_upper = os.path.join(data_dir, f'lrsd/{phase}/')
    if not os.path.exists(save_path_upper):
        os.makedirs(save_path_upper)
    #get tar ims from lrsd
    th_std = 5 # 原来是5
    get_tar_ims(data_dir, save_path_upper, th_std, phase) # 得到LRSD粗检测结果并完成从mask到框的转换并保存为txt文件
    # 将每张图的检测结果保存为 .txt 文件，存放在 /root/autodl-tmp//AircraftDataset23/lrsd/train/<序列名>/coords_unfilt/ 目录下
    data_list = os.listdir(save_path_upper)
    data_list = [i for i in data_list if '.' not in i]  # lhg
    # data_list = data_list[:3]  # for debug
    data_list.sort()
    for dfk in range(0, len(data_list)):
        data_f = data_list[dfk]
        ########新建文件夹
        data_folder = os.path.join(save_path_upper, data_f, 'coords_unfilt') # 粗检测结果存放路径，还未经过sort过滤
        #####
        save_folder = os.path.join(save_path_upper, data_f)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_txt_folder = os.path.join(save_folder, 'coords_update')
        if not os.path.exists(save_txt_folder):
            os.mkdir(save_txt_folder)
        save_txt_folder_ori = os.path.join(save_folder, 'coords_filt')
        if not os.path.exists(save_txt_folder_ori):
            os.mkdir(save_txt_folder_ori)
        ###################
        ######
        im_list = os.listdir(data_folder)
        im_list = [i for i in im_list if i.endswith('.txt')]
        im_list.sort()
        ######
        mot_tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.1) # SORT 利用卡尔曼滤波和匈牙利匹配算法，将不同帧的检测框关联起来，赋予每个目标唯一的 ID
        ids = []
        trajs = []
        ######
        dets_all_ori = []
        for i_im in range(0, len(im_list)): # 将跟踪结果组织成轨迹列表 trajs。每条轨迹包含该目标在不同帧的 [帧索引, x1, y1, x2, y2] 信息
            im_name = os.path.join(im_list[i_im])
            im_path = os.path.join(data_folder, im_name)
            det_coords = np.loadtxt(im_path).reshape(-1,6)

            dets_all_ori.append(det_coords)
            track_bbs_ids = mot_tracker.update(det_coords)
            for it in range(track_bbs_ids.shape[0]):
                id = track_bbs_ids[it,-1]
                coord = track_bbs_ids[it,:4]
                if id not in ids:
                    ids.append(track_bbs_ids[it,-1])
                    trajs.append([])
                index = ids.index(id)
                trajs[index].append([i_im]+coord.tolist())

        trajs_filt = []
        for traj_i in trajs: # 轨迹过滤（核心逻辑）： 遍历所有生成的轨迹，应用以下两条硬性规则 
                            # 1、时长过滤：如果一条轨迹包含的帧数少于 15帧 (len < 15)，则被丢弃。这过滤了瞬时的噪声。
                            # 2、速度过滤：计算目标相邻帧中心点的位移距离。
                                # 计算平均速度 v_mean。
                                # 如果 v_mean < 0.55 像素/帧，则被视为静止物体或极慢速干扰，予以丢弃
            if len(traj_i)<15:
                continue
            a = np.array(traj_i)
            ct = (a[:,3:5]+a[:,1:3])/2
            d = ct[1:, :]-ct[:-1,:]
            d = (d[:,0]**2+d[:,1]**2)**0.5
            v = d/(a[1:,0]-a[:-1,0])
            v_mean = abs(v).mean()
            # print(v_mean)
            if v_mean<0.55:
                continue
            trajs_filt.append(traj_i)

        det_for_images = [[] for i in range(len(im_list))]
        images = [i for i in range(len(im_list))]
        count = 0
        for i_traj in trajs_filt:
            count=count+1
            for i_trajkk in i_traj:
                index = images.index(i_trajkk[0])
                det_for_images[index].append(i_trajkk[1:]+[count])
        ##################

        for kk in range(len(det_for_images)):
            #######
            txt_save_name = os.path.join(save_txt_folder, im_list[kk].replace('.jpg', '.txt'))
            fid_txt = open(txt_save_name, 'w')
            #
            txt_save_name_ori = os.path.join(save_txt_folder_ori, im_list[kk].replace('.jpg', '.txt')) # 实际上过滤了本身就是txt了，jpg替换未执行
            fid_txt_ori = open(txt_save_name_ori, 'w')
            #####update
            for coord in det_for_images[kk]:
                fid_txt.write('%d\t%d\t%d\t%d\t0\t%d\n'%(coord[0], coord[1], coord[2], coord[3], coord[4]))
            #####filt
            for coord11 in det_for_images[kk]:
                fid_txt_ori.write('%d\t%d\t%d\t%d\t0\t%d\n'%(coord11[0], coord11[1], coord11[2], coord11[3], coord11[4]))
            fid_txt.close()
            fid_txt_ori.close()
        print('folder', data_f, 'get filtered lrsd results done!!!')


if __name__ == '__main__':
    data_dir = '/media/wellwork/L/xc/datasets/RsCarData'
    generate_labels(data_dir)





