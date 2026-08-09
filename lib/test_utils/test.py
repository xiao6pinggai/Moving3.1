from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.models.stNet import get_det_net, load_model, save_model
from lib.dataset.dataset_factory import get_dataset
from lib.utils_eval.evaluation_final_func import eval_func_final
from lib.utils_eval.evaluation_coco_json_func import eval_coco_json_f1
from lib.utils_eval.evaluation_net1_point import eval_net1_points
from lib.test_utils.show_imgs import *
from lib.test_utils.process_img_dets import *
from lib.test_utils.vis_hook_utils import VisHookManager
from lib.utils1.save_img import save_net1_output
import GPUtil
import scipy.io as scio
import gc

COCO_STAT_NAMES = [
    'ap', 'ap50', 'ap75', 'ap_small', 'ap_medium', 'ap_large',
    'ar_max1', 'ar_max10', 'ar_max100', 'ar_small', 'ar_medium', 'ar_large'
]

def _add_coco_stats(results_return, split_name, stats):
    # COCOeval.summarize() 的 12 项原样展开保存，便于实验日志检索。
    prefix = 'all' if split_name == 'all' else split_name
    for stat_name, value in zip(COCO_STAT_NAMES, stats):
        results_return[f'{prefix}_coco_{stat_name}'] = float(value)
    if split_name == 'all':
        results_return['ap50'] = float(stats[1])

def test(opt, split, modelPath, show_flag, results_name, save_mat=False, i_th=3, DataVal=None):
    test_net1 = False # 生成坐标txt
    run_eval_net1 = False # 计算坐标与bbox的recall和iou覆盖率
    run_ap = opt.metric['run_ap']
    run_f1 = opt.metric['run_f1']
    save_json = opt.metric['save_json']
    save_mat = opt.metric['save_mat']
    f1_mode = opt.metric['f1_mode'] # ['iou', 'dis']
    f1_source = opt.metric.get('f1_source', 'json')
    eval_splits = opt.metric.get('eval_splits', ['all'])
    inference = opt.metric['inference']
    need_result_json = save_json or run_ap or (run_f1 and f1_source == 'json')
    # save_mat = False
    xmlname = opt.xmlname # 'xml_det' # xml1new # xml_det 

    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    # opt.test_large_size = True

    print(opt.model_name)
    if DataVal is None:
        dataset = get_dataset(opt)
        DataVal = dataset(opt, split)
    # dataset = get_dataset(opt)

    # DataVal = dataset(opt, split)
    model_end_name = modelPath.split('/')[-1].split('.')[0]
    model = None
    if inference:
        if opt.off_flag:
            head = {'hm': DataVal.num_classes, 'wh': 2, 'reg': 2}
        else:
            head = {'hm': DataVal.num_classes, 'wh': 2}
        model = get_det_net(head, opt.model_name, DataVal.resolution, opt.seqLen, opt, thresh=i_th)  # 建立模型 model的resolution变了
        model = load_model(model, modelPath)
        model = model.to(opt.device)
        model.eval()


    return_time = True
    # num_classes = dataset.num_classes
    num_classes = DataVal.num_classes
    max_per_image = opt.K

    # 评估结果统一放在 results_dir；mat 只作为可选兼容产物。
    save_mat_path_upper = os.path.join(opt.save_results_dir, results_name)
    os.makedirs(save_mat_path_upper, exist_ok=True)
    result_json_path = os.path.join(save_mat_path_upper, 'results_{}.json'.format(model_end_name))
    if save_mat:
        print(f'mkdirs:{save_mat_path_upper}')

    # ── 特征图可视化：创建目录并实例化 hook 管理器 ──────────────────────────
    vis_features = getattr(opt, 'vis_features', False)
    if vis_features:
        vis_mode       = getattr(opt, 'vis_mode', 'mean')
        vis_layers     = getattr(opt, 'vis_layers', [])
        vis_max_frames = getattr(opt, 'vis_max_frames', 10)  # 每段视频最多保存帧数，可在 opt 中覆盖
        # vis 文件夹名称关联 vis_mode：mean→vis_mean，max→vis_max，数字→vis_numX
        try:
            _ch = int(vis_mode)
            _vis_suffix = f'vis_num{_ch}'
        except ValueError:
            _vis_suffix = f'vis_{vis_mode}'
        save_vis_path_upper = os.path.join(opt.save_results_dir, results_name, _vis_suffix)
        if not os.path.exists(save_vis_path_upper):
            os.mkdir(save_vis_path_upper)
            print(f'mkdirs:{save_vis_path_upper}')
        vis_mgr = VisHookManager(
            model               = model,
            vis_layers          = vis_layers,
            vis_mode            = vis_mode,
            save_vis_path_upper = save_vis_path_upper,
            max_vis_frames      = vis_max_frames,
            patch_len           = opt.seqLen,
        )
    # ── end vis_features setup ───────────────────────────────────────────────

    if opt.datasetname == 'aircraft' or opt.datasetname == 'sdm_car' or opt.datasetname == 'mir':
        test_upper_path = opt.data_dir + 'images/test/'
    elif opt.datasetname == 'rs_car' or opt.datasetname == 'rs_car_new':
        test_upper_path = opt.data_dir + 'images/test1024/'


    data_folder_list = os.listdir(test_upper_path)
    data_folder_list = [
        f for f in data_folder_list
        if not f.startswith('.') and os.path.isdir(os.path.join(test_upper_path, f))
    ]
    data_folder_list = [i for i in data_folder_list if '.' not in i]  # lhg
    data_folder_list.sort()
    # 加快调试进度
    if False:
        data_folder_list = data_folder_list[:2]  # for debug
    patch_len = opt.seqLen

    time_all = []
    preprocess_time_all = []
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]
    im_id = 1 # 用于保存json
    results = {}
    his_img_num =0
    results_return = {} # 写txt结果
    component_stats = {}
    image_id_map = None
    if need_result_json and hasattr(DataVal, 'coco'):
        image_id_map = {}
        for image_info in DataVal.coco.dataset.get('images', []):
            ann_rel_path = image_info.get('file_name', '').replace('\\', '/').lstrip('./')
            for prefix in ('images/test/', 'images/test1024/'):
                if ann_rel_path.startswith(prefix):
                    ann_rel_path = ann_rel_path[len(prefix):]
                    break
            image_id = int(image_info['id'])
            image_id_map[ann_rel_path] = image_id
            image_id_map[os.path.splitext(ann_rel_path)[0]] = image_id
    # 前向推理
    # 如果不进行前向推理，需要给出ap coco dir 以及 xml dir
    if inference:
        total_sampling_rate = 0.0  # 累积采样率
        num_batches = 0  # 批次数
        for ii in range(len(data_folder_list)):
            data_folder_path = os.path.join(test_upper_path, data_folder_list[ii], 'img1')
            if save_mat:
                save_mat_folder = os.path.join(save_mat_path_upper, data_folder_list[ii])
                os.makedirs(save_mat_folder, exist_ok=True)
            # ── 为当前视频注册特征图 hook ────────────────────────────────────
            if vis_features:
                vis_mgr.register(data_folder_list[ii])

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
            if test_net1:
                # [MODIFIED] 初始化该视频序列的点云缓存字典
                # 结构: { global_frame_id (int): np.array([[y, x], ...]) }
                video_coords_cache = {}
            for pk in range(patch_num):
                time_start = time.time()
                if overlap_flag and pk==patch_num-1:
                    patch_ims = img_list[imgs_number-patch_len:imgs_number]
                    video_frame_id = imgs_number-patch_len+1
                else:
                    video_frame_id = pk*patch_len+1
                    patch_ims = img_list[pk*patch_len : (pk+1)*patch_len]
                patch_ims_path = [os.path.join(data_folder_path, i) for i in patch_ims]
                patch_xml_path = [
                                    f.replace("img1", xmlname).rsplit('.', 1)[0] + ".xml" 
                                    for f in patch_ims_path
                                ] # rsplit('.', 1) 表示从右边开始，以 . 为分隔符，只切分 1 次。
                
                # time_start1 = time.time() # 合理的位置
                time_start0 = time.time()
                batch_dict, meta, patch_imgs, input_batch = preprocess(patch_ims_path, DataVal, patch_xml_path)
                # patch_imgs: 未归一化的原始图像；input_batch: 归一化RGB图像和未归一化的灰度图(字典)
                for k in input_batch:
                    if k == 'batch_size':
                        continue
                    input_batch[k] = torch.from_numpy(input_batch[k]).to(opt.device)
                time_start1 = time.time() # 放在这里不合理
                output, dets, forward_time = process(model, input_batch, return_time, opt, opt.K) # output==z
                # ── 捕获特征图并保存（每段视频前 vis_max_frames 张）─────────
                if vis_features:
                    vis_mgr.save_batch(patch_ims)
                # ── end vis ─────────────────────────────────────────────────
                if False:
                    # 累加当前批次的采样率
                    total_sampling_rate += output['sampling_rate'].item()
                    num_batches += 1
                if False:
                    os.makedirs(save_mat_folder+'vis', exist_ok=True)
                    for ik in range(len(patch_ims)):
                        imgpath = os.path.join(save_mat_folder+'vis', os.path.splitext(patch_ims[ik])[0] + '.png')
                        save_net1_output(output['soft_mask'][0][0][ik], imgpath,mode='255')
                # dets.shape=torch.Size([20, 128, 6]) # 表示20张图像，每张图像最多128个检测框，每个检测框6个值(x1,y1,x2,y2,score,class:0)
                # output是模型的原始输出，包含热力图、wh、reg、mask_all、voxel_coords、lasso等信息
                torch.cuda.synchronize()
                time_start3 = time.time()
                # 后处理
                rets, dets_post = post_process(dets, meta, num_classes, max_per_image=max_per_image)
                # 后处理实际上做了张量转成字典（内部为20个array，每个array包含128个array，每个array有一个6元素的列表）的变形
                # 由于都采用了top128所以二者形状是相同的，只是dets_post是直接topK，而rets经过nms改变了score数值，因此排序发生了可一定变化
                # 计算指标用NMS以后的

                time_end = time.time()
                forward_elapsed = forward_time - time_start1
                time_all.append(forward_elapsed)
                preprocess_time_all.append((time_start1 - time_start0))
                if need_result_json:
                    for im_id_i, ret in enumerate(rets):
                        img_rel_path = os.path.relpath(
                            patch_ims_path[im_id_i], test_upper_path).replace(os.sep, '/')
                        if image_id_map is not None:
                            im_id = image_id_map.get(img_rel_path)
                            if im_id is None:
                                im_id = image_id_map.get(os.path.splitext(img_rel_path)[0])
                            if im_id is None:
                                raise KeyError(
                                    f'Cannot find COCO image_id for test image: {img_rel_path}')
                        else:
                            im_id = video_frame_id + im_id_i + his_img_num
                        results[im_id] = ret
                    # print(len(results))
                    im_id += patch_len
                # 假设 output['voxel_coords'] shape 为 [N, 4] -> (batch_idx, time_idx, y, x)
                # 其中 batch_idx 在测试时通常为 0 (单Batch推理)
                if test_net1 and 'voxel_coords' in output:
                    coords_tensor = output['voxel_coords']
                    
                    if coords_tensor.shape[0] > 0:
                        # 转为 numpy, (N, 4)
                        coords_np = coords_tensor.detach().cpu().numpy()
                        
                        # 遍历当前 Patch 中的每一帧 (通常 time_idx 范围是 0 到 seqLen-1)
                        # 获取当前 Patch 包含的局部时间步
                        unique_time_indices = np.unique(coords_np[:, 1])
                        
                        for local_t in unique_time_indices:
                            # 计算全局帧 ID
                            # video_frame_id 是当前 Patch 第一帧的物理文件名序号 (如 1, 6, 11...)
                            # local_t 是 Patch 内部的偏移量 (0, 1, 2, 3, 4)
                            global_fid = int(video_frame_id + local_t)
                            
                            # 提取属于该帧的所有点 (y, x)
                            # mask: 筛选当前 Batch (通常是0) 和当前 Time
                            # 注意：如果 batch_size > 1，这里需要加上 batch_idx 的判断，但测试通常 batch=1
                            mask = (coords_np[:, 1] == local_t)
                            
                            # 取 [y, x] 部分 (Indices 2 and 3)
                            current_frame_points = coords_np[mask, 2:4].astype(np.int32)
                        
                            video_coords_cache[global_fid] = current_frame_points


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
                    # view_dets(dets, conf_th=0.25,save_flag=1, fig_save_name = fig_save_name2)
                    # plt.close('all') # lhg
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
                        ret = rets[ik] # {1~2255:(128,5)}
                        A = np.array(ret[1])# [1]是key
                        scio.savemat(mat_save_name, {'A':A})
                        del A, ret
            # [MODIFIED] 单个视频所有 Patch 处理完后，统一保存坐标 txt
            if test_net1 and len(video_coords_cache) > 0:
                save_txt_path_upper = os.path.join(opt.save_results_dir, results_name)
                txt_save_folder = os.path.join(save_txt_path_upper, data_folder_list[ii])
                os.makedirs(txt_save_folder, exist_ok=True)
                txt_save_path = os.path.join(txt_save_folder, f'{data_folder_list[ii]}.txt')
                
                # 准备写入数据列表
                all_points_list = []
                
                # 按帧ID排序，保证 txt 是有序的
                sorted_frame_ids = sorted(video_coords_cache.keys())
                
                for fid in sorted_frame_ids:
                    points = video_coords_cache[fid]
                    # 如果该帧有点
                    if len(points) > 0:
                        # 构造 (N, 3) 数组: [fid, y, x]
                        fids_col = np.full((len(points), 1), fid, dtype=np.int32)
                        frame_data = np.hstack((fids_col, points))
                        all_points_list.append(frame_data)
                
                if len(all_points_list) > 0:
                    # 拼接所有帧的数据
                    final_data = np.concatenate(all_points_list, axis=0)

                    # 保存
                    np.savetxt(txt_save_path, final_data, fmt='%d', delimiter=',', header='frame_id,y,x')
                    print(f"Saved coords to {txt_save_path}, total points: {final_data.shape[0]}")
                else:
                    print(f"No points detected for {data_folder_list[ii]}")

                # 清空缓存，释放内存
                video_coords_cache.clear()
                del batch_dict, meta, patch_imgs, input_batch, output, dets, rets, dets_post
            his_img_num += imgs_number
            # ── 移除当前视频的 hook ──────────────────────────────────────────
            if vis_features:
                vis_mgr.remove()
            # break
        if hasattr(model, 'print_dataset_stage1_summary'):
            model.print_dataset_stage1_summary()
        time_mean = np.array(time_all).mean()
        time_preprocess_mean = np.array(preprocess_time_all).mean()
        print('forward_time_mean:', time_mean/patch_len, 'forward frames per second: ', 1/time_mean*patch_len)
        print('time_preprocess_mean:', time_preprocess_mean / patch_len, 'frames per second: ', 1 / time_preprocess_mean * patch_len)
        results_return['total_time_mean'] = time_mean/patch_len
        results_return['frames_per_second'] = 1/time_mean*patch_len
        results_return['forward_time_mean'] = time_mean/patch_len
        results_return['forward_frames_per_second'] = 1/time_mean*patch_len
        results_return['total_preprocess_time_mean'] = time_preprocess_mean/patch_len
        results_return['preprocess_frames_per_second'] = 1/time_preprocess_mean*patch_len
        if hasattr(model, 'get_component_stats'):
            component_stats = model.get_component_stats(per_frame_divisor=patch_len)
            if component_stats:
                print('===== Five-Stage Component Profiling =====')
                print('Stage                 runtime(s)  runtime(%)  Params      GFLOPS')
                component_order = getattr(model, 'runtime_component_names', ())
                for component_name in component_order:
                    stats_dict = component_stats[component_name]
                    display_name = stats_dict.get('display_name', component_name)
                    print(
                        f"{display_name:<20} {stats_dict['runtime']:.6f}    "
                        f"{stats_dict['runtime_percent']:>7.2f}%  {stats_dict['params']:<10d} {stats_dict['gflops']:.6f}"
                    )
                    key_prefix = display_name.replace(' ', '_').replace('/', '_').replace('.', '').replace('%', 'pct')
                    results_return[f'{key_prefix}_runtime_mean'] = stats_dict['runtime']
                    results_return[f'{key_prefix}_runtime_percent'] = stats_dict['runtime_percent']
                    results_return[f'{key_prefix}_params'] = stats_dict['params']
                    results_return[f'{key_prefix}_GFLOPS'] = stats_dict['gflops']
                total_stats = component_stats.get('Total')
                if total_stats is not None:
                    print(
                        f"{'Total':<20} {total_stats['runtime']:.6f}    "
                        f"{total_stats['runtime_percent']:>7.2f}%  {total_stats['params']:<10d} {total_stats['gflops']:.6f}"
                    )
                    results_return['Total_runtime_mean'] = total_stats['runtime']
                    results_return['Total_runtime_percent'] = total_stats['runtime_percent']
                    results_return['Total_params'] = total_stats['params']
                    results_return['Total_GFLOPS'] = total_stats['gflops']
                print('=========================================')
                if hasattr(model, '_component_profile_printed'):
                    model._component_profile_printed = True
        if False:
            # 循环结束后计算平均
            average_sampling_rate = total_sampling_rate / num_batches
            print(f"整个测试集的平均采样率: {average_sampling_rate * 100:.5f}%")
            results_return['average_sampling_rate'] = average_sampling_rate * 100
        if need_result_json and results:
            DataVal.save_results(results, save_mat_path_upper, model_end_name)
    else:
        # 不推理时，AP 和 json-F1 均复用已有 COCO 预测 json。
        if (run_ap or (run_f1 and f1_source == 'json')) and not os.path.exists(result_json_path):
            raise FileNotFoundError(result_json_path)


    if run_ap:
        results_return['time'] = 1 / 60.
        if hasattr(DataVal, 'run_eval_splits'):
            split_stats = DataVal.run_eval_splits(result_json_path, eval_splits)
            for split_name, stats in split_stats.items():
                _add_coco_stats(results_return, split_name, stats)
        else:
            if eval_splits != ['all']:
                raise Exception('This dataset does not support sim/real AP splits.')
            if inference:
                stats1, _ = DataVal.run_eval(results, save_mat_path_upper, model_end_name)
            else:
                stats1, _ = DataVal.run_eval_just(save_mat_path_upper, 'results_{}.json'.format(model_end_name), 0.5)
            _add_coco_stats(results_return, 'all', stats1)

    if run_f1:
        if f1_source == 'json':
            conf_results = eval_coco_json_f1(
                DataVal.coco, result_json_path, save_mat_path_upper,
                f1_mode=f1_mode, eval_splits=eval_splits, opt=opt,
                data_dir=test_upper_path, xmlname=xmlname)
            for eval_mode_metric, mode_results in conf_results.items():
                best_by_split = {split_name: -1 for split_name in eval_splits}
                for conf, split_results in mode_results.items():
                    for split_name, v_d in split_results.items():
                        re = v_d['avg']['recall']
                        pre = v_d['avg']['prec']
                        f1 = v_d['avg']['f1']
                        prefix = '' if split_name == 'all' else f'{split_name}_'
                        results_return[f'{prefix}{eval_mode_metric}_conf_%.2f' % conf + '_avg_recall'] = re
                        results_return[f'{prefix}{eval_mode_metric}_conf_%.2f' % conf + '_avg_prec'] = pre
                        results_return[f'{prefix}{eval_mode_metric}_conf_%.2f' % conf + '_avg_f1'] = f1
                        if split_name == 'all':
                            results_return[f'all_{eval_mode_metric}_conf_%.2f' % conf + '_avg_recall'] = re
                            results_return[f'all_{eval_mode_metric}_conf_%.2f' % conf + '_avg_prec'] = pre
                            results_return[f'all_{eval_mode_metric}_conf_%.2f' % conf + '_avg_f1'] = f1
                        if best_by_split[split_name] < f1:
                            best_by_split[split_name] = f1
                for split_name, best in best_by_split.items():
                    if split_name == 'all':
                        results_return[f'{eval_mode_metric}_f1_best'] = best
                        results_return[f'all_{eval_mode_metric}_f1_best'] = best
                    else:
                        results_return[f'{split_name}_{eval_mode_metric}_f1_best'] = best
        elif f1_source == 'mat':
            if eval_splits != ['all']:
                raise Exception('mat f1_source only supports all split.')
            for eval_mode_metric in f1_mode:
                conf_results = eval_func_final([os.path.join(opt.save_results_dir, results_name + '/')], data_dir=test_upper_path, opt=opt,
                                            eval_mode_metric=eval_mode_metric,xmlname=xmlname)
                best = -1
                for conf, v_c in conf_results.items():
                    for k_m, v_m in v_c.items():
                        for k_d, v_d in v_m.items():
                            re = v_d['avg']['recall']
                            pre = v_d['avg']['prec']
                            f1 = v_d['avg']['f1']
                            results_return[f'{eval_mode_metric}_conf_%.2f'%conf + '_avg_recall'] = re
                            results_return[f'{eval_mode_metric}_conf_%.2f' % conf + '_avg_prec'] = pre
                            results_return[f'{eval_mode_metric}_conf_%.2f' % conf + '_avg_f1'] = f1
                            if best < f1:
                                best = f1
                results_return[f'{eval_mode_metric}_f1_best'] = best
        else:
            raise Exception('Not a valid f1_source!!')

    if run_ap or run_f1:
        results_tol_txt = os.path.join(opt.save_results_dir, results_name, 'results_tol.txt')
        results_tol_txt_fid = open(results_tol_txt, 'w+')
        results_tol_txt_fid.write(results_name+'\n')
        for k, v in results_return.items():
            results_tol_txt_fid.write(k+': %.5f\n' % v)
            # F1 最优值后空一行，方便人眼区分不同指标组。
            if k.endswith('_f1_best'):
                results_tol_txt_fid.write('\n')
        results_tol_txt_fid.close()

        time_txt = open(os.path.join(opt.save_results_dir, results_name, 'time.txt'), 'w')
        if inference:
            time_txt.write('forward_time_mean: %.4f\t forward frames per second: %.2f\n' % (time_mean/patch_len, 1/time_mean*patch_len))
            time_txt.write('total_preprocess_time_mean: %.4f\t preprocess frames per second: %.2f\n' % (time_preprocess_mean/patch_len, 1/time_preprocess_mean*patch_len))
            if component_stats:
                time_txt.write('===== Five-Stage Component Profiling =====\n')
                component_order = getattr(model, 'runtime_component_names', ())
                for component_name in component_order:
                    stats_dict = component_stats[component_name]
                    display_name = stats_dict.get('display_name', component_name)
                    time_txt.write(
                        f"{display_name}: runtime={stats_dict['runtime']:.6f}s, "
                        f"runtime_percent={stats_dict['runtime_percent']:.2f}%, "
                        f"Params={stats_dict['params']}, GFLOPS={stats_dict['gflops']:.6f}\n"
                    )
                total_stats = component_stats.get('Total')
                if total_stats is not None:
                    time_txt.write(
                        f"Total: runtime={total_stats['runtime']:.6f}s, "
                        f"runtime_percent={total_stats['runtime_percent']:.2f}%, "
                        f"Params={total_stats['params']}, GFLOPS={total_stats['gflops']:.6f}\n"
                    )
                time_txt.write('=========================================\n')
        time_txt.close()

    if model is not None:
        del model
    gc.collect()
    plt.close('all') # 强制关闭所有 figure，防止残留
    torch.cuda.empty_cache() # 清理 GPU 临时显存
    if run_eval_net1:
        eval_net1_points([os.path.join(opt.save_results_dir, results_name + '/')],data_dir=test_upper_path,xmlname=xmlname)
    return results_return