import os
import json
import numpy as np

from lib.utils1.utils_eval import eval_metric


def _image_video_name(image_info):
    file_name = image_info.get('file_name', '').replace('\\', '/').lstrip('./')
    for prefix in ('images/test/', 'images/test1024/'):
        if file_name.startswith(prefix):
            file_name = file_name[len(prefix):]
            break
    parts = file_name.split('/')
    for part in parts:
        if part:
            return part
    return str(image_info['id'])


def _xywh_to_xyxy(box):
    x, y, w, h = box[:4]
    return [x, y, x + w, y + h]


def eval_coco_json_f1(coco_gt, pred_json_path, save_dir, conf_ths=None, f1_mode=None,
                      eval_splits=None, opt=None, write_flag=True, data_dir=None, xmlname='xml1'):
    # 使用 COCO json 直接计算旧 F1 指标，避免再落盘 mat/xml。
    if conf_ths is None:
        conf_ths = [0.2, 0.25, 0.3, 0.35, 0.4]
    if f1_mode is None:
        f1_mode = ['iou']
    if eval_splits is None:
        eval_splits = ['all']

    with open(pred_json_path, 'r') as f:
        detections = json.load(f)

    images = coco_gt.dataset['images']
    image_by_id = {int(img['id']): img for img in images}
    video_by_img = {int(img['id']): _image_video_name(img) for img in images}

    gt_reader = eval_metric()
    gt_by_img = {}
    for image_id, image_info in image_by_id.items():
        if data_dir is not None:
            # F1 口径沿用旧逻辑的 XML 标注，只把检测结果从 mat 改为 json。
            video_name = video_by_img[image_id]
            stem = os.path.splitext(image_info.get('file_name', '').replace('\\', '/').split('/')[-1])[0]
            xml_path = os.path.join(data_dir, video_name, xmlname, stem + '.xml')
            gt_by_img[image_id] = gt_reader.getGtFromXml(xml_path).astype(np.float32).reshape(-1, 4)
        else:
            ann_ids = coco_gt.getAnnIds(imgIds=[image_id])
            gt_by_img[image_id] = np.array(
                [_xywh_to_xyxy(ann['bbox']) for ann in coco_gt.loadAnns(ann_ids)],
                dtype=np.float32
            ).reshape(-1, 4)

    det_by_img = {}
    for det in detections:
        image_id = int(det['image_id'])
        det_by_img.setdefault(image_id, []).append([
            *_xywh_to_xyxy(det['bbox']),
            float(det['score'])
        ])
    for image_id in image_by_id:
        det_by_img.setdefault(image_id, [])

    split_to_img_ids = {}
    all_img_ids = sorted(image_by_id)
    split_to_img_ids['all'] = all_img_ids
    if 'sim' in eval_splits or 'real' in eval_splits:
        split_to_img_ids['real'] = [
            image_id for image_id in all_img_ids
            if video_by_img[image_id].startswith('realobject')
        ]
        split_to_img_ids['sim'] = [
            image_id for image_id in all_img_ids
            if not video_by_img[image_id].startswith('realobject')
        ]

    if opt is not None and opt.datasetname == 'aircraft':
        img_size = [512, 512]
    elif opt is not None and opt.datasetname == 'sdm_car':
        img_size = [1080, 1920]
    elif opt is not None and opt.datasetname == 'mir':
        img_size = [352, 416]
    else:
        img_size = [1024, 1024]

    all_results = {}
    for eval_mode_metric in f1_mode:
        mode_results = {}
        for conf_thresh in conf_ths:
            conf_results = {}
            for split_name in eval_splits:
                videos = {}
                for image_id in split_to_img_ids[split_name]:
                    video_name = video_by_img[image_id]
                    videos.setdefault(video_name, []).append(image_id)

                video_results = {}
                for video_name, img_ids in sorted(videos.items()):
                    if eval_mode_metric == 'dis':
                        det_metric = eval_metric(dis_th=5, iou_th=0.5, eval_mode='dis')
                        thresh = 5
                    elif eval_mode_metric == 'iou':
                        det_metric = eval_metric(dis_th=5, iou_th=0.5, eval_mode='iou')
                        thresh = 0.5
                    else:
                        raise Exception('Not a valid eval mode!!')

                    for image_id in img_ids:
                        gt = gt_by_img[image_id]
                        det = np.array(det_by_img[image_id], dtype=np.float32).reshape(-1, 5)
                        if det.shape[0] > 0:
                            score = det[:, -1]
                            inds = np.argsort(-score)
                            det = det[inds]
                            det = det[det[:, -1] > conf_thresh]
                        det_metric.update(gt, det)

                    result = det_metric.get_result(img_size=img_size, seq_len=len(img_ids))
                    video_results[video_name] = result

                metrics = np.array([
                    [v['recall'], v['prec'], v['f1'], v['pd'], v['fa_1'], v['fa_2']]
                    for v in video_results.values()
                ])
                avg = np.mean(metrics, 0)
                video_results['avg'] = {
                    'recall': avg[0],
                    'prec': avg[1],
                    'f1': avg[2],
                    'pd': avg[3],
                    'fa1': avg[4],
                    'fa2': avg[5],
                }
                conf_results[split_name] = video_results

                if write_flag:
                    txt_name = 'reuslts_%s_%.2f_F1_%s.txt' % (
                        eval_mode_metric, conf_thresh, split_name)
                    with open(os.path.join(save_dir, txt_name), 'w+') as fid:
                        fid.write(save_dir + '\n')
                        fid.write('split=%s, evalmode=%s, conf_thresh=%.2f, thresh=%.2f\n' % (
                            split_name, eval_mode_metric, conf_thresh, thresh))
                        fid.write('video\trecall\tprecision\tf1\tpd\tfa_1\tfa_2\n')
                        for video_name, result in video_results.items():
                            if video_name == 'avg':
                                continue
                            fid.write('%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5e\t%.5e\n' % (
                                video_name, result['recall'], result['prec'], result['f1'],
                                result['pd'], result['fa_1'], result['fa_2']))
                        fid.write('avg\t%.5f\t%.5f\t%.5f\t%.5f\t%.5e\t%.5e\n' % (
                            avg[0], avg[1], avg[2], avg[3], avg[4], avg[5]))

            mode_results[conf_thresh] = conf_results
        all_results[eval_mode_metric] = mode_results

    return all_results
