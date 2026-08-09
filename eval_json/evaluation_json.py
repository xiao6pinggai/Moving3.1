import argparse
import contextlib
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)
from utils_eval import eval_metric
from per_video_coco_eval import run_per_video_coco_eval


COCO_STAT_NAMES = [
    'ap', 'ap50', 'ap75', 'ap_small', 'ap_medium', 'ap_large',
    'ar_max1', 'ar_max10', 'ar_max100', 'ar_small', 'ar_medium', 'ar_large'
]


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in text.split(',') if x.strip()]


def default_save_dir(pred_json):
    pred_dir = os.path.dirname(os.path.abspath(pred_json))
    pred_name = os.path.splitext(os.path.basename(pred_json))[0]
    # 未指定 save_dir 时，按预测 json 文件名生成同级评估目录，便于多模型结果并列查看。
    return os.path.join(pred_dir, 'eval_%s' % pred_name)


def image_video_name(image_info):
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


def xywh_to_xyxy(box):
    x, y, w, h = box[:4]
    return [x, y, x + w, y + h]


def get_split_img_ids(coco_gt, eval_splits, exclude_videos=None):
    exclude_videos = set(exclude_videos or [])
    all_img_ids = []
    for image_info in coco_gt.dataset['images']:
        video_name = image_video_name(image_info)
        if video_name not in exclude_videos:
            all_img_ids.append(int(image_info['id']))
    split_img_ids = {'all': sorted(all_img_ids)}
    if 'sim' in eval_splits or 'real' in eval_splits:
        real_img_ids, sim_img_ids = [], []
        for image_info in coco_gt.dataset['images']:
            video_name = image_video_name(image_info)
            if video_name in exclude_videos:
                continue
            if video_name.startswith('realobject'):
                real_img_ids.append(int(image_info['id']))
            else:
                sim_img_ids.append(int(image_info['id']))
        split_img_ids['real'] = sorted(real_img_ids)
        split_img_ids['sim'] = sorted(sim_img_ids)
    return split_img_ids


def run_coco_eval(coco_gt, pred_json, eval_splits, save_dir, exclude_videos=None):
    coco_dt = coco_gt.loadRes(pred_json)
    split_img_ids = get_split_img_ids(coco_gt, eval_splits, exclude_videos)
    coco_results = {}
    for split_name in eval_splits:
        print('\n========== %s ==========' % split_name)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = split_img_ids[split_name]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_results[split_name] = coco_eval.stats.copy()

    path = os.path.join(save_dir, 'coco_results.txt')
    with open(path, 'w') as f:
        f.write('split\tmetric\tvalue\n')
        for split_name in eval_splits:
            for name, value in zip(COCO_STAT_NAMES, coco_results[split_name]):
                f.write('%s\t%s\t%.5f\n' % (split_name, name, value))
    return coco_results


def get_img_size(coco_gt, img_ids):
    image = coco_gt.loadImgs([img_ids[0]])[0]
    return [int(image.get('height', 512)), int(image.get('width', 512))]


def run_json_f1(coco_gt, pred_json, gt_json, eval_splits, f1_modes, conf_ths, save_dir, exclude_videos=None):
    with open(pred_json, 'r') as f:
        detections = json.load(f)

    image_by_id = {int(img['id']): img for img in coco_gt.dataset['images']}
    video_by_img = {image_id: image_video_name(img) for image_id, img in image_by_id.items()}

    gt_by_img = {}
    for image_id in image_by_id:
        ann_ids = coco_gt.getAnnIds(imgIds=[image_id])
        gt_by_img[image_id] = np.array(
            [xywh_to_xyxy(ann['bbox']) for ann in coco_gt.loadAnns(ann_ids)],
            dtype=np.float32
        ).reshape(-1, 4)

    det_by_img = defaultdict(list)
    for det in detections:
        image_id = int(det['image_id'])
        det_by_img[image_id].append([*xywh_to_xyxy(det['bbox']), float(det['score'])])

    split_img_ids = get_split_img_ids(coco_gt, eval_splits, exclude_videos)
    result_summary = {}
    for eval_mode in f1_modes:
        for conf_th in conf_ths:
            for split_name in eval_splits:
                videos = defaultdict(list)
                for image_id in split_img_ids[split_name]:
                    videos[video_by_img[image_id]].append(image_id)

                video_results = {}
                for video_name, img_ids in sorted(videos.items()):
                    if eval_mode == 'iou':
                        metric = eval_metric(dis_th=5, iou_th=0.5, eval_mode='iou')
                        thresh = 0.5
                    elif eval_mode == 'dis':
                        metric = eval_metric(dis_th=5, iou_th=0.5, eval_mode='dis')
                        thresh = 5
                    else:
                        raise ValueError('unsupported f1 mode: %s' % eval_mode)

                    for image_id in img_ids:
                        gt = gt_by_img[image_id]
                        det = np.array(det_by_img.get(image_id, []), dtype=np.float32).reshape(-1, 5)
                        if det.shape[0] > 0:
                            score = det[:, -1]
                            inds = np.argsort(-score)
                            det = det[inds]
                            det = det[det[:, -1] > conf_th]
                        metric.update(gt, det)

                    video_results[video_name] = metric.get_result(
                        img_size=get_img_size(coco_gt, img_ids), seq_len=len(img_ids))

                metrics = np.array([
                    [v['recall'], v['prec'], v['f1'], v['pd'], v['fa_1'], v['fa_2']]
                    for v in video_results.values()
                ])
                avg = np.mean(metrics, 0)
                video_results['avg'] = {
                    'recall': avg[0], 'prec': avg[1], 'f1': avg[2],
                    'pd': avg[3], 'fa_1': avg[4], 'fa_2': avg[5]
                }
                result_summary[(eval_mode, conf_th, split_name)] = video_results['avg']

                txt_name = 'results_%s_%.2f_F1_%s.txt' % (eval_mode, conf_th, split_name)
                with open(os.path.join(save_dir, txt_name), 'w') as f:
                    f.write('pred_json=%s\n' % pred_json)
                    f.write('gt_json=%s\n' % gt_json)
                    f.write('split=%s, evalmode=%s, conf_thresh=%.2f, thresh=%.2f\n' % (
                        split_name, eval_mode, conf_th, thresh))
                    f.write('video\trecall\tprecision\tf1\tpd\tfa_1\tfa_2\n')
                    for video_name, result in video_results.items():
                        if video_name == 'avg':
                            continue
                        f.write('%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5e\t%.5e\n' % (
                            video_name, result['recall'], result['prec'], result['f1'],
                            result['pd'], result['fa_1'], result['fa_2']))
                    f.write('avg\t%.5f\t%.5f\t%.5f\t%.5f\t%.5e\t%.5e\n' % (
                        avg[0], avg[1], avg[2], avg[3], avg[4], avg[5]))

    with open(os.path.join(save_dir, 'f1_results.txt'), 'w') as f:
        f.write('evalmode\tconf\tsplit\trecall\tprecision\tf1\tpd\tfa_1\tfa_2\n')
        for (eval_mode, conf_th, split_name), result in sorted(result_summary.items()):
            f.write('%s\t%.2f\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5e\t%.5e\n' % (
                eval_mode, conf_th, split_name, result['recall'], result['prec'], result['f1'],
                result['pd'], result['fa_1'], result['fa_2']))
    return result_summary


def load_track_disp(mot_json, disp_bins):
    """从 MOT json 加载轨迹 id，并按每条轨迹的平均帧间位移(中心距/帧间隔)分箱。

    返回:
      mot_ann_by_img: image_id -> 该图的 MOT 标注列表
      cat_by_img:     image_id -> 与 mot_ann_by_img 对齐的类别索引列表
      cat_names:      类别名列表(最后一个是 no_disp)
      video_of_img:   image_id -> video_id
      track_cat:      (video_id, track_id) -> 类别索引
    """
    with open(mot_json, 'r') as f:
        mot = json.load(f)
    frame_of_img = {int(i['id']): int(i['frame_id']) for i in mot['images']}
    video_of_img = {int(i['id']): int(i['video_id']) for i in mot['images']}

    tracks = defaultdict(list)  # (vid, tid) -> [(frame_id, cx, cy)]
    mot_ann_by_img = defaultdict(list)
    for a in mot['annotations']:
        image_id = int(a['image_id'])
        vid = video_of_img[image_id]
        tid = int(a['track_id'])
        x, y, w, h = a['bbox']
        tracks[(vid, tid)].append((frame_of_img[image_id], x + w / 2.0, y + h / 2.0))
        mot_ann_by_img[image_id].append(a)

    bin_edges = sorted(disp_bins)

    def classify(disp):
        if not np.isfinite(disp):
            return len(bin_edges) + 1
        for i, hi in enumerate(bin_edges):
            if disp < hi:
                return i
        return len(bin_edges)

    track_cat = {}
    for (vid, tid), items in tracks.items():
        items = sorted(items)
        if len(items) < 2:  # 仅 1 帧无法计算位移 -> no_disp
            track_cat[(vid, tid)] = len(bin_edges) + 1
            continue
        per_frame = []
        for (f0, x0, y0), (f1, x1, y1) in zip(items[:-1], items[1:]):
            gap = f1 - f0
            d = np.hypot(x1 - x0, y1 - y0)
            per_frame.append(d / gap if gap > 0 else np.nan)
        track_cat[(vid, tid)] = classify(float(np.nanmean(per_frame)))

    cat_by_img = {}
    for image_id, anns in mot_ann_by_img.items():
        vid = video_of_img[image_id]
        cat_by_img[image_id] = [track_cat[(vid, int(a['track_id']))] for a in anns]

    cat_names = []
    for i, hi in enumerate(bin_edges):
        lo = bin_edges[i - 1] if i > 0 else 0
        cat_names.append('%g<=disp<%g' % (lo, hi))
    cat_names.append('disp>=%g' % bin_edges[-1])
    cat_names.append('no_disp')
    return mot_ann_by_img, cat_by_img, cat_names, video_of_img, track_cat


def nearest_gt_cat(det_box, gt, cats):
    """将无匹配的检测框(fp)归属到本帧中心距离最近的 gt 框所属类别(近似)。
    无 gt 时归入最后一个类别(no_disp)。"""
    if gt.shape[0] == 0:
        return len(cats) - 1
    dc = np.array([(det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2])
    gc = np.stack([(gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2], 1)
    dist = np.sum((gc - dc) ** 2, 1)
    return int(cats[np.argmin(dist)])


def run_json_f1_disp(coco_gt, pred_json, gt_json, mot_json, disp_bins, eval_splits, f1_modes,
                     conf_ths, save_dir, exclude_videos=None, fp_attr='none'):
    """按轨迹帧间位移类别评估 F1 类指标(新增功能，不影响原有 per-video F1 流程)。

    检测框无轨迹 id，因此逐帧匹配后:
      - tp/fn 由被匹配/未匹配的 gt 框类别精确定义;
      - fp 仅在 fp_attr=='nearest' 时按最近 gt 框启发式归属(近似)。
    """
    mot_ann_by_img, cat_by_img, cat_names, video_of_img, track_cat = load_track_disp(mot_json, disp_bins)
    n_cat = len(cat_names)
    use_fp = fp_attr == 'nearest'

    with open(pred_json, 'r') as f:
        detections = json.load(f)
    image_by_id = {int(img['id']): img for img in coco_gt.dataset['images']}

    # gt 框改用 MOT json(与 gt_json 标注一致且含轨迹 id)，与 cat_by_img 天然对齐
    gt_by_img = {}
    for image_id in image_by_id:
        anns = mot_ann_by_img.get(image_id, [])
        gt_by_img[image_id] = np.array(
            [xywh_to_xyxy(a['bbox']) for a in anns], dtype=np.float32).reshape(-1, 4)

    det_by_img = defaultdict(list)
    for det in detections:
        image_id = int(det['image_id'])
        det_by_img[image_id].append([*xywh_to_xyxy(det['bbox']), float(det['score'])])

    split_img_ids = get_split_img_ids(coco_gt, eval_splits, exclude_videos)

    rows = []
    print('\n========== F1 per displacement category ==========')
    for eval_mode in f1_modes:
        metric = eval_metric(dis_th=5, iou_th=0.5, eval_mode=eval_mode)
        for conf_th in conf_ths:
            for split_name in eval_splits:
                img_ids = split_img_ids[split_name]
                n_frames = len(img_ids)
                cat_tp = np.zeros(n_cat)
                cat_fn = np.zeros(n_cat)
                cat_fp = np.zeros(n_cat)
                track_total = defaultdict(int)  # (vid,tid) -> 该轨迹参与评估的帧数
                track_hit = defaultdict(int)    # (vid,tid) -> 被命中的帧数

                for image_id in img_ids:
                    anns = mot_ann_by_img.get(image_id, [])
                    cats = cat_by_img.get(image_id, [])
                    gt = gt_by_img[image_id]
                    vid = video_of_img.get(image_id, -1)

                    det = np.array(det_by_img.get(image_id, []), dtype=np.float32).reshape(-1, 5)
                    if det.shape[0] > 0:
                        score = det[:, -1]
                        inds = np.argsort(-score)
                        det = det[inds]
                        det = det[det[:, -1] > conf_th]

                    if gt.shape[0] > 0:
                        for a in anns:
                            track_total[(vid, int(a['track_id']))] += 1

                    matched_pairs = metric.update(gt, det, return_match=True)
                    if matched_pairs.shape[0] > 0:
                        matched_gt = set(matched_pairs[:, 1].tolist())
                        for gi, c in enumerate(cats):
                            if gi in matched_gt:
                                cat_tp[c] += 1
                                track_hit[(vid, int(anns[gi]['track_id']))] += 1
                            else:
                                cat_fn[c] += 1
                    elif gt.shape[0] > 0:
                        for gi, c in enumerate(cats):
                            cat_fn[c] += 1

                    if use_fp and det.shape[0] > matched_pairs.shape[0]:
                        matched_det = set(matched_pairs[:, 0].tolist())
                        for di in range(det.shape[0]):
                            if di not in matched_det:
                                cat_fp[nearest_gt_cat(det[di, :4], gt, cats)] += 1

                img_size = get_img_size(coco_gt, img_ids)
                area = img_size[0] * img_size[1]
                cat_track_count = np.zeros(n_cat)
                for key in track_total:
                    cat_track_count[track_cat[key]] += 1

                print('\n--- split=%s eval_mode=%s conf=%.2f ---' % (split_name, eval_mode, conf_th))
                print('%-16s %8s %9s %9s %10s %9s' % ('disp_cat', 'n_tracks', 'n_gtbox', 'recall_pd', 'track_hit', 'prec'))
                for c in range(n_cat):
                    n_gt = cat_tp[c] + cat_fn[c]
                    recall = cat_tp[c] / n_gt if n_gt > 0 else np.nan
                    keys = [k for k in track_total if track_cat[k] == c]
                    if keys:
                        hit_rate = float(np.mean([track_hit[k] / track_total[k] for k in keys]))
                    else:
                        hit_rate = np.nan
                    if use_fp:
                        prec = cat_tp[c] / (cat_tp[c] + cat_fp[c]) if (cat_tp[c] + cat_fp[c]) > 0 else np.nan
                        f1 = 2 * recall * prec / (recall + prec) if (
                            np.isfinite(recall) and np.isfinite(prec) and (recall + prec) > 0) else np.nan
                        fa_1 = cat_fp[c] / (area * n_frames) if n_frames else np.nan
                        fa_2 = cat_fp[c] / n_frames if n_frames else np.nan
                    else:
                        prec = f1 = fa_1 = fa_2 = np.nan
                    print('%-16s %8d %9d %9s %10s %9s' % (
                        cat_names[c], cat_track_count[c], n_gt,
                        '%.2f' % (recall * 100) if np.isfinite(recall) else 'NA',
                        '%.2f' % (hit_rate * 100) if np.isfinite(hit_rate) else 'NA',
                        '%.2f' % (prec * 100) if np.isfinite(prec) else 'NA'))
                    rows.append((eval_mode, conf_th, split_name, cat_names[c], cat_track_count[c], n_gt,
                                 recall, hit_rate, prec, f1, fa_1, fa_2))

    path = os.path.join(save_dir, 'f1_disp_results.txt')
    with open(path, 'w') as f:
        f.write('pred_json=%s\ngt_json=%s\nmot_json=%s\n' % (pred_json, gt_json, mot_json))
        f.write('evalmode\tconf\tsplit\tdisp_cat\tn_tracks\tn_gt_boxes\tbox_recall_pd\ttrack_hit_rate\tprecision\tf1\tfa_1\tfa_2\n')
        for row in rows:
            # 前 4 个比例指标(recall/hit_rate/prec/f1)输出为两位小数百分数，fa_1/fa_2 保持计数
            vals = []
            for i, v in enumerate(row[6:12]):
                if isinstance(v, float) and np.isfinite(v):
                    vals.append('%.2f' % (v * 100) if i < 4 else '%.5f' % v)
                else:
                    vals.append('NA')
            f.write('%s\t%.2f\t%s\t%s\t%d\t%d\t%s\n' % (
                row[0], row[1], row[2], row[3], row[4], row[5], '\t'.join(vals)))
    print('\ndisplacement-category F1 results saved to: %s' % path)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate prediction COCO json with COCOeval and F1 metrics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pred_json', default='/root/autodl-tmp/SGNet/weights/aircraft_multi/UNet3D/UNet3D_163264128_snack111555_wobn_L5_womask_seglen10_2026_08_09_18_34_18/results_model_best_ap50/results_model_best_ap50.json')
    parser.add_argument('--gt_json', default='/root/autodl-tmp/AircraftDataset27/annotations/annotations_test_new.json')
    parser.add_argument('--save_dir', default=None, help='directory to save metric txt files; None means eval_<pred_json_stem> beside pred_json')
    parser.add_argument('--eval_splits', default='all,real,sim', help='comma list: all,real,sim')
    parser.add_argument('--f1_mode', default='iou', help='comma list: iou,dis')
    parser.add_argument('--exclude_videos', default='realobject_1083_2_2_5_10')
    parser.add_argument('--conf_ths', default='0.2,0.25,0.3,0.35,0.4', help='comma separated confidence thresholds')
    parser.add_argument('--no_coco', action='store_true', default=False, help='skip COCOeval metrics')
    parser.add_argument('--run_video_coco', action='store_true', default=False, help='run COCOeval for each video and print compact percentage table')
    parser.add_argument('--no_f1', action='store_true', default=True, help='skip F1 metrics')
    parser.add_argument('--mot_json', default='/root/autodl-tmp/AircraftDataset27/annotations/test_mot_new.json',
                        help='MOT json with track ids, bbox consistent with gt_json; used only for --disp_bins')
    parser.add_argument('--disp_bins', default='5,10,20', help='comma list of per-frame displacement bin edges (px), e.g. 5,10,20; empty disables per-track displacement classification')
    parser.add_argument('--fp_attr', default='none', choices=['none', 'nearest'],
                        help="how to attribute false positives to displacement categories; 'nearest'=assign to nearest gt box category (approximate precision/F1)")
    args = parser.parse_args()

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = default_save_dir(args.pred_json)
    os.makedirs(save_dir, exist_ok=True)

    coco_gt = COCO(args.gt_json)
    eval_splits = parse_str_list(args.eval_splits)
    f1_modes = parse_str_list(args.f1_mode)
    conf_ths = parse_float_list(args.conf_ths)
    exclude_videos = parse_str_list(args.exclude_videos)

    if not args.no_coco:
        run_coco_eval(coco_gt, args.pred_json, eval_splits, save_dir, exclude_videos)
    if args.run_video_coco:
        coco_dt = coco_gt.loadRes(args.pred_json)
        run_per_video_coco_eval(coco_gt, coco_dt, image_video_name, save_dir, exclude_videos)
    if not args.no_f1:
        run_json_f1(coco_gt, args.pred_json, args.gt_json, eval_splits, f1_modes, conf_ths, save_dir, exclude_videos)

    disp_bins = parse_float_list(args.disp_bins)
    if disp_bins:
        run_json_f1_disp(coco_gt, args.pred_json, args.gt_json, args.mot_json, disp_bins,
                         eval_splits, f1_modes, conf_ths, save_dir, exclude_videos, args.fp_attr)

    print('\nresults saved to: %s' % save_dir)


if __name__ == '__main__':
    main()
