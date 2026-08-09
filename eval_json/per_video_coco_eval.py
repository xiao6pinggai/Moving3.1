import contextlib
import io
import os
from collections import defaultdict

from pycocotools.cocoeval import COCOeval


VIDEO_COCO_STAT_NAMES = ['AP', 'AP50', 'AP75', 'AR100']
VIDEO_COCO_STAT_INDEXES = [0, 1, 2, 8]


def collect_video_img_ids(coco_gt, image_video_name_func, exclude_videos=None):
    exclude_videos = set(exclude_videos or [])
    videos = defaultdict(list)
    for image_info in coco_gt.dataset['images']:
        video_name = image_video_name_func(image_info)
        if video_name in exclude_videos:
            continue
        videos[video_name].append(int(image_info['id']))
    return {name: sorted(img_ids) for name, img_ids in videos.items()}


def run_per_video_coco_eval(coco_gt, coco_dt, image_video_name_func, save_dir, exclude_videos=None):
    videos = collect_video_img_ids(coco_gt, image_video_name_func, exclude_videos)
    name_width = max(len('video'), max(len(name) for name in videos))
    header = '%*s  %8s  %8s  %8s  %8s' % ((name_width, 'video') + tuple(VIDEO_COCO_STAT_NAMES))
    lines = [header]

    print('\n========== per-video coco ==========' )
    print(header)
    # 每段视频单独设置 imgIds 跑 COCOeval，屏蔽原生长输出，只保留便于横向比较的主指标。
    for video_name, img_ids in sorted(videos.items()):
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = img_ids
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        values = [coco_eval.stats[index] * 100 for index in VIDEO_COCO_STAT_INDEXES]
        line = '%*s  %8.3f  %8.3f  %8.3f  %8.3f' % ((name_width, video_name) + tuple(values))
        print(line)
        lines.append(line)

    path = os.path.join(save_dir, 'coco_video_results.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path
