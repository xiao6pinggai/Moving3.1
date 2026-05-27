from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#general
from lib.dataset.dataset.coco import COCO
from lib.dataset.dataset.pascal import PascalVOC
from lib.dataset.dataset.kitti import KITTI
from lib.dataset.dataset.coco_hp import COCOHP
#remote sensing
from lib.dataset.dataset.coco_rs_car import COCO_rs_car
from lib.dataset.dataset.coco_rs_car_new import COCO_rs_car_new
from lib.dataset.dataset.coco_sdm_car import COCO_sdm_car
from lib.dataset.dataset.coco_mir import COCO_mir
from lib.dataset.dataset.coco_aircraft import COCO_aircraft
from lib.dataset.dataset.coco_rs_challenge import COCO_rschallenge

from lib.dataset.sample.ctdet_sample import CTDetDataset


dataset_factory = {
  'rs_car': COCO_rs_car,
  'rs_car_new': COCO_rs_car_new,
  'sdm_car': COCO_sdm_car,
  'mir':  COCO_mir,
  'aircraft': COCO_aircraft,
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
}
#
_sample_factory = {
  'ctdet': CTDetDataset,
}
#
task_factory = {
    'ctdet_points':'ctdet',
}

def get_dataset(opt):
  dataset = opt.datasetname
  task = task_factory[opt.task]
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
