from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from lib.models.Net1Only import Net1
from lib.models.spconv_centerDet_minus import sp_centerDet_minus
from lib.models.LightweightUnet3DDynamic import UNet
from lib.models.I2PSOD import I2PSOD
from lib.models.I2PSOD_test import I2PSOD_test
from lib.models.Net1_SpDetHead import Net1_SpDetHead
from lib.models.Net1_Net2 import Net1_Net2
from lib.models.UNet3D import UNet3D
def model_lib(model_chose):
    model_factory = {
                    'sp_centerDet_minus': sp_centerDet_minus,
                    'LightweightUnet3DDynamic': UNet,
                    'I2PSOD': I2PSOD,
                    'Net1': Net1,
                    'I2PSOD_test': I2PSOD_test,
                    'Net1_SpDetHead': Net1_SpDetHead,
                    'Net1_Net2': Net1_Net2,
                    "UNet3D": UNet3D,
                     }
    return model_factory[model_chose]

def get_det_net(heads, model_name, img_size, img_num, opt, thresh=None):
    model_func = model_lib(model_name)
    if model_name == 'sp_centerDet_minus':
        model = model_func(heads, img_size, img_num, layers=opt.layers, thresh=thresh)
    elif model_name == 'LightweightUnet3DDynamic':
        model = model_func(out_channel_list=[32, 64, 128], num_classes=1)
    elif model_name in {'I2PSOD', 'Net1', 'I2PSOD_test', 'Net1_SpDetHead', 'Net1_Net2'}:
        model_kwargs = dict(
            image_size=img_size,
            img_num=opt.seqLen,
            layers=opt.layers,
            thresh=opt.thresh,
            feat_channels=opt.feat_channels,
            T_pooling=opt.T_pooling,
            groups=opt.groups,
            downsample_mode=opt.downsample_mode,
            net1name=opt.net1name,
        )
        if model_name == 'Net1':
            model = model_func(heads, **model_kwargs, opt=opt)
        else:
            model = model_func(heads, **model_kwargs, opt=opt)
    elif model_name == 'UNet3D':
        model = model_func(heads, input_channels=3, feat_channels=opt.feat_channels, T_pooling=opt.T_pooling, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, Snack_skip=opt.Snack_skip, Snack_max_offset=opt.Snack_max_offset, UNet3D_skip=opt.UNet3D_skip, TZSConv_skip=opt.TZSConv_skip)
    else:
        model = model_func(heads)
    return model


def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
