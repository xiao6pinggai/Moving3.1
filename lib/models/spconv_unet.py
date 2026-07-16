from functools import partial

import numpy as np
import torch
import torch.nn as nn
import os, sys

# 向上查找项目根目录并加入 sys.path（支持 autodl/本地 Windows 双环境）
_cur = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_cur, 'path_setup.py')):
    _cur = os.path.dirname(_cur)
if _cur not in sys.path:
    sys.path.insert(0, _cur)

from lib.models.spconv_utils import replace_feature, spconv
# from lib.utils import common_utils
from lib.models.spconv_backbone import GroupedDilatedBlock, GroupedDilatedBlock2, post_act_block
from lib.models.cos_matchv10 import SparseSymmetricCosineAttention
from lib.models.cos_dis_v11 import SparseSymmetricCosineAttention as SparseSymmetricCosineAttentionV11
from lib.models.cos_update_v12 import SparseSymmetricCosineAttention as SparseSymmetricCosineAttentionV12
from lib.models.cos_update_v13 import FrameRestrictedSparseTrajectoryTransformer
from lib.models.cos_update_v14 import ObjectCentricAssociationTokenFusion
from lib.models.cos_update_v15 import SparseTrajectoryTokenModule
from lib.models.cos_update_v16 import TripletMotionConsistencySparseConv as TripletMotionConsistencySparseConvV16
from lib.models.cos_update_v17 import TripletMotionConsistencyCosineSparseConv
from lib.models.cos_update_v18 import TripletMotionConsistencyMotionPairSparseConv
from lib.models.cos_update_v20 import TripletMotionConsistencyMotionPairSparseConv as TripletMotionConsistencyMotionPairSparseConvV20
from lib.models.cos_update_v21 import TripletMotionConsistencyMotionPairSparseConv as TripletMotionConsistencyMotionPairSparseConvV21
from lib.models.cos_update_v22 import TripletMotionConsistencyMotionPairSparseConv as TripletMotionConsistencyMotionPairSparseConvV22
from lib.models.cos_update_v23 import TripletMotionConsistencyMotionPairSparseConv as TripletMotionConsistencyMotionPairSparseConvV23
from lib.models.cos_update_v24 import TripletMotionConsistencyMotionPairSparseConv as TripletMotionConsistencyMotionPairSparseConvV24
from lib.models.cos_update_v25 import TripletMotionConsistencyMotionPairSparseConv as TripletMotionConsistencyMotionPairSparseConvV25
from lib.models.cos_update_v26 import TripletMotionConsistencyMotionPairSparseConv as TripletMotionConsistencyMotionPairSparseConvV26
from lib.models.cos_update_v27 import TripletMotionConsistencyMotionPairSparseConv as TripletMotionConsistencyMotionPairSparseConvV27
from lib.models.cos_update_v28 import TripletMotionConsistencyMotionPairSparseConv as TripletMotionConsistencyMotionPairSparseConvV28
from lib.models.cos_update_v29 import TripletMotionConsistencyMotionPairSparseConv as TripletMotionConsistencyMotionPairSparseConvV29
from lib.models.cos_update_v30 import TripletMotionConsistencyMotionPairSparseConv as TripletMotionConsistencyMotionPairSparseConvV30
# from lib.models.se import SparseSymmetricCosineAttention, SparseSEModule


MFE_MODULE_NAMES = (
    'cosv10', 'cosv11', 'cosv12', 'cosv13', 'frstt',
    'cosv14', 'ocatf', 'object_token',
    'cosv15', 'sttm', 'sparse_traj_token',
    'cosv16', 'tmc', 'tmc_sconv',
    'cosv17', 'tmc_cos', 'tmc_cos_sconv',
    'cosv18', 'tmc_motion_pair', 'tmc_motion_pair_sconv',
    'cosv20', 'tmc_motion_pair_attn', 'tmc_motion_pair_attn_sconv',
    'cosv21', 'tmc_motion_pair_branch_gate', 'tmc_motion_pair_branch_gate_sconv',
    'cosv22', 'tmc_motion_pair_repeat', 'tmc_motion_pair_repeat_sconv',
    'cosv23', 'tmc_feature_pair_marginal_attn', 'tmc_feature_pair_marginal_attn_sconv',
    'cosv24', 'tmc_feature_pair_multidilation_attn', 'tmc_feature_pair_multidilation_attn_sconv',
    'cosv25', 'tmc_feature_neighbor_attn', 'tmc_feature_neighbor_attn_sconv',
    'cosv26', 'tmc_neighbor_sa_cross_attn', 'tmc_neighbor_sa_cross_attn_sconv',
    'cosv27', 'tmc_feature_pair_temp_attn', 'tmc_feature_pair_temp_attn_sconv',
    'cosv28', 'tmc_feature_pair_bin_attn', 'tmc_feature_pair_bin_attn_sconv',
    'cosv29', 'tmc_feature_pair_sigmoid_attn', 'tmc_feature_pair_sigmoid_attn_sconv',
    'cosv30', 'tmc_feature_pair_sigmoid_9d_attn', 'tmc_feature_pair_sigmoid_9d_attn_sconv',
)
BOTTLE_MFE_WITH_2_BLOCK = 'mfew2block'


def parse_optional_module_name(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == '':
            return None
        if text.lower() in ('none', 'null'):
            return None
        return text.lower()
    return value


def parse_triplet_values(value, cast, name):
    if value is None:
        return (cast(3), cast(3), cast(3))
    if isinstance(value, str):
        text = value.strip().strip('[]()')
        if ',' in text:
            tokens = [token.strip() for token in text.split(',') if token.strip()]
        else:
            tokens = text.split()
            if len(tokens) == 1:
                tokens = tokens * 3
    else:
        tokens = list(value) if isinstance(value, (tuple, list)) else [value] * 3
    if len(tokens) == 1:
        tokens = tokens * 3
    if len(tokens) != 3:
        raise ValueError(f'{name} should contain 1 or 3 values for conv1/2/3, got {value}')
    return tuple(cast(token) for token in tokens)


def parse_mfe_skip(value):
    if value is None:
        return (True, True, True)
    if isinstance(value, str):
        text = value.strip().strip('[]()')
        if ',' in text:
            tokens = [token.strip() for token in text.split(',') if token.strip()]
        else:
            tokens = text.split()
            if len(tokens) == 1 and len(tokens[0]) == 3 and all(ch in '01' for ch in tokens[0]):
                tokens = list(tokens[0])
    else:
        tokens = list(value)
    if len(tokens) != 3:
        raise ValueError(f'MFE_skip should contain 3 bool values for conv1/2/3, got {value}')

    def to_bool(token):
        if isinstance(token, bool):
            return token
        if isinstance(token, (int, float)):
            return bool(token)
        token = str(token).strip().lower()
        if token in ('1', 'true', 't', 'yes', 'y'):
            return True
        if token in ('0', 'false', 'f', 'no', 'n'):
            return False
        raise ValueError(f'Cannot parse MFE_skip value: {token}')

    return tuple(to_bool(token) for token in tokens)


def build_mfe_module(mfe_name, *args, opt=None, **kwargs):
    tmc_level = kwargs.pop('tmc_level', None)
    if mfe_name == 'cosv12':
        mfe_cls = SparseSymmetricCosineAttentionV12
    elif mfe_name in ('cosv13', 'frstt'):
        mfe_cls = FrameRestrictedSparseTrajectoryTransformer
    elif mfe_name in ('cosv14', 'ocatf', 'object_token'):
        mfe_cls = ObjectCentricAssociationTokenFusion
    elif mfe_name in ('cosv15', 'sttm', 'sparse_traj_token'):
        mfe_cls = SparseTrajectoryTokenModule
    elif mfe_name == 'cosv11':
        mfe_cls = SparseSymmetricCosineAttentionV11
    elif mfe_name == 'cosv10':
        mfe_cls = SparseSymmetricCosineAttention
    elif mfe_name in ('cosv16', 'tmc', 'tmc_sconv'):
        mfe_cls = TripletMotionConsistencySparseConvV16
    elif mfe_name in ('cosv17', 'tmc_cos', 'tmc_cos_sconv'):
        mfe_cls = TripletMotionConsistencyCosineSparseConv
    elif mfe_name in ('cosv18', 'tmc_motion_pair', 'tmc_motion_pair_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConv
    elif mfe_name in ('cosv20', 'tmc_motion_pair_attn', 'tmc_motion_pair_attn_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConvV20
    elif mfe_name in ('cosv21', 'tmc_motion_pair_branch_gate', 'tmc_motion_pair_branch_gate_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConvV21
    elif mfe_name in ('cosv22', 'tmc_motion_pair_repeat', 'tmc_motion_pair_repeat_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConvV22
    elif mfe_name in ('cosv23', 'tmc_feature_pair_marginal_attn', 'tmc_feature_pair_marginal_attn_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConvV23
    elif mfe_name in ('cosv24', 'tmc_feature_pair_multidilation_attn', 'tmc_feature_pair_multidilation_attn_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConvV24
    elif mfe_name in ('cosv25', 'tmc_feature_neighbor_attn', 'tmc_feature_neighbor_attn_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConvV25
    elif mfe_name in ('cosv26', 'tmc_neighbor_sa_cross_attn', 'tmc_neighbor_sa_cross_attn_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConvV26
    elif mfe_name in ('cosv27', 'tmc_feature_pair_temp_attn', 'tmc_feature_pair_temp_attn_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConvV27
    elif mfe_name in ('cosv28', 'tmc_feature_pair_bin_attn', 'tmc_feature_pair_bin_attn_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConvV28
    elif mfe_name in ('cosv29', 'tmc_feature_pair_sigmoid_attn', 'tmc_feature_pair_sigmoid_attn_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConvV29
    elif mfe_name in ('cosv30', 'tmc_feature_pair_sigmoid_9d_attn', 'tmc_feature_pair_sigmoid_9d_attn_sconv'):
        mfe_cls = TripletMotionConsistencyMotionPairSparseConvV30
    else:
        raise ValueError(f'Unknown mfe_name: {mfe_name}')
    if opt is not None and mfe_name in ('cosv13', 'frstt', 'cosv14', 'ocatf', 'object_token', 'cosv15', 'sttm', 'sparse_traj_token'):
        kwargs['num_frames'] = int(opt.seqLen)
    if opt is not None and mfe_name in ('cosv16', 'tmc', 'tmc_sconv', 'cosv17', 'tmc_cos', 'tmc_cos_sconv', 'cosv18', 'tmc_motion_pair', 'tmc_motion_pair_sconv', 'cosv20', 'tmc_motion_pair_attn', 'tmc_motion_pair_attn_sconv', 'cosv21', 'tmc_motion_pair_branch_gate', 'tmc_motion_pair_branch_gate_sconv', 'cosv22', 'tmc_motion_pair_repeat', 'tmc_motion_pair_repeat_sconv', 'cosv23', 'tmc_feature_pair_marginal_attn', 'tmc_feature_pair_marginal_attn_sconv', 'cosv24', 'tmc_feature_pair_multidilation_attn', 'tmc_feature_pair_multidilation_attn_sconv', 'cosv25', 'tmc_feature_neighbor_attn', 'tmc_feature_neighbor_attn_sconv', 'cosv26', 'tmc_neighbor_sa_cross_attn', 'tmc_neighbor_sa_cross_attn_sconv', 'cosv27', 'tmc_feature_pair_temp_attn', 'tmc_feature_pair_temp_attn_sconv', 'cosv28', 'tmc_feature_pair_bin_attn', 'tmc_feature_pair_bin_attn_sconv', 'cosv29', 'tmc_feature_pair_sigmoid_attn', 'tmc_feature_pair_sigmoid_attn_sconv', 'cosv30', 'tmc_feature_pair_sigmoid_9d_attn', 'tmc_feature_pair_sigmoid_9d_attn_sconv'):
        topk_values = parse_triplet_values(opt.tmc_topk, int, 'tmc_topk')
        window_values = parse_triplet_values(opt.tmc_window_size, int, 'tmc_window_size')
        idx = int(tmc_level) if tmc_level is not None else 0
        kwargs.setdefault('topk', int(topk_values[idx]))
        kwargs.setdefault('window_size', int(window_values[idx]))
        kwargs.setdefault('hidden_ratio', float(opt.tmc_hidden_ratio))
        kwargs.setdefault('pos_hidden', int(opt.tmc_pos_hidden))
        kwargs.setdefault('pos_scale', float(opt.tmc_pos_scale))
        is_v25_or_v26 = mfe_name in ('cosv25', 'tmc_feature_neighbor_attn', 'tmc_feature_neighbor_attn_sconv', 'cosv26', 'tmc_neighbor_sa_cross_attn', 'tmc_neighbor_sa_cross_attn_sconv')
        is_v27_or_v28 = mfe_name in ('cosv27', 'tmc_feature_pair_temp_attn', 'tmc_feature_pair_temp_attn_sconv', 'cosv28', 'tmc_feature_pair_bin_attn', 'tmc_feature_pair_bin_attn_sconv', 'cosv29', 'tmc_feature_pair_sigmoid_attn', 'tmc_feature_pair_sigmoid_attn_sconv', 'cosv30', 'tmc_feature_pair_sigmoid_9d_attn', 'tmc_feature_pair_sigmoid_9d_attn_sconv')
        if not is_v25_or_v26:
            kwargs.setdefault('chunk_size', int(opt.tmc_chunk_size))
        if not is_v25_or_v26 and not is_v27_or_v28:
            kwargs.setdefault('ffn_position', opt.tmc_ffn_position)
            kwargs.setdefault('topk_relu', opt.topk_relu)
        if mfe_name in ('cosv24', 'tmc_feature_pair_multidilation_attn', 'tmc_feature_pair_multidilation_attn_sconv', 'cosv25', 'tmc_feature_neighbor_attn', 'tmc_feature_neighbor_attn_sconv', 'cosv26', 'tmc_neighbor_sa_cross_attn', 'tmc_neighbor_sa_cross_attn_sconv'):
            kwargs.setdefault('tpdilation', opt.tpdilation)
        if mfe_name in ('cosv22', 'tmc_motion_pair_repeat', 'tmc_motion_pair_repeat_sconv'):
            repeat_values = parse_triplet_values(opt.MFErepeat, int, 'MFErepeat')
            kwargs.setdefault('repeat', int(repeat_values[idx]))
    return mfe_cls(*args, **kwargs)
  

class UNetV2_3_T_nodown_v2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg  # None
        self.opt = kwargs['opt']
        try:
            self.MFE = self.opt.MFE
        except:
            self.MFE = None
            print("Warning: MFE is not specified in opt, set to None.")
        self.bottle_enhancement = parse_optional_module_name(getattr(self.opt, 'bottle_enhancement', None))
        self.mfe_skip = parse_mfe_skip(getattr(self.opt, 'MFE_skip', (True, True, True)))
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )

        block = post_act_block

        # ------------------------------------------------------------------
        # Encoder
        # stem:
        #   conv_input: Cin -> 16
        #
        # encoder:
        #   conv1: 16 -> 16, SubM + SubM
        #   conv2: 16 -> 32, SP(stride=(1,2,2)) + SubM
        #   conv3: 32 -> 64, SP(stride=(1,2,2)) + SubM
        #   conv4: 64 -> 64, SP(stride=(1,2,2)) + SubM
        # ------------------------------------------------------------------

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(
                16, 32, 3,
                norm_fn=norm_fn,
                stride=(1, 2, 2),
                padding=1,
                indice_key='spconv2',
                conv_type='spconv',
            ),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(
                32, 64, 3,
                norm_fn=norm_fn,
                stride=(1, 2, 2),
                padding=1,
                indice_key='spconv3',
                conv_type='spconv',
            ),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.x_bottle_down = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=(1, 2, 2), padding=1, indice_key='spconv_b', conv_type='spconv'),
        )

        if self.bottle_enhancement == 'block':
            self.x_bottle_enhancement = block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_3')
        elif self.bottle_enhancement == BOTTLE_MFE_WITH_2_BLOCK:
            bottle_mfe_name = parse_optional_module_name(getattr(self.opt, 'MFE', None))
            if bottle_mfe_name not in MFE_MODULE_NAMES:
                bottle_mfe_name = 'cosv18'
            self.x_bottle_pre_block = block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_mfe_pre')
            self.x_bottle_enhancement = build_mfe_module(
                bottle_mfe_name, opt=self.opt, tmc_level=2, in_channels=128,
                kernel_size=3, use_qkv=False, use_maxpool=False, alpha=0.5,
                use_biqkv=False, temporal_dilation=1, conv=spconv.SparseSequential())
            self.x_bottle_post_block = block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_mfe_post')
        elif self.bottle_enhancement in MFE_MODULE_NAMES:
            self.x_bottle_enhancement = build_mfe_module(
                self.bottle_enhancement, opt=self.opt, tmc_level=2, in_channels=128,
                kernel_size=3, use_qkv=False, use_maxpool=False, alpha=0.5,
                use_biqkv=False, temporal_dilation=1, conv=spconv.SparseSequential())
        elif self.bottle_enhancement is None:
            self.x_bottle_enhancement = None
        else:
            raise ValueError(
                f'Unknown bottle_enhancement: {self.bottle_enhancement}. '
                f'Expected None, "block", "{BOTTLE_MFE_WITH_2_BLOCK}", or one of {MFE_MODULE_NAMES}.'
            )

        self.x_bottle_up = spconv.SparseSequential(
            block(128, 64, 3, stride=(1, 2, 2), padding=1, norm_fn=norm_fn, indice_key='spconv_b', conv_type='inverseconv'),
        )

        # ------------------------------------------------------------------
        # MFE: keep the original structure.
        # 这里不新增 shortcut4，也不单独写 apply 方法。
        # 原始 MFE 只控制 x_conv1 / x_conv2 / x_conv3 三条 skip。
        # ------------------------------------------------------------------

        if self.MFE in MFE_MODULE_NAMES:
            self.sptial2d1 = block(16, 16, (1, 3, 3), norm_fn=norm_fn,
                dialtion=(1, 1, 1), padding=(0, 1, 1), indice_key='subm1_1') if False else spconv.SparseSequential()
            self.sptial2d2 = block(32, 32, (1, 3, 3), norm_fn=norm_fn,
                dialtion=(1, 1, 1), padding=(0, 1, 1), indice_key='subm2_2') if False else spconv.SparseSequential()
            self.sptial2d3 = block(64, 64, (1, 3, 3), norm_fn=norm_fn,
                dialtion=(1, 1, 1), padding=(0, 1, 1), indice_key='subm3_3') if False else spconv.SparseSequential()


            self.shortcut1 = build_mfe_module(self.MFE, opt=self.opt,
                tmc_level=0, in_channels=16, kernel_size=9,
                use_qkv=False, use_maxpool=False, alpha=0.5,
                use_biqkv=False, temporal_dilation=1, conv=self.sptial2d1)
            self.shortcut2 = build_mfe_module(self.MFE, opt=self.opt,
                tmc_level=1, in_channels=32, kernel_size=7,
                use_qkv=False, use_maxpool=False, alpha=0.5,
                use_biqkv=False, temporal_dilation=1, conv=self.sptial2d2)
            self.shortcut3 = build_mfe_module(self.MFE, opt=self.opt,
                tmc_level=2, in_channels=64, kernel_size=5,
                use_qkv=False, use_maxpool=False, alpha=0.5,
                use_biqkv=False, temporal_dilation=1, conv=self.sptial2d3)


        # if self.MFE in ('cosv10', 'cosv11', 'cosv12', 'cosv13', 'frstt', 'cosv14', 'ocatf', 'object_token'):
        #     self.shortcut1fusion = GroupedDilatedBlock(
        #         16, 16, 3,
        #         dilations=[1, 2, 3, 4],
        #         indice_key='subm1',
        #     )
        #     self.shortcut2fusion = GroupedDilatedBlock(
        #         32, 32, 3,
        #         dilations=[1, 2, 3, 4],
        #         indice_key='subm2',
        #     )
        #     self.shortcut3fusion = GroupedDilatedBlock(
        #         64, 64, 3,
        #         dilations=[1, 2, 3, 4],
        #         indice_key='subm3',
        #     )


        # ------------------------------------------------------------------
        # Decoder
        #
        # conv4 已被真正的 bottleneck 替换。
        # x_conv3 作为最深的 skip，x_bottle 作为 bottom 分支。
        #
        # up3:
        #   skip3 64 + bottle 64 -> 64 -> inv to level2
        #
        # up2:
        #   skip2 32 + up3 32 -> 32 -> inv to level1
        #
        # up1:
        #   skip1 16 + up2 16 -> 16 -> output
        # ------------------------------------------------------------------
        # 跳线连接前
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn) if False else spconv.SparseSequential()
        self.conv_up_m3 = block(
            128, 64, 3,
            norm_fn=norm_fn,
            padding=1,
            indice_key='subm3',
        )
        self.inv_conv3 = block(64, 32, 3, stride=(1, 2, 2), padding=1, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')
        # 跳线连接前
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn) if False else spconv.SparseSequential()
        self.conv_up_m2 = block(
            64, 32, 3,
            norm_fn=norm_fn,
            padding=1,
            indice_key='subm2',
        )
        self.inv_conv2 = block(32, 16, 3, stride=(1, 2, 2), padding=1, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')
        # 跳线连接前
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn) if False else spconv.SparseSequential()
        self.conv_up_m1 = block(
            32, 16, 3,
            norm_fn=norm_fn,
            padding=1,
            indice_key='subm1',
        )

        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))

        self.num_point_features = 16
        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:
            SparseConvTensor with reduced channel dimension.
        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                voxel_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]

        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )

        x = self.conv_input(input_sp_tensor)

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)

        
        if self.bottle_enhancement == 'block':
            x_bottle = self.x_bottle_down(x_conv3)
            x_bottle = self.x_bottle_enhancement(x_bottle)
            x_bottle = self.x_bottle_up(x_bottle)
        elif self.bottle_enhancement == BOTTLE_MFE_WITH_2_BLOCK:
            x_bottle = self.x_bottle_down(x_conv3)
            x_bottle = self.x_bottle_pre_block(x_bottle)
            x_bottle = replace_feature(x_bottle, self.x_bottle_enhancement(x_bottle)[0])
            x_bottle = self.x_bottle_post_block(x_bottle)
            x_bottle = self.x_bottle_up(x_bottle)
        elif self.bottle_enhancement in MFE_MODULE_NAMES:
            x_bottle = self.x_bottle_down(x_conv3)
            x_bottle = replace_feature(x_bottle, self.x_bottle_enhancement(x_bottle)[0])
            x_bottle = self.x_bottle_up(x_bottle)
        elif not self.bottle_enhancement:
            x_bottle = spconv.SparseSequential(x_conv3)

        # ------------------------------------------------------------------
        # Decoder
        # x_conv3 is the deepest skip and x_bottle is the bottom branch.
        # ------------------------------------------------------------------
        if self.MFE in MFE_MODULE_NAMES and self.mfe_skip[2]:
            x_conv3 = replace_feature(x_conv3, self.shortcut3(x_conv3)[0])
            # x_conv3 = self.shortcut3fusion(x_conv3)

        x_up3 = self.UR_block_forward(
            x_conv3,
            x_bottle,
            self.conv_up_t3,
            self.conv_up_m3,
            self.inv_conv3,
        )

        if self.MFE in MFE_MODULE_NAMES and self.mfe_skip[1]:
            x_conv2 = replace_feature(x_conv2, self.shortcut2(x_conv2)[0])
            # x_conv2 = self.shortcut2fusion(x_conv2)

        x_up2 = self.UR_block_forward(
            x_conv2,
            x_up3,
            self.conv_up_t2,
            self.conv_up_m2,
            self.inv_conv2,
        )

        if self.MFE in MFE_MODULE_NAMES and self.mfe_skip[0]:
            x_conv1 = replace_feature(x_conv1, self.shortcut1(x_conv1)[0])
            # x_conv1 = self.shortcut1fusion(x_conv1)

        x_up1 = self.UR_block_forward(
            x_conv1,
            x_up2,
            self.conv_up_t1,
            self.conv_up_m1,
            self.conv5,
        )

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1

        # 这里保留你原代码的设置，避免影响后续模块接口。
        # 但从当前 U-Net 输出分辨率看，x_up1 已经回到 conv1/input 同级分辨率。
        # 如果后续模块严格使用真实 stride，这里理论上应检查是否需要改为 1。
        batch_dict['encoded_spconv_tensor_stride'] = 8

        return batch_dict
###################################################################################################################
# 阅读代码时，以下请忽略，本工程中仅使用UNetV2_3_T_nodown_v2类，以下保留仅为参考。
# 阅读代码时，以下请忽略，本工程中仅使用UNetV2_3_T_nodown_v2类，以下保留仅为参考。
# 阅读代码时，以下请忽略，本工程中仅使用UNetV2_3_T_nodown_v2类，以下保留仅为参考。
######################v2备份，新版增加unet的宽度，每层3卷积##########################


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out
class SparseBasicBlock2D3D(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock2D3D, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=(1,3,3), stride=stride, padding=(0,1,1), bias=False, indice_key=f'{indice_key}_temp'
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

class UNetV2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.num_point_features = 16

        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict

class UNetV2_3(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # None
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.num_point_features = 16

        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv3)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict
    

class UNetV2_2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        # self.conv3 = spconv.SparseSequential(
        #     # [800, 704, 21] <- [400, 352, 11]
        #     block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        # )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        # self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        # self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        # self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.num_point_features = 16

        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv2)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        # x_up3 = self.UR_block_forward(x_conv3, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_conv2, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict
    

class UNetV2_3_32(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 32, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(32),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(128, 256, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(256),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(128, 128, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(128, 64, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(64, 64, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(128, 64, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(32, 32, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.num_point_features = 32

        self.out_channel = 32

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv3)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict

class UNetV2_3_T_nodown(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # None
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.num_point_features = 16

        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv3)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict



class UNetV2_3_T_nodown_maxpool(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # None
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.use_maxpool = True
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # === 核心修改：构建下采样 Block 的辅助函数 ===
        # === 核心修改：构建下采样 Block 的辅助函数 ===
        def make_down_block(in_ch, out_ch, key, stride):
            if self.use_maxpool:
                return spconv.SparseSequential(
                    # 1. 修正点：将 indice_key 给 MaxPool
                    # 这样 InverseConv 才能通过这个 key 找到下采样时的坐标映射
                    spconv.SparseMaxPool3d(kernel_size=3, stride=stride, padding=1, indice_key=key),
                    
                    # 2. 修正点：后面的 SubMConv 必须用一个新的 key
                    # 如果复用 key，会覆盖掉 MaxPool 的索引信息，导致报错。
                    # 这里简单的加上 '_subm' 后缀即可
                    block(in_ch, out_ch, 3, norm_fn=norm_fn, padding=1, indice_key=key + '_subm', conv_type='subm') 
                )
            else:
                return block(in_ch, out_ch, 3, norm_fn=norm_fn, stride=stride, padding=1, 
                             indice_key=key, conv_type='spconv')
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            # block(16, 32, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv2', conv_type='spconv'),
            make_down_block(16,32,'spconv2',stride=(1,2,2)),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            # block(32, 64, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv3', conv_type='spconv'),
            make_down_block(32,64,'spconv3',stride=(1,2,2)),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.num_point_features = 16

        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv3)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict
    



class UNetV2_3_T_nodown_v2_bf(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # None
        self.opt = kwargs['opt']
        self.MFE = self.opt.MFE
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        # self.cos_match2 = SparseSymmetricCosineAttention(kernel_size=9)
        # self.cos_match3 = SparseSymmetricCosineAttention(kernel_size=7)
        # self.cos_match4 = SparseSymmetricCosineAttention(kernel_size=5)
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv2', conv_type='spconv'), # d1
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv3', conv_type='spconv'), #d2
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )
        # self.x_bottle = spconv.SparseSequential(
        #     block(64, 128, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv_b', conv_type='spconv'),#d3
        #     # 1. 升维: 64 -> 128 (使用 SubMConv 保持形状，或者 stride=1 的 Conv)
        #     # block(64, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_1'),
        #     # 真正的bottle
        #     # 2. (可选) 中间深层处理: 128 -> 128
        #     # block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_2'),
        #     # 3. 降维: 128 -> 64
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_3'), 

        #     block(128, 64, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv_b', conv_type='inverseconv')
        # )
        
        self.x_bottle = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_3'),
            block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_3'), 
        )
        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]

        # self.conv_up_t1 = SparseSymmetricCosineAttention(kernel_size=9)
        # self.se1 = SparseSEModule(channel=16, reduction=2)
        # self.se2 = SparseSEModule(channel=32, reduction=2)
        # self.se3 = SparseSEModule(channel=64, reduction=2)
        # self.sptial2d1 = block(16, 16, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm1_1') # 3d
        # self.sptial2d2 = block(32, 32, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm2_2')
        # self.sptial2d3 = block(64, 64, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm3_3')


        # mfe # V2!!!
        if self.MFE in ('cosv10', 'cosv13', 'frstt', 'cosv14', 'ocatf', 'object_token', 'cosv15', 'sttm', 'sparse_traj_token'):
            if self.MFE in ('cosv14', 'ocatf', 'object_token', 'cosv15', 'sttm', 'sparse_traj_token'):
                self.sptial2d1 = spconv.SparseSequential()
                self.sptial2d2 = spconv.SparseSequential()
                self.sptial2d3 = spconv.SparseSequential()
            else:
                self.sptial2d1 = block(16, 16, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm1_1') # 2d
                self.sptial2d2 = block(32, 32, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm2_2')
                self.sptial2d3 = block(64, 64, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm3_3')
            self.shortcut1 = build_mfe_module(self.MFE, opt=self.opt, in_channels=16, kernel_size=41, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=1, conv=self.sptial2d1)
            self.shortcut2 = build_mfe_module(self.MFE, opt=self.opt, in_channels=32, kernel_size=21, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=1, conv=self.sptial2d2)
            self.shortcut3 = build_mfe_module(self.MFE, opt=self.opt, in_channels=64, kernel_size=11, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=1, conv=self.sptial2d3)
        # mfe
        
        
        # self.shortcut1fusion = block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        # self.shortcut2fusion = block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2')
        # self.shortcut3fusion = block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')


        # mfe
        if self.MFE in ('cosv10', 'cosv13', 'frstt'):
            self.shortcut1fusion = GroupedDilatedBlock(16, 16, 3, dilations=[1, 2, 3,4], indice_key='subm1')
            self.shortcut2fusion = GroupedDilatedBlock(32, 32, 3, dilations=[1, 2, 3,4], indice_key='subm2')
            self.shortcut3fusion = GroupedDilatedBlock(64, 64, 3, dilations=[1, 2, 3,4], indice_key='subm3')
        # mfe


        # self.shortcut1fusion = SparseBasicBlock2D3D(16, 16, norm_fn=norm_fn, indice_key='subm1')
        # self.shortcut2fusion = SparseBasicBlock2D3D(32, 32, norm_fn=norm_fn, indice_key='subm2')
        # self.shortcut3fusion = SparseBasicBlock2D3D(64, 64, norm_fn=norm_fn, indice_key='subm3')
        
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.num_point_features = 16

        self.out_channel = 16

    # def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
    #     x_trans = conv_t(x_lateral)
    #     x_trans = x_lateral
    #     x = x_trans
    #     x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
    #     x_m = conv_m(x)
    #     x = self.channel_reduction(x, x_m.features.shape[1])
    #     x = replace_feature(x, x_m.features + x.features)
    #     x = conv_inv(x)
    #     return x
    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x
    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        ###
        # enhanced_feat1,_ = self.cos_match2(x_conv1.indices, x_conv1.features)
        # x_conv1 = replace_feature(x_conv1, enhanced_feat1)

        x_conv2 = self.conv2(x_conv1)
        ###
        # enhanced_feat2,_ = self.cos_match3(x_conv2.indices, x_conv2.features)
        # x_conv2 = replace_feature(x_conv2, enhanced_feat2)

        x_conv3 = self.conv3(x_conv2)
        ###
        # enhanced_feat3,_ = self.cos_match4(x_conv3.indices, x_conv3.features)
        # x_conv3 = replace_feature(x_conv3, enhanced_feat3)
        # x_conv4 = self.conv4(x_conv3)

        
        x_bottle = self.x_bottle(x_conv3)
        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11

        # w/o shortcut process
        # x_conv3 = self.se3(x_conv3)
        if self.MFE in ('cosv10', 'cosv13', 'frstt', 'cosv14', 'ocatf', 'object_token', 'cosv15', 'sttm', 'sparse_traj_token'):
            x_conv3 = replace_feature(x_conv3, self.shortcut3(x_conv3)[0])
            if self.MFE == 'cosv10':
                x_conv3 = self.shortcut3fusion(x_conv3)
        x_up3 = self.UR_block_forward(x_conv3, x_bottle, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        # w/o shortcut process
        # x_conv2 = self.se2(x_conv2)
        if self.MFE in ('cosv10', 'cosv13', 'frstt', 'cosv14', 'ocatf', 'object_token', 'cosv15', 'sttm', 'sparse_traj_token'):
            x_conv2 = replace_feature(x_conv2, self.shortcut2(x_conv2)[0])
            if self.MFE == 'cosv10':
                x_conv2 = self.shortcut2fusion(x_conv2)
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        # w/o shortcut process
        # x_conv1 = self.se1(x_conv1)
        if self.MFE in ('cosv10', 'cosv13', 'frstt', 'cosv14', 'ocatf', 'object_token', 'cosv15', 'sttm', 'sparse_traj_token'):
            x_conv1 = replace_feature(x_conv1, self.shortcut1(x_conv1)[0])
            if self.MFE == 'cosv10':
                x_conv1 = self.shortcut1fusion(x_conv1)
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict
class UNetV2_3_T_nodown_v2_bfbf(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg  # None
        self.opt = kwargs['opt']
        self.MFE = self.opt.MFE
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )

        block = post_act_block

        # ------------------------------------------------------------------
        # Encoder
        # stem:
        #   conv_input: Cin -> 16
        #
        # encoder:
        #   conv1: 16 -> 16, SubM + SubM
        #   conv2: 16 -> 32, SP(stride=(1,2,2)) + SubM
        #   conv3: 32 -> 64, SP(stride=(1,2,2)) + SubM
        #   conv4: 64 -> 64, SP(stride=(1,2,2)) + SubM
        # ------------------------------------------------------------------

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(
                16, 32, 3,
                norm_fn=norm_fn,
                stride=(1, 2, 2),
                padding=1,
                indice_key='spconv2',
                conv_type='spconv',
            ),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(
                32, 64, 3,
                norm_fn=norm_fn,
                stride=(1, 2, 2),
                padding=1,
                indice_key='spconv3',
                conv_type='spconv',
            ),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(
                64, 64, 3,
                norm_fn=norm_fn,
                stride=(1, 2, 2),
                padding=1,
                indice_key='spconv4',
                conv_type='spconv',
            ),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        # ------------------------------------------------------------------
        # MFE: keep the original structure.
        # 这里不新增 shortcut4，也不单独写 apply 方法。
        # 原始 MFE 只控制 x_conv1 / x_conv2 / x_conv3 三条 skip。
        # ------------------------------------------------------------------

        if self.MFE in ('cosv10', 'cosv13', 'frstt', 'cosv14', 'ocatf', 'object_token', 'cosv15', 'sttm', 'sparse_traj_token'):
            self.sptial2d1 = block(16, 16, (1, 3, 3), norm_fn=norm_fn,
                dialtion=(1, 1, 1), padding=(0, 1, 1), indice_key='subm1_1')
            self.sptial2d2 = block(32, 32, (1, 3, 3), norm_fn=norm_fn,
                dialtion=(1, 1, 1), padding=(0, 1, 1), indice_key='subm2_2')
            self.sptial2d3 = block(64, 64, (1, 3, 3), norm_fn=norm_fn,
                dialtion=(1, 1, 1), padding=(0, 1, 1), indice_key='subm3_3') if False else spconv.SparseSequential()


            self.shortcut1 = build_mfe_module(self.MFE, opt=self.opt,
                in_channels=16,
                kernel_size=9,
                use_qkv=False,
                use_maxpool=False,
                alpha=0.5,
                use_biqkv=False,
                temporal_dilation=1,
                conv=self.sptial2d1,
            )
            self.shortcut2 = build_mfe_module(self.MFE, opt=self.opt,
                in_channels=32,
                kernel_size=7,
                use_qkv=False,
                use_maxpool=False,
                alpha=0.5,
                use_biqkv=False,
                temporal_dilation=1,
                conv=self.sptial2d2,
            )
            self.shortcut3 = build_mfe_module(self.MFE, opt=self.opt,
                in_channels=64,
                kernel_size=5,
                use_qkv=False,
                use_maxpool=False,
                alpha=0.5,
                use_biqkv=False,
                temporal_dilation=1,
                conv=self.sptial2d3,
            )


        # if self.MFE == 'cosv10':
        #     self.shortcut1fusion = GroupedDilatedBlock(
        #         16, 16, 3,
        #         dilations=[1, 2, 3, 4],
        #         indice_key='subm1',
        #     )
        #     self.shortcut2fusion = GroupedDilatedBlock(
        #         32, 32, 3,
        #         dilations=[1, 2, 3, 4],
        #         indice_key='subm2',
        #     )
        #     self.shortcut3fusion = GroupedDilatedBlock(
        #         64, 64, 3,
        #         dilations=[1, 2, 3, 4],
        #         indice_key='subm3',
        #     )


        # ------------------------------------------------------------------
        # Decoder
        #
        # 去掉 bottle 后，最底部直接使用 x_conv4。
        #
        # up4:
        #   x_lateral = x_conv4
        #   x_bottom  = x_conv4
        #   concat: 64 + 64 = 128
        #   conv_m4: 128 -> 64
        #   inv_conv4: 64 -> 64, level4 -> level3
        #
        # up3:
        #   skip3 64 + up4 64 -> 64 -> inv to level2
        #
        # up2:
        #   skip2 32 + up3 32 -> 32 -> inv to level1
        #
        # up1:
        #   skip1 16 + up2 16 -> 16 -> output
        # ------------------------------------------------------------------
        # 跳线连接前
        self.conv_up_t4 = SparseBasicBlock(
            64, 64,
            indice_key='subm4',
            norm_fn=norm_fn,
        ) if False else spconv.SparseSequential()

        self.conv_up_m4 = block(
            128, 64, 3,
            norm_fn=norm_fn,
            padding=1,
            indice_key='subm4',
        )
        self.inv_conv4 = block(
            64, 64, 3,
            stride=(1, 2, 2),
            padding=1,
            norm_fn=norm_fn,
            indice_key='spconv4',
            conv_type='inverseconv',
        )
        # 跳线连接前
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn) if False else spconv.SparseSequential()
        self.conv_up_m3 = block(
            128, 64, 3,
            norm_fn=norm_fn,
            padding=1,
            indice_key='subm3',
        )
        self.inv_conv3 = block(64, 32, 3, stride=(1, 2, 2), padding=1, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')
        # 跳线连接前
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn) if False else spconv.SparseSequential()
        self.conv_up_m2 = block(
            64, 32, 3,
            norm_fn=norm_fn,
            padding=1,
            indice_key='subm2',
        )
        self.inv_conv2 = block(32, 16, 3, stride=(1, 2, 2), padding=1, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')
        # 跳线连接前
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn) if False else spconv.SparseSequential()
        self.conv_up_m1 = block(
            32, 16, 3,
            norm_fn=norm_fn,
            padding=1,
            indice_key='subm1',
        )

        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))

        self.num_point_features = 16
        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:
            SparseConvTensor with reduced channel dimension.
        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                voxel_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]

        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )

        x = self.conv_input(input_sp_tensor)

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # ------------------------------------------------------------------
        # Decoder
        # No explicit bottle.
        # The deepest encoder feature x_conv4 is used as both lateral and bottom.
        # ------------------------------------------------------------------
        x_up4 = self.UR_block_forward(
            x_conv4,
            x_conv4,
            self.conv_up_t4,
            self.conv_up_m4,
            self.inv_conv4,
        )

        if self.MFE in ('cosv10', 'cosv13', 'frstt', 'cosv14', 'ocatf', 'object_token', 'cosv15', 'sttm', 'sparse_traj_token'):
            x_conv3 = replace_feature(x_conv3, self.shortcut3(x_conv3)[0])
            # x_conv3 = self.shortcut3fusion(x_conv3)

        x_up3 = self.UR_block_forward(
            x_conv3,
            x_up4,
            self.conv_up_t3,
            self.conv_up_m3,
            self.inv_conv3,
        )

        if self.MFE in ('cosv10', 'cosv13', 'frstt', 'cosv14', 'ocatf', 'object_token', 'cosv15', 'sttm', 'sparse_traj_token'):
            x_conv2 = replace_feature(x_conv2, self.shortcut2(x_conv2)[0])
            # x_conv2 = self.shortcut2fusion(x_conv2)

        x_up2 = self.UR_block_forward(
            x_conv2,
            x_up3,
            self.conv_up_t2,
            self.conv_up_m2,
            self.inv_conv2,
        )

        if self.MFE in ('cosv10', 'cosv13', 'frstt', 'cosv14', 'ocatf', 'object_token', 'cosv15', 'sttm', 'sparse_traj_token'):
            x_conv1 = replace_feature(x_conv1, self.shortcut1(x_conv1)[0])
            # x_conv1 = self.shortcut1fusion(x_conv1)

        x_up1 = self.UR_block_forward(
            x_conv1,
            x_up2,
            self.conv_up_t1,
            self.conv_up_m1,
            self.conv5,
        )

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1

        # 这里保留你原代码的设置，避免影响后续模块接口。
        # 但从当前 U-Net 输出分辨率看，x_up1 已经回到 conv1/input 同级分辨率。
        # 如果后续模块严格使用真实 stride，这里理论上应检查是否需要改为 1。
        batch_dict['encoded_spconv_tensor_stride'] = 8

        return batch_dict
  

class UNetV2_3_T_nodown_v3(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # None
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        # self.cos_match2 = SparseSymmetricCosineAttention(kernel_size=9)
        # self.cos_match3 = SparseSymmetricCosineAttention(kernel_size=7)
        # self.cos_match4 = SparseSymmetricCosineAttention(kernel_size=5)
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv2', conv_type='spconv'), # d1
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv3', conv_type='spconv'), #d2
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )
        self.x_bottle = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv_b', conv_type='spconv'),#d3
            # 1. 升维: 64 -> 128 (使用 SubMConv 保持形状，或者 stride=1 的 Conv)
            # block(64, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_1'),
            # 真正的bottle
            # 2. (可选) 中间深层处理: 128 -> 128
            # block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_2'),
            # 3. 降维: 128 -> 64
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_3'),

            block(128, 64, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv_b', conv_type='inverseconv')
        )
        

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]

        # self.conv_up_t1 = SparseSymmetricCosineAttention(kernel_size=9)
        # self.se1 = SparseSEModule(channel=16, reduction=2)
        # self.se2 = SparseSEModule(channel=32, reduction=2)
        # self.se3 = SparseSEModule(channel=64, reduction=2)
        # self.sptial2d1 = block(16, 16, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm1_1') # 3d
        # self.sptial2d2 = block(32, 32, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm2_2')
        # self.sptial2d3 = block(64, 64, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm3_3')


        # 
        self.sptial2d1 = block(16, 16, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm1_1') # 2d
        self.sptial2d2 = block(32, 32, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm2_2')
        self.sptial2d3 = block(64, 64, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm3_3')

        # self.sptial2d1 = GroupedDilatedBlock2(16, 16, 3, dilations=[1, 2, 3], indice_key='subm1') # 2d
        # self.sptial2d2 = GroupedDilatedBlock2(32, 32, 3, dilations=[1, 2, 3], indice_key='subm2')
        # self.sptial2d3 = GroupedDilatedBlock2(64, 64, 3, dilations=[1, 2, 3], indice_key='subm3')
        
        self.shortcut1 = SparseSymmetricCosineAttention(in_channels=16, kernel_size=13, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=2, conv=self.sptial2d1)
        self.shortcut2 = SparseSymmetricCosineAttention(in_channels=32, kernel_size=9, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=2, conv=self.sptial2d2)
        self.shortcut3 = SparseSymmetricCosineAttention(in_channels=64, kernel_size=5, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=2, conv=self.sptial2d3)
        # 13-->6 9-->4*2  5-->2*4
        
        
        # self.shortcut1fusion = block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        # self.shortcut2fusion = block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2')
        # self.shortcut3fusion = block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')


        # 
        self.shortcut1fusion = GroupedDilatedBlock2(16, 16, 3, dilations=[1, 2, 3, 4], indice_key='subm1')
        self.shortcut2fusion = GroupedDilatedBlock2(32, 32, 3, dilations=[1, 2, 3, 4], indice_key='subm2')
        self.shortcut3fusion = GroupedDilatedBlock2(64, 64, 3, dilations=[1, 2, 3, 4], indice_key='subm3')
        # 


        # self.shortcut1fusion = SparseBasicBlock2D3D(16, 16, norm_fn=norm_fn, indice_key='subm1')
        # self.shortcut2fusion = SparseBasicBlock2D3D(32, 32, norm_fn=norm_fn, indice_key='subm2')
        # self.shortcut3fusion = SparseBasicBlock2D3D(64, 64, norm_fn=norm_fn, indice_key='subm3')
        
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.num_point_features = 16

        self.out_channel = 16

    # def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
    #     x_trans = conv_t(x_lateral)
    #     x_trans = x_lateral
    #     x = x_trans
    #     x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
    #     x_m = conv_m(x)
    #     x = self.channel_reduction(x, x_m.features.shape[1])
    #     x = replace_feature(x, x_m.features + x.features)
    #     x = conv_inv(x)
    #     return x
    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x
    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        ###
        # enhanced_feat1,_ = self.cos_match2(x_conv1.indices, x_conv1.features)
        # x_conv1 = replace_feature(x_conv1, enhanced_feat1)

        x_conv2 = self.conv2(x_conv1)
        ###
        # enhanced_feat2,_ = self.cos_match3(x_conv2.indices, x_conv2.features)
        # x_conv2 = replace_feature(x_conv2, enhanced_feat2)

        x_conv3 = self.conv3(x_conv2)
        ###
        # enhanced_feat3,_ = self.cos_match4(x_conv3.indices, x_conv3.features)
        # x_conv3 = replace_feature(x_conv3, enhanced_feat3)
        # x_conv4 = self.conv4(x_conv3)

        
        x_bottle = self.x_bottle(x_conv3)
        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11

        # w/o shortcut process
        # x_conv3 = self.se3(x_conv3)
        x_conv3 = replace_feature(x_conv3, self.shortcut3(x_conv3)[0])
        x_up3 = self.UR_block_forward(x_conv3, x_bottle, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        # w/o shortcut process
        # x_conv2 = self.se2(x_conv2)
        x_conv2 = replace_feature(x_conv2, self.shortcut2(x_conv2)[0])
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        # w/o shortcut process
        # x_conv1 = self.se1(x_conv1)
        x_conv1 = replace_feature(x_conv1, self.shortcut1(x_conv1)[0])
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict

if __name__ == '__main__':
    image_size = [512,512]
    img_num = 10
    grid_size = np.array([image_size[1], image_size[0], img_num - 1])
    model = UNetV2_3_T_nodown_v3(1,grid_size)
    print(model)
