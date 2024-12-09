import torch
import torch.nn as nn
from models.utils import get_knn_pts, index_points
from einops import repeat, rearrange
from models.pointops.functions import pointops
from modules2.SE_block import SE
from modules2.CBAM_blocks import CBAM
import torch.nn.functional as F


class Point3DConv(nn.Module):
    def __init__(self, args):
        super(Point3DConv, self).__init__()

        self.k = args.k
        self.args = args
        self.conv_delta = nn.Sequential(
            nn.Conv2d(3, args.growth_rate, 1),
            nn.BatchNorm2d(args.growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv_feats = nn.Sequential(
            nn.Conv2d(args.bn_size * args.growth_rate, args.growth_rate, 1),
            nn.BatchNorm2d(args.growth_rate),
            nn.ReLU(inplace=True)
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(args.growth_rate, args.growth_rate, 1),
            nn.BatchNorm2d(args.growth_rate),
            nn.ReLU(inplace=True)
        )
        # TODO
        self.senet = SE(in_chnls=args.feat_dim, ratio=2)  # 试试这个效果如何args.feat_dim // 2

    def forward(self, feats, pts, knn_idx=None):
        # input: (b, c, n)

        if knn_idx == None:
            # (b, 3, n, k), (b, n, k)
            knn_pts, knn_idx = get_knn_pts(self.k, pts, pts, return_idx=True)
        else:
            knn_pts = index_points(pts, knn_idx)
        # (b, 3, n, k)
        knn_delta = knn_pts - pts[..., None]
        # (b, c, n, k)
        knn_delta = self.conv_delta(knn_delta)
        # (b, c, n, k)
        knn_feats = index_points(feats, knn_idx)
        # (b, c, n, k)
        knn_feats = self.conv_feats(knn_feats)
        # multiply: (b, c, n, k)
        new_feats = knn_delta * knn_feats

        # (b, c, n, k)
        new_feats = self.post_conv(new_feats)
        # TODO 添加SE
        new_feats = self.senet(new_feats)
        # print(f"输入的形状为{new_feats.shape}") # [32, 32, 1024, 16]
        # sum: (b, c, n)
        new_feats = new_feats.sum(dim=-1)
        return new_feats


class DenseLayer(nn.Module):
    """
    Conv1d+P3DConv
    """

    def __init__(self, args, input_dim):
        super(DenseLayer, self).__init__()

        self.conv_bottle = nn.Sequential(
            nn.Conv1d(input_dim, args.bn_size * args.growth_rate, 1),
            nn.BatchNorm1d(args.bn_size * args.growth_rate),
            nn.ReLU(inplace=True)
        )
        self.point_conv = Point3DConv(args)

    def forward(self, feats, pts, knn_idx=None):
        # input: (b, c, n)

        new_feats = self.conv_bottle(feats)
        # (b, c, n)
        new_feats = self.point_conv(new_feats, pts, knn_idx)
        # concat
        return torch.cat((feats, new_feats), dim=1)


class DenseUnit(nn.Module):
    """
    DenseBlock = 3个denseLawyer
    """

    def __init__(self, args):
        super(DenseUnit, self).__init__()

        self.dense_layers = nn.ModuleList([])
        for i in range(args.layer_num):
            self.dense_layers.append(DenseLayer(args, args.feat_dim + i * args.growth_rate))

    def forward(self, feats, pts, knn_idx=None):
        # input: (b, c, n)

        for dense_layer in self.dense_layers:
            new_feats = dense_layer(feats, pts, knn_idx)
            feats = new_feats
        return feats


class Transition(nn.Module):
    def __init__(self, args):
        super(Transition, self).__init__()

        input_dim = args.feat_dim + args.layer_num * args.growth_rate
        self.trans = nn.Sequential(
            nn.Conv1d(input_dim, args.feat_dim, 1),
            nn.BatchNorm1d(args.feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        # input: (b, c, n)

        new_feats = self.trans(feats)
        return new_feats


class FeatureExtractorTry001(nn.Module):
    def __init__(self, args):
        super(FeatureExtractorTry001, self).__init__()

        self.k = args.k
        self.conv_init = nn.Sequential(
            nn.Conv1d(3, args.feat_dim, 1),
            nn.BatchNorm1d(args.feat_dim),
            nn.ReLU(inplace=True)
        )
        self.dense_blocks = nn.ModuleList([])
        for i in range(args.block_num):
            self.dense_blocks.append(nn.ModuleList([
                DenseUnit(args),
                Transition(args)
            ]))
        # TODO
        self.cbam = CBAM(args.feat_dim)

    def forward(self, pts):
        # print(f"输入的形状为{pts.shape}") # [32,3,1024]
        # input: (b, 3, n)
        # get knn_idx: (b, n, 3)
        pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
        # (b, m, k)
        knn_idx = pointops.knnquery_heap(self.k, pts_trans, pts_trans).long()
        # (b, c, n) 第一个MLP
        init_feats = self.conv_init(pts)

        local_feats = []
        local_feats.append(init_feats)
        # local features
        for dense_block, trans in self.dense_blocks:
            new_feats = dense_block(init_feats, pts, knn_idx)
            new_feats = trans(new_feats)
            # # TODO 添加空间注意力机制和通道注意力机制的结合
            # print(f"形状为：{new_feats.shape}") # [32,64,1024]
            new_feats = new_feats.unsqueeze(-1)  # 变为形状 [32, 64, 1024, 1]
            new_feats = self.cbam(new_feats)
            new_feats = new_feats.squeeze(-1)  # 恢复为 [32, 64, 1024]
            init_feats = new_feats
            local_feats.append(init_feats)
        # global features: (b, c)  四个局部特征最大池化的特征

        global_feats = init_feats.max(dim=-1)[0]
        # max_feats = init_feats.max(dim=-1)[0]
        # avg_feats = init_feats.mean(dim=-1)
        # global_feats = torch.cat((max_feats, avg_feats), dim=1)  # 合并
        return global_feats, local_feats
