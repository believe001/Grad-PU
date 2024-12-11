import torch
import torch.nn as nn
from einops import repeat
from models.FeatureExtractorTry001 import FeatureExtractorTry001
from models.P2PRegressor import P2PRegressor
from models.utils import get_knn_pts, index_points
class P2PNet(nn.Module):
    def __init__(self, args):
        super(P2PNet, self).__init__()

        self.args = args
        # 原来
        # self.feature_extractor = FeatureExtractor(args)

        self.feature_extractor = FeatureExtractorTry001(args)
        self.p2p_regressor = P2PRegressor(args)
    def extract_feature(self, original_pts):
        # input: (b, 3, n)

        # global_feats: (b, c), local_feats: list (b, c, n)
        global_feats, local_feats = self.feature_extractor(original_pts)
        return global_feats, local_feats


    def interpolate_feature(self, original_pts, query_pts, local_feat):
        """
        求查询点的k近邻点特征加权值
        计算查询点p的k个邻居点neighbor_i。
        根据距离算出来权重w_i,然后w_i * neighbor_feature = feat_i。最后将p的所有的feat_i相加
        """
        k = 3
        # interpolation: (b, 3, n, k), (b, n, k)
        knn_pts, knn_idx = get_knn_pts(k, original_pts, query_pts, return_idx=True)
        # dist
        repeat_query_pts = repeat(query_pts, 'b c n -> b c n k', k=k)
        # (b, n, k) 对应论文中的偏移量
        dist = torch.norm(knn_pts - repeat_query_pts, p=2, dim=1)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        # (b, n, k)
        weight = dist_recip / norm
        # (b, c, n, k)
        knn_feat = index_points(local_feat, knn_idx)
        # (b, c, n, k)
        interpolated_feat = knn_feat * weight.unsqueeze(1)
        # (b, c, n)： 可以反映查询点的局部环境。
        interpolated_feat = torch.sum(interpolated_feat, dim=-1)
        return interpolated_feat


    def regress_distance(self, original_pts, query_pts, global_feats, local_feats):
        """
        根据插值特征，输入到回归函数中去得到点对点距离。
        """
        # pts: (b, 3, n) global_feats: (b, c), local_feats: list (b, c, n)

        # (b, c, n)
        global_feats = repeat(global_feats, 'b c -> b c n', n=query_pts.shape[-1])
        # interpolated local feats
        interpolated_local_feats = []
        for feat in local_feats:
            # (b, c, n)
            interpolated_feat = self.interpolate_feature(original_pts, query_pts, feat)
            interpolated_local_feats.append(interpolated_feat)
        # (b, c*(block_num+1), n)
        agg_local_feats = torch.cat(interpolated_local_feats, dim=1)
        # (b, 3+c*(block_num+2), m)
        agg_feats = torch.cat((query_pts, agg_local_feats, global_feats), dim=1)
        # (b, 1, n)
        p2p = self.p2p_regressor(agg_feats)
        return p2p


    def forward(self, original_pts, query_pts):
        # input: (b, 3, n)

        # global_feats: (b, c), local_feats: list (b, c, n)
        global_feats, local_feats = self.extract_feature(original_pts)
        # (b, 1, n)
        p2p = self.regress_distance(original_pts, query_pts, global_feats, local_feats)
        return p2p