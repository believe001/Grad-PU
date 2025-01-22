import torch
import torch.nn as nn
from einops import rearrange

from models.DenseGenerator import DenseGenerator
from models.P2PNet import P2PNet
from models.utils import get_query_points


class InterpDistNet(nn.Module):
    def __init__(self, args):
        super(InterpDistNet, self).__init__()
        self.dense = DenseGenerator()

        self.p2pnet = P2PNet(args)
        self.args = args

    def forward(self, x):
        '''
        :param x: [B,3,N]
        :return: [B,4N,3]
        '''

        feature, coord = self.dense(rearrange(x, 'b c n -> b n c').contiguous().float().cuda())
        # query points:添加噪声后的点
        query_pts = get_query_points(coord, self.args)
        pred_p2p = self.p2pnet(coord, query_pts)  # TODO

        return query_pts, pred_p2p
