import torch
import torch.nn as nn
import math
import models.pointops.functions.pointops as pointops


class DenseGenerator(nn.Module):
    def __init__(self):
        super(DenseGenerator, self).__init__()

        self.feature_extractor = FeatureExtractionUnit(num_neighbors=16, G=12)

        self.feature_expansion = FeatureExpansionUnit(use_cuda=True, upsacle_factor=4)

        self.coordinate_regression = nn.Sequential(nn.Conv1d(339 + 2, 256, 1),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv1d(256, 64, 1),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv1d(64, 3, 1))

    def forward(self, x):
        '''
        :param x:[B,N,3]
        :return: [B,3,4*N]
        '''

        y = self.feature_extractor(x.permute(0, 2, 1))  # B,C,N
        y = self.feature_expansion(y)

        return y, self.coordinate_regression(y)


class FeatureExtractionComponent(nn.Module):
    def __init__(self, pre_channel, num_neighbors, G=12):
        super(FeatureExtractionComponent, self).__init__()

        self.num_neighbors = num_neighbors
        self.pre = nn.Sequential(nn.Conv1d(pre_channel, 24, 1),
                                 nn.BatchNorm1d(24),
                                 nn.ReLU(inplace=True))

        self.grouper = pointops.QueryAndGroup(nsample=num_neighbors + 1, return_idx=True, use_xyz=False)

        in_channel = 48
        self.shared_mlp_1 = nn.Sequential(nn.Conv2d(in_channel, G, 1, bias=False),
                                          nn.BatchNorm2d(G),
                                          nn.ReLU(inplace=True))
        in_channel += G
        self.shared_mlp_2 = nn.Sequential(nn.Conv2d(in_channel, G, 1, bias=False),
                                          nn.BatchNorm2d(G),
                                          nn.ReLU(inplace=True))
        in_channel += G
        self.shared_mlp_3 = nn.Conv2d(in_channel, G, 1, bias=False)

        self.maxpool = nn.MaxPool2d([1, self.num_neighbors])

    def forward(self, x, xyz=None):
        '''
        :param x: [B,C,N]
        :return: [B,C',N]
        '''
        if xyz == None:
            xyz = x.detach()
        # pre layer [B,24,N]
        pre_points = self.pre(x)
        # knn [B,24,N,K]
        grouped_feature, grouped_xyz, indices = self.grouper(xyz=xyz.permute(0, 2, 1), features=pre_points)
        grouped_feature = grouped_feature[..., 1:]  # 去除自身点
        grouped_xyz = grouped_xyz[..., 1:]
        indices = indices[..., 1:]
        # print(grouped_feature.shape,grouped_xyz.shape,indices.shape)
        # torch.Size([2, 24, 256, 16]) torch.Size([2, 3, 256, 16]) torch.Size([2, 256, 16])

        y = torch.cat([pre_points.unsqueeze(-1).repeat(1, 1, 1, self.num_neighbors), grouped_feature], dim=1)
        # print(y.shape) #torch.Size([2, 48, 256, 16])

        y = torch.cat([self.shared_mlp_1(y), y], dim=1)  # torch.Size([2, 60, 256, 16])
        # print(y.shape)

        y = torch.cat([self.shared_mlp_2(y), y], dim=1)  # torch.Size([2, 72, 256, 16])
        # print(y.shape)

        y = torch.cat([self.shared_mlp_3(y), y], dim=1)  # torch.Size([2, 84, 256, 16])
        # print(y.shape)

        y = self.maxpool(y)  # torch.Size([2, 84, 256, 1])
        # print(y.shape)

        y = torch.cat([y.squeeze(-1), x], dim=1)
        return y


class FeatureExtractionUnit(nn.Module):
    def __init__(self, num_neighbors, G=12):
        super(FeatureExtractionUnit, self).__init__()

        self.feature_extractor_1 = FeatureExtractionComponent(pre_channel=3, num_neighbors=num_neighbors, G=G)

        self.feature_extractor_2 = FeatureExtractionComponent(pre_channel=84 + 3, num_neighbors=num_neighbors, G=G)

        self.feature_extractor_3 = FeatureExtractionComponent(pre_channel=84 + 84 + 3, num_neighbors=num_neighbors, G=G)

        self.feature_extractor_4 = FeatureExtractionComponent(pre_channel=84 + 84 + 84 + 3, num_neighbors=num_neighbors,
                                                              G=G)

    def forward(self, x):
        '''
        :param x: [B,3,N]
        :return: [B,339,N]
        '''
        y = self.feature_extractor_1(x)  # torch.Size([2, 87, 256])
        # print(f'FE 1 : {y.shape}')
        y = self.feature_extractor_2(y, x)  # torch.Size([2, 171, 256])
        # print(f'FE 2 : {y.shape}')
        y = self.feature_extractor_3(y, x)  # torch.Size([2, 255, 256])
        # print(f'FE 3 : {y.shape}')
        y = self.feature_extractor_4(y, x)  # torch.Size([2, 339, 256])
        # print(f'FE 4 : {y.shape}')
        return y


class FeatureExpansionUnit(nn.Module):
    def __init__(self, use_cuda=True, upsacle_factor=4):
        super(FeatureExpansionUnit, self).__init__()
        self.use_cuda = use_cuda
        self.upsacle_factor = upsacle_factor

    def gen_grid(self, grid_size):
        """
        output [2, grid_size x grid_size]
        """
        x = torch.linspace(-0.2, 0.2, grid_size, dtype=torch.float32)
        # grid_size x grid_size
        x, y = torch.meshgrid(x, x)

        # 2 x grid_size x grid_size
        grid = torch.stack([x, y], dim=0).view(2,
                                               grid_size * grid_size)  # [2, grid_size, grid_size] -> [2, grid_size*grid_size]
        return grid

    def cat_code(self, x):
        B, C, N = x.shape

        # code_plus = torch.ones(B,1,N)
        # code_minus = -torch.ones(B,1,N)

        code = self.gen_grid(math.ceil(math.sqrt(self.upsacle_factor * N))).expand(B, -1, -1)

        if code.shape[2] != self.upsacle_factor * N:
            # 将最后一个维度变为self.upsacle_factor * N
            code = code[:, :, :self.upsacle_factor * N]
        if self.use_cuda:
            # code_plus = code_plus.cuda()
            # code_minus = code_minus.cuda()
            code = code.cuda()


        # plus = torch.cat([x,code_plus],dim=1)
        # minus = torch.cat([x,code_minus],dim=1)

        return torch.cat([x.repeat(1, 1, self.upsacle_factor), code], dim=1)

    def forward(self, x):
        '''
        :param x:[B,C,N]
        :return: [B,C+2,4N]
        '''
        return self.cat_code(x)
