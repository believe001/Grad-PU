import os
import torch
import sys

sys.path.append(os.getcwd())
import time
from models.InterpDistNet import InterpDistNet
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import PUDataset
from models.P2PNet import P2PNet
from args.pu1k_args import parse_pu1k_args
from args.pugan_args import parse_pugan_args
from args.utils import str2bool
from models.utils import *
import argparse


def min_max_normalize(loss, min_val, max_val, epsilon=1e-8):
    return (loss - min_val) / (max_val - min_val + epsilon)


def train(args):
    set_seed(args.seed)
    start = time.time()

    # load data
    train_dataset = PUDataset(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)

    # set up folders for checkpoints and logs
    str_time = datetime.now().isoformat()
    output_dir = os.path.join(args.out_path, str_time)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    logger = get_logger('train', log_dir)
    logger.info('Experiment ID: %s' % (str_time))

    # create model
    logger.info('========== Build Model ==========')
    model = InterpDistNet(args)
    model = model.cuda()
    # get the parameter size
    para_num = sum([p.numel() for p in model.parameters()])
    logger.info("=== The number of parameters in model: {:.4f} K === ".format(float(para_num / 1e3)))
    # log
    logger.info(args)
    logger.info(repr(model))
    # set model state
    model.train()
    # loss
    chamfer_dist = chamfer_3DDist()
    chamfer_dist.cuda()

    start_epoch = 0
    # resume model
    if args.resume != 'none':
        # 加载预训练模型，继续训练
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer = checkpoint['optimizer']
        scheduler_steplr = checkpoint['scheduler_steplr']
        logger.info('Load epoch %s success! ' % start_epoch)
        logger.info('========== Resume Training ==========')
    else:
        # optimizer
        assert args.optim in ['adam', 'sgd']
        if args.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        # lr scheduler
        scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
        # train
        logger.info('========== Begin Training ==========')

    # 初始化归一化参数
    # min_p2p_loss = float('inf')
    # max_p2p_loss = float('-inf')
    # min_cd_loss = float('inf')
    # max_cd_loss = float('-inf')
    epsilon = 1e-8
    for epoch in range(start_epoch, args.epochs):
        logger.info('********* Epoch %d *********' % (epoch + 1))
        # epoch loss
        epoch_p2p_loss = 0.0
        epoch_cd_loss = 0.0
        epoch_total_loss = 0.0
        for i, (input_pts, gt_pts, radius) in enumerate(train_loader):
            # (b, n, 3)
            input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
            gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()
            dense_pts, pred_p2p = model(input_pts)

            # calculate cd loss
            cd_loss, cd_loss_rev, _, _ = chamfer_dist(dense_pts, gt_pts)
            cd_loss = torch.mean(cd_loss)
            cd_loss_rev = torch.mean(cd_loss_rev)
            cd_loss = cd_loss + cd_loss_rev
            # calculate p2p loss
            p2p_loss = get_p2p_loss(args, pred_p2p, dense_pts, gt_pts)

            # # 更新归一化参数
            # min_p2p_loss = min(min_p2p_loss, p2p_loss.item())
            # max_p2p_loss = max(max_p2p_loss, p2p_loss.item())
            # min_cd_loss = min(min_cd_loss, cd_loss.item())
            # max_cd_loss = max(max_cd_loss, cd_loss.item())

            # 归一化损失
            # p2p_loss_normalized = min_max_normalize(p2p_loss, min_p2p_loss, max_p2p_loss)
            # cd_loss_normalized = min_max_normalize(cd_loss, min_cd_loss, max_cd_loss)

            # total_loss = args.alpha * p2p_loss_normalized + (1.0 - args.alpha) * cd_loss_normalized

            # 动态权重调整 (Dynamic Weight Adjustment)
            lambda_cd = 1.0 / (cd_loss.item() + epsilon)
            lambda_p2p = 1.0 / (p2p_loss.item() + epsilon)

            # Normalize weights
            total_weight = lambda_cd + lambda_p2p
            lambda_cd /= total_weight
            lambda_p2p /= total_weight

            # Total loss
            total_loss = lambda_cd * cd_loss + lambda_p2p * p2p_loss
            # total_loss = args.alpha * p2p_loss_normalized + (1.0 - args.alpha) * cd_loss_normalized

            # update parameters
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_p2p_loss += p2p_loss.item()
            epoch_cd_loss += cd_loss.item()
            epoch_total_loss += total_loss.item()

            # log
            writer.add_scalar('train/loss', total_loss, i)
            writer.flush()
            if (i + 1) % args.print_rate == 0:
                logger.info("epoch: %d/%d, iters: %d/%d, p2p_loss: %f, cd_loss: %f, total loss: %f" %
                            (epoch + 1, args.epochs, i + 1, len(train_loader), epoch_p2p_loss / (i + 1),
                             epoch_cd_loss / (i + 1), epoch_total_loss / (i + 1)))

        # lr scheduler
        scheduler_steplr.step()

        # log
        interval = time.time() - start
        logger.info(
            "epoch: %d/%d, avg epoch p2p loss: %f, avg epoch cd loss: %f, avg epoch total loss: %f, time: %d mins %.1f secs" %
            (epoch + 1, args.epochs, epoch_p2p_loss / len(train_loader), epoch_cd_loss / len(train_loader),
             epoch_total_loss / len(train_loader), interval / 60, interval % 60))

        # save checkpoint
        if (epoch + 1) % args.save_rate == 0:
            model_name = 'ckpt-epoch-%d.pth' % (epoch + 1)
            model_path = os.path.join(ckpt_dir, model_name)
            torch.save(model.state_dict(), model_path)


def parse_train_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    parser.add_argument('--dataset', default='pugan', type=str, help='pu1k or pugan')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--print_rate', default=200, type=int, help='loss print frequency in each epoch')
    parser.add_argument('--save_rate', default=10, type=int, help='model save frequency')
    parser.add_argument('--out_path', default='./output', type=str, help='the checkpoint and log save path')
    parser.add_argument('--alpha', default=0.1, type=float, help='Weight for p2p_loss in total loss')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train_args = parse_train_args()
    assert train_args.dataset in ['pu1k', 'pugan']

    if train_args.dataset == 'pu1k':
        model_args = parse_pu1k_args()
    else:
        model_args = parse_pugan_args()

    reset_model_args(train_args, model_args)

    train(model_args)
