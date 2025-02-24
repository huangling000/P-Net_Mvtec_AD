# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       trick
   Project Name:    segOCT
   Author :         Kang ZHOU
   Date:            2018/11/6
-------------------------------------------------
   Change Activity:
                   2018/11/6:
-------------------------------------------------
"""
import os
import warnings
import shutil
import math

import torch
import numpy as np
import scipy.stats as st
import glob
import pdb
import time


def adjust_lr(original_lr, optimizer, epoch, adjust_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every $e_freq epochs"""
    if (epoch + 1) in adjust_epoch:
        # new_lr = pow(0.1, adjust_epoch.index(epoch) + 1) * original_lr
        new_lr = pow(0.1, 1) * original_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    return original_lr


def l1_reg(model):
    l1_loss = 0
    cnt = 0
    for p in model.parameters():
        cnt += 1
        l1_loss += p.abs().sum()
    return l1_loss / (cnt + 0.000001)


def save_ckpt(version, state, epoch, is_best=None, args=None):

    ckpt_dir = os.path.join(args.output_root, version, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    """
    save latest model after every epoch
    """
    ckpt_file_path = os.path.join(ckpt_dir, 'latest_ckpt.pth.tar')
    torch.save(state, ckpt_file_path)

    """
    after [save_freq * 2] epoch, save model every [save_freq] epochs
    """
    if epoch > args.save_model_freq and epoch % args.save_model_freq == 0:
        ckpt_file_path = os.path.join(ckpt_dir, 'Epoch_{}.pth.tar'.format(epoch))
        torch.save(state, ckpt_file_path)

    """
    after [save_freq * 2] epoch, save best model
    """
    if is_best:
        best_dic = glob.glob(os.path.join(ckpt_dir, 'Model_best*'))
        for i in best_dic:
            os.remove(i)
        best_file_path = os.path.join(ckpt_dir, 'Model_best@{}.pth.tar'.format(epoch))
        torch.save(state, best_file_path)


def cuda_visible(gpu_list):
    if gpu_list == None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        warnings.warn('You should better speicify the gpu id. The default gpu is 0.')
    elif len(gpu_list) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_list[0])
    elif len(gpu_list) == 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{},{}'.format(gpu_list[0], gpu_list[1])
    elif len(gpu_list) == 3:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{},{},{}'.format(gpu_list[0], gpu_list[1], gpu_list[2])
    elif len(gpu_list) == 4:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{},{},{},{}'. \
            format(gpu_list[0], gpu_list[1], gpu_list[2], gpu_list[3])
    elif len(gpu_list) == 5:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{},{},{},{},{}'. \
            format(gpu_list[0], gpu_list[1], gpu_list[2], gpu_list[3], gpu_list[4])
    else:
        raise ValueError('wrong in gpu list')


def print_args(args):
    print('\n', '*' * 30, 'Args', '*' * 30)
    print('Args: \n{}\n'.format(args))


def calc_confidence_interval(samples, confidence_value=0.95):
    # samples should be a numpy array
    if type(samples) is list:
        samples = np.asarray(samples)
    assert isinstance(samples, np.ndarray), 'args: samples {} should be np.array'.format(samples)
    # print('Results List:', samples)
    stat_accu = st.t.interval(confidence_value, len(samples)-1, loc=np.mean(samples), scale=st.sem(samples))
    center = (stat_accu[0]+stat_accu[1])/2
    deviation = (stat_accu[1]-stat_accu[0])/2
    return center, deviation


def diff_cut(image_diff, cut_rate, constant=None):
    n = image_diff.size(0)
    output = torch.zeros_like(image_diff)
    if constant == 0 or cut_rate == 0:
        output = image_diff
    elif not constant:
        for i in range(n):
            pix_list = sorted(image_diff[i].view(-1))
            threshold = pix_list[int(len(pix_list) * cut_rate)]
            output[i] = (image_diff[i] > threshold).float() * image_diff[i]
    elif constant != 0:
        threshold = constant
        output = (image_diff > threshold).float() * image_diff

    return output


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, num_top = 10):
        self.reset()
        _array = np.zeros(shape=(num_top)) + 0.01
        self.top_list = _array.tolist()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def average(self):
        return self.avg

    def top_update_calc(self, val):
        # update the lowest or NOT
        if val > self.top_list[0]:
            self.top_list[0] = val
            # [lowest, ..., highest]
            self.top_list.sort()
        # update mean, deviation
        mean, deviation = calc_confidence_interval(self.top_list)
        return mean, deviation


class LastAvgMeter(object):
    """ Compute the average of last LENGHT new value"""
    def __init__(self, length=20):
        self.val_list = None
        self.avg = None
        self.length = length
        self.reset()

    def reset(self):
        self.val_list = [0] * self.length
        self.avg = 0

    def update(self, val):
        # delete the 0-th element and add the length-th element
        del(self.val_list[0])
        self.val_list += [val]

        self.avg = sum(self.val_list) / self.length
        self.std=0
        for i in self.val_list:
            self.std+=(i-self.avg)**2
        self.std= math.sqrt(self.std/len(self.val_list))


def main():

    pass


if __name__ == '__main__':
    main()
