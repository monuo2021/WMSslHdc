"""
Metric calculation for the OOD detection task. Code credit to: https://github.com/pokaxpoka/deep_Mahalanobis_detector
"""
from __future__ import print_function,division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_metric(known, novel):
    stype = ""
    # tp 和 fp 是用于存储真阳性和假阳性计数的字典。
    tp, fp = dict(), dict()
    # tnr_at_tpr95 用于存储 TNR at TPR95 的值。
    tnr_at_tpr95 = dict()
    
    known.sort()
    novel.sort()
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp[stype] = -np.ones([num_k+num_n+1], dtype=int)    # tp[i] 表示在第 i 个阈值时，被正确识别为已知类的样本数量（即真阳性数量）。
    fp[stype] = -np.ones([num_k+num_n+1], dtype=int)    # fp[i] 表示在第 i 个阈值时，被错误识别为已知类的新类样本数量（即假阳性数量）。
    # 初始化 tp 和 fp 的首元素为 num_k 和 num_n，是因为在初始状态下（即在没有应用任何阈值时），
    # 所有的已知类样本都被假设为正例（因此 tp[0] = num_k），所有的新类样本都被误判为正例（因此 fp[0] = num_n）。
    tp[stype][0], fp[stype][0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[stype][l+1:] = tp[stype][l]
            fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
            break
        elif n == num_n:
            tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
            fp[stype][l+1:] = fp[stype][l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[stype][l+1] = tp[stype][l]
                fp[stype][l+1] = fp[stype][l] - 1
            else:
                k += 1
                tp[stype][l+1] = tp[stype][l] - 1
                fp[stype][l+1] = fp[stype][l]
    tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()    # tp数组的每个元素除以num_k，再减去0.95，找到最接近 0.95的位置。
    tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    results = dict()
    results[stype] = dict()
    
    # TNR
    mtype = 'TNR'
    results[stype][mtype] = tnr_at_tpr95[stype]
    
    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
    fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
    results[stype][mtype] = -np.trapz(1.-fpr, tpr)
    
    # DTACC：衡量的是在不同决策阈值下，模型的准确性。它考虑了真阳性率和真阴性率。
    mtype = 'DTACC'
    results[stype][mtype] = .5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max()
    
    # AUIN
    mtype = 'AUIN'
    denom = tp[stype]+fp[stype]
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
    results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
    
    # AUOUT
    mtype = 'AUOUT'
    denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
    results[stype][mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
    
    return results[stype]

def print_results(results):
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('')
    for mtype in mtypes:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    print('')
    

def get_curve(dir_name, stypes = ['Baseline', 'Gaussian_LDA']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known = np.loadtxt('{}/confidence_{}_In.txt'.format(dir_name, stype), delimiter='\n')
        novel = np.loadtxt('{}/confidence_{}_Out.txt'.format(dir_name, stype), delimiter='\n')
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    
    return tp, fp, tnr_at_tpr95

def metric(dir_name, stypes = ['Bas', 'Gau'], verbose=False):
    tp, fp, tnr_at_tpr95 = get_curve(dir_name, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = -np.trapz(1.-fpr, tpr)
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = .5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max()
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
            print('')
    
    return results