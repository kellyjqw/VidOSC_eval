import os

import tqdm
import torch
import torch.nn as nn
# import pytorch_lightning as pl
# import numpy as np
from data_scripts.evaluator import StatePrec1
from argparse import Namespace
from dataset import HowToChangeFeatDataset
from evaluator import Evaluator

from task import FrameCls
import pickle
import json

from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

vocab = {'background': 0, 'rolling': 1, 'squeezing': 2, 'mashing': 3, 'roasting': 4, 'peeling': 5, 
        'chopping': 6, 'crushing': 7, 'melting': 8, 'mincing': 9, 'slicing': 10, 'grating': 11, 
        'sauteing': 12, 'frying': 13, 'blending': 14, 'coating': 15, 'browning': 16, 'grilling': 17, 
        'shredding': 18, 'whipping': 19, 'zesting': 20}
sc_list = ['rolling', 'squeezing', 'mashing', 'roasting', 'peeling', 'chopping', 'crushing', 
            'melting', 'mincing', 'slicing','grating', 'sauteing', 'frying', 'blending', 'coating', 
            'browning', 'grilling', 'shredding', 'whipping', 'zesting']
category_num = len(vocab) - 1
vocab_size = 3 * category_num + 1
input_dim = 768 
args = Namespace(
    # data args
    ann_dir='./data_files',
    pseudolabel_dir='./videoclip_pseudolabel',
    feat_dir='./data',
    sc_list="all",
    # model args
    transformer_heads=4,
    transformer_layers=3,
    transformer_dim=512,
    transformer_dropout=0.1,
    # training args
    gpus=1,
    lr=5e-4,
    wd=0.0001,
    batch_size=64,
    num_workers=8,
    n_epochs=50,
    log_dir='./logs',
    log_name='debug',
    ckpt="checkpoint/multitask.ckpt",
    use_gt_action=False,
    det=0
)

def infer_state_idx(prob):
    pred_idx = torch.argmax(prob, dim=0).cpu().numpy()
    pred_idx = pred_idx[1:]
    return pred_idx
# recommended simple way
task = FrameCls.load_from_checkpoint(checkpoint_path="checkpoint/multitask.ckpt", args=args)
model = task.model  # this is your FeatTimeTransformer
vocab = task.vocab

model.eval()
test_dataset = HowToChangeFeatDataset(args)
E = Evaluator(end_label=3, out_dir="vidOSC_results")
i = 0
torch.set_printoptions(threshold=torch.inf, precision=2, sci_mode=False) 

with torch.no_grad():

    for batch in tqdm.tqdm(test_dataset):
    # for batch in test_dataset:
        feat, label, osc, is_novel, video_id, video_name = batch
        
        sc_name = osc.split("_")[0]
        feat = feat.unsqueeze(0)
        pred = model(feat)
        prob = torch.softmax(pred, dim=-1)
        gt_category_id = vocab[sc_name]
        st_prob = prob[:, [0, 3 * gt_category_id - 2, 3 * gt_category_id - 1, 3 * gt_category_id]]

        pred_4 = st_prob.argmax(dim=-1)  # (T,)
        gt_4 = label.view(-1).long()

        # print(f"{video_name=}, {osc=}, {is_novel=}")
        # # print(f"{st_prob=}")
        # print(f"{pred_4=}")
        # print(f"{gt_4=}")

        # print(prob)

        pred_bin = E.bin(pred_4)
        gt_bin = E.bin(gt_4)
        pred_3 = E.tern(pred_4)
        gt_3 = E.tern(gt_4)
        
        E.record_bin_metrics(pred_bin, gt_bin, is_novel)
        E.record_tern_metrics(pred_3, gt_3, is_novel)
        E.record_IoU(pred_4, gt_4, is_novel)
        E.record_framediff(pred_4, gt_4, is_novel)

        # break
        i+= 1
        # if i % 300 == 0:
        #     E.save_result()
        #     break
E.save_result()
print(f"num samples: {i}")
