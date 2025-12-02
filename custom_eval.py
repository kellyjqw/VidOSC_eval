import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import lookforthechange
from torchmetrics.classification import MulticlassPrecision, MulticlassF1Score

from loader import construct_loader
from model import FeatTimeTransformer
from data_scripts.evaluator import StatePrec1
from typing import SimpleNamespace
from dataset import HowToChangeFeatDataset

from task import FrameCls
vocab = {'background': 0, 'rolling': 1, 'squeezing': 2, 'mashing': 3, 'roasting': 4, 'peeling': 5, 'chopping': 6, 'crushing': 7, 
                        'melting': 8, 'mincing': 9, 'slicing': 10, 'grating': 11, 'sauteing': 12, 'frying': 13, 'blending': 14, 'coating': 15, 
                        'browning': 16, 'grilling': 17, 'shredding': 18, 'whipping': 19, 'zesting': 20}
sc_list = ['rolling', 'squeezing', 'mashing', 'roasting', 'peeling', 'chopping', 'crushing', 'melting', 'mincing', 'slicing', 
                        'grating', 'sauteing', 'frying', 'blending', 'coating', 'browning', 'grilling', 'shredding', 'whipping', 'zesting']
category_num = len(vocab) - 1
vocab_size = 3 * category_num + 1
input_dim = 768 
args = SimpleNamespace(
    # data args
    ann_dir='./data_files',
    pseudolabel_dir='./videoclip_pseudolabel',
    feat_dir='./data',
    sc_list=sc_list,
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

model.eval()
test_dataset = HowToChangeFeatDataset(args)
i = 0
with torch.no_grad():
    for batch in test_dataset:
        feat, label, osc, is_novel = batch
        sc_name = osc.split("_")[0]
        pred = model(feat)

        prob = torch.softmax(pred, dim=-1)
        gt_category_id = vocab[sc_name]
        st_prob = prob[:, [0, 3 * gt_category_id - 2, 3 * gt_category_id - 1, 3 * gt_category_id]]

        pred_idx = infer_state_idx(st_prob)
        print(pred_idx)
        label_np = label.cpu().numpy().flatten()
        
        # pred_idx excludes the first frame (frames 1 to N-1), so align with label[1:]
        # Extract end state predictions and ground truth (class 3)
        # pred_idx[0] corresponds to label_np[1], pred_idx[1] to label_np[2], etc.
        pred_end_state = (pred_idx == 3).astype(int)
        gt_end_state_full = (label_np == 3).astype(int)
        gt_end_state = gt_end_state_full[1:]  # Align with pred_idx (exclude first frame)

        print(f"{pred_end_state=}")
        print(f"{gt_end_state_full=}")
        print(f"{gt_end_state=}")
        i+= 1
        if i == 3:
            break

'''
class NewFrameCls(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.infer_ordering = False  # use causual ordering constraint during inference
        self.vocab = {'background': 0, 'rolling': 1, 'squeezing': 2, 'mashing': 3, 'roasting': 4, 'peeling': 5, 'chopping': 6, 'crushing': 7, 
                        'melting': 8, 'mincing': 9, 'slicing': 10, 'grating': 11, 'sauteing': 12, 'frying': 13, 'blending': 14, 'coating': 15, 
                        'browning': 16, 'grilling': 17, 'shredding': 18, 'whipping': 19, 'zesting': 20}
        self.sc_list = ['rolling', 'squeezing', 'mashing', 'roasting', 'peeling', 'chopping', 'crushing', 'melting', 'mincing', 'slicing', 
                        'grating', 'sauteing', 'frying', 'blending', 'coating', 'browning', 'grilling', 'shredding', 'whipping', 'zesting']
        self.category_num = len(self.vocab) - 1
        args.vocab_size = 3 * self.category_num + 1
        args.input_dim = 768 * (1 + self.args.det)
        self.model = FeatTimeTransformer(args)
        # self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.eval_setting = ['known', 'novel', 'all']
        self.metric_name_list = ['avg_f1_known', 'avg_f1_novel', 'avg_prec_known', 'avg_prec_novel']
        
        # Initialize metric accumulators for end state evaluation
        self.end_state_metrics = {
            'first_frame_diff': {'all': [], 'known': [], 'novel': []},
            'iou': {'all': [], 'known': [], 'novel': []},
            'accuracy': {'all': [], 'known': [], 'novel': []},
            'recall': {'all': [], 'known': [], 'novel': []},
            'precision': {'all': [], 'known': [], 'novel': []},
            'f1': {'all': [], 'known': [], 'novel': []}
        } 


    def training_step(self, batch, batch_idx):
        pass

    def infer_state_idx(self, prob):
        pred_idx = torch.argmax(prob, dim=0).cpu().numpy()
        pred_idx = pred_idx[1:]
        return pred_idx

    def validation_step(self, batch, batch_idx):
        feat, label, osc, is_novel = batch
        osc = osc[0]
        sc_name = osc.split('_')[0]

        name = 'novel' if is_novel.item() else 'known'
        pred = self.model(feat)
        prob = torch.softmax(pred, dim=-1)
        category_pred = prob[:, 1:].reshape(-1, self.category_num, 3).sum(dim=0).sum(dim=-1)
        inferred_catgeory_id = category_pred.argmax().item() + 1
        key = sc_name
        gt_category_id = self.vocab[key]

        # self.log('category_acc', inferred_catgeory_id == gt_category_id, on_step=False, on_epoch=True)
        category_id = gt_category_id
        st_prob = prob[:, [0, 3 * category_id - 2, 3 * category_id - 1, 3 * category_id]]

        pred_idx = self.infer_state_idx(st_prob)
        print(pred_idx)
        label_np = label.cpu().numpy().flatten()
        
        # pred_idx excludes the first frame (frames 1 to N-1), so align with label[1:]
        # Extract end state predictions and ground truth (class 3)
        # pred_idx[0] corresponds to label_np[1], pred_idx[1] to label_np[2], etc.
        pred_end_state = (pred_idx == 3).astype(int)
        gt_end_state_full = (label_np == 3).astype(int)
        gt_end_state = gt_end_state_full[1:]  # Align with pred_idx (exclude first frame)

        print(f"{pred_end_state=}")
        print(f"{gt_end_state_full=}")
        print(f"{gt_end_state=}")

        
        # # Metric 1: Average difference between first predicted end state and first ground truth end state
        # # Note: pred_idx indices need to be offset by 1 to match label_np frame indices
        # pred_first_end_idx = np.where(pred_end_state == 1)[0]
        # gt_first_end_idx = np.where(gt_end_state == 1)[0]
        # gt_first_end_idx_full = np.where(gt_end_state_full == 1)[0]
        
        # if len(pred_first_end_idx) == 0:
        #     pred_first_end_frame = len(label_np)  # frame after last frame
        # else:
        #     pred_first_end_frame = pred_first_end_idx[0] + 1  # +1 because pred_idx starts from frame 1
        
        # if len(gt_first_end_idx_full) == 0:
        #     gt_first_end_frame = len(label_np)  # frame after last frame
        # else:
        #     gt_first_end_frame = gt_first_end_idx_full[0]
        
        # first_frame_diff = abs(pred_first_end_frame - gt_first_end_frame)
        # self.end_state_metrics['first_frame_diff']['all'].append(first_frame_diff)
        # self.end_state_metrics['first_frame_diff'][name].append(first_frame_diff)
        
        # # Metric 2: IoU for videos with end state interval
        # if len(gt_first_end_idx_full) > 0:  # Only compute IoU if video has end state interval
        #     intersection = np.logical_and(pred_end_state, gt_end_state).sum()
        #     union = np.logical_or(pred_end_state, gt_end_state).sum()
        #     iou = intersection / union if union > 0 else 0.0
        #     self.end_state_metrics['iou']['all'].append(iou)
        #     self.end_state_metrics['iou'][name].append(iou)
        
        # # Metric 3: Accuracy, recall, precision, and F1 of end state class
        # # Binary classification: end state (1) vs not end state (0)
        # # Compare pred_end_state with gt_end_state (both aligned, excluding first frame)
        # tp = np.logical_and(pred_end_state == 1, gt_end_state == 1).sum()
        # fp = np.logical_and(pred_end_state == 1, gt_end_state == 0).sum()
        # fn = np.logical_and(pred_end_state == 0, gt_end_state == 1).sum()
        # tn = np.logical_and(pred_end_state == 0, gt_end_state == 0).sum()
        
        # accuracy = (tp + tn) / len(pred_end_state) if len(pred_end_state) > 0 else 0.0
        # precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # self.end_state_metrics['accuracy']['all'].append(accuracy)
        # self.end_state_metrics['accuracy'][name].append(accuracy)
        # self.end_state_metrics['recall']['all'].append(recall)
        # self.end_state_metrics['recall'][name].append(recall)
        # self.end_state_metrics['precision']['all'].append(precision)
        # self.end_state_metrics['precision'][name].append(precision)
        # self.end_state_metrics['f1']['all'].append(f1)
        # self.end_state_metrics['f1'][name].append(f1)

    def on_validation_epoch_end(self):
        # Log end state metrics
        for metric_name in ['first_frame_diff', 'iou', 'accuracy', 'recall', 'precision', 'f1']:
            for key in self.eval_setting:
                if len(self.end_state_metrics[metric_name][key]) > 0:
                    avg_value = np.mean(self.end_state_metrics[metric_name][key])
                    self.log(f'end_state_{metric_name}_{key}', avg_value, on_step=False, on_epoch=True, prog_bar=True)
                # Reset for next epoch
                self.end_state_metrics[metric_name][key] = []
        
        for i, sc_name in enumerate(self.sc_list):
            for key in self.eval_setting:
                val_prec1 = self.state_prec1[sc_name][key].compute()
                self.log(f'{sc_name}_avg_prec1_{key}', val_prec1['avg'], on_step=False, on_epoch=True, prog_bar=True)
                self.state_prec1[sc_name][key].reset()

        if len(self.sc_list) > 1:
            avg_result = np.zeros((len(self.sc_list), 6))
            value_name = ['avg_f1_known', 'avg_f1_novel', 'avg_prec_known', 'avg_prec_novel', 'avg_prec1_known',
                        'avg_prec1_novel']
            for i, sc_name in enumerate(self.sc_list):
                value_list = [self.trainer.callback_metrics.get(f'{sc_name}_{v}').item() for v in value_name]
                avg_result[i] = value_list
            avg_result = avg_result.mean(axis=0)
            for i, v in enumerate(value_name):
                self.log(f'{v}', avg_result[i], on_step=False, on_epoch=True, prog_bar=True)

        val_name = [f'{self.sc_list[0]}_avg_f1_known', f'{self.sc_list[0]}_avg_f1_novel',
                    f'{self.sc_list[0]}_avg_prec_known', f'{self.sc_list[0]}_avg_prec_novel',
                    f'{self.sc_list[0]}_avg_prec1_known', f'{self.sc_list[0]}_avg_prec1_novel']
        value_list = [round(self.trainer.callback_metrics.get(v).item() * 100, 2) for v in val_name]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def train_dataloader(self):
        return construct_loader(self.args, "train")

    def val_dataloader(self):
        return construct_loader(self.args, "val")

'''