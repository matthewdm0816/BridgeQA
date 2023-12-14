""" 
Modified from: https://github.com/daveredrum/ScanRefer/blob/master/lib/solver.py
"""

import os
import re
import sys
import time
import torch
import wandb
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LinearLR, CosineAnnealingLR
import torch.nn as nn
#import torch.distributed as dist

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from lib.loss_helper import get_loss 
from lib.eval_helper import get_eval
from utils.eta import decode_eta
from utils.multilr import MultiLR
from lib.pointnet2.pytorch_utils import BNMomentumScheduler
import pandas as pd

ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_vote_loss: {train_vote_loss}
[loss] train_objectness_loss: {train_objectness_loss}
[loss] train_box_loss: {train_box_loss}
[loss] train_sem_cls_loss: {train_sem_cls_loss}
[loss] train_ref_loss: {train_ref_loss}
[loss] train_lang_loss: {train_lang_loss}
[loss] train_answer_loss: {train_answer_loss}
[loss] train_align_loss: {train_align_loss}
[loss] train_mae_loss: {train_mae_loss}

[sco.] train_obj_acc: {train_obj_acc}
[sco.] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[sco.] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
[sco.] train_ref_acc: {train_ref_acc}
[sco.] train_lang_acc: {train_lang_acc}
[sco.] train_answer_acc@1: {train_answer_acc_at1}
[sco.] train_answer_acc@10: {train_answer_acc_at10}

[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[train] train_loss: {train_loss}
[train] train_vote_loss: {train_vote_loss}
[train] train_objectness_loss: {train_objectness_loss}
[train] train_box_loss: {train_box_loss}
[train] train_sem_cls_loss: {train_sem_cls_loss}
[train] train_ref_loss: {train_ref_loss}
[train] train_lang_loss: {train_lang_loss}
[train] train_answer_loss: {train_answer_loss}
[train] train_align_loss: {train_align_loss}
[train] train_mae_loss: {train_mae_loss}

[train] train_obj_acc: {train_obj_acc}
[train] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[train] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
[train] train_ref_acc: {train_ref_acc}
[train] train_lang_acc: {train_lang_acc}
[train] train_answer_acc@1: {train_answer_acc_at1}
[train] train_answer_acc@10: {train_answer_acc_at10}

[val]   val_loss: {val_loss}
[val]   val_vote_loss: {val_vote_loss}
[val]   val_objectness_loss: {val_objectness_loss}
[val]   val_box_loss: {val_box_loss}
[val]   val_sem_cls_loss: {train_sem_cls_loss}
[val]   val_ref_loss: {val_ref_loss}
[val]   val_lang_loss: {val_lang_loss}
[val]   val_answer_loss: {val_answer_loss}

[val]   val_obj_acc: {val_obj_acc}
[val]   val_pos_ratio: {val_pos_ratio}, val_neg_ratio: {val_neg_ratio}
[val]   val_iou_rate_0.25: {val_iou_rate_25}, val_iou_rate_0.5: {val_iou_rate_5}
[val]   val_ref_acc: {val_ref_acc}
[val]   val_lang_acc: {val_lang_acc}
[val]   val_answer_acc@1: {val_answer_acc_at1}
[val]   val_answer_acc@10: {val_answer_acc_at10}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[loss] vote_loss: {vote_loss}
[loss] objectness_loss: {objectness_loss}
[loss] box_loss: {box_loss}
[loss] sem_cls_loss: {sem_cls_loss}
[loss] ref_loss: {ref_loss}
[loss] lang_loss: {lang_loss}
[loss] answer_loss: {answer_loss}
[loss] align_loss: {align_loss}
[loss] mae_loss: {mae_loss}

[sco.] obj_acc: {obj_acc}
[sco.] pos_ratio: {pos_ratio}, neg_ratio: {neg_ratio}
[sco.] iou_rate_0.25: {iou_rate_25}, iou_rate_0.5: {iou_rate_5}
[sco.] ref_acc: {ref_acc}
[sco.] lang_acc: {lang_acc}
[sco.] answer_acc@1: {answer_acc_at1}
[sco.] answer_acc@10: {answer_acc_at10}
"""

LOG_SCORE_KEYS = {
    "loss": ["loss", "vote_loss", "objectness_loss", "box_loss", "sem_cls_loss", "ref_loss", "lang_loss", "answer_loss", "align_loss", "mae_loss"],
    "score": ["obj_acc", "pos_ratio", "neg_ratio", "iou_rate_0.25", "iou_rate_0.5",
                "ref_acc", "lang_acc", "answer_acc_at1", "answer_acc_at10", 
                "answer_acc_at1_scene", "answer_acc_at10_scene", 
                "answer_acc_at1_2d", "answer_acc_at10_2d", 
                "answer_acc_at1_2d3d", "answer_acc_at10_2d3d",
                "answer_acc_at1_2d_over_3d", "answer_acc_at1_3d_over_2d",  
            ]
}

class Solver():
    def __init__(self, model, config, dataloader, optimizer, stamp, val_step=10, epoch=None,
                cur_criterion="answer_acc_at1", detection=True, use_reference=True, use_lang_classifier=True, use_answer=True, 
                max_grad_norm=None, lr_decay_step=None, lr_decay_step_2d=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None, loss_weights=None, use_vlm_align=False, scheduler_type="step",
                begin_align_epoch=0, save_pred=False, ddp=False, reinit_epoch=-1,
    ):
        self.epoch = epoch
        self.verbose = 0
        self.ddp = ddp
        
        self.model = model

        if ddp:
            self.model_inner = model.module
        else:
            self.model_inner = model

        self.config = config
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step
        self.cur_criterion = cur_criterion

        self.answerable_data_size = {}
        self.all_data_size = {}
        for phase in dataloader.keys():
            self.answerable_data_size[phase] = dataloader[phase].dataset.answerable_data_size
            self.all_data_size[phase] = dataloader[phase].dataset.all_data_size

        self.detection = detection
        self.use_reference = use_reference
        self.use_answer = use_answer
        self.use_lang_classifier = use_lang_classifier
        self.use_vlm_align = use_vlm_align

        self.max_grad_norm = max_grad_norm
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.bn_decay_step = bn_decay_step
        self.bn_decay_rate = bn_decay_rate

        self.loss_weights = loss_weights
        self.reinit_epoch = reinit_epoch

        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "ref_loss": float("inf"),
            "answer_loss": float("inf"),
            "lang_loss": float("inf"),
            "objectness_loss": float("inf"),
            "vote_loss": float("inf"),
            "box_loss": float("inf"),
            "sem_cls_loss": float("inf"),            
            "answer_acc_at1": -float("inf"),
            "answer_acc_at10": -float("inf"),
            "answer_acc_at1_scene": -float("inf"),
            "answer_acc_at10_scene": -float("inf"),
            "answer_acc_at1_2d": -float("inf"),
            "answer_acc_at10_2d": -float("inf"),
            "answer_acc_at1_2d3d": -float("inf"),
            "answer_acc_at10_2d3d": -float("inf"),
            "answer_acc_at1_2d_over_3d": -float("inf"),
            "answer_acc_at1_3d_over_2d": -float("inf"),  
            "lang_acc": -float("inf"),
            "ref_acc": -float("inf"),            
            "obj_acc": -float("inf"),
            "pos_ratio": -float("inf"),
            "neg_ratio": -float("inf"),
            "iou_rate_0.25": -float("inf"),
            "iou_rate_0.5": -float("inf"),
            "align_loss": -float("inf"),
            "mae_loss": -float("inf"),
        }

        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }
        
        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # lr scheduler
        # total_iters = self.epoch
        self.scheduler_type = scheduler_type
        if scheduler_type == "step":
            if lr_decay_step and lr_decay_rate:
                if isinstance(lr_decay_step, list):
                    self.lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
                else:
                    self.lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
            else:
                self.lr_scheduler = None
        # elif scheduler_type == "stepwarmup":
        #     # step with warmup
        #     assert self.epoch is not None, "Provide epochs first!"
        elif scheduler_type == "step_except_2d":
            assert isinstance(lr_decay_step, list)
            self.lr_scheduler = MultiLR(optimizer,[
                lambda opt: MultiStepLR(opt, lr_decay_step_2d, lr_decay_rate), # blip, decay faster
                lambda opt: MultiStepLR(opt, lr_decay_step, lr_decay_rate), # blip3d
                lambda opt: MultiStepLR(opt, lr_decay_step, lr_decay_rate), # other
            ])

        elif scheduler_type == "linear":
            assert self.epoch is not None, "Provide epochs first!"
            self.lr_scheduler = LinearLR(optimizer, 1, 0.001, total_iters=self.epoch)
        elif scheduler_type == "cosine":
            assert self.epoch is not None, "Provide epochs first!"
            self.lr_scheduler = CosineAnnealingLR(optimizer, self.epoch)
        else:
            self.lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate**(int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)
        else:
            self.bn_scheduler = None

        self.begin_align_epoch = begin_align_epoch
        self.save_pred = save_pred

    def set_answer_vocab(self, answer_vocab):
        self.model_inner.answer_vocab = answer_vocab
        self.model_inner.load_soft_label()

    def __call__(self, epoch, verbose):
        self._start()
        # setting
        self.epoch = epoch
        self.verbose = verbose


        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * self.val_step

        for epoch_id in range(epoch):
            if epoch_id == self.reinit_epoch and self.reinit_epoch > 0:
                print("reinit model parameters...")
                self.model_inner.reinit_params()
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))
                if self.ddp:
                    self.dataloader["train"].sampler.set_epoch(epoch_id)
                # feed 
                self._feed(self.dataloader["train"], "train", epoch_id)

                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

                # update lr scheduler
                if self.lr_scheduler:
                    print("update learning rate --> {}\n".format(self.lr_scheduler.get_lr()))
                    self.lr_scheduler.step()

                # update bn scheduler
                if self.bn_scheduler:
                    print("update batch normalization momentum --> {}\n".format(self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                    self.bn_scheduler.step()
                
            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _start(self):
        # save commandline 
        cmd = " ".join([v for v in sys.argv])
        cmd_file = os.path.join(CONF.PATH.OUTPUT, self.stamp, "cmdline.txt")
        open(cmd_file, 'w').write(cmd)
        wandb.save(cmd_file)   

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str)

    def _reset_log(self, phase):
        self.log[phase] = {
            # info
            "forward": [],
            "backward": [],
            "eval": [],
            "fetch": [],
            "iter_time": [],
            # loss
            "loss": [],
            "ref_loss": [],
            "answer_loss": [],
            "lang_loss": [],
            "objectness_loss": [],
            "vote_loss": [],
            "box_loss": [],
            "sem_cls_loss": [],
            "align_loss": [],
            "mae_loss": [],
            # scores
            "answer_acc_at1": [],
            "answer_acc_at10": [],
            "answer_acc_at1_scene": [],
            "answer_acc_at10_scene": [],
            "answer_acc_at1_2d": [],
            "answer_acc_at10_2d": [],
            "answer_acc_at1_2d3d": [],
            "answer_acc_at10_2d3d": [],
            "answer_acc_at1_2d_over_3d": [],
            "answer_acc_at1_3d_over_2d": [],    
            "lang_acc": [],
            "ref_acc": [],
            "obj_acc": [],            
            "pos_ratio": [],
            "neg_ratio": [],
            "iou_rate_0.25": [],
            "iou_rate_0.5": [],
            # pred_answers
            "pred_lang": [],
            "pred_answer": [],
            "pred_answer_scores": [],
            "pred_answer_scores_2d": [],
            "pred_answer_scores_scene": [],
            "scene_id": [],
            "question_id": [],
        }

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict)
        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()

        # gradient clipping
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.max_grad_norm)

        self.optimizer.step()

    def _compute_loss(self, data_dict, epoch):
        _, data_dict = get_loss(
            data_dict=data_dict, 
            config=self.config, 
            detection=self.detection,
            use_reference=self.use_reference, 
            use_answer=self.use_answer,
            use_lang_classifier=self.use_lang_classifier,
            loss_weights=self.loss_weights,
            use_vlm_align=self.use_vlm_align and epoch >= self.begin_align_epoch,
        )

        # dump
        self._running_log["ref_loss"] = data_dict["ref_loss"]
        self._running_log["answer_loss"] = data_dict["answer_loss"]
        self._running_log["lang_loss"] = data_dict["lang_loss"]
        self._running_log["objectness_loss"] = data_dict["objectness_loss"]
        self._running_log["vote_loss"] = data_dict["vote_loss"]
        self._running_log["box_loss"] = data_dict["box_loss"]
        self._running_log["sem_cls_loss"] = data_dict["sem_cls_loss"]
        self._running_log["align_loss"] = data_dict["align_loss"]
        self._running_log["mae_loss"] = data_dict["mae_loss"]
        self._running_log["loss"] = data_dict["loss"]

    def _eval(self, data_dict):
        data_dict = get_eval(
            data_dict=data_dict,
            config=self.config,
            answer_vocab=self.dataloader["train"].dataset.answer_vocab,
            use_reference=True, 
            use_lang_classifier=self.use_lang_classifier
        )

        # dump
        if "ref_acc" in data_dict:
            self._running_log["ref_acc"] = np.mean(data_dict["ref_acc"])     
        if "lang_acc" in data_dict:
            self._running_log["lang_acc"] = data_dict["lang_acc"].item()
        # self._running_log["answer_acc_at1"] = data_dict["answer_acc_at1"].item()
        # self._running_log["answer_acc_at10"] = data_dict["answer_acc_at10"].item()
        # self._running_log["answer_acc_at1_scene"] = data_dict["answer_acc_at1_scene"].item()
        # self._running_log["answer_acc_at10_scene"] = data_dict["answer_acc_at10_scene"].item()
        # self._running_log["answer_acc_at1_2d"] = data_dict["answer_acc_at1_2d"].item()
        # self._running_log["answer_acc_at10_2d"] = data_dict["answer_acc_at10_2d"].item()
        # self._running_log["answer_acc_at1_2d3d"] = data_dict["answer_acc_at1_2d3d"].item()
        # self._running_log["answer_acc_at10_2d3d"] = data_dict["answer_acc_at10_2d3d"].item()
        for k, v in data_dict.items():
            if "answer_acc" in k:
                self._running_log[k] = v.item()

        self._running_log["obj_acc"] = data_dict["obj_acc"].item()
        self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
        self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()
        if "ref_iou_rate_0.25" in data_dict:
            self._running_log["iou_rate_0.25"] = np.mean(data_dict["ref_iou_rate_0.25"])
        if "ref_iou_rate_0.5" in data_dict:
            self._running_log["iou_rate_0.5"] = np.mean(data_dict["ref_iou_rate_0.5"])

    def _feed(self, dataloader, phase, epoch_id):
        # switch mode
        self._set_phase(phase)

        # re-init log
        self._reset_log(phase)

        scene_number_to_id = dataloader.dataset.scene_number_to_id

        # change dataloader
        dataloader = dataloader if phase == "train" else tqdm(dataloader)

        for i, data_dict in enumerate(dataloader):
            # move to cuda
            for key in data_dict:
                if type(data_dict[key]) is dict:
                    data_dict[key] = {k:v.cuda() for k, v in data_dict[key].items()}
                elif type(data_dict[key]) is list or type(data_dict[key]) is str:
                    # keep list as-is
                    ...
                else:
                    data_dict[key] = data_dict[key].cuda()
            
            # add phase info
            data_dict["phase"] = phase
            data_dict["iteration"] = i # + len(dataloader) * epoch_id

            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                "ref_loss": 0,
                "answer_loss": 0,
                "lang_loss": 0,
                "objectness_loss": 0,
                "vote_loss": 0,
                "box_loss": 0,
                "sem_cls_loss": 0, 
                "align_loss": 0,
                "mae_loss": 0,
                # score
                "ref_acc": 0,
                "lang_acc": 0, 
                "answer_acc_at1": 0, 
                "answer_acc_at10": 0,      
                "answer_acc_at1_scene": 0, 
                "answer_acc_at10_scene": 0,    
                "answer_acc_at1_2d": 0, 
                "answer_acc_at10_2d": 0, 
                "answer_acc_at1_2d3d": 0, 
                "answer_acc_at10_2d3d": 0, 
                "answer_acc_at1_2d_over_3d": 0,
                "answer_acc_at1_3d_over_2d": 0,                           
                "obj_acc": 0,
                "pos_ratio": 0,
                "neg_ratio": 0,
                "iou_rate_0.25": 0,
                "iou_rate_0.5": 0,
            }

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

            with torch.autograd.set_detect_anomaly(True):
                # forward
                start = time.time()
                data_dict = self._forward(data_dict)
                self._compute_loss(data_dict, epoch_id)
                self.log[phase]["forward"].append(time.time() - start)

                # backward
                if phase == "train":
                    start = time.time()
                    self._backward()
                    self.log[phase]["backward"].append(time.time() - start)

            # eval
            start = time.time()

            self._eval(data_dict)
            self.log[phase]["eval"].append(time.time() - start)

            # record log
            for key in self._running_log.keys():
                value = self._running_log[key] # score or loss
                if type(value) == torch.Tensor:
                    value = value.item() # if loss
                # average if multiple gpus
                if torch.distributed.is_initialized():
                    value = torch.tensor(value).cuda()
                    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                    value = value.item() / torch.distributed.get_world_size()
                self.log[phase][key].append(value)
            answerable_rate = self.answerable_data_size[phase] / self.all_data_size[phase]

            if "pred_langs" in data_dict:
                self.log[phase]["pred_lang"] += data_dict["pred_langs"].argmax(axis=1).tolist() 

            if "pred_answers" in data_dict:
                self.log[phase]["pred_answer"] += data_dict["pred_answers"].tolist()

            # record all scores
            for key, value in data_dict.items():
                if "pred_answer_scores" in key:
                    print(phase, value.shape)
                    self.log[phase][key] += value.cpu().detach().tolist()

            self.log[phase]["scene_id"] += [scene_number_to_id[scene_number] for scene_number in data_dict["scene_id"].tolist()]
            self.log[phase]["question_id"] += data_dict["question_id"].tolist()

            # report
            if phase == "train":
                iter_time = self.log[phase]["fetch"][-1]
                iter_time += self.log[phase]["forward"][-1]
                iter_time += self.log[phase]["backward"][-1]
                iter_time += self.log[phase]["eval"][-1]
                self.log[phase]["iter_time"].append(iter_time)
                
                if (self._global_iter_id + 1) % self.verbose == 0:
                    self._train_report(epoch_id)

                # evaluation
                if self._global_iter_id % self.val_step == 0:
                    print("evaluating...")
                    # val
                    self._feed(self.dataloader["val"], "val", epoch_id)
                    self._dump_log("val")
                    self._set_phase("train")
                    self._epoch_report(epoch_id)

                # dump log
                self._dump_log("train")
                self._global_iter_id += 1

        # check best
        if phase == "val":
            cur_best = np.mean(self.log[phase][self.cur_criterion])
            if torch.distributed.is_initialized():
                cur_best = torch.tensor(cur_best).cuda(torch.distributed.get_rank())
                torch.distributed.all_reduce(cur_best, op=torch.distributed.ReduceOp.SUM)
                cur_best = cur_best.item() / torch.distributed.get_world_size()
            if cur_best > self.best[self.cur_criterion]:
                self._log("best val_{} achieved: {}".format(self.cur_criterion, cur_best))
                self._log("current train_loss: {}".format(np.mean(self.log["train"]["loss"])))
                self._log("current val_loss: {}".format(np.mean(self.log["val"]["loss"])))
                self._log("current val_iou_rate_0.5: {}".format(np.mean(self.log["val"]["iou_rate_0.5"])))
                self._log("current val_iou_rate_0.5: {}".format(np.mean(self.log["val"]["iou_rate_0.5"])))
                self.best["epoch"] = epoch_id + 1

                for key in LOG_SCORE_KEYS["loss"] + LOG_SCORE_KEYS["score"]:
                    self.best[key] = np.mean(self.log[phase][key])

                # WandB logging of best_val_score
                for key, value in self.best.items():
                    wandb.log({"best_val/{}".format(key): round(value, 5)}, step=self._global_iter_id)

                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)                

                if "pred_answer" in self.log[phase]:
                    self._log("saving best predictions...\n")
                    pred_answer_idxs = self.log[phase]["pred_answer"]
                    pred_answers = [self.dataloader["val"].dataset.answer_vocab.itos(pred_answer_idx) for pred_answer_idx in pred_answer_idxs]

                    qa_id_df = pd.DataFrame([self.log[phase]["scene_id"], self.log[phase]["question_id"]]).T
                    qa_id_df.columns = ["scene_id", "question_id"]                    
                    if len(self.log[phase]["pred_lang"]) != 0:
                        pred_lang_idxs = self.log[phase]["pred_lang"]

                        # dataloader.iterable
                        pred_langs = [self.dataloader["val"].dataset.label2raw[pred_lang_idx] for pred_lang_idx in pred_lang_idxs]
                        pred_ansewr_df = pd.DataFrame([pred_lang_idxs, pred_langs, pred_answer_idxs, pred_answers]).T
                        pred_ansewr_df.columns = ["pred_lang_idx", "pred_lang", "pred_answer_idx", "pred_answer"]                        
                    else:
                        pred_ansewr_df = pd.DataFrame([pred_answer_idxs, pred_answers]).T
                        pred_ansewr_df.columns = ["pred_answer_idx", "pred_answer"]

                    pred_ansewr_df = pd.concat([qa_id_df, pred_ansewr_df], axis=1)

                    # predicted answer scores
                    columns = ["pred_answer_scores", "pred_answer_scores_2d", "pred_answer_scores_scene"]
                    columns = [column for column in columns if len(self.log[phase][column]) != 0] # filter out empty columns
                    # answer_scores_df = pd.DataFrame([self.log[phase][column] for column in columns]).T
                    answer_scores_df = pd.DataFrame({column: self.log[phase][column] for column in columns})
                    # answer_scores_df.columns = columns
                    pred_ansewr_df = pd.concat([pred_ansewr_df, answer_scores_df], axis=1)

                    if torch.distributed.is_initialized():
                        local_rank = f"_{torch.distributed.get_rank()}"
                    else:
                        local_rank = ""
                    # save pred_answers
                    pred_ansewr_df.to_csv(os.path.join(model_root, f"best_val_pred_answers{local_rank}.csv"), index=False)

                # save best model
                if not torch.distributed.is_initialized() or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
                    torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))


    def _dump_log(self, phase):
        for loss_or_score in ["loss", "score"]:
            for key in LOG_SCORE_KEYS[loss_or_score]:
                value = np.mean([v for v in self.log[phase][key]])
                # TensorBoard
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(loss_or_score, key),
                    value,
                    self._global_iter_id
                )
                # WandB
                # phase, key, item -> val/score/ref_acc
                wandb.log({"{}/{}/{}".format(phase, loss_or_score, key): value}, step=self._global_iter_id)


    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save check point
        self._log("saving checkpoint...\n")
        save_dict = {
            "epoch": epoch_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))

        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.dataloader["val"]) * np.ceil(self._total_iter["train"] / self.val_step) * mean_est_val_time
        eta = decode_eta(eta_sec)

        iter_report_dic = {}
        phase = "train"
        for key in LOG_SCORE_KEYS["loss"] + LOG_SCORE_KEYS["score"]:
            iter_report_dic[phase+"_"+re.sub('0.','',key)] = round(np.mean([v for v in self.log[phase][key]]), 5)
        iter_report_dic["epoch_id"] = epoch_id + 1
        iter_report_dic["iter_id"] = self._global_iter_id + 1
        iter_report_dic["total_iter"] = self._total_iter[phase]
        iter_report_dic["mean_fetch_time"] = round(np.mean(fetch_time), 5)
        iter_report_dic["mean_forward_time"] = round(np.mean(forward_time), 5)
        iter_report_dic["mean_backward_time"] = round(np.mean(backward_time), 5)
        iter_report_dic["mean_eval_time"] = round(np.mean(eval_time), 5)
        iter_report_dic["mean_iter_time"] = round(np.mean(iter_time), 5)
        iter_report_dic["eta_h"]=eta["h"]
        iter_report_dic["eta_m"]=eta["m"]
        iter_report_dic["eta_s"]=eta["s"]        

        iter_report = self.__iter_report_template.format(**iter_report_dic)
        self._log(iter_report)


    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        epoch_report_dic = {}
        for phase in ["train", "val"]:
            for key in LOG_SCORE_KEYS["loss"] + LOG_SCORE_KEYS["score"]:
                epoch_report_dic[phase + "_" + re.sub('0.', '', key)] = round(np.mean([v for v in self.log[phase][key]]), 5)
        epoch_report = self.__epoch_report_template.format(**epoch_report_dic)
        self._log(epoch_report)


    def _best_report(self):
        self._log("training completed...")
        best_report_dic = {re.sub('0.', '', k):v for k, v in self.best.items()}
        best_report = self.__best_report_template.format(**best_report_dic)
        # WandB logging of best_val_score
        for key, value in self.best.items():
            wandb.log({"best_val/{}".format(key): round(value, 5)})

        self._log(best_report)
        best_report_file = os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt")
        with open(best_report_file, "w") as f:
            f.write(best_report)
        wandb.save(best_report_file)
