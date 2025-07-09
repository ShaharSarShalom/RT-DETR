"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
################ Shahar changes 
import wandb

import wandb
import numpy as np
from ..data.dataset.coco_eval import CocoEvaluator

def log_coco_metrics_to_wandb(coco_evaluator: CocoEvaluator, epoch: int, prefix: str = "val"):
    """
    Extract COCO evaluation metrics and log them to wandb
    
    Args:
        coco_evaluator: CocoEvaluator instance after evaluation
        epoch: Current epoch number
        prefix: Prefix for metric names (e.g., 'val', 'test')
    """
    metrics = {}
    
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        # Get the stats from COCOeval
        stats = coco_eval.stats
        
        if iou_type == "bbox":
            # COCO detection metrics
            metrics.update({
                f"{prefix}/mAP": stats[0],              # AP @ IoU=0.50:0.95
                f"{prefix}/mAP_50": stats[1],           # AP @ IoU=0.50
                f"{prefix}/mAP_75": stats[2],           # AP @ IoU=0.75
                f"{prefix}/mAP_small": stats[3],        # AP @ IoU=0.50:0.95 (small)
                f"{prefix}/mAP_medium": stats[4],       # AP @ IoU=0.50:0.95 (medium)
                f"{prefix}/mAP_large": stats[5],        # AP @ IoU=0.50:0.95 (large)
                f"{prefix}/AR_1": stats[6],             # AR @ IoU=0.50:0.95 (max 1 det)
                f"{prefix}/AR_10": stats[7],            # AR @ IoU=0.50:0.95 (max 10 det)
                f"{prefix}/AR_100": stats[8],           # AR @ IoU=0.50:0.95 (max 100 det)
                f"{prefix}/AR_small": stats[9],         # AR @ IoU=0.50:0.95 (small)
                f"{prefix}/AR_medium": stats[10],       # AR @ IoU=0.50:0.95 (medium)
                f"{prefix}/AR_large": stats[11],        # AR @ IoU=0.50:0.95 (large)
            })
        
        elif iou_type == "segm":
            # Instance segmentation metrics
            metrics.update({
                f"{prefix}/segm_mAP": stats[0],
                f"{prefix}/segm_mAP_50": stats[1],
                f"{prefix}/segm_mAP_75": stats[2],
                f"{prefix}/segm_AR_100": stats[8],
            })
        
        elif iou_type == "keypoints":
            # Keypoint detection metrics
            metrics.update({
                f"{prefix}/kp_mAP": stats[0],
                f"{prefix}/kp_mAP_50": stats[1],
                f"{prefix}/kp_mAP_75": stats[2],
                f"{prefix}/kp_AR_100": stats[8],
            })
    
    # Log all metrics to wandb
    wandb.log(metrics, step=epoch)
    
    return metrics

def log_coco_summary_to_wandb(coco_evaluator: CocoEvaluator, epoch: int):
    """
    Log a comprehensive summary of COCO evaluation results to wandb
    """
    summary_text = []
    
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        summary_text.append(f"\n=== {iou_type.upper()} Evaluation ===")
        
        # Capture the summary output
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            coco_eval.summarize()
        summary_output = f.getvalue()
        
        summary_text.append(summary_output)
    
    # Log as text to wandb
    wandb.log({"coco_evaluation_summary": "\n".join(summary_text)}, step=epoch)
################ Shahar changes 

class DetSolver(BaseSolver):
    
    ################ Shaahr changes 
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        
        self.wandb_run = None
        self.output_dir = None
        
        # cfg.use_wandb
        if getattr(self.cfg, "use_wandb", True):
            self.wandb_run = wandb.init(
                project=getattr(self.cfg, "wandb_project", "rtdetr_experiments"),
                name=getattr(self.cfg, "wandb_run_name", None),
                tags=["rtdetrv2", "detection"],
                config=self.cfg.__dict__,
                dir=str(self.output_dir) if self.output_dir else None,
                resume="allow"
            )

    def _wandb_log(self, log_stats, step=None):
        if self.wandb_run is not None:
            wandb.log(log_stats, step=step)
    ################ Shaahr changes 
    
    # In fit(), after log_stats is created and before writing to log.txt, add:
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )
            
            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device
            )
            
            ################ Shaahr changes 
            # Log all train_stats values to wandb under "train/" namespace
            if self.wandb_run is not None:
                wandb_log_dict = {f"train/{k}": v for k, v in train_stats.items()}
                self.wandb_run.log(wandb_log_dict, step=epoch)
            
            log_coco_metrics_to_wandb(coco_evaluator, epoch=epoch, prefix="val")
            ################ Shaahr changes 
            
            # TODO 
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
            
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
