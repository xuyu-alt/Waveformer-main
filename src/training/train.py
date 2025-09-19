"""
The main training script for training on synthetic data
"""

import argparse
import multiprocessing
import os
import logging
from pathlib import Path
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm  # pylint: disable=unused-import
try:
    from torchmetrics.audio.snr import (
        scale_invariant_signal_noise_ratio as si_snr,
        signal_noise_ratio as snr,
    )
except Exception:  # Fallback for older torchmetrics
    import warnings as _tm_warnings
    _tm_warnings.filterwarnings("ignore", category=FutureWarning)
    from torchmetrics.functional import (
        scale_invariant_signal_noise_ratio as si_snr,
        signal_noise_ratio as snr,
    )

from src.helpers import utils
from src.training.eval import test_epoch
from src.training.synthetic_dataset import tensorboard_add_sample, FSDSoundScapesDataset
from src.training.folder_jams_dataset import FolderJAMSDataset

def train_epoch(model: nn.Module, device: torch.device,
                optimizer: optim.Optimizer,
                train_loader: torch.utils.data.dataloader.DataLoader,
                n_items: int, epoch: int = 0,
                writer: Optional[SummaryWriter] = None, data_params = None,
                overfit_single_batch: bool = False) -> dict:

    """
    Train a single epoch.
    """
    # Set the model to training.
    model.train()

    # Training loop
    losses = []
    metrics = {}

    with tqdm(total=len(train_loader), desc='Train', ncols=100) as t:
        if overfit_single_batch:
            single_batch = next(iter(train_loader))
        for batch_idx, (mixed, label, gt) in enumerate(train_loader if not overfit_single_batch else [single_batch]*len(train_loader)):
            mixed = mixed.to(device)
            label = label.to(device)
            gt = gt.to(device)

            # Reset grad
            optimizer.zero_grad()

            # 跳过无效批次：无前景标签或 gt 能量过低
            try:
                gt_energy_all = (gt ** 2).mean().item()
                label_sum_all = label.sum().item()
            except Exception:
                gt_energy_all = 0.0
                label_sum_all = 0.0
            if (gt_energy_all < 1e-8) or (label_sum_all <= 0):
                if batch_idx % 100 == 0:
                    logging.warning(f"Skip batch {batch_idx}: label_sum={label_sum_all:.1f}, gt_energy={gt_energy_all:.2e}")
                t.update(1)
                continue

            # 详细的数据预处理检查
            if batch_idx == 0:
                logging.info("=== 数据预处理检查 ===")
                logging.info(f"mixed shape: {mixed.shape}, dtype: {mixed.dtype}")
                logging.info(f"label shape: {label.shape}, dtype: {label.dtype}")
                logging.info(f"gt shape: {gt.shape}, dtype: {gt.dtype}")
                
                # 检查音频数据范围
                logging.info(f"mixed range: [{mixed.min().item():.6f}, {mixed.max().item():.6f}]")
                logging.info(f"gt range: [{gt.min().item():.6f}, {gt.max().item():.6f}]")
                
                # 检查标签数据
                logging.info(f"label sum: {label.sum().item():.6f}, non-zero count: {(label > 0).sum().item()}")
                logging.info(f"label values: {label[0].cpu().numpy()}")
                
                # 检查音频是否包含有效信号
                mixed_energy = (mixed ** 2).mean().item()
                gt_energy = (gt ** 2).mean().item()
                logging.info(f"mixed energy: {mixed_energy:.8f}, gt energy: {gt_energy:.8f}")
                
                # 检查是否有静音或异常值
                if mixed_energy < 1e-8:
                    logging.warning("WARNING: mixed audio has very low energy!")
                if gt_energy < 1e-8:
                    logging.warning("WARNING: ground truth audio has very low energy!")

            # Run through the model
            output = model(mixed, label)

            # 详细的模型输出检查
            if batch_idx == 0:
                logging.info("=== 模型输出检查 ===")
                logging.info(f"output shape: {output.shape}, dtype: {output.dtype}")
                logging.info(f"output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
                logging.info(f"output mean: {output.mean().item():.6f}, std: {output.std().item():.6f}")
                
                # 检查输出是否合理
                output_energy = (output ** 2).mean().item()
                logging.info(f"output energy: {output_energy:.8f}")
                
                if output_energy < 1e-8:
                    logging.warning("WARNING: model output has very low energy!")
                
                # 检查输出是否包含NaN或Inf
                if torch.isnan(output).any():
                    logging.error("ERROR: model output contains NaN!")
                if torch.isinf(output).any():
                    logging.error("ERROR: model output contains Inf!")
                
                # 检查输出与输入的尺寸匹配
                if output.shape != gt.shape:
                    logging.error(f"ERROR: output shape {output.shape} != gt shape {gt.shape}")

            # Debug: tensor stats
            if batch_idx == 0:
                logging.info(f"mixed: mean={mixed.mean().item():.6f}, std={mixed.std().item():.6f}, l2={mixed.norm().item():.6f}")
                logging.info(f"output: mean={output.mean().item():.6f}, std={output.std().item():.6f}, l2={output.norm().item():.6f}")
                logging.info(f"gt: mean={gt.mean().item():.6f}, std={gt.std().item():.6f}, l2={gt.norm().item():.6f}")

            # Compute loss
            loss = network.loss(output, gt)

            losses.append(loss.item())

            # Backpropagation
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            # Update the weights
            optimizer.step()

            metrics_batch = network.metrics(mixed.detach(), output.detach(),
                                            gt.detach())
            for k in metrics_batch.keys():
                if not k in metrics:
                    metrics[k] = []
                # 确保metrics_batch[k]是列表
                if isinstance(metrics_batch[k], list):
                    metrics[k].extend(metrics_batch[k])
                else:
                    metrics[k].append(metrics_batch[k])

            if writer is not None and batch_idx == 0:
                tensorboard_add_sample(
                    writer, tag='Train',
                    sample=(mixed.detach()[:8], label.detach()[:8],
                            gt.detach()[:8], output.detach()[:8]),
                    step=epoch, params=data_params)

            # Show current loss in the progress meter
            t.set_postfix(loss='%.05f'%loss.item())
            t.update()

            if n_items is not None and batch_idx == n_items:
                break

    avg_metrics = {k: np.mean(metrics[k]) for k in metrics.keys()}
    avg_metrics['loss'] = np.mean(losses)
    avg_metrics_str = "Train:"
    for m in avg_metrics.keys():
        avg_metrics_str += ' %s=%.04f' % (m, avg_metrics[m])
    logging.info(avg_metrics_str)

    return avg_metrics


def train(args: argparse.Namespace):
    """
    Train the network.
    """

    # Load dataset
    # 获取实际数据集中的标签列表
    fg_root = Path('my_dataset/foreground')
    labels_82 = [d.name for d in fg_root.iterdir() if d.is_dir()]
    labels_82.sort()  # 确保标签顺序一致
    
    data_train = FolderJAMSDataset(
        root_dir=args.train_data['input_dir'],
        label_list=labels_82,
        sr=args.train_data['sr'],
        resample_rate=args.train_data['resample_rate'])
    logging.info("Loaded train dataset at %s containing %d elements" %
                 (args.train_data['input_dir'], len(data_train)))
    data_val = FolderJAMSDataset(
        root_dir=args.val_data['input_dir'],
        label_list=labels_82,
        sr=args.val_data['sr'],
        resample_rate=args.val_data['resample_rate'])
    logging.info("Loaded val dataset at %s containing %d elements" %
                 (args.val_data['input_dir'], len(data_val)))

    # Set up the device and workers.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        gpu_ids = args.gpu_ids if args.gpu_ids is not None\
                        else range(torch.cuda.device_count())
        device_ids = [_ for _ in gpu_ids]
        data_parallel = len(device_ids) > 1
        device = 'cuda:%d' % device_ids[0]
        torch.cuda.set_device(device_ids[0])
        logging.info("Using CUDA devices: %s" % str(device_ids))
    else:
        data_parallel = False
        device = torch.device('cpu')
        logging.info("Using device: CPU")

    # Set multiprocessing params
    num_workers = min(multiprocessing.cpu_count(), args.n_workers)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    } if use_cuda else {}

    # Set up data loaders
    #print(args.batch_size, args.eval_batch_size)
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=args.eval_batch_size,
                                             **kwargs)

    # Set up model 
    # 动态对齐 label_len 为实际前景类别数，避免与数据不一致
    try:
        args.model_params['label_len'] = len(labels_82)
    except Exception:
        pass
    model = network.Net(**args.model_params)

    # Add graph to tensorboard with example train samples
    # _mixed, _label, _ = next(iter(val_loader))
    # args.writer.add_graph(model, (_mixed, _label))

    if use_cuda and data_parallel:
        model = nn.DataParallel(model, device_ids=device_ids)
        logging.info("Using data parallel model")
    model.to(device)

    # Set up the optimizer
    logging.info("Initializing optimizer with %s" % str(args.optim))
    optimizer = network.optimizer(model, **args.optim, data_parallel=data_parallel)
    logging.info('Learning rates initialized to:' + utils.format_lr_info(optimizer))

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **args.lr_sched)
    logging.info("Initialized LR scheduler with params: fix_lr_epochs=%d %s"
                 % (args.fix_lr_epochs, str(args.lr_sched)))

    base_metric = args.base_metric
    train_metrics = {}
    val_metrics = {}
    
    # Early stopping variables
    best_val_metric = float('-inf')
    patience_counter = 0
    early_stop_patience = 10

    # Load the model if `args.start_epoch` is greater than 0. This will load the
    # model from epoch = `args.start_epoch - 1`
    assert args.start_epoch >=0, "start_epoch must be greater than 0."
    if args.start_epoch > 0:
        checkpoint_path = os.path.join(args.exp_dir,
                                       '%d.pt' % (args.start_epoch - 1))
        _, train_metrics, val_metrics = utils.load_checkpoint(
            checkpoint_path, model, optim=optimizer, lr_sched=lr_scheduler,
            data_parallel=data_parallel)
        logging.info("Loaded checkpoint from %s" % checkpoint_path)
        logging.info("Learning rates restored to:" + utils.format_lr_info(optimizer))

    # Training loop
    try:
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        for epoch in range(args.start_epoch, args.epochs + 1):
            logging.info("Epoch %d:" % epoch)
            checkpoint_file = os.path.join(args.exp_dir, '%d.pt' % epoch)
            # 若已存在则覆盖旧权重，避免重复运行时中断
            if os.path.exists(checkpoint_file):
                logging.warning("Checkpoint %s exists, will overwrite." % checkpoint_file)
            #print("---- begin trianivg")
            # 训练一个 epoch；若出现无效批（gt能量为0），由数据集侧重试过滤
            curr_train_metrics = train_epoch(model, device, optimizer,
                                             train_loader, args.n_train_items,
                                             epoch=epoch, writer=args.writer,
                                             data_params=args.train_data,
                                             overfit_single_batch=args.overfit_single_batch)
            
            # 添加调试信息
            logging.info(f"Epoch {epoch} train metrics: {curr_train_metrics}")
            #raise KeyboardInterrupt
            curr_test_metrics = test_epoch(model, device, val_loader,
                                           args.n_test_items, network.loss,
                                           network.metrics, epoch=epoch,
                                           writer=args.writer,
                                           data_params=args.val_data)
            
            # 添加调试信息，检查test_metrics的内容
            logging.info(f"Epoch {epoch} test metrics: {curr_test_metrics}")
            if 'loss' in curr_test_metrics:
                logging.info(f"Test loss value: {curr_test_metrics['loss']}")
                if curr_test_metrics['loss'] == 0:
                    logging.warning("Test loss is 0! This might indicate a problem.")
            else:
                logging.error("No 'loss' key found in test metrics!")
                logging.error(f"Available keys: {list(curr_test_metrics.keys())}")
            
            # LR scheduler
            if epoch >= args.fix_lr_epochs:
                lr_scheduler.step(curr_test_metrics[base_metric])
                logging.info(
                    "LR after scheduling step: %s" %
                    [_['lr'] for _ in optimizer.param_groups])

            # Write metrics to tensorboard
            args.writer.add_scalars('Train', curr_train_metrics, epoch)
            args.writer.add_scalars('Val', curr_test_metrics, epoch)
            args.writer.flush()

            for k in curr_train_metrics.keys():
                if not k in train_metrics:
                    train_metrics[k] = [curr_train_metrics[k]]
                else:
                    train_metrics[k].append(curr_train_metrics[k])

            for k in curr_test_metrics.keys():
                if not k in val_metrics:
                    val_metrics[k] = [curr_test_metrics[k]]
                else:
                    val_metrics[k].append(curr_test_metrics[k])

            # 添加调试信息，检查 base_metric 是否存在
            if base_metric not in val_metrics:
                logging.error(f"base_metric '{base_metric}' not found in val_metrics. Available keys: {list(val_metrics.keys())}")
                # 如果找不到 base_metric，使用第一个可用的指标
                if val_metrics:
                    base_metric = list(val_metrics.keys())[0]
                    logging.info(f"Using '{base_metric}' as fallback metric")
                else:
                    logging.error("No validation metrics available!")
                    continue

            if max(val_metrics[base_metric]) == val_metrics[base_metric][-1]:
                logging.info("Found best validation %s!" % base_metric)

            utils.save_checkpoint(
                checkpoint_file, epoch, model, optimizer, lr_scheduler,
                train_metrics, val_metrics, data_parallel)
            logging.info("Saved checkpoint at %s" % checkpoint_file)

            utils.save_graph(train_metrics, val_metrics, args.exp_dir)

            
            # Early stopping check
            current_val_metric = curr_test_metrics[base_metric]
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                patience_counter = 0
                logging.info(f'New best validation {base_metric}: {best_val_metric:.6f}')
            else:
                patience_counter += 1
                logging.info(f'Validation {base_metric} not improved for {patience_counter} epochs (best: {best_val_metric:.6f})')
                
            if patience_counter >= early_stop_patience:
                logging.info(f'Early stopping triggered after {patience_counter} epochs without improvement')
                break

        return train_metrics, val_metrics


    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:  # pylint: disable=broad-except
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('exp_dir', type=str,
                        default='./experiments/fsd_mask_label_mult',
                        help="Path to save checkpoints and logs.")

    parser.add_argument('--n_train_items', type=int, default=None,
                        help="Number of items to train on in each epoch")
    parser.add_argument('--n_test_items', type=int, default=None,
                        help="Number of items to test.")
    parser.add_argument('--start_epoch', type=int, default=0,
                        help="Start epoch")
    parser.add_argument('--pretrain_path', type=str,
                        help="Path to pretrained weights")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help="List of GPU ids used for training. "
                        "Eg., --gpu_ids 2 4. All GPUs are used by default.")
    parser.add_argument('--detect_anomaly', dest='detect_anomaly',
                        action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--wandb', dest='wandb', action='store_true',
                        help="Whether to sync tensorboard to wandb")
    parser.add_argument('--overfit_single_batch', dest='overfit_single_batch', action='store_true',
                        help="Overfit a single training batch for debugging optimization path.")

    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    random.seed(230)
    np.random.seed(230)
    if args.use_cuda:
        torch.cuda.manual_seed(230)

    # Set up checkpoints
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    utils.set_logger(os.path.join(args.exp_dir, 'train.log'))

    # Load model and training params
    params = utils.Params(os.path.join(args.exp_dir, 'config.json'))
    for k, v in params.__dict__.items():
        vars(args)[k] = v

    # Initialize tensorboard writer
    tensorboard_dir = os.path.join(args.exp_dir, 'tensorboard')
    args.writer = SummaryWriter(tensorboard_dir, purge_step=args.start_epoch)
    if args.wandb:
        import wandb
        wandb.init(
            project='Semaudio', sync_tensorboard=True,
            dir=tensorboard_dir, name=os.path.basename(args.exp_dir))

    exec("import %s as network" % args.model)
    logging.info("Imported the model from '%s'." % args.model)

    train(args)

    args.writer.close()
    if args.wandb:
        wandb.finish()
