"""
Test script to evaluate the model.
"""

import argparse
import importlib
import multiprocessing
import os, glob
import logging

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm  # pylint: disable=unused-import
try:
    from torchmetrics.audio.snr import (
        scale_invariant_signal_noise_ratio as si_snr,
        signal_noise_ratio as snr,
    )
    from torchmetrics.audio.sdr import (
        signal_distortion_ratio as sdr,
        scale_invariant_signal_distortion_ratio as si_sdr,
    )
except Exception:  # Fallback for older torchmetrics
    import warnings as _tm_warnings
    _tm_warnings.filterwarnings("ignore", category=FutureWarning)
    from torchmetrics.functional import (
        scale_invariant_signal_noise_ratio as si_snr,
        signal_noise_ratio as snr,
        signal_distortion_ratio as sdr,
        scale_invariant_signal_distortion_ratio as si_sdr,
    )

from src.helpers import utils
from src.training.synthetic_dataset import FSDSoundScapesDataset, tensorboard_add_metrics
from src.training.synthetic_dataset import tensorboard_add_sample
from src.training.folder_jams_dataset import FolderJAMSDataset
from pathlib import Path

def test_epoch(model: nn.Module, device: torch.device,
               test_loader: torch.utils.data.dataloader.DataLoader,
               n_items: int, loss_fn, metrics_fn,
               profiling: bool = False, epoch: int = 0,
               writer: SummaryWriter = None, data_params = None) -> float:
    """
    Evaluate the network.
    """
    model.eval()
    metrics = {}
 
    with torch.no_grad():
        for batch_idx, (mixed, label, gt) in \
                enumerate(tqdm(test_loader, desc='Test', ncols=100)):
            mixed = mixed.to(device)
            label = label.to(device)
            gt = gt.to(device)

            # 跳过无效批次：无前景标签或 gt 能量过低（与训练保持一致）
            try:
                gt_energy_all = (gt ** 2).mean().item()
                label_sum_all = label.sum().item()
            except Exception:
                gt_energy_all = 0.0
                label_sum_all = 0.0
            if (gt_energy_all < 1e-8) or (label_sum_all <= 0):
                if batch_idx % 100 == 0:
                    logging.warning(f"Skip test batch {batch_idx}: label_sum={label_sum_all:.1f}, gt_energy={gt_energy_all:.2e}")
                continue

            # Run through the model
            with profile(activities=[ProfilerActivity.CPU],
                         record_shapes=True) as prof:
                with record_function("model_inference"):
                    output = model(mixed, label)
            if profiling:
                logging.info(
                    prof.key_averages().table(sort_by="self_cpu_time_total",
                                              row_limit=20))

            # Compute loss
            loss = loss_fn(output, gt)
            
            # 添加调试信息
            if batch_idx == 0:  # 只对第一个batch打印调试信息
                logging.info(f"Test batch {batch_idx}:")
                logging.info(f"  Output shape: {output.shape}, Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
                logging.info(f"  GT shape: {gt.shape}, GT range: [{gt.min().item():.6f}, {gt.max().item():.6f}]")
                logging.info(f"  Loss value: {loss.item():.6f}")
                logging.info(f"  Output mean: {output.mean().item():.6f}, GT mean: {gt.mean().item():.6f}")
                logging.info(f"  Output std: {output.std().item():.6f}, GT std: {gt.std().item():.6f}")

            # Compute metrics
            metrics_batch = metrics_fn(mixed, output, gt)
            metrics_batch['loss'] = [loss.item()]
            metrics_batch['runtime'] = [prof.profiler.self_cpu_time_total/1000]
            for k in metrics_batch.keys():
                if not k in metrics:
                    metrics[k] = metrics_batch[k]
                else:
                    metrics[k] += metrics_batch[k]

            if writer is not None:
                if batch_idx == 0:
                    tensorboard_add_sample(
                        writer, tag='Test',
                        sample=(mixed[:8], label[:8], gt[:8], output[:8]),
                        step=epoch, params=data_params)
                tensorboard_add_metrics(
                    writer, tag='Test', metrics=metrics_batch, label=label,
                    step=epoch)

            if n_items is not None and batch_idx == (n_items - 1):
                break

        avg_metrics = {k: np.mean(metrics[k]) for k in metrics.keys()}
        avg_metrics_str = "Test:"
        for m in avg_metrics.keys():
            avg_metrics_str += ' %s=%.04f' % (m, avg_metrics[m])
        logging.info(avg_metrics_str)

        return avg_metrics

def evaluate(network, args: argparse.Namespace):
    """
    Evaluate the model on a given dataset.
    """

    # Load dataset from jams+wav folder
    # 获取实际数据集中的标签列表
    fg_root = Path('my_dataset/foreground')
    labels_82 = [d.name for d in fg_root.iterdir() if d.is_dir()]
    labels_82.sort()  # 确保标签顺序一致
    
    data_test = FolderJAMSDataset(
        root_dir=args.test_data['input_dir'],
        label_list=labels_82,
        sr=args.test_data['sr'],
        resample_rate=args.test_data['resample_rate'])
    logging.info("Loaded test dataset at %s containing %d elements" %
                 (args.test_data['input_dir'], len(data_test)))
 
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

    # Set up data loader
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.eval_batch_size,
                                              **kwargs)

    # Set up model
    model = network.Net(**args.model_params)
    if use_cuda and data_parallel:
        model = nn.DataParallel(model, device_ids=device_ids)
        logging.info("Using data parallel model")
    model.to(device)

    # Load weights
    if args.pretrain_path == "best":
        ckpts = glob.glob(os.path.join(args.exp_dir, '*.pt'))
        ckpts.sort(
            key=lambda _: int(os.path.splitext(os.path.basename(_))[0]))
        val_metrics = torch.load(ckpts[-1])['val_metrics'][args.base_metric]
        best_epoch = max(range(len(val_metrics)), key=val_metrics.__getitem__)
        args.pretrain_path = os.path.join(args.exp_dir, '%d.pt' % best_epoch)
        logging.info(
            "Found 'best' validation %s=%.02f at %s" %
            (args.base_metric, val_metrics[best_epoch], args.pretrain_path))
    if args.pretrain_path != "":
        utils.load_checkpoint(
            args.pretrain_path, model, data_parallel=data_parallel)
        logging.info("Loaded pretrain weights from %s" % args.pretrain_path)

    # Evaluate
    try:
        return test_epoch(
            model, device, test_loader, args.n_items, network.loss,
            network.metrics, args.profiling)
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:  # pylint: disable=broad-except
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('experiments', nargs='+', type=str,
                        default=None,
                        help="List of experiments to evaluate. "
                        "Provide only one experiment when providing "
                        "pretrained path. If pretrianed path is not "
                        "provided, epoch with best validation metric "
                        "is used for evaluation.")
    parser.add_argument('--results', type=str, default="",
                        help="Path to the CSV file to store results.")

    # System params
    parser.add_argument('--n_items', type=int, default=None,
                        help="Number of items to test.")
    parser.add_argument('--pretrain_path', type=str, default="best",
                        help="Path to pretrained weights")
    parser.add_argument('--profiling', dest='profiling', action='store_true',
                        help="Enable or disable profiling.")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help="List of GPU ids used for training. "
                        "Eg., --gpu_ids 2 4. All GPUs are used by default.")
    args = parser.parse_args()

    results = []

    for exp_dir in args.experiments:
        eval_args = argparse.Namespace(**vars(args))
        eval_args.exp_dir = exp_dir

        utils.set_logger(os.path.join(exp_dir, 'eval.log'))
        logging.info("Evaluating %s ..." % exp_dir)

        # Load model and training params
        params = utils.Params(os.path.join(exp_dir, 'config.json'))
        for k, v in params.__dict__.items():
            vars(eval_args)[k] = v

        network = importlib.import_module(eval_args.model)
        logging.info("Imported the model from '%s'." % eval_args.model)

        curr_res = evaluate(network, eval_args)
        curr_res['experiment'] = os.path.basename(exp_dir)
        results.append(curr_res)

        del eval_args

    if args.results != "":
        print("Writing results to %s" % args.results)
        pd.DataFrame(results).to_csv(args.results, index=False)
