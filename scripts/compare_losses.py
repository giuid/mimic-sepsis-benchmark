import os
import glob
from tensorboard.backend.event_processing import event_accumulator

def get_last_metrics(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    res = {}
    for tag in ['train/loss_imp', 'train/loss_cls', 'train/loss_total']:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            res[tag] = events[-1].value
    return res

dirs = sorted(glob.glob('outputs/logs/joint_saits_Vanilla/version_*'))
for d in dirs:
    print(f"--- {d} ---")
    metrics = get_last_metrics(d)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
