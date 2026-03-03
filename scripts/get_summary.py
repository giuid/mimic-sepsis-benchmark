import os
import glob
from tensorboard.backend.event_processing import event_accumulator

def get_summary(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    if 'val/auprc' not in ea.Tags()['scalars']:
        return None
    events = ea.Scalars('val/auprc')
    best_auprc = max([e.value for e in events])
    last_epoch = len(events) - 1
    return {"best_auprc": best_auprc, "last_epoch": last_epoch}

print("SAITS Vanilla (versions):")
for v in range(4):
    d = f"outputs/logs/joint_saits_Vanilla/version_{v}"
    if os.path.exists(d):
        s = get_summary(d)
        if s:
            print(f"  Version {v}: Best AUPRC {s['best_auprc']:.4f} at epoch unknown (total {s['last_epoch']} epochs)")

print("\nSAITS KGI (versions):")
for v in range(3):
    d = f"outputs/logs/joint_saits_KGI/version_{v}"
    if os.path.exists(d):
        s = get_summary(d)
        if s:
            print(f"  Version {v}: Best AUPRC {s['best_auprc']:.4f} at epoch unknown (total {s['last_epoch']} epochs)")
