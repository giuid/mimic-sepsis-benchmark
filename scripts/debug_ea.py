import os
from tensorboard.backend.event_processing import event_accumulator

log_file = 'outputs/logs/joint_saits_Vanilla/version_2/'
print(f"Loading {log_file}...")
ea = event_accumulator.EventAccumulator(log_file)
ea.Reload()

print("Tags found:", ea.Tags().keys())
if 'scalars' in ea.Tags():
    print("Scalars:", ea.Tags()['scalars'])
    if 'val/auprc' in ea.Tags()['scalars']:
        events = ea.Scalars('val/auprc')
        print(f"AUPRC events: {len(events)}")
        for e in events[-5:]:
            print(f"Step {e.step}, Value {e.value:.4f}")
    if 'val/loss' in ea.Tags()['scalars']:
        events = ea.Scalars('val/loss')
        print(f"Loss events: {len(events)}")
        for e in events[-5:]:
            print(f"Step {e.step}, Value {e.value:.4f}")
