import os
import glob
from tensorboard.backend.event_processing import event_accumulator

def list_tags(log_dir):
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        print(f"Tags in {log_dir}:")
        print(ea.Tags()['scalars'])
    except Exception as e:
        print(f"Error in {log_dir}: {e}")

if __name__ == '__main__':
    dirs = glob.glob('outputs/logs/joint_saits_Vanilla/version_*')
    for d in dirs:
        list_tags(d)
