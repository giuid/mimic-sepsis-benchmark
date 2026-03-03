import os
import glob
from tensorboard.backend.event_processing import event_accumulator

def extract_metrics(log_dir):
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        metrics = {}
        tags = ea.Tags()['scalars']
        for tag in tags:
            events = ea.Scalars(tag)
            metrics[tag] = [(e.step, e.value) for e in events]
        return metrics
    except Exception as e:
        return {"error": str(e)}

def analyze_folder(name):
    log_dirs = glob.glob(f'outputs/logs/{name}/version_*')
    best_log = None
    max_epochs = 0
    for d in log_dirs:
        m = extract_metrics(d)
        if 'val/auprc' in m:
            n = len(m['val/auprc'])
            if n > max_epochs:
                max_epochs = n
                best_log = d
                
    if best_log:
        print(f"\n=== Result for {name} (Log: {best_log}) ===")
        m = extract_metrics(best_log)
        
        for metric in ['val/loss', 'val/auprc']:
            if metric in m:
                vals = m[metric]
                print(f"{metric}:")
                print(f"  First: {vals[0][1]:.4f} (Step {vals[0][0]})")
                if len(vals) > 1:
                    print(f"  Middle: {vals[len(vals)//2][1]:.4f} (Step {vals[len(vals)//2][0]})")
                    print(f"  Last:  {vals[-1][1]:.4f} (Step {vals[-1][0]})")
                    
                    # Trend check
                    if len(vals) > 5:
                        last_avg = sum(v[1] for v in vals[-3:]) / 3
                        prev_avg = sum(v[1] for v in vals[-6:-3]) / 3
                        diff = last_avg - prev_avg
                        direction = "decreasing" if diff < 0 else "increasing"
                        print(f"  Trend (last 6 eps): {direction} (delta: {diff:.6f})")
    else:
        print(f"No valid logs found for {name}")

if __name__ == '__main__':
    analyze_folder("joint_saits_Vanilla")
    analyze_folder("joint_saits_KGI")
    analyze_folder("joint_mrnn_Vanilla")
    analyze_folder("joint_brits_Vanilla")
