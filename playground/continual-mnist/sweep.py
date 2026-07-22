#!/usr/bin/env python3
"""
Fast sweep — runs each experiment in sequence.
Each 20-task run takes ~60-90s.

GPU: RTX 4090 — runs at full utilization.
"""

import os, sys, json, subprocess, time
from datetime import datetime

SCRIPT = os.path.join(os.path.dirname(__file__), 'experiment.py')
RESULTS = os.path.join(os.path.dirname(__file__), 'results')
PYTHON = '/home/mmurua/scratch/NeuroAI/Project/NMA-NeuroAI-Online-Learning-Project/.venv/bin/python'
os.makedirs(RESULTS, exist_ok=True)

def run(name, args, n_tasks=20):
    output = os.path.join(RESULTS, f"{name}.json")
    if os.path.exists(output):
        print(f"  SKIP (exists): {name}")
        return
    cmd = [PYTHON, SCRIPT, '--n-tasks', str(n_tasks), '--epochs', '1',
           '--batch-size', '500', '--output', output] + args
    t0 = time.time()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {name}...", end=' ', flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"FAILED ({elapsed:.0f}s)")
        print(r.stderr[-500:])
        return None
    # Parse result
    for line in r.stdout.split('\n'):
        if 'Final avg acc' in line:
            acc = float(line.split(':')[1].strip())
        if 'Avg forgetting' in line:
            forget = float(line.split(':')[1].strip())
        if 'Avg BWT' in line:
            bwt = float(line.split(':')[1].strip())
    print(f"acc={acc:.4f} forget={forget:.4f} bwt={bwt:.4f} ({elapsed:.0f}s)")
    return {'acc': acc, 'forget': forget, 'bwt': bwt}

def main():
    print("="*60)
    print("CONTINUAL MNIST SWEEP — Fast Mode")
    print(f"GPU: RTX 4090")
    print(f"Time per run: ~60-90s (20 tasks)")
    print("="*60)

    run("bp_adam", ["--method", "bp"])
    run("bp_adam_lr3e-4", ["--method", "bp", "--lr", "0.0003"])
    run("pc_vanilla", ["--method", "pc", "--T", "20"])
    run("pc_vanilla_lr3e-4", ["--method", "pc", "--T", "20", "--lr", "0.0003"])

    # CLASSP sweep — BP
    for p in [1.0, 2.0, 3.0]:
        for th in [0.0, 1e-6, 1e-5]:
            run(f"bp_classp_p{p}_th{th}", ["--method", "bp", "--classp",
                                            "--classp-p", str(p),
                                            "--classp-threshold", str(th)])

    # CLASSP sweep — PC
    for p in [1.0, 2.0, 3.0]:
        for th in [0.0, 1e-6, 1e-5]:
            run(f"pc_classp_p{p}_th{th}", ["--method", "pc", "--T", "20", "--classp",
                                            "--classp-p", str(p),
                                            "--classp-threshold", str(th)])

    # iPC
    run("pc_ipc", ["--method", "pc", "--T", "20", "--ipc"])
    run("pc_ipc_lr3e-4", ["--method", "pc", "--T", "20", "--ipc", "--lr", "0.0003"])

    # 50-task benchmarks
    for name, args in [
        ("bp_adam_50tasks", ["--method", "bp"]),
        ("pc_vanilla_50tasks", ["--method", "pc", "--T", "20"]),
        ("pc_vanilla_lr3e-4_50tasks", ["--method", "pc", "--T", "20", "--lr", "0.0003"]),
        ("pc_classp_p2_th1e-5_50tasks", ["--method", "pc", "--T", "20", "--classp",
                                          "--classp-p", "2.0", "--classp-threshold", "1e-5"]),
        ("pc_ipc_50tasks", ["--method", "pc", "--T", "20", "--ipc"]),
        ("bp_classp_p2_th1e-5_50tasks", ["--method", "bp", "--classp",
                                          "--classp-p", "2.0", "--classp-threshold", "1e-5"]),
    ]:
        run(name, args, n_tasks=50)

    # ====== SUMMARY ======
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    rows = []
    for f in sorted(os.listdir(RESULTS)):
        if not f.endswith('.json') or 'ipc' not in f.lower():
            continue
        with open(os.path.join(RESULTS, f)) as fh:
            d = json.load(fh)
        rows.append((f.replace('.json',''), d['final_avg_acc'], d['avg_forgetting'], d['avg_bwt'],
                     d.get('elapsed_seconds', 0)))

    print(f"{'Experiment':<40} {'Acc':>8} {'Forget':>8} {'BWT':>8} {'Time':>8}")
    print("-"*75)
    for n, a, f, b, t in sorted(rows, key=lambda x: -x[1]):
        print(f"{n:<40} {a:>8.4f} {f:>8.4f} {b:>8.4f} {t:>7.0f}s")

    print(f"\nResults: {RESULTS}")


if __name__ == '__main__':
    main()
