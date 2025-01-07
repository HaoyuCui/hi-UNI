import os
import subprocess
import sys

# os.chdir('..')

total_fold = len(os.listdir('kf')) // 2

for fold in range(1, total_fold + 1):
    args = [sys.executable] + ['train.py'] + ['--fold', str(fold)]
    print(f"Running: {' '.join(args)}")
    subprocess.run(args)