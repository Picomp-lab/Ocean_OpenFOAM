#!/usr/bin/env python3
"""
生成 AMP_WK x SLP 的 3x3 参数组合算例目录（基准组 TK94 已跑，跳过）。

严格原则：只改 input.txt 里的 AMP_WK 和 SLP 两行，其余逐字节保持与 TK94 一致。
gauges.txt 原样复制；可执行文件用软链接指向 TK94 已编译的二进制。
"""
import os, shutil, re

BASE = os.path.dirname(os.path.abspath(__file__))   # data/fwv，脚本自己所在的目录
SRC = os.path.join(BASE, 'TK94')
EXE = 'funwave--gnu-parallel-single'

AMPS = [('0635', '0.0635'), ('0610', '0.0610'), ('0585', '0.0585')]
SLPS = [('325', '1:32.5', '0.030769231'),
        ('350', '1:35',   '0.028571429'),
        ('375', '1:37.5', '0.026666667')]

BASELINE = ('0635', '350')          # 已跑完 = TK94

src_input = open(os.path.join(SRC, 'input.txt')).read()

SBATCH = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=share
#SBATCH --constraint=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# AMP_WK = {amp}   SLP = {slp} ({slpname})
# input.txt 里 PX=1 PY=1，所以 ntasks 必须 = 1
# 基准组(AMP_WK=0.0635, 1:35)在 EPYC 单核实测 14556 s (4h02m)

module load openmpi/4.0_gcc-10

cd "$SLURM_SUBMIT_DIR"
mkdir -p output

mpirun -np 1 ./{exe} input.txt
"""

made = []
for atag, aval in AMPS:
    for stag, sname, sval in SLPS:
        if (atag, stag) == BASELINE:
            continue
        name = 'H%s_S%s' % (atag, stag)
        d = os.path.join(BASE, name)
        os.makedirs(os.path.join(d, 'output'), exist_ok=True)

        # ---- input.txt：只替换两行 ----
        txt = src_input
        txt, n1 = re.subn(r'(?m)^AMP_WK = 0\.0635\s*$', 'AMP_WK = %s' % aval, txt)
        txt, n2 = re.subn(r'(?m)^SLP = 0\.028571429\s*$', 'SLP = %s' % sval, txt)
        assert n1 == 1, '%s: AMP_WK 替换了 %d 处' % (name, n1)
        assert n2 == 1, '%s: SLP 替换了 %d 处' % (name, n2)
        open(os.path.join(d, 'input.txt'), 'w').write(txt)

        # ---- gauges.txt 原样 ----
        shutil.copy2(os.path.join(SRC, 'gauges.txt'), os.path.join(d, 'gauges.txt'))

        # ---- 可执行文件软链接 ----
        link = os.path.join(d, EXE)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.join(SRC, EXE), link)

        # ---- 作业脚本 ----
        open(os.path.join(d, 'run.sbatch'), 'w').write(
            SBATCH.format(name=name, amp=aval, slp=sval, slpname=sname, exe=EXE))

        made.append((name, aval, sname, sval))

print('已生成 %d 个算例目录：\n' % len(made))
print('  %-12s %-9s %-9s %s' % ('目录', 'AMP_WK', '坡度', 'SLP'))
print('  ' + '-'*46)
for name, aval, sname, sval in made:
    print('  %-12s %-9s %-9s %s' % (name, aval, sname, sval))
print('\n  (基准组 AMP_WK=0.0635 / 1:35 = 已跑完的 TK94，跳过)')
