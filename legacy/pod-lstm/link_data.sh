#!/bin/bash

# POD / LSTM 那条线的数据在仓库外（case 23 G、各 *_results 共几 G），没进版本库。
OCEAN_DATA="${OCEAN_DATA:-$HOME/hpc-share/ocean_project}"
for dir in /nfs/hpc/share/coast-lab/OpenFOAM/BA_TingKirby1994_3D_spilling/BA02_9M_Smag_old/*/; do
    name=$(basename "$dir")
    if [[ "$name" != "constant" && "$name" != "system" ]]; then
        ln -s "$dir" "$OCEAN_DATA/case/$name" 2>/dev/null
    fi
done
