#!/usr/bin/env sh

python train.py --data_path=data --save_dir=checkpoints --dropout=0.2 --add_position_features --epochs=2 --max_len=150
