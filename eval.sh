#!/usr/bin/env sh

python evaluate.py --split=test --model_path=checkpoints/best_ckpt.pt --data_path=data --batch_size=32 --ref_path=refs/ref.txt --result_path=results/eval.txt --beam_size 1
