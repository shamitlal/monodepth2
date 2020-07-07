# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

# python train.py --model_name mono_model --png --data_path /projects/katefgroup/datasets/shamit_carla_correct/npys --dataset carla --run_name lr4
# python train.py --model_name mono_model --png --data_path /projects/katefgroup/datasets/shamit_carla_correct/npys --dataset carla --run_name lr3 --learning_rate 1e-3
# python train.py --model_name mono_model --png --data_path /projects/katefgroup/datasets/shamit_carla_correct/npys --dataset carla --run_name lr2 --learning_rate 1e-2

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
