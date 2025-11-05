#!/bin/bash

PYTHON="/mnt/c/Users/USER/anaconda3/envs/ognn/python.exe"

$PYTHON finetune.py --fold 1 --seed 2023
$PYTHON finetune.py --fold 2 --seed 2023
$PYTHON finetune.py --fold 3 --seed 2023
$PYTHON finetune.py --fold 4 --seed 2023
$PYTHON finetune.py --fold 5 --seed 2023

$PYTHON make_sub.py
