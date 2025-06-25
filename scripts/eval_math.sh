#!/bin/bash

# Task Selection
TASK="math" # Available options: mbpp2, math, ai2_arc

# First Stage Inference: Classification Expert
# Set to 'None' if not using cls expert
CLS_EXPERT_PATH="None"

# Second Stage: Expert Models
# Set your actual model paths
CODE_EXPERT_PATH="models/code_expert.pt"
MATH_EXPERT_PATH="models/math_expert.pt"
REASONING_EXPERT_PATH="models/reasoning_expert.pt"

# Start evaluation!
CUDA_VISIBLE_DEVICES=0,1 python svd_reinforce_hydra.py \
    base_model@_global_=tinyllama \
    task@_global_=$TASK \
    mode@_global_=eval \
    prompt_based_eval=True \
    experts_path_dict.code=$CODE_EXPERT_PATH \
    experts_path_dict.math=$MATH_EXPERT_PATH \
    experts_path_dict.reasoning=$REASONING_EXPERT_PATH \
    load_ckpt=$CLS_EXPERT_PATH
