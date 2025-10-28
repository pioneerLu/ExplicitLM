#!/bin/bash
# Simple test to validate the script without actually running training

# Set test variables
EXP_ID="test_exp"
EXP_DESC="Test experiment for Hydra-Zen script validation"
DATASET_VERSION=""
VAL_DATASET_VERSION=""
EMBEDDING_VERSION=""
DATABASE_VERSION=""
CACHE_VERSION=""
TRAIN_ARGS="training.epochs=1 model.knowledge_num=1024"

# Source the script to check for syntax errors
set -e
source experiments/scripts/_run_experiment_core_hydra_zen.sh