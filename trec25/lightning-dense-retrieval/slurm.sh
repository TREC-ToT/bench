#!/usr/bin/bash

srun \
    --container-image=mam10eks/trec-tot-lightning-ir-baseline:dev-0.0.1 \
    --mem=120g \
    -c 6 \
    --container-remap-root \
    --gres=gpu:ampere:1 \
    --pty bash -i
