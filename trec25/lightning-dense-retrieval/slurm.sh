#!/usr/bin/bash

srun \
    --container-image=mam10eks/trec-tot-lightning-ir-baseline:dev-0.0.1 \
    --mem=60g \
    -c 3 \
    --container-remap-root \
    --gres=gpu:ampere:1 \
    --pty bash -i
