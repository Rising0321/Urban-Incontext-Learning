# Urban In-Context Learning: Bridging Pretraining and Inference through Masked Diffusion for Urban Profiling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper "Urban In-Context Learning: Bridging Pretraining and Inference through Masked Diffusion for Urban Profiling" 

## Overview

Urban In-Context Learning (UIL) is a novel framework that bridges the gap between pretraining and inference in urban computing through masked diffusion. 

## Requirements

```bash
torch>=1.9.0
numpy
pandas
tqdm
diffusers
swanlab
transformers
```

## Quick Start

Currently, only the test code is available for model evaluation. The full training code and data processing pipeline will be released upon paper acceptance.

To test the model:

```bash
python test_main.py \
    --dataset Manhattan \
    --model d10 \
    --sample_step 100 \
    --source UrbanVLP
```

## Model Checkpoints

We provide pretrained model checkpoints for the Manhattan dataset with three different tasks:
- `Manhattan_carbon_best_model.pth`: For carbon emission prediction
- `Manhattan_crash_best_model.pth`: For traffic accident prediction
- `Manhattan_house_best_model.pth`: For house price prediction

## Directory Structure

```
MUD/
├── checkpoints/           # Pretrained model checkpoints
├── data/                 # Dataset files (partial)
│   └── Manhattan/       
├── embeddings/           # Urban embeddings
│   └── UrbanVLP/        
├── model/               # Model architecture
│   └── MDT.py          # Masked Diffusion Transformer
├── utils/               # Utility functions
└── test_main.py        # Testing script
```

## License

This project is licensed under the MIT License.

## Notes

- The current release only includes the test code and model checkpoints
- Full training code and data processing pipeline will be made available upon paper acceptance
- For any questions about the implementation, please open an issue
