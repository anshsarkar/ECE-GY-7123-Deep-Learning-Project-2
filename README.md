# ECE-GY-7123-Deep-Learning-Project-2

**Team GradesAreAllYouNeed**  
Ansh Sarkar (as20363@nyu.edu)  
Princy Doshi (pd2672@nyu.edu)  
Simran Kucheria (sk11645@nyu.edu)  
New York University

## Overview

This project implements parameter-efficient fine-tuning (PEFT) using Low-Rank Adaptation (LoRA) on the AGNews dataset, targeting text classification with under 1 million trainable parameters and a Roberta base. LoRA freezes pretrained weights and injects trainable low-rank matrices, enabling efficient adaptation and hardware savings.

Main LoRA parameters:
- `r` (rank): dimension of the low-rank matrices
- `alpha`: scaling factor for the LoRA update

**Final Results:**  
- Validation Accuracy: **93.79%**  
- Test Accuracy: **85.05%**  

## Methodology

- **Dataset:** AGNews (train/val split from Huggingface, test from Kaggle)
- **Experiments:** LoRA configs, regularization, training stabilization (DoRA, rsLoRA), weight initialization
- **Preprocessing:** Tried stopword/punctuation removal, stemming, lemmatization (not used—hurt performance)

## LoRA Configurations

| Rank | Alpha | Target Modules              | Val. Loss | Val. Accuracy (%) |
|------|-------|----------------------------|-----------|-------------------|
| 8    | 16    | ["query", "value"]         | 0.174     | 94.32             |
| 4    | 16    | ["query", "value"]         | 0.174     | 94.43             |
| 4    | 8     | ["query", "value"]         | 0.180     | 94.13             |
| 4    | 32    | ["query", "value"]         | 0.215     | 92.58             |
| 4    | 2     | ["query", "value"]         | 0.170     | 94.49             |
| 4    | 8     | ["query", "output.dense"]  | 0.177     | 94.42             |
| 2    | 4     | ["query", "value", "output.dense"] | 0.179 | 94.13      |

**Selected:** Rank 4, Alpha 8, ["query", "value"] (best test performance)

## Advanced Techniques

- **DoRA:** Decomposes weights into magnitude and direction for more stable, generalizable training
- **rsLoRA:** Adjusts scaling to stabilize training, especially at higher ranks

## Regularization

- **LoRA Dropout:** 0.6 (best performance)
- **L2 Regularization:** 0.1 (weight decay)
- **Learning Rate Scheduler:** Cosine (with 10% warmup)
- **Early Stopping:** Patience 3, threshold 0.01

## Weight Initialization

- **Tried:** Default (Kaiming), Gaussian, PiSSA (SVD-based)
- **Used:** Gaussian (best stability, avoids overfitting)

## Training Hyperparameters

- **Learning Rate:** 2e-4
- **Batch Size:** 32
- **Epochs:** 10 (early stopping usually at 6)
- **DoRA:** True
- **rsLoRA:** True

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685
- Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", arXiv:2402.09353
- Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA", arXiv:2312.03732