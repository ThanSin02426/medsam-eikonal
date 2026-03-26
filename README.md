# MedSAM-PINN: Physics-Informed Neural Networks for Zero-Shot Medical Image Segmentation

## Overview
This repository contains the implementation of MedSAM-PINN, a framework that integrates the Eikonal equation into the MedSAM architecture as a spatial regularization constraint. While foundation models like MedSAM demonstrate remarkable capabilities in promptable segmentation, they often struggle with boundary ambiguity, acoustic shadows, and speckle noise inherent in modalities like ultrasound. MedSAM-PINN addresses this by enforcing mathematical continuity across ambiguous tissue boundaries, allowing the network to learn continuous spatial geometry rather than relying solely on pixel textures.

## Key Features
* **Physics-Informed Architecture:** Integrates a custom convolutional projection head to replace discrete mask decoder outputs, predicting a continuous Signed Distance Field (SDF).
* **Eikonal Equation Regularization:** Enforces a physical constraint where the gradient magnitude of the true SDF must equal exactly 1 almost everywhere.
* **Zero-Shot Generalization:** Demonstrates strong generalization capabilities across unseen biological structures and datasets, provided the underlying imaging physics remains consistent.

## Architecture Details
The base MedSAM architecture relies on a Vision Transformer (ViT-B) image encoder, a prompt encoder for spatial cues, and a mask decoder. To preserve the generalized feature extraction of the foundation model, both the image encoder and prompt encoder remain entirely frozen during training. Modifications are strictly confined to the mask decoder.

Instead of binary logits optimized via cross-entropy loss, the modified projection head performs continuous regression to predict an SDF, denoted as $\phi(x)$. In this field, pixels inside the target yield negative distance values, outside pixels yield positive values, and the target boundary is implicitly defined by the zero-level set.

### Loss Function 
The total loss function balances the Mean Squared Error (MSE) against the ground-truth SDF and the physics constraint, with $\lambda$ set to 0.1 to stabilize training dynamics. 

The Eikonal physics loss is formulated as:
```math
\mathcal{L}_{Eikonal} = \frac{1}{N} \sum_{i=1}^{N} \left( \Vert \nabla \phi(x_i) \Vert - 1 \right)^2
```

## Training Curriculum
To prevent model collapse under the dual complexity of clinical noise and physical constraints, training is conducted in two phases:

1. **Phase 1 (Synthetic Pre-training):** The model is trained on a synthetic dataset of geometrically perfect shapes (solid circles and ellipses). This isolates the environment, allowing the projection head to learn the mathematical gradients of the Eikonal equation without medical image noise.
2. **Phase 2 (Clinical Fine-tuning):** The physics-aware weights are fine-tuned on clinical ultrasound data to map mathematical continuous boundaries to realistic, noisy human tissue. This phase was executed on a distributed cluster of 4 NVIDIA L4 GPUs using Distributed Data Parallel (DDP), utilizing dynamic bounding box prompts with a spatial jitter of $\pm15$ pixels to simulate clinical variance.

## Datasets
The pipeline utilized one dataset for Phase 2 fine-tuning and three distinct datasets for strict zero-shot evaluation:

* **Fine-tuning:** Breast Ultrasound Images (BUSI)
* **Test 1 (Zero-Shot):** HC18 (Fetal Head Circumference)
* **Test 2 (Zero-Shot):** Kaggle Brachial Plexus Nerve
* **Test 3 (Zero-Shot):** Kvasir-SEG (GI Polyps)

*Note: For the Kaggle dataset, empty masks (58% of the data) were explicitly filtered out during evaluation to prevent artificial score inflation via bounding-box shortcut learning.*

## Results
MedSAM-PINN significantly outperformed the baseline MedSAM on unseen ultrasound organs. The model experienced a mathematically consistent minor degradation on the Kvasir-SEG dataset, as the optical physics of a GI camera (endoscopy) differ from the acoustic scattering physics of ultrasound, confirming the model successfully encoded modality-specific physics.

| Dataset | Modality / Target | Baseline | Our PINN |
| :--- | :--- | :--- | :--- |
| **BUSI** | Ultrasound / Breast (Trained) | 0.6715 | Pending |
| **HC18** | Ultrasound / Fetal Head | 0.8606 | 0.8953 |
| **Kaggle BP** | Ultrasound / Neck Nerve | 0.7065 | 0.7275 |
| **Kvasir-SEG** | Endoscopy / GI Polyps | 0.7749 | 0.7516 |
