# Comparative Analysis of U-Net-Based Architectures for Brain Tumor Segmentation

A research-oriented comparative study of **2D and 3D U-Net-based semantic segmentation architectures** for automated brain tumor segmentation in multi-modal MRI. The project investigates how architectural choices—including **residual learning, attention mechanisms, multi-scale feature extraction, volumetric modeling, and ensemble learning**—affect segmentation performance on the **BraTS 2020** dataset.

The repository contains the complete experimental notebooks used to develop, train, evaluate, and visualize the investigated models.

> **Research question:** How do different U-Net-based architectures and 2D/3D representations compare for multi-modal brain tumor segmentation, and what effect do attention, residual connections, volumetric context, and ensemble learning have on segmentation performance?

---

## 1. Research Objective

Semantic segmentation of brain tumors from MRI is challenging because tumor regions vary substantially in **shape, size, intensity, location, and appearance across patients**. In addition, different tumor components can have very different visual characteristics.

This project was designed as a **comparative experimental study**, rather than as the implementation of a single segmentation model.

The study investigates:

- **2D vs. 3D segmentation** and the importance of volumetric spatial context.
- **Vanilla U-Net vs. residual U-Net architectures**.
- The contribution of **attention mechanisms** to feature selection.
- The use of **multi-scale feature extraction** through ASPP-style modules.
- Whether an **ensemble of specialized segmentation models** can improve performance.
- Differences in performance across **Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET)** regions.

The experiments are implemented in separate Jupyter notebooks so that each architecture can be studied independently and compared through its training behavior and evaluation results.

---

# 2. Experimental Architecture Set

The project investigates the following model families:

| Model | Dimensionality | Main Idea |
|---|---:|---|
| **U-Net** | 2D | Baseline slice-based segmentation |
| **Attention ResUNet** | 2D | Residual learning + attention + multi-scale features |
| **Attention ResUNet Ensemble** | 2D | Multiple specialized models combined for segmentation |
| **U-Net** | 3D | Volumetric baseline preserving 3D spatial context |
| **ResUNet** | 3D | Residual learning for volumetric segmentation |
| **Attention ResUNet** | 3D | Attention + residual learning in a 3D architecture |

The purpose is not simply to build several models, but to **experimentally examine architectural differences under medical-image segmentation settings**.

---

# 3. Dataset

The experiments use the **BraTS 2020 (Brain Tumor Segmentation)** dataset.

BraTS provides multi-modal brain MRI volumes together with expert tumor segmentation annotations.

Each subject contains four MRI modalities:

- **T1**
- **T1ce / T1Gd**
- **T2**
- **FLAIR**

The segmentation labels distinguish tumor components:

| Label | Tumor Component |
|---:|---|
| 1 | Necrotic / Non-enhancing Tumor (NCR/NET) |
| 2 | Peritumoral Edema (ED) |
| 4 | Enhancing Tumor (ET) |

For the 3D experiments, these labels are transformed into three binary segmentation targets:

- **WT — Whole Tumor**
- **TC — Tumor Core**
- **ET — Enhancing Tumor**

The MRI volumes are loaded from NIfTI files using `nibabel`.

### Dataset source

The notebooks reference the BraTS 2020 training dataset:

https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

The dataset is **not included in this repository** because of its size and licensing/distribution considerations.

---

# 4. Experimental Design

The study follows an end-to-end segmentation workflow:
<img width="1376" height="768" alt="Medical_image_segmentation_resea…_20260920200843" src="https://github.com/user-attachments/assets/c6a51014-9cec-4564-a7c2-db2d9e479c83" />



The notebooks implement different experimental configurations around this common workflow.

---

# 5. Repository Structure

```text
Brain-Tumor-Segmentation-main/
│
├── AttentionResunet-2D.ipynb
├── Ensemble-AttentionResunet-2D.ipynb
├── Unet-2D.ipynb
├── Unet-3D.ipynb
├── ResUnet-3D.ipynb
├── AttentionUnet-3D.ipynb
└── README.md
```

### Notebook-to-experiment mapping

| Notebook | Experiment |
|---|---|
| `Unet-2D.ipynb` | 2D U-Net baseline |
| `AttentionResunet-2D.ipynb` | 2D Attention + Residual U-Net |
| `Ensemble-AttentionResunet-2D.ipynb` | 2D ensemble / mixture-of-experts style experiment |
| `Unet-3D.ipynb` | 3D U-Net baseline |
| `ResUnet-3D.ipynb` | 3D Residual U-Net |
| `AttentionUnet-3D.ipynb` | 3D Attention U-Net / attention-residual experiment |

Each notebook contains the corresponding preprocessing, architecture implementation, training procedure, evaluation, and visualization experiments.

---

# 6. Preprocessing

The project uses multi-modal MRI data as the input to the segmentation models.

The preprocessing pipeline includes:

- Loading NIfTI MRI volumes.
- Loading corresponding segmentation masks.
- Combining the four MRI modalities into multi-channel inputs.
- MRI intensity normalization.
- Converting BraTS labels into binary target masks.
- Creating PyTorch `Dataset` and `DataLoader` pipelines.
- Preparing 2D slices for the 2D experiments.
- Preserving complete volumetric information for the 3D experiments.
- Visualization of MRI modalities and corresponding segmentation masks.

For the 3D experiments, a recorded validation sample has the form:

```text
Image: (1, 4, 155, 240, 240)
Mask:  (1, 3, 155, 240, 240)
```

representing four MRI input modalities and three tumor-region targets.

---

# 7. Data Splitting

The 3D experimental pipeline uses metadata from the BraTS dataset and performs stratified splitting based on age groups.

The recorded split contains:

| Split | Samples |
|---|---:|
| Training | 201 |
| Validation | 34 |
| Test | 133 |
| Total | 368 |

The test partition in the notebook is constructed from subjects with missing age information in the merged metadata. Consequently, this should be understood as an **experiment-specific split**, rather than a conventional randomly sampled held-out test set.

This distinction is important when interpreting and reproducing the reported results.

---

# 8. Model Architectures

## 8.1 2D U-Net — Baseline

`Unet-2D.ipynb` establishes a slice-based U-Net baseline.

The model follows the conventional encoder-decoder structure:
<img width="1376" height="768" alt="U-Net_architecture_medical_image…_20260920201524" src="https://github.com/user-attachments/assets/77c3dd45-53ef-4bec-a1c4-09215da309b5" />


The purpose of this experiment is to establish a baseline for comparison with more sophisticated architectures.

---

## 8.2 2D Attention ResUNet

`AttentionResunet-2D.ipynb` investigates whether combining several architectural improvements can improve segmentation.

The architecture incorporates:

- Residual blocks
- Squeeze-and-Excitation attention
- Attention-based decoder components
- ASPP / multi-scale feature extraction
- U-Net-style skip connections

Conceptually:
<img width="1376" height="768" alt="Attention_residual_U-Net_block_d…_20260920201914" src="https://github.com/user-attachments/assets/d2312062-5cf7-42b4-b876-bbde5e8afbcd" />


This experiment investigates whether improved feature representation and selective attention can provide advantages over a conventional U-Net.

---

## 8.3 2D Attention ResUNet Ensemble

`Ensemble-AttentionResunet-2D.ipynb` extends the 2D experiment through an ensemble strategy.

Instead of relying on a single segmentation model, multiple independently trained models are used for different target channels.

The recorded experiment trains four specialized models and combines their predictions into a final multi-channel segmentation output.

This provides an experimental investigation of whether **model specialization and ensemble learning** can improve segmentation performance.

---

## 8.4 3D U-Net

`Unet-3D.ipynb` extends the U-Net architecture from individual 2D slices to complete 3D MRI volumes.

The architecture uses:

- 3D convolutions
- Group Normalization
- ReLU activations
- 3D pooling
- Volumetric decoder layers
- U-Net skip connections
<img width="1376" height="768" alt="Schematic_of_3D_U-Net_architecture_20260920202044" src="https://github.com/user-attachments/assets/e1050087-63b0-4928-86ec-06513269ec2a" />

The main motivation is to preserve spatial relationships between adjacent MRI slices that are lost when treating each slice independently.

---

## 8.5 3D ResUNet

`ResUnet-3D.ipynb` introduces residual learning into the 3D U-Net framework.

The architecture incorporates:

- 3D residual convolution blocks
- Group Normalization
- LeakyReLU activation
- Residual skip paths
- 3D encoder-decoder structure
- Volumetric U-Net skip connections

The experiment investigates whether residual feature learning can improve optimization and segmentation quality when working with high-dimensional MRI volumes.

---

## 8.6 3D Attention U-Net

`AttentionUnet-3D.ipynb` investigates attention-based feature selection in a volumetric segmentation setting.

The attention mechanism learns spatial weighting maps so that informative regions can receive greater emphasis during decoding.

The notebook also includes an **Optuna-based hyperparameter search** exploring parameters such as:

- Learning rate
- Batch size
- Number of base channels
- Gradient accumulation
- Number of training epochs
<img width="1376" height="768" alt="Schematic_of_3D_attention_U-Net_20260920202142" src="https://github.com/user-attachments/assets/637144ab-894b-4f48-8c77-f78d3a73f130" />

The recorded notebook contains the architecture and optimization framework, but does not contain a complete final evaluation comparable to the completed 3D U-Net and 3D ResUNet experiments. Therefore, no final benchmark score is reported for this experiment.

---

# 9. Training Methodology

The experiments primarily use PyTorch and GPU acceleration.

The segmentation objective combines Binary Cross-Entropy and Dice loss:

```text
Total Loss = BCE Loss + Dice Loss
```

Dice loss is particularly relevant for medical segmentation because it directly optimizes the overlap between predicted and ground-truth tumor regions.

The notebooks also implement:

- Training/validation loops
- Model checkpointing
- Loss tracking
- Dice tracking
- Model serialization
- Prediction pipelines
- GPU execution
- Runtime measurements

---

# 10. Evaluation Methodology

The models are evaluated using several complementary metrics.

### Dice Coefficient

```text
Dice = 2 × |Prediction ∩ Ground Truth|
       --------------------------------
       |Prediction| + |Ground Truth|
```

### Intersection over Union

```text
IoU = |Prediction ∩ Ground Truth|
      ----------------------------
      |Prediction ∪ Ground Truth|
```

Additional metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Class-specific Dice
- Class-specific Jaccard / IoU

For the 3D experiments, the evaluation considers:

```text
WT — Whole Tumor
TC — Tumor Core
ET — Enhancing Tumor
```

This is important because tumor sub-regions differ substantially in size and difficulty.

---

# 11. Experimental Results

The following results are the metrics recorded in the notebooks.

Because the experiments use different representations, preprocessing choices, training configurations, and evaluation procedures, these results should be interpreted as **experimental observations rather than a single controlled leaderboard**.

## 11.1 Summary

| Experiment | Recorded Result |
|---|---|
| **2D U-Net** | Test IoU: **0.6715** |
| **2D Attention ResUNet** | Test IoU: **0.6715** |
| **2D Attention ResUNet Ensemble** | Mean IoU: **0.7791** |
| **2D Ensemble — tumor channels** | Mean IoU: **0.7071** |
| **3D U-Net** | Accuracy: **99.56%**, Precision: **52.14%**, Recall: **88.88%**, F1: **65.73%** |
| **3D ResUNet** | Accuracy: **99.90%**, Precision: **91.41%**, Recall: **87.40%**, F1: **89.36%** |
| **3D Attention U-Net** | Architecture + training/Optuna experiment; no complete final metric recorded |

---

## 11.2 2D U-Net

The recorded validation Dice improved from approximately:

```text
0.4802 → 0.9868
```

during the 20-epoch experiment.

Validation loss decreased from approximately:

```text
0.4848 → 0.0131
```

The notebook reports:

```text
Mean IoU on test data: 0.6715
```

---

## 11.3 2D Attention ResUNet

The Attention ResUNet experiment reached a recorded maximum validation Dice of approximately:

```text
0.8922
```

The notebook reports:

```text
Mean IoU on test data: 0.6715
```

This experiment was intended to investigate the effect of combining residual connections, attention, and multi-scale feature extraction in a 2D segmentation architecture.

---

## 11.4 2D Ensemble

The four-model ensemble produced the following recorded test IoU values:

| Channel | IoU |
|---|---:|
| Channel 1 | **0.9950** |
| Channel 2 | **0.6983** |
| Channel 3 | **0.6406** |
| Channel 4 | **0.7824** |

Overall recorded results:

```text
Mean IoU:           0.7791
Mean tumor IoU:     0.7071
```

The ensemble experiment provides evidence for further investigation of specialized-model approaches for multi-channel medical segmentation.

---

# 12. 3D U-Net Results

The recorded evaluation produced:

| Metric | Result |
|---|---:|
| Accuracy | **99.56%** |
| Precision | **52.14%** |
| Recall | **88.88%** |
| F1-score | **65.73%** |

The notebook additionally evaluates WT, TC, and ET segmentation using Dice and Jaccard metrics.

Example recorded validation cases:

| Case | WT Dice | WT IoU | TC Dice | TC IoU | ET Dice | ET IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.8592 | 0.7531 | 0.8809 | 0.7872 | 0.5208 | 0.3521 |
| 9 | 0.8211 | 0.6965 | 0.8473 | 0.7350 | 0.5114 | 0.3435 |
| 27 | 0.8946 | 0.8093 | 0.5719 | 0.4005 | 0.4952 | 0.3291 |
| 2 | 0.9040 | 0.8248 | 0.7589 | 0.6114 | 0.6267 | 0.4564 |
| 20 | 0.4164 | 0.2630 | 0.1182 | 0.0628 | 0.0884 | 0.0462 |

These results also illustrate why accuracy alone is not sufficient for evaluating medical image segmentation: the background occupies a large portion of a brain volume, so overlap-based metrics provide more meaningful information about tumor delineation.

---

# 13. 3D ResUNet Results

The recorded evaluation produced:

| Metric | Result |
|---|---:|
| Accuracy | **99.90%** |
| Precision | **91.41%** |
| Recall | **87.40%** |
| F1-score | **89.36%** |

Selected recorded validation results:

| Case | WT Dice | WT IoU | TC Dice | TC IoU | ET Dice | ET IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.9250 | 0.8605 | 0.8743 | 0.7767 | 0.7017 | 0.5405 |
| 9 | 0.8198 | 0.6947 | 0.8730 | 0.7746 | 0.5446 | 0.3742 |
| 27 | 0.9313 | 0.8714 | 0.9173 | 0.8473 | 0.5294 | 0.3600 |
| 2 | 0.9068 | 0.8294 | 0.9257 | 0.8616 | 0.8164 | 0.6898 |
| 20 | 0.9590 | 0.9213 | 0.9488 | 0.9026 | 0.9247 | 0.8600 |

The recorded validation prediction time was approximately:

```text
7.43 seconds
```

within the notebook's evaluation environment.

---

# 14. What the Comparison Suggests

The experiments provide several research observations.

### 2D vs. 3D representation

The 2D models process MRI slices independently, which simplifies computation but does not directly model the spatial relationship between neighboring slices.

The 3D models operate on volumetric data and therefore have access to spatial context across all three dimensions.

The recorded experiments show that the 3D ResUNet achieved substantially stronger precision and F1 results than the recorded 3D U-Net experiment.

### Residual learning

The comparison between the 3D U-Net and 3D ResUNet experiments provides an investigation into the effect of residual connections on volumetric segmentation.

The recorded 3D ResUNet result was:

```text
F1 = 89.36%
Precision = 91.41%
Recall = 87.40%
```

compared with the recorded 3D U-Net result:

```text
F1 = 65.73%
Precision = 52.14%
Recall = 88.88%
```

These results motivate further controlled experiments to isolate the contribution of residual learning.

### Attention and multi-scale features

The Attention ResUNet experiments investigate whether attention mechanisms and multi-scale feature extraction can help the network focus on informative anatomical regions and tumor structures.

### Ensemble learning

The 2D ensemble experiment demonstrates another research direction: using multiple specialized segmentation models instead of relying on a single model to predict every target channel.

---

# 15. Computational Analysis

The 3D U-Net notebook records the following timings:

| Operation | Time |
|---|---:|
| Data loading | 0.206 s |
| Data preprocessing | 0.193 s |
| Stratification | 0.038 s |
| Model training | 3320.32 s |
| Tumor prediction | 7.20 s |
| Tumor visualization | ~7.23 s |
| Total recorded wall time | 3421.46 s |

The notebook records GPU execution on an **NVIDIA Quadro RTX 8000**.

These values are provided as experiment-specific runtime measurements. They depend on hardware, storage, CUDA/PyTorch versions, batch configuration, and notebook state and should not be interpreted as universal benchmarks.

---

# 16. Visualization

The notebooks provide qualitative visualization in addition to numerical evaluation.

Implemented visualizations include:

- MRI modality slices.
- Ground-truth tumor masks.
- Predicted segmentation masks.
- Prediction/ground-truth overlays.
- Training and validation curves.
- Confusion matrices.
- 3D tumor-region scatter visualizations.

The 3D visualizations represent the predicted tumor components in volumetric space, allowing qualitative inspection of tumor localization and shape.

---

# 17. Technologies

The project uses:

- **Python**
- **PyTorch**
- **CUDA**
- **NumPy**
- **Pandas**
- **SciPy**
- **scikit-learn**
- **scikit-image**
- **Nibabel**
- **Nilearn**
- **Albumentations**
- **Matplotlib**
- **Seaborn**
- **Plotly**
- **Optuna**
- **TorchMetrics**
- **tqdm**

---

# 18. Reproducing the Experiments

## Step 1 — Obtain the dataset

Download the BraTS 2020 dataset from the source referenced in the notebooks.

The dataset is not included in this repository.

## Step 2 — Configure the dataset path

Open the desired notebook and update the dataset paths to match the local environment.

## Step 3 — Install dependencies

A starting environment can be installed with:

```bash
pip install numpy pandas scipy scikit-learn scikit-image
pip install nibabel nilearn pydicom
pip install albumentations matplotlib seaborn plotly
pip install optuna torchmetrics tqdm tabulate
```

Install the appropriate PyTorch/CUDA build for the available GPU.

## Step 4 — Run an experiment

Recommended progression:

```text
1. Unet-2D.ipynb
2. AttentionResunet-2D.ipynb
3. Ensemble-AttentionResunet-2D.ipynb
4. Unet-3D.ipynb
5. ResUnet-3D.ipynb
6. AttentionUnet-3D.ipynb
```

Starting with the 2D U-Net provides a baseline before moving to the more computationally demanding architectures.

---

# 19. Reproducibility and Experimental Limitations

This repository represents a research/experimental implementation. Several factors should be considered before drawing conclusions from the reported numbers.

### Different experimental configurations

The notebooks were developed as separate experiments and do not constitute a single perfectly controlled benchmark. Differences exist in:

- Preprocessing
- Data representation
- Training configuration
- Batch size
- Number of epochs
- Evaluation procedure
- Dataset splitting

Therefore, the results should not be interpreted as a definitive ranking of all architectures.

### Test-set construction

The 3D pipeline uses age metadata for stratification and constructs a test partition from subjects with missing age information. This is an experiment-specific design choice and should be replaced with a standardized independent test protocol for a formal benchmark.

### Incomplete Attention U-Net experiment

The 3D Attention U-Net notebook contains the architecture, training framework, and Optuna optimization code, but the saved notebook does not contain a complete final evaluation equivalent to the completed 3D U-Net and 3D ResUNet experiments.

### No clinical validation

The experiments use a public research dataset and are not clinically validated.

---

# 20. Future Research Directions

The project can be extended into a more rigorous research benchmark through:

1. **Unified experimental protocol**  
   Train every architecture using identical splits, preprocessing, augmentation, and optimization settings.

2. **Cross-validation**  
   Evaluate every architecture across multiple folds and report mean ± standard deviation.

3. **Standard BraTS evaluation**  
   Include official WT, TC, and ET Dice and Hausdorff Distance metrics.

4. **Surface-based metrics**  
   Add Hausdorff Distance and Average Symmetric Surface Distance.

5. **Ablation studies**  
   Separately measure the contribution of residual connections, attention, ASPP, and ensemble learning.

6. **Controlled 2D vs. 3D comparison**  
   Keep all variables fixed except dimensionality to quantify the effect of volumetric context.

7. **Attention analysis**  
   Visualize learned attention maps to investigate which anatomical regions influence predictions.

8. **Generalization testing**  
   Evaluate trained models on an independent dataset or external validation cohort.

9. **Experiment tracking**  
   Introduce structured experiment tracking using TensorBoard, Weights & Biases, or similar tools.

10. **Modular implementation**  
    Refactor reusable datasets, models, losses, and evaluation functions from notebooks into Python modules for easier reproducibility.

---

# 21. Research Contribution

The primary contribution of this repository is an **experimental comparison of U-Net-based architectures for brain tumor segmentation**, covering both **slice-based 2D and volumetric 3D approaches**.

The study brings together:

- Baseline U-Net segmentation
- Residual U-Net architectures
- Attention-based feature selection
- Squeeze-and-Excitation mechanisms
- ASPP / multi-scale feature extraction
- 2D ensemble segmentation
- 3D volumetric segmentation
- Quantitative WT/TC/ET evaluation
- Hyperparameter optimization
- Qualitative 2D and 3D visualization

Rather than focusing on a single proposed architecture, the project explores **how different architectural design choices affect medical image segmentation performance** and establishes a foundation for more controlled future experiments.

---

# 22. Conclusion

This project investigates brain tumor segmentation as a comparative deep-learning problem using the BraTS 2020 multi-modal MRI dataset.

The experiments cover a progression from a conventional **2D U-Net baseline** to architectures incorporating **attention, residual learning, multi-scale feature extraction, ensemble learning, and 3D volumetric processing**.

The recorded experiments demonstrate meaningful differences between the investigated approaches, particularly in the 3D experiments. The 3D ResUNet achieved a recorded **89.36% F1-score**, with **91.41% precision** and **87.40% recall**, while the 2D ensemble recorded a **77.91% mean IoU**.

The results should be viewed as exploratory evidence rather than a final architectural ranking. A controlled benchmark with identical data splits, preprocessing, training budgets, and statistical evaluation would be the appropriate next step for drawing stronger comparative conclusions.

---

## Disclaimer

This project is intended for **research and educational purposes**. The models have not been clinically validated and should not be used for diagnosis or medical decision-making. Clinical deployment would require independent validation, expert evaluation, regulatory approval, and appropriate safety testing.

---

## Author

**Usman Khan**

Computer Science / Artificial Intelligence Research

This repository contains the experimental notebooks and implementations used for the comparative study.
