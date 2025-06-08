
# CNN Image Classification: Flowers Dataset

This project focuses on classifying flower species using Convolutional Neural Networks (CNNs). Two key approaches were used: training a custom VGG-style CNN from scratch, and fine-tuning a pretrained ResNet50. The results of both were compared using various performance metrics.

---

## Dataset

- **Source**: [Kaggle - Flowers Multiclass Dataset](https://www.kaggle.com/datasets/alsaniipe/flowers-multiclass-datasets)
- **Structure**:
    ```
    flower_photos/
        train/
        validation/
        test/
    ```
- The dataset was preprocessed (resized to 224x224 and normalized), and augmented during training to improve generalization.

---

## Image Preprocessing

- **Resize**: 224x224 pixels
- **Normalization**: Pixel values scaled to [0, 1]
- **Label Formats**:
  - VGG: One-hot encoding
  - ResNet: One-hot encoding

---

## Data Augmentation

Applied only on the training set using:
- Random rotation (±20°)
- Width/height shift (10%)
- Shear and zoom (10%)
- Brightness variation
- Horizontal flips

---

## Model 1: VGG-Style CNN

- Built from scratch using stacked Conv2D, MaxPooling, Flatten, and Dense layers.
- Trained for 10 epochs on the augmented training data.
- Optimizer: Adam (LR=1e-4)
- Loss: Categorical Crossentropy

---

## Model 2: Fine-Tuned ResNet50

1. **Stage 1**: Freeze base layers, train only new classifier (5–10 epochs)
2. **Stage 2**: Unfreeze last ResNet block (e.g., layer4), train it with classifier (5–10 epochs)
3. **Stage 3**: Unfreeze all layers, fine-tune entire model (5–10 epochs)

---

## Result Comparison

- **Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC Curve and AUC score

- **Tools**:
  - `scikit-learn`: Metrics and confusion matrix
  - `matplotlib`, `seaborn`: Visualization

---

## Dependencies

```bash
tensorflow
matplotlib
seaborn
scikit-learn
```

---