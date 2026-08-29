# Adversarial Machine Learning Investigation

## Project Overview

This project investigates how a deep learning image classification model behaves under adversarial attacks and how its predictions can be explained and improved.

The project is divided into five phases. Each phase is implemented as a separate Python class, and `Main.py` is used to run all phases together and pass the required outputs from one phase to the next.

The main model used in the project is a **pretrained ResNet-34** trained and fine-tuned on the **Caltech101** dataset.

---

## Project Structure

```text
Project/
│
├── Phase_One.py
├── Phase_Two.py
├── Phase_Three.py
├── Phase_Four.py
├── Phase_Five.py
├── Main.py
└── data/
```

Each phase has its own class, while `Main.py` acts as the main controller for the complete project.

---

# Phase 1 – Model Training

The first phase is responsible for preparing the dataset and building the original classification model.

### Dataset and Preprocessing

The Caltech101 dataset is downloaded using `torchvision`. The images are:

* Resized to `224 × 224`
* Converted to 3-channel grayscale
* Converted to PyTorch tensors

The dataset is divided into:

* 70% Training
* 15% Validation
* 15% Testing

A fixed random seed is used for the split to make the experiments reproducible.

### Model

A pretrained **ResNet-34** is used as the base model.

Most of the pretrained layers are frozen, while the last convolutional block (`layer4`) is unfrozen for fine-tuning.

The original fully connected layer is replaced with a new layer whose output size is equal to the number of Caltech101 classes.

The model is trained using:

* Cross Entropy Loss
* Adam Optimizer
* Learning rate = `0.0001`
* 5 epochs

After training, the model is evaluated on the test set and the trained weights are saved.

The `run()` function returns the trained model, dataset, data loaders, criterion, device, and test accuracy so they can be used by the following phases.

---

# Phase 2 – Adversarial Attack

The second phase tests whether the trained model can be fooled by modifying the input images.

The implemented attack is **Fast Gradient Sign Method (FGSM)**.

FGSM uses the gradient of the loss with respect to the input image to create a small perturbation:

```text
x_adv = x + ε × sign(∇x J(θ, x, y))
```

Different epsilon values are tested to see how the attack strength affects the model.

The project uses:

```text
0.001
0.005
0.01
0.02
```

For each epsilon, adversarial images are generated and the model accuracy is calculated.

The epsilon that produces the lowest adversarial accuracy is selected as the strongest attack and is passed to the later phases.

---

# Phase 3 – Model Explainability

The third phase is used to understand what the model is focusing on when making its predictions.

Two XAI techniques are implemented:

### 1. Saliency Maps

The gradient of the predicted class with respect to the input image is calculated.

This helps identify the pixels that have the strongest influence on the model's prediction.

### 2. Grad-CAM

Grad-CAM uses the gradients of the target class with respect to feature maps from the convolutional layers.

The resulting heatmap is used to show the important regions of the image that contributed to the prediction.

Both techniques are used to visually investigate the model's decisions.

---

# Phase 4 – Forensic Analysis

The fourth phase combines the adversarial attack and explainability methods.

For selected test images, the analysis compares:

1. The original clean image.
2. The model's prediction on the clean image.
3. The XAI visualization for the clean prediction.
4. The adversarial version of the image.
5. The new prediction after the attack.
6. The adversarial perturbation/noise.
7. The XAI visualization for the incorrect prediction.

This allows us to investigate not only **that** the model was fooled, but also **how its attention changed after the attack**.

The strongest epsilon found in Phase 2 is used for this analysis.

---

# Phase 5 – Defense

The final phase focuses on making the model more resistant to adversarial attacks.

The selected defense is **Adversarial Training**.

A copy of the original model is created and trained using both clean and adversarial examples.

During training, FGSM examples are generated using the selected epsilon. The model is then trained to correctly classify the adversarial examples.

After training, the hardened model is evaluated on two versions of the test set:

* Clean test images
* Adversarial test images

The results are compared with the original model.

The final comparison includes:

```text
Original Clean Accuracy
Original Adversarial Accuracy
Hardened Clean Accuracy
Hardened Adversarial Accuracy
Adversarial Accuracy Improvement
Clean Accuracy Change
```

The hardened model is also saved after training.

---

# Main Program

`Main.py` connects all five phases together.

The general workflow is:

```text
Phase 1
   ↓
Train and evaluate original model
   ↓
Phase 2
   ↓
Generate FGSM attacks and select strongest epsilon
   ↓
Phase 3
   ↓
Generate Saliency Maps and Grad-CAM
   ↓
Phase 4
   ↓
Perform forensic analysis
   ↓
Phase 5
   ↓
Adversarial training and final comparison
```

The important part is that the model and other required objects from Phase 1 are passed to the following phases instead of creating a completely new model for every phase.

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* ResNet-34
* Caltech101
* FGSM
* Saliency Maps
* Grad-CAM
* Adversarial Training
* Matplotlib
* NumPy

---

## Final Goal

The main goal of the project is to understand the vulnerability of a deep learning model to adversarial examples, visualize the reasons behind its predictions, perform a forensic analysis of its failures, and finally evaluate whether adversarial training can improve its robustness.
