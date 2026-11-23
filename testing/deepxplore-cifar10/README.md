# DeepXplore-inspired Testing on CIFAR-10

This project implements a simplified version of [DeepXplore](https://github.com/peikexin9/deepxplore) using the CIFAR-10 dataset and two ResNet50 models trained under different configurations. The goal is to generate inputs that cause the models to disagree in their predictions.

---

## 📁 Directory Structure

```
CIFAR10/
├── train_model1.py          # Train ResNet50 with Adam → Model1.h5
├── train_model2.py          # Train ResNet50 with SGD  → Model2.h5
├── gen_diff.py              # Generate disagreement-inducing inputs
├── Model1.h5                # Saved model 1
├── Model2.h5                # Saved model 2
├── generated_inputs/        # Output disagreement images
```

---

## 🧪 How to Use

### 1️⃣ Train the two models:
```bash
python train_model1.py
python train_model2.py
```

### 2️⃣ Generate inputs that induce prediction disagreements:
```bash
python gen_diff.py --seeds 20 --grad_iterations 30 --step 0.01 --weight_diff 0.5
```
→ Images will be saved in `generated_inputs/`

---

## 🧠 Implementation Notes
- Uses ResNet50 models trained from scratch on CIFAR-10
- Inputs are optimized using gradient ascent to maximize disagreement between the two models’ outputs
- This approach follows the spirit of DeepXplore: testing DNNs via differential behavior

---

## 💻 Environment
- Python 3.10
- TensorFlow 2.10.0 (GPU)
- CUDA 11.2 + cuDNN 8.1
- GPU: NVIDIA GeForce RTX 4090
- OS: Windows 11

---

## 🔗 Reference
This project is based on the original DeepXplore:
> https://github.com/peikexin9/deepxplore
