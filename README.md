# Trustworthy AI Exercises

Small, self-contained exercises on adversarial attacks, DNN testing, and formal verification, mostly written for the *Trustworthy AI* graduate course at the University of Seoul (Spring 2025).

Each folder was originally its own repository; the commit history of every one is preserved here.

## Contents

### Adversarial attacks

| Folder | Attack | Dataset | Model |
|---|---|---|---|
| [attacks/fgsm-mnist-untargeted](attacks/fgsm-mnist-untargeted) | FGSM, untargeted | MNIST | CNN |
| [attacks/fgsm-mnist-targeted](attacks/fgsm-mnist-targeted) | FGSM, targeted | MNIST | CNN |
| [attacks/pgd-mnist-untargeted](attacks/pgd-mnist-untargeted) | PGD, untargeted | MNIST | CNN |
| [attacks/pgd-mnist-targeted](attacks/pgd-mnist-targeted) | PGD, targeted | MNIST | CNN |
| [attacks/fgsm-cifar10-untargeted](attacks/fgsm-cifar10-untargeted) | FGSM, untargeted | CIFAR-10 | ConvNeXt-Tiny |
| [attacks/fgsm-cifar10-targeted](attacks/fgsm-cifar10-targeted) | FGSM, targeted | CIFAR-10 | ConvNeXt-Tiny |
| [attacks/pgd-cifar10-untargeted](attacks/pgd-cifar10-untargeted) | PGD, untargeted | CIFAR-10 | ConvNeXt-Tiny |
| [attacks/pgd-cifar10-targeted](attacks/pgd-cifar10-targeted) | PGD, targeted | CIFAR-10 | ConvNeXt-Tiny |

### Testing

| Folder | What it does |
|---|---|
| [testing/deepxplore-cifar10](testing/deepxplore-cifar10) | Simplified DeepXplore: generates inputs on which two ResNet50 models disagree |

### Verification

| Folder | What it does |
|---|---|
| [verification/marabou-fashion-mnist](verification/marabou-fashion-mnist) | Trains a Fashion-MNIST model, exports it to ONNX, and checks local robustness with Marabou |

## Usage

Every folder has its own `README.md` and `requirements.txt`. Install and run from inside the folder you want:

```bash
cd attacks/fgsm-mnist-untargeted
pip install -r requirements.txt
python test.py
```

## Related

- [project_pigeon](https://github.com/jhkang-rsrch/project_pigeon): course project on neuron activation coverage, which later became K-SNAC (ASE 2025 Workshop).
