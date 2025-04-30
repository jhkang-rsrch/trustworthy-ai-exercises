import argparse
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from imageio import imwrite
import os

# Argument parsing
parser = argparse.ArgumentParser(description='DeepXplore-style input generation for CIFAR-10')
parser.add_argument('--weight_diff', type=float, default=0.5)
parser.add_argument('--step', type=float, default=0.01)
parser.add_argument('--seeds', type=int, default=10)
parser.add_argument('--grad_iterations', type=int, default=20)
parser.add_argument('--save_dir', type=str, default='generated_inputs')
args = parser.parse_args()

# CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32') / 255.
y_test = to_categorical(y_test, 10)

input_shape = (32, 32, 3)
input_tensor = Input(shape=input_shape)

# Load models
model1 = load_model("Model1.h5")
model2 = load_model("Model2.h5")

# Create loss based on difference
orig_input = K.placeholder(shape=(1, 32, 32, 3))
output1 = model1(orig_input)
output2 = model2(orig_input)

loss = args.weight_diff * K.mean(K.square(output1 - output2))
grads = K.gradients(loss, orig_input)[0]
grads = K.l2_normalize(grads)
iterate = K.function([orig_input], [loss, grads, output1, output2])

# Output directory
os.makedirs(args.save_dir, exist_ok=True)

print("\nGenerating disagreement-inducing inputs...")
for i in range(args.seeds):
    seed = np.expand_dims(x_test[i], axis=0)
    modified = np.copy(seed)

    for j in range(args.grad_iterations):
        loss_value, grads_value, pred1, pred2 = iterate([modified])
        modified += grads_value * args.step
        pred1_label = np.argmax(pred1)
        pred2_label = np.argmax(pred2)

        if pred1_label != pred2_label:
            img_uint8 = (modified[0] * 255).astype(np.uint8)
            filename = f"{args.save_dir}/diff_{i}_m1_{pred1_label}_m2_{pred2_label}.png"
            imwrite(filename, img_uint8)
            print(f"[{i}] Disagreement found: model1={pred1_label}, model2={pred2_label} → saved {filename}")
            break
    else:
        print(f"[{i}] No disagreement found")
