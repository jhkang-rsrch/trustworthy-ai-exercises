from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# 1. 데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. 모델 정의
base_model = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax')
])

# 3. 컴파일
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. 학습
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 5. 저장
model.save("Model1.h5")
