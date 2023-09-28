import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping  # 添加 Early Stopping 的導入

# 載入數據集
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 只保留狗的圖片 (label為5)
train_dogs = train_images[train_labels[:,0] == 5]
test_dogs = test_images[test_labels[:,0] == 5]

# 創建非狗的數據集
train_not_dogs = train_images[train_labels[:,0] != 5][:len(train_dogs)]
test_not_dogs = test_images[test_labels[:,0] != 5][:len(test_dogs)]

# 合併數據集並標記
train_images = np.concatenate([train_dogs, train_not_dogs])
train_labels = np.array([1] * len(train_dogs) + [0] * len(train_not_dogs))

test_images = np.concatenate([test_dogs, test_not_dogs])
test_labels = np.array([1] * len(test_dogs) + [0] * len(test_not_dogs))

# 正規化圖片
train_images = train_images / 255.0
test_images = test_images / 255.0

# 創建模型
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 創建 Early Stopping 回調函數
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 訓練模型，添加 Early Stopping 回調函數
#model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels), callbacks=[early_stopping])

# 訓練模型
model.fit(train_images, train_labels, epochs=11000, validation_data=(test_images, test_labels))


# 儲存模型
model.save("dog_detector_model.keras2")





