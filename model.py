import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. LOAD DATASET

df = pd.read_csv("Labels.csv")

label_map = {
    "GON+": 1,
    "GON-": 0
}

df["label_numeric"] = df["Label"].map(label_map)

images = []
labels = []

for index, row in df.iterrows():

    image_path = os.path.join("Images", row["Image Name"])

    img = Image.open(image_path)
    img = img.resize((224,224))

    img_array = np.array(img) / 255.0

    images.append(img_array)
    labels.append(row["label_numeric"])

X = np.array(images)
y = np.array(labels)

print("Dataset loaded")
print("Images shape:", X.shape)

# 2. SPLIT DATASET


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

# 3. BUILD CNN MODEL

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1,activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. TRAIN MODEL

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# 5. EVALUATE MODEL

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)

# 6. SAVE MODEL

model.save("glaucoma_model.h5")

print("Model saved successfully!")