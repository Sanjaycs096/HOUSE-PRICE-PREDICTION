import tensorflow as tf
import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
df = pd.read_csv('dataset/socal2.csv')
image_folder = 'dataset/house_images'

# Image processing
X, y = [], []
for _, row in df.iterrows():
    img_path = os.path.join(image_folder,f"{row['image_id']}.jpg")
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (100, 100))
        X.append(img)
        y.append(row['price'])

X = np.array(X) / 255.0
y = np.array(y)

# Scale y (price)
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Save the scaler
joblib.dump(scaler, 'models/price_scaler.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save('models/image_model.h5')
