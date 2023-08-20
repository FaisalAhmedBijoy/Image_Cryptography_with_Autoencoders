import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Define constants
input_shape = (256, 256, 1)
batch_size = 32
epochs = 5
data_dir = 'datasets/test'  # Update with your dataset directory

# Load and preprocess the dataset
image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
images = []

for image_path in image_paths:
    img = load_img(image_path, color_mode='grayscale', target_size=input_shape[:2])
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    images.append(img_array)

images = np.array(images)

# Split dataset into training and validation sets
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
print('Training shape: ', train_images.shape)
print('Validation shape: ', val_images.shape)


# Define input shape
input_img = Input(shape=input_shape)


# Define encoder layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((4, 4), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((4, 4), padding='same')(x)
encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

# Define decoder layers
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((4, 4))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((4, 4))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Define autoencoder model
autoencoder = Model(input_img, decoded, name='autoencoder')
print(autoencoder.summary())

# ... Rest of the architecture ...

# Define autoencoder model
autoencoder = Model(input_img, decoded, name='autoencoder')
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
history=autoencoder.fit(train_images, 
                        train_images, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(val_images, val_images))

# Save the trained model
autoencoder.save('autoencoder_model.h5')

print("Training complete.")
