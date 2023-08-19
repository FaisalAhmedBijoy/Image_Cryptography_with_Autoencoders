import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Define the input shape
input_shape = (256, 256, 3)  # Assuming RGB images

# Encoder
input_img = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((8, 8),padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((4, 4),padding='same')(x)
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2))(x)  # Changed to (4, 4) for 10x10 encoding

# Decoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((4, 4))(x)  # Changed to (4, 4) for upsampling
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((4, 4))(x)  # Changed to (4, 4) for upsampling
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Autoencoder model
autoencoder = Model(input_img, decoded)
print(autoencoder.summary())
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Load your image data and preprocess it (assuming you have the images in a directory)
# images = load_and_preprocess_images("path_to_images")

# Split the images into training and validation sets
# train_images, val_images = split_train_val(images)

# Train the autoencoder
# autoencoder.fit(train_images, train_images, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_images))

# Once trained, you can use the autoencoder to encode and decode images
# encoded_images = autoencoder.predict(images)

# You can also visualize the original, encoded, and decoded images to see the results
# visualize_results(images, encoded_images)
