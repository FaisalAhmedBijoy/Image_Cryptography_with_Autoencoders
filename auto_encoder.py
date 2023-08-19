from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Define input shape
input_img = Input(shape=(256, 256, 1))

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
# Compile model
# autoencoder.compile(optimizer='sgd', loss='mse')

# # Train model
# autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# # Use encoder to compress input image
# compressed_image = encoder.predict(input_image)

# # Use decoder to reconstruct original image from compressed representation
# reconstructed_image = decoder.predict(compressed_image)