   # Compile model
    # autoencoder.compile(optimizer='sgd', loss='mse')

    # # Train model
    # autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    # # Use encoder to compress input image
    # compressed_image = encoder.predict(input_image)

    # # Use decoder to reconstruct original image from compressed representation
    # reconstructed_image = decoder.predict(compressed_image)