import matplotlib.pyplot as plt
from data_loader import load_data
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model, Model

if __name__ == '__main__':
    # Use encoder to compress input image
    dataset_dir = 'datasets/test'
    saved_model_path = 'logs/autoencoder_model.h5'
    saved_original_image_path = 'images/model_architecture_and_performances/original_image.png'
    saved_compressed_encoded_image_path = 'images/model_architecture_and_performances/compressed_encoded_image.png'
    saved_decompressed_decoded_image_path = 'images/model_architecture_and_performances/decompressed_decoded_image.png'
    train_images, val_images = load_data(dataset_dir)

    input_image = val_images[0]
    autoencoder = load_model(saved_model_path, compile=False)
    # print(autoencoder.summary())

    # Use the encoder sub-model to encode the input data
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder_output').output)
    encoded_data = encoder.predict(input_image.reshape(1, 256, 256, 1))
    print('encoded data shape: ', encoded_data.shape)

    # Use the decoder sub-model to decode the encoded data
    decoder_input = Input(shape=(16, 16, 128))
    decoder_layers = autoencoder.layers[6:]
    decoder_output = decoder_input
    print(decoder_layers)
    for layer in decoder_layers:
        decoder_output = layer(decoder_output)
    decoder = Model(inputs=decoder_input, outputs=decoder_output)
    decoded_data = decoder.predict(encoded_data)
    print('decoded data shape: ', decoded_data.shape)

    # Visualize original and reconstructed images

    plt.title('original image: 256x256')
    plt.imshow(input_image.reshape(256, 256), cmap='gray')
    plt.savefig(saved_original_image_path, dpi=500)



    plt.title('encoded compressed image: 16x16')
    plt.imshow(encoded_data.reshape(16,16,128)[:,:,0], cmap='gray')
    plt.savefig(saved_compressed_encoded_image_path, dpi=500)

  
    plt.title('decoded reconstructed image: 256x256')
    plt.imshow(decoded_data.reshape(256, 256), cmap='gray')
    plt.savefig(saved_decompressed_decoded_image_path, dpi=500)

    # plt.show()