from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D


def autoencoder_architecture(input_image_shape,saved_architecture_path):

    # Define input shape
    input_img = Input(input_image_shape)

    # Define encoder layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((4, 4), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((4, 4), padding='same')(x)
    encoded = Conv2D(128, (3, 3), activation='relu', padding='same',name='encoder_output')(x)

    # Define decoder layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same',name='decoder_input')(encoded)
    x = UpSampling2D((4, 4))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((4, 4))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Define autoencoder model
    autoencoder = Model(input_img, decoded, name='autoencoder')
    plot_model(autoencoder, to_file=saved_architecture_path, show_shapes=True)
    print(autoencoder.summary())

    # Use the encoder to compress the input data
    # encoder = Model(input_img, encoded, name='encoder')
    # # encoded_imgs = encoder.predict(x_test)

    # # Use the decoder to reconstruct the input data
    # decoder = Model(encoder.output, decoded, name='decoder')
    # # decoded_imgs = decoder.predict(encoded_imgs)
    return autoencoder
 

if __name__=='__main__':
    input_image_shape=(256,256,1)
    saved_architecture_path="images/model_architecture_and_performances/autoencoder_architecture.png"
    auto_encoder=autoencoder_architecture(input_image_shape,saved_architecture_path)
    