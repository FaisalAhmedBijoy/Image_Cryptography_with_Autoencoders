import matplotlib.pyplot as plt
from data_loader import load_data
from tensorflow.keras.models import load_model
if __name__ == '__main__':

# Use encoder to compress input image
    dataset_dir='datasets/test'
    saved_model_path='logs/autoencoder_model.h5'
    train_images,val_images=load_data(dataset_dir)
    encoder = load_model(saved_model_path, 
                         compile=False)
    input_image = val_images[0]
    compressed_image = encoder.predict(input_image.reshape(1, 256, 256, 1))
    # Use decoder to reconstruct original image from compressed representation
    decoder = load_model('logs/autoencoder_model.h5', compile=False)
    reconstructed_image = decoder.predict(compressed_image)
    # Visualize original and reconstructed images
    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(input_image.reshape(256, 256), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Reconstructed image')
    plt.imshow(reconstructed_image.reshape(256, 256), cmap='gray')
    plt.show()