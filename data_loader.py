import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(dataset_dir):
   # Load and preprocess the dataset
    images = []
    input_shape=(256,256,1)
    image_paths = [os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir)]

    for image_path in image_paths:
        img = load_img(image_path, color_mode='grayscale', target_size=input_shape[:2])
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        images.append(img_array)

    images = np.array(images)

    # Split dataset into training and validation sets
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    return train_images,val_images

if __name__=='__main__':
    dataset_dir='datasets/test'
    train_images,val_images=load_data(dataset_dir)
    print('Training shape: ', train_images.shape)
    print('Validation shape: ', val_images.shape)