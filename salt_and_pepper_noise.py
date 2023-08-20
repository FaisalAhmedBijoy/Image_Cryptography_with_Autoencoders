import cv2
import numpy as np

def add_salt_and_pepper_noise(image, amount):
    noisy_image = np.copy(image)
    height, width = noisy_image.shape[:2]
    
    # Calculate the number of pixels to add noise to
    num_pixels = int(amount * height * width)
    
    # Generate random coordinates to add noise to
    coords = [np.random.randint(0, height, num_pixels),
              np.random.randint(0, width, num_pixels)]
    
    # Set random pixels to salt or pepper values
    noisy_image[coords[0], coords[1]] = [255, 255, 255]  # Salt (white)
    noisy_image[coords[0], coords[1]] = 0  # Pepper (black)
    
    return noisy_image


if __name__ == '__main__':
    # Load the image
    image_path = 'images/input_samples/lena2.tif'
    output_noisy_image_path='images/noisy_images/lena_noisy.png'
    image = cv2.imread(image_path)

    # Add salt and pepper noise with a noise amount of 0.02 (2% of pixels)
    noisy_image = add_salt_and_pepper_noise(image, amount=0.02)
    cv2.imwrite(output_noisy_image_path,noisy_image)

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
