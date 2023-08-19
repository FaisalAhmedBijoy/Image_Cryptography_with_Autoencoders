import cv2
import numpy as np
import keygen as kg
from generate_chaotic_map_sequence import logistic_map

def chaotic_map_sequece_generation(image_path,x0=0.5, l=3.75):
    image =cv2.imread(image_path,0)
    print('input image shape: ',image.shape)
    # generate chaotic keys
    height=image.shape[0]
    width=image.shape[1]
    N = height*width # size of flattened input image
    chaotic_map_sequence=logistic_map(x0, l, N)
    print('chaotic map sequence: ',len(chaotic_map_sequence))
    return image, chaotic_map_sequence

def image_encryption_generation(image,chaotic_map_sequence,image_encrypted_path):
    
    # image encryption with chaotic sequence
    height = image.shape[0]
    width = image.shape[1]
    encrypted_image=np.zeros(shape=[height,width,3],dtype=np.uint8)
    print('encrypting image: ',encrypted_image.shape)
    z=0
    for i in range(height):
        for j in range(width):
            encrypted_image[i][j]=image[i,j] ** chaotic_map_sequence[z]
            z=z+1
    cv2.imwrite(image_encrypted_path,encrypted_image)
    return encrypted_image

def image_decrypted_image(encrypted_image,chaotic_map_sequence,image_decrypted_path):
    # decrpy the image
    height = encrypted_image.shape[0]
    width = encrypted_image.shape[1]
    decrypted_image=np.zeros(shape=[height,width,3],dtype=np.uint8)
    z=0
    for i in range(height):
        for j in range(width):
            decrypted_image[i][j]= encrypted_image[i,j] ** (1 / chaotic_map_sequence[z])
            z=z+1
    cv2.imwrite(image_decrypted_path,decrypted_image)
    return decrypted_image

if __name__ == '__main__':
    x0 = 0.5
    l = 3.75
    image_path='images/input_samples/lena2.tif'
    image_encrypted_path='images/encrypted_decrypted_images/Lena_encrypted_image.png'
    image_decrypted_path='images/encrypted_decrypted_images/Lena_decrypted_image.png'
    image,chaotic_map_sequence=chaotic_map_sequece_generation(image_path,x0, l)
    encrypted_image=image_encryption_generation(image,chaotic_map_sequence,image_encrypted_path)
    decrypted_image=image_decrypted_image(encrypted_image,chaotic_map_sequence,image_decrypted_path)

    cv2.imshow('input image',image)
    cv2.imshow('encrypted image',encrypted_image)
    cv2.imshow('decrypted image',decrypted_image)
    cv2.waitKey()
    cv2.destroyAllWindows()