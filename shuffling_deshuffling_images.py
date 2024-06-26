import cv2
import numpy as np
from generate_chaotic_map_sequence import logistic_map

def shuffle_image(image, S):
    """
    Shuffle the input image using a chaotic map generated in step 4.1.
    """
    N = image.shape[0]
    shuffling_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            shuffling_matrix[i][j] = int(S[i] * N)
    shuffled_image = np.zeros_like(image)
    for i in range(N):
        row = image[i]
        max_row = np.max(row)
        max_col = np.max(shuffling_matrix[i])
        if max_row > max_col:
            shifted_row = np.roll(row, int(-max_col))
        else:
            shifted_row = np.roll(row, int(max_col))
        shuffled_image[i] = shifted_row
    return shuffled_image

def deshuffle_image(image, S):
    """
    Deshuffle the input image using a chaotic map generated in step 4.1.
    """
    N = image.shape[0]
    shuffling_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            shuffling_matrix[i][j] = int(S[i] * N)
    deshuffled_image = np.zeros_like(image)
    for i in range(N):
        row = image[i]
        max_row = np.max(row)
        max_col = np.max(shuffling_matrix[i])
        if max_row > max_col:
            shifted_row = np.roll(row, int(max_col))
        else:
            shifted_row = np.roll(row, int(-max_col))
        deshuffled_image[i] = shifted_row
    return deshuffled_image

def image_shuffling_deshuffling_generation(input_image_path, output_shuffle_image_path, output_deshuffle_image_path):
    
    # read input image
    image = cv2.imread(input_image_path, 0)
    print('image shape: ', image.shape)
    
    # generate chaotic sequence
    S = logistic_map(x0=0.5, l=3.9, N=image.shape[0] * image.shape[1])
    print('chaotic sequence shape: ', len(S))

    # shuffle the image
    shuffled_image = shuffle_image(image, S)
    cv2.imshow('input image', image)
    cv2.imshow('shuffled_image', shuffled_image)
    cv2.imwrite(output_shuffle_image_path, shuffled_image)

    # deshuffle the image
    deshuffled_image = deshuffle_image(shuffled_image, S)
    cv2.imshow('deshuffled_image', deshuffled_image)
    cv2.imwrite(output_deshuffle_image_path, deshuffled_image)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__=='__main__':
   
    input_image_path = 'images/input_samples/faisal_2.jpg'
    output_shuffle_image_path = 'images/shuffled_deshuffled_image/faisal_2_shuffled_image.png'
    output_deshuffle_image_path = 'images/shuffled_deshuffled_image/faisal_2_deshuffled_image.png'
    image_shuffling_deshuffling_generation(input_image_path, output_shuffle_image_path, output_deshuffle_image_path)