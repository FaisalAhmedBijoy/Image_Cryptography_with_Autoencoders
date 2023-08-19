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
    # print(shuffled_image)
    for i in range(N):
        row = image[i]
        max_row = np.max(row)
        max_col = np.max(shuffling_matrix[i])
        # print('max row: ',max_row, ' max_col' ,max_col)
        
        if max_row > max_col:
            shifted_row = np.roll(row, int(max_col))
        else:
            shifted_row = np.roll(row, -int(max_col))
        shuffled_image[i] = shifted_row
    return shuffled_image

def shuffling_image_generation(input_image_path, output_image_path):
    
    # image = np.random.rand(28, 28) # example input image
    image = cv2.imread(input_image_path,0)
    print('image shape: ',image.shape)
    S = logistic_map(x0=0.5, l=3.9, N=256) # example sequence S generated using the chaotic map
    print('chaotic sequence shape: ',len(S))

    shuffled_image = shuffle_image(image, S)
    cv2.imshow('input image',image)
    cv2.imshow('shuffled_image',shuffled_image)
    cv2.imwrite(output_image_path,shuffled_image)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__=='__main__':
   
    input_image_path='images/input_samples/lena1.tif'
    output_image_path='images/shuffled_image/Lena_shuffled_image.png'
    shuffling_image_generation(input_image_path, output_image_path)
    
    
