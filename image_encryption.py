import cv2
from generate_chaotic_map_sequence import logistic_map

def image_encrption_chaotic_sequence(input_image_path,chaotic_map_sequence):
    """
    pressed image is encrypted by applying bitxor operation. It is defined as follows:
    E = bitxor(C, S)
    where C indicates compressed image, S indicates chaotic
    sequence generated in step 4.1 and E indicates the final
    encrypted image, which is then transmitted.
    """
    compressed_image=cv2.imread(input_image_path,0)
    compressed_image=cv2.resize(compressed_image,(10,10))
    compressed_image=compressed_image.flatten()


    print('C shape: ',compressed_image.shape)
    # convert image to byte array
    image_bytes = bytearray(compressed_image)
    print('Image bytes: ',image_bytes)


    print('chaotic sequence shape: ',len(chaotic_map_sequence))

    E=cv2.bitwise_xor(image_bytes,chaotic_map_sequence)
    cv2.imshow('input image',compressed_image)
    cv2.imshow('encrypted image',E)
    cv2.waitKey()
    cv2.destroyAllWindows()
if __name__=='__main__':
   
    input_image_path='images/input_samples/lena1.tif'
    x0 = 0.5
    l = 3.5
    N = 100 # size of flattened input image
    chaotic_map_sequence=logistic_map(x0, l, N)
    print('chaotic map sequence: ',len(chaotic_map_sequence))
    image_encrption_chaotic_sequence(input_image_path, chaotic_map_sequence)