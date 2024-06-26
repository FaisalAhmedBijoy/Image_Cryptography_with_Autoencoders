import cv2
import numpy as np

def UACI_comparision(image1, image2,saved_UACI_image_path):
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    num_pixels = gray1.shape[0] * gray1.shape[1]
    uaci = np.sum(np.abs(gray1.astype("float") - gray2.astype("float"))) / (num_pixels * 255)
    # print("UACI score: {}".format(uaci))

    diff = cv2.absdiff(gray1, gray2)
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(saved_UACI_image_path,diff_normalized)

    cv2.imshow("Gray 1", gray1)
    cv2.imshow("Gray 2", gray2)
    cv2.imshow("Difference Image", diff_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return uaci

if __name__ == '__main__':

    image_1_path="images/encrypted_decrypted_images/faisal_decrypted_image.png"
    image_2_path="images/input_samples/faisal.jpg"
    saved_UACI_image_path="images/UACI_images/UACI_difference_faisal.png"

    # image_1_path="images/shuffled_deshuffled_image/Lena_deshuffled_image.png"
    # image_2_path="images/input_samples/lena1.tif"

    image1 = cv2.imread(image_1_path)
    image2 = cv2.imread(image_2_path)
    uaci_score=UACI_comparision(image1, image2,saved_UACI_image_path)
    print("UACI score: ",uaci_score)