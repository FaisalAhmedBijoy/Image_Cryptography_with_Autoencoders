import cv2
import numpy as np


def visualization_NPCR_comparision(gray1, gray2,output_diff_image_1_path,output_diff_image_2_path):
    diff = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_HOT)
    result1 = cv2.addWeighted(image1, 0.7, heatmap, 0.3, 0)
    result2 = cv2.addWeighted(image2, 0.7, heatmap, 0.3, 0)
    cv2.imwrite(output_diff_image_1_path,result1)
    cv2.imwrite(output_diff_image_2_path,result2)
    cv2.imshow("Image 1", result1)
    cv2.imshow("Image 2", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def NPCR_comparision(image1, image2,output_diff_image_1_path,output_diff_image_2_path):
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray 1", gray1)
    cv2.imshow("Gray 2", gray2)

    num_pixels = gray1.shape[0] * gray1.shape[1]
    npcr = np.sum(gray1 != gray2) / num_pixels * 100
    # print("NPCR score: {}%".format(npcr))
    visualization_NPCR_comparision(gray1, gray2,output_diff_image_1_path,output_diff_image_2_path)

    return npcr

if __name__ == '__main__':
 

    image_1_path="images/encrypted_decrypted_images/faisal_decrypted_image.png"
    image_2_path="images/input_samples/faisal.jpg"

    # image_1_path="images/shuffled_deshuffled_image/Lena_deshuffled_image.png"
    # image_2_path="images/input_samples/lena1.tif"

    output_diff_image_1_path="images/NPCR_images/NPCR_difference_1_faisal.png"
    output_diff_image_2_path="images/NPCR_images/NPCR_difference_2_faisal.png"

    image1 = cv2.imread(image_1_path)
    image2 = cv2.imread(image_2_path)
    npcr_score=NPCR_comparision(image1, image2,output_diff_image_1_path,output_diff_image_2_path)
    print("NPCR score: ",npcr_score)