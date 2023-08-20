import cv2
from skimage.metrics import structural_similarity as ssim

def visualize_ssim_differences(imageA, imageB, diff):
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    cv2.imshow("Image 1", imageA)
    cv2.imshow("Image 2", imageB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def  compare_structural_similarity(imageA, imageB):
    # print('imaage A shape: ',imageA.shape)
    # print('image B shape: ',imageB.shape)
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (score, diff) = ssim(grayA, grayB, full=True)
    print("SSIM score: {}".format(score))

    visualize_ssim_differences(imageA, imageB, diff)
    return score,diff

if __name__ == '__main__':

    # imageA = cv2.imread("images/shuffled_deshuffled_image/Lena_deshuffled_image.png")
    # imageB = cv2.imread("images/input_samples/lena1.tif")
    
    imageA = cv2.imread("images/encrypted_decrypted_images/Lena_decrypted_image.png")
    imageB = cv2.imread("images/input_samples/lena2.tif")
    score,diff=compare_structural_similarity(imageA, imageB)
    
    