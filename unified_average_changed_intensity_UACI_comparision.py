import cv2
import numpy as np
image1 = cv2.imread("images/encrypted_decrypted_images/Lena_decrypted_image.png")
image2 = cv2.imread("images/input_samples/lena2.tif")

# image1 = cv2.imread("images/shuffled_deshuffled_image/Lena_deshuffled_image.png")
# image2 = cv2.imread("images/input_samples/lena1.tif")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
num_pixels = gray1.shape[0] * gray1.shape[1]
uaci = np.sum(np.abs(gray1.astype("float") - gray2.astype("float"))) / (num_pixels * 255)
print("UACI score: {}".format(uaci))

diff = cv2.absdiff(gray1, gray2)
diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow("Gray 1", gray1)
cv2.imshow("Gray 2", gray2)
cv2.imshow("Difference Image", diff_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()