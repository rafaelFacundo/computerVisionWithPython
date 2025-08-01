import cv2

image = cv2.imread("./1966.jpg")

alpha = 2  # Increase contrast
beta = -100    # No change in brightness

# Apply the linear transformation
# new_image = alpha * image + beta
enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

cv2.imshow("frame", enhanced_image)

cv2.waitKey(0)
cv2.destroyAllWindows()