import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread("test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
dst = cv2.Canny(gray, 0, 150)
blured = cv2.blur(dst, (5,5), 0)
lower_red = np.array([0, 78, 255])
upper_red = np.array([0, 98, 255])
mask_red = cv2.inRange(gray, lower_red, upper_red)
Contours, imgContours = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in Contours:
    if cv2.contourArea(contour) > 40:
        [X, Y, W, H] = cv2.boundingRect(contour)
cropped_image = img[Y:Y+H, X:X+W]
print([X,Y,W,H])
plt.imshow(cropped_image)
plt.show()
plt.imshow(img)
plt.show()
cv2.imwrite('contour1.png', cropped_image)