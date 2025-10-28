import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("img/johnson.jfif")
image = cv2.resize(image, None, fx=0.5, fy=0.5)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.equalizeHist(gray)
blured_gray = cv2.GaussianBlur(gray, (11, 11), 0)
width = image.shape[1]
height = image.shape[0]

edges = cv2.Canny(blured_gray, 60, 70) # 100 130
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
black_image = np.zeros((height, width, 3), dtype=np.uint8)
cv2.drawContours(black_image, contours, -1, (255, 255, 255), 2)
drawing = cv2.bitwise_not(black_image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
#plt.title('Оригинальное изображение')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
#plt.title('Размытие по Гауссу)')
plt.imshow(cv2.cvtColor(black_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
#plt.title('Выделение границ')
plt.imshow(cv2.cvtColor(drawing, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()



# cv2.imshow('Original Image', image)
# cv2.imshow('Canny Edges', black_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#cv2.imwrite("a.jpg", edges)