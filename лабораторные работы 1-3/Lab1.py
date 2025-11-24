import cv2
import numpy as np
import matplotlib.pyplot as plt

original = cv2.imread("img/elphs.jpg")
blurred = cv2.GaussianBlur(original, (19, 19), 0)
edges = cv2.Sobel(original, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
edges = cv2.convertScaleAbs(edges)

sharpened = cv2.addWeighted(original, 1.5, blurred, -0.5, 0)

combined = cv2.addWeighted(blurred, 0.5, edges, 0.5, 0)
combined = cv2.addWeighted(combined, 0.5, sharpened, 0.5, 0)

plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.title('Оригинальное изображение')
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Размытие по Гауссу)')
plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Выделение границ')
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Повышение резкости')
plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Комбинация изображений')
plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()