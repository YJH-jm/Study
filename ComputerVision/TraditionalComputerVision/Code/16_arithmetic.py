import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

src1 = cv2.imread("images/lenna256.bmp", cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread("images/square.bmp", cv2.IMREAD_GRAYSCALE)

print(src1.shape)
print(src2.shape)
if src1 is None or src2 is None:
    print("Image load failed!")
    sys.exit()

dst1 = cv2.add(src1, src2, dtype=cv2.CV_8U)
dst2 = cv2.addWeighted(src1, 0.5, src2, 0.5, 0.0)
dst3 = cv2.subtract(src1, src2)
dst4 = cv2.absdiff(src1, src2)

plt.subplot(231), plt.axis("off"), plt.imshow(src1, "gray"), plt.title("src1")
plt.subplot(232), plt.axis("off"), plt.imshow(src2, "gray"), plt.title("src2")
plt.subplot(233), plt.axis("off"), plt.imshow(dst1, "gray"), plt.title("add")
plt.subplot(234), plt.axis("off"), plt.imshow(dst2, "gray"), plt.title("addWeighted")
plt.subplot(235), plt.axis("off"), plt.imshow(dst3, "gray"), plt.title("subtract")
plt.subplot(236), plt.axis("off"), plt.imshow(dst4, "gray"), plt.title("absdiff")
plt.show()

# cv2.waitKey()
# cv2.destroyAllWindows()