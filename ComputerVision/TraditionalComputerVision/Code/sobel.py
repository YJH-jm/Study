import sys
import numpy as np
import cv2


src = cv2.imread("./images/lenna.bmp", cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed!")
    sys.exit()

'''
# case 1
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

dx = cv2.filter2D(src, -1, kernel) # grayscale이 되므로 미분값이 음수가 되면 0으로 saturation 되어서 변화가 보이지 않음
dx = cv2.filter2D(src, -1, kernel,delta=128)
'''

# case 2

dx = cv2.Sobel(src, -1, 1, 0, delta=128)
dy = cv2.Sobel(src, -1, 0, 1, delta=128)


cv2.imshow("src", src)
cv2.imshow("dx", dx)
cv2.imshow("dy", dy)

cv2.waitKey()
cv2.destroyAllWindows()
