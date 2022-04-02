import numpy as np
import cv2

# 새 영상 생성
img1 = np.empty((240, 320), dtype = np.uint8) # grayscale image
img2 = np.zeros((240, 320, 3), dtype=np.uint8)
img3 = np.ones((240, 320, 3), dtype = np.uint8)
img4 = np.full((240, 320), 128, dtype = np.uint8)

# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('img3', img3)
# cv2.imshow('img4', img4)

# cv2.waitKey()
# cv2.destroyAllWindows()

# 영상 복사
img1 = cv2.imread('images/HappyFish.jpg')
img2 = img1 # img1과 img2이 같은 것을 공유, 참조와 같은 
img3 = img1.copy() # 메모리 새로 할당

img1[:, :] = (0, 255, 225)

# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('img3', img3)

# cv2.waitKey()
# cv2.destroyAllWindows()

# 부분 영상추출
img1 = cv2.imread('images/HappyFish.jpg')
img2= img1[40:120, 30:150]
img3 = img1[40:120, 30:150].copy()

# img1[:, :] = (0, 255, 225)
img2.fill(0)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)

cv2.waitKey()
cv2.destroyAllWindows()

