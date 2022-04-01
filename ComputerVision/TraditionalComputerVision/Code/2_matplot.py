import matplotlib.pyplot as plt
import cv2

# 컬러 영상 출력
imgBGR = cv2.imread('Code\ch01\cat.bmp') # BGR 순서로 읽어옴

imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB) # Matplotlib는 RGB 형태로 사용되기 때문에 변환 과정 필요

plt.axis('off') # 가로 세로 숫자 눈금 표시 안하겠다는 의미
plt.imshow(imgRGB)
plt.show()


# 그레이스케일 영상 출력
imgGray = cv2.imread('Code\ch01\cat.bmp', cv2.IMREAD_GRAYSCALE) # 밝기값만 저장

plt.axis('off') 
plt.imshow(imgGray, cmap='gray')
plt.show()


# 두 개의 영상을 함께 출력
plt.subplot(121), plt.axis('off'), plt.imshow(imgRGB)
plt.subplot(122), plt.axis('off'), plt.imshow(imgGray,cmap='gray')
plt.show()