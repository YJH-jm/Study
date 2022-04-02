import cv2

# 마스트 영상을 이용한 영상 합성
src = cv2.imread('images/airplane.bmp', cv2.IMREAD_COLOR)
mask = cv2.imread('images/mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('images/field.bmp', cv2.IMREAD_COLOR)

cv2.copyTo(src, mask, dst) # src, dst는 둘 다 color 이거나 gray이여야 함
# dst = cv2.copyTo(src, mask) # 이 경우는 mask의 ROI 부분만 src에서 채워줌
# dst[mask > 0] = src[mask > 0] # 이 코드도 cv2.copyTo(src, mask, dst)와 같은 결과 나옴

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.imshow('mask', mask)

# cv2.waitKey()
# cv2.destroyAllWindows()


# 알파채널이 존재하는 png 파일 합성
src = cv2.imread('images/opencv-logo-white.png', cv2.IMREAD_UNCHANGED) # 알파채널이 있는 png 파일을 읽어오기 위함
mask = src[:, :, -1]
src = src[:, :, 0:3]
dst = cv2.imread('images/field.bmp', cv2.IMREAD_COLOR)

h, w = src.shape[:2]
crop = dst[0:h, 0:w]
cv2.copyTo(src, mask, crop)


cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)

cv2.waitKey()
cv2.destroyAllWindows()