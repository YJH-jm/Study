import cv2
import sys

# 그레이 스케일 영상의 히스토그램 평탄화
src = cv2.imread("images/Hawkes.jpg", cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed!")
    sys.exit()


dst = cv2.equalizeHist(src)

# 컬러 영상의 히스토그램 평탄화

src = cv2.imread("images/field.bmp")
if src is None:
    print("Image load failed!")
    sys.exit()

src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
planes = cv2.split(src_ycrcb)
planes[0] = cv2.equalizeHist(planes[0])

dst_ycrcb = cv2.merge(planes)
dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()