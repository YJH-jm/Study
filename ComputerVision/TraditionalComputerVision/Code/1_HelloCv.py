import sys
import cv2

print("Hello OpenCV", cv2.__version__)

# img = cv2.imread('Code\ch01\cat.bmp', cv2.IMREAD_COLOR)
# img = cv2.imread('Code\ch01\cat.bmp') 
img = cv2.imread('Code\ch01\cat.bmp', cv2.IMREAD_GRAYSCALE)
# print(type(img)) # <class 'numpy.ndarray'>

if img is None:
    print("Image load failed!")
    sys.exit()

cv2.imwrite('Code\ch01\cat_gray.png', img)
cv2.namedWindow('image') # 생략해도 만들어줌
cv2.imshow('image', img)


# cv2.waitKey()
while True: # 특정 키를 누르면 종료
    if cv2.waitKey() == 27: # ord('a')
        break

cv2.destroyAllWindows()
# cv2.destroyWindow('image')