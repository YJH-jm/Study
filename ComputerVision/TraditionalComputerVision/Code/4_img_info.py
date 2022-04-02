import sys
import cv2

img1 = cv2.imread('images/cat.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('images/cat.bmp')

if img1 is None or img2 is None:
    print("Image load failed!")
    sys.exit()

print(type(img1)) # <class 'numpy.ndarray'>
print(img1.shape) # (480, 640)
print(img1.dtype) # uint8

print(img2.shape) # (480, 640, 3)
print(img2.dtype) # uint8


if img1.ndim == 2: # img1.ndim = len(img1.shape)
    print('img1 is a grayscale image')


h, w = img1.shape
print('w x h = {} x {}'.format(w, h))


h, w = img2.shape[:2]
print('w x h = {} x {}'.format(w, h))

# 픽셀값 참조
x = 20
y = 10
p1 = img1[y, x]
print(p1)
img1[y, x] = 0

p2 = img2[y, x]
print(p2)  # [237 242 232] = [B, G, R]
img2[y, x] = [0, 0, 255]

# 아래 코드는 시간이 너무 많이 걸리기 때문에 쓰지 말 것 
for y in range(h):
    for x in range(w):
        img1[y, x] = 0
        img2[y, x] = (0, 0, 255)

img1[:, :] = 0
img2[:, :, :] = (0, 255, 255)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey()
cv2.destroyAllWindows()