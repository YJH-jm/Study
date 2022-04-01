import sys
import cv2
import glob

img_files = glob.glob('.\\images\\slide_images\\*.jpg')

for f in img_files:
    print(f)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cnt = len(img_files)
idx = 0
while True:
    img = cv2.imread(img_files[idx])
    
    if img is None:
        print("Image load failed!")
        break
    
    cv2.imshow('image', img)
    if cv2.waitKey(1000) == 27:
        break
    
    idx += 1
    if idx >= cnt:
        idx = 0

cv2.destroyAllWindows()