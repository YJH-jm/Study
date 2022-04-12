import sys
import cv2
import numpy as np
import time

img = cv2.imread("images/hongkong.jpg")

if img is None:
    print("Image load failed!")
    sys.exit()

tm = cv2.TickMeter()
tm.start()
t1 = time.time()


edge = cv2.Canny(img, 50, 150)


tm.stop()
t2 = time.time()
print("Elapsed time : {}ms".format(tm.getTimeMilli()))
print("time : {}".format(t2-t1))

cv2.imshow("image", edge)
cv2.waitKey()
cv2.destroyAllWindows()