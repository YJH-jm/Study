import sys
import cv2
import numpy as np

# 기존 코드
# cap = cv2.VideoCapture()
# cap.open(0) 


# cap = cv2.VideoCapture(0) # 위의 두 줄을 합친 코드
cap = cv2.VideoCapture('images/video1.mp4')
if not cap.isOpened():
    print("camera open failed!")
    sys.exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 속성 받아 올 수 있음
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(w, h)

# w, h 값에 모든 값을 넣을 수 있는 것이 아님, 비디오일 때는 안 하고 카메라 일 때 함
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    edge = cv2.Canny(frame, 50, 150)
    cv2.imshow('frame', frame)
    cv2.imshow('edge', edge)
    
    if cv2.waitKey(20) == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()

# Zed 이용
# cap = cv2.VideoCapture()
# cap.open(1) 

# if not cap.isOpened():
#     print("camera open failed!")
#     sys.exit()

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break
    
#     l_r_image = np.split(frame, 2, axis=1)
#     cv2.imshow('frame', l_r_image[1])
#     if cv2.waitKey(20) == 27: # ESC
#         break

# cap.release()
# cv2.destroyAllWindows()