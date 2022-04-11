import sys
import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"DIVX") # *"DIVX" == "D", "I", "V", "X" # *는 문자열을 풀어써주는 의미
delay = round(1000 / fps) # 한 프레임과 그 다음 프레임간의 간격을 계산하기 위한 코드 

out = cv2.VideoWriter("output.avi", fourcc, fps, (w, h))

if not out.isOpened():
    print("File opend failed")
    cap.release()
    sys.exit()

while True:
    # 카메라에서 한 frame 받아오는 코드
    ret, frame = cap.read()

    if not ret:
        break

    inversed = ~frame # 프레임을 반전하는 코드
    edge = cv2.Canny(frame, 50, 150)
    
    out.write(frame)
    cv2.imshow("frame", frame)
    cv2.imshow("inversed", inversed)
    cv2.imshow("edge", edge)

    if cv2.waitKey(delay) == 27:
        break

cap.release()
