## 마스크 연산과 ROI
### ROI (Region of Interest)
- 관심영역
- 영상에서 특정 영산을 수행하고자 하는 임의의 부분 영역

### 마스크 연산
- OpenCV는 일분 함수에 대해 ROI 연산 지원
    - 마스크 영상을 인자로 함께 전달해야 함
    - `cv2.copyTo()`, `cv2.calcHist()`, `cv2.bitwise_or()`, `cv2.matchTemplate()` 등등 
- 마스크 영상은 cv2.CV_8UC1 타입 (grayscale)
- 마스트 영상의 픽셀값이 0이 아닌 위치에서만 연산이 수행
    - 보통 마스크 영상으로는 0 또는 255로 구성된 이진 영상


## OpenCV 그리기 함수
- OpenCV 영상에서 선, 도형, 문자열을 출력하는 그리기 함수 제공
    - 선 그리기 : 직선, 화살표, 마커 등
    - 도형 그리기 : 사각형, 원, 타원, 다각형 등
    - 문자열 출력
- 그리기 알고리즈을 이용하여 영상 픽셀 값 자체를 변경
    - 원본 영상이 필요하면 복사본 만들어 그리기 & 출력
- Grayscale 영상에는 컬러로 그리기 안됨
    - `cv2.cvtColor()` 함수 이용하여 BGR 컬러 영상으로 변환 후 그리기 함수 호출 

### 직선 그리기
`cv2.line(img, pt1, pt2, color, thickness=None, lineType=None, shift=None) -> img`
- img : 그림 그릴 영상
- pt1, pt2 : 직선의 시작점과 끝점
- color : 선 색상 또는 밝기. () 