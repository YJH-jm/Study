## 2D Human Pose Estimation
- 입력 이미지로부터, 사람의 관절을 2D 공간에 위치화 시키는 알고리즘

    <br>

    <p align=center><img src="./images/1.png" width = 50%></p>

    <br>

    - 


<br>
<br>



## Challenges of 2D Human Pose Estimation
- Occulusion
    - 콘서트장이나 운동장 같은 곳 
    - 지금도 해결해 나가고 있는 문제
- 복잡한 자세
    - 일상 생활 외의 자세 (요가, 필라테스 등등)
- 작은 해상도
    - CCTV를 이용한 범죄 감지 ㄷ,ㅇ에 자주 발생
- 모션 블러
    - 입력 이미지의 정보가 불안전
    - 사람이 빨리 움직이거나 또는 사진 찍으면서 손이 흔들리는 경우
- 이미지 잘림
    - Human Pose Estimation의 경우 관절 Joint의 개수가 정해져 있기 때문에 이상하게 결과가 나오는 경우 존재

<br>
<br>


## Top-Down approach
- **Human detection + Single person pose estimation**
- Bottom-up 방식보다 더 뛰어난 정확성
- 최근 발표된 yolo 등의 매우 정확한 human detection network 존재
    - Bottleneck은 single person pose estimation에서 발생
    - Human detection이 실패하면 아예 single person pose estimation이 불가능하다는 단점이 존재하나 detector의 성능이 좋아져 어느정도 해결
- Human pose estimation에 쓰이는 이미지가 매우 고해상도
    - Human detection 한 후 resize 하기 때문에 해상도가 낮아 손목, 발목 등이 잘 안보이는 단점 해결 
- Bottom-up approach들보다 비효율적
    - 2개의 분리된 시스템이기 때문
- 대표적인 방식 
    - Mask R-CNN (ICCV 2017)
    - Simple Baseline (ECCV 2018)
    - HRNet (CVPR 2019)
<br>
<br>


## Bottom-Up approach
- **Joint detection + Grouping**
- Top-down approach들보다 낮은 정확성
- Human pose estimation에 쓰이는 사람 입력 이미지가 저해상도일 가능성 존재
    - Top down 방식처럼 resize하지 않기 때문
    - 여러 scale의 사람들을 다뤄야하기 때문에 network에 부담이 갈 수 있음
- Top-down approach 들보다 더 효율적
- 대표적인 방식 
    - Associative Embdeeing
    - HigherHRNet
<br>
<br>


