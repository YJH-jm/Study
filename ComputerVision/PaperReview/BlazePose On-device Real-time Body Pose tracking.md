# BlazePose: On-device Real-time Body Pose tracking
(2020년 연구)

<br>
<br>

## Introduction 
- Human body pose estimation은 health tracking 과 같은 분야에서 많이 사용 
- 하지만 pose의 다양성, 수많은 자유도 그리고 occlusion과 같은 상황 때문에 어려움이 있음
- 최근 pose estimation의 연구는 각 관절 좌표의 offset을 조정하기 위한 heatmap 생성 방식을 많이 이용
- 하지만 최소한의 overhead로 많은 사람들에게 확장이 되지만 한 사람에 대하여 모바일 폰으로 실시간으로 추론하기에는 크기가 큼
- Heatmap 기반 방식과는 대조적으로, regression 방식의 접근 방법은 계산이 덜 필요하고 확장 가능성이 높지만 애매한 포즈를 잘 예측하지 못하는 문제 존재
- Stacked Hourglass 구조로 적은 수의 파라미터를 사용하여 예측 성능 향상시켰고 이 구조를 확장하여 encoder-decoder 구조를 이용하여 모든 좌표의 heatmap을 예측하고 또 다른 encoder를 이용하여 regression을 이용하여 직접적으로 모든 관절의 좌표를 예측
- 이 논문의 핵심 아이디어는 heatmap branch는 추론 시에 사용하지 않게 만들어 실시간으로 모바일 폰에서 추론 할 수 있도록 만든다는 것


<br>
<br>

## 2. Model Architecture and Pipeline Design
### 2.1 Inference pipeline
- Detector-Tracker 구조를 채택
    - hand landmark 예측, dence face landmark 예측과 같은 다양한 작업에서 좋은 실시간 성능을 보여줌
- 이 pipeline은 가벼운 body pose detector와 pose tracker 순서로 구성된 네트워크
- Tracker는 관절 좌표, 현재 프레임에서의 사람의 존재 유무, 그리고 현재 프레임의 ROI를 예측
- 현재 프레임에 사람이 없다고 예측했을 때, 다음 프레임에서 detector를 다시 수행

<br>
<br>

### 2.2 Person detector
- 현재 대부분의 object detection 방식들은 NMS (Non-Maximum Suppression) 알고리즘을 후처리 방식으로 사용 
- 위의 방식은 사람들이 악수를 하거나 안는 등의 연결된 동작을 하고 있을 떄 잘 작동하지 않음
- 이 문제점을 해결하기 위해 얼굴이나 몸통처럼 상대적으로 움직이기 힘든 몸의 부분을 찾는 것에 집중
- Neural network에서 몸통을 찾는데 가장 핵심적인 부분은 사람의 얼굴이라는 것을 많은 경우에서 발견
    <!-- - 얼굴은 high-contrast fea -->
- Detector를 빠르고 가볍게 하기 위해서, single person 경우에 항상 사람의 머리는 모인다고 가정
- 결과적으로, face detector를 person detector로 사용 
- 이 face detector는 추가적으로 alignment parameters 예측
    - 사람 엉덩이의 중심점, 사람 전체를 포함하는 원의 반지름, 어깨의 중심점과 엉덩이 중심점을 연결하는 선의 기울기 

<br>
<br>

### 2.3 Topology
- 33개의 사람의 body point를 찾는 새로운 topology 제공
    - BlazeFace, BlazePalm, Coco를 모두 포함하는 topology
- Openpose와 Kinetic topology와 다르게 

<br>
<br>

### 2.4 Dataset
- 



