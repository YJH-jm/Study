
## Abstract 
- 기존의 많은 skeleton based action recognition 방식에서는 GCNs을 선택하여 skeleton 특징 추출
- 하지만 GCN 을 기본으로 한 방법들은 robustness, interoperability, scalibility 에 약함
- 이 논문에서는 새로운 방식의 Skeleton based action recongnition 방식을 제시하고 이를  **PoseConv3D** 라 함
- PoseConv3D는 3D heatmap volume에 의존함
- GCN 방식과 비교하여 PoseConv3D 방식은
    - 시공간의 특징을 학습하는데 더 효과적
    - pose estimation noise 에 더욱 강함
    - cross-dataset 에 대하여 더욱 일반화가 잘됨
    - multi-person 에 대하여 추가적인 계산 비용 없음


<br>
<br>

## 1. Introduction
- 사람의 skeleton은 영상에서 주로 관절들의 좌표 리스트 []의 시퀀스로 나타남
    - 좌표는 pose estimator에서 extract
- 관절에 대한 정보만 들어있기 때문에 배경의 다양성이나 빛의 변화와 같은 contextual nuisances에 강함
- Skeleton-based action recognition에서 가장 많이 사용되고 있는 방법은 GCN(graph convolutional networks) 
- GCN은 모든 timestep마다 모든 사람의 관절이 노드가 됨


GCN에 대한 내용 공부하고 추가 할 것 

- GCN을 기본으로 한 방식들을 몇 가지 한계가 존재
    1. Robustness
        - GCN은 사람의 좌표를 직접적으로 다루기 때문에 좌표 이동의 분포에 영향을 받음
    2. Interoperability
        - RGB, optical flow 와 같은 다른 modalities를 결합하여 사용하면 성능 향상이 가능함을 예전 연구들에서 증명
        - 하지만 GCN의 경우는 skeleton들의 그래프를 사용하기 때문에 결합이 어려움
    3. Scalability
        - GCN은 모든 사람의 관절을 노드로 하기 때문에 GCN의 complexith는 사람의 수에 따라 선형적으로 증가함

<br>
<br>

- 이 논문에서는 GCN 방식보다 경쟁력있는 새로운 프레임워크인 **PoseConv3D** 제시

<br> 

<p align=center><img src = "./images/1.png" width = 50%></p>

<br>

- PoseConv3D는 2D pose estimator에서 얻은 2D pose를 input으로 사용
- HRNet은 skeleton의 관절들의 heatmap들을 쌓아 2D pose 표현 
- 다른 timestpe의 heatmap들은 3D heatmap volume을 만들기 위해 시간축으로 쌓음
- PoseConv3D는 3d Convolutional neural network 사용
- GCN과의 차이는 아래의 표에 정리

<br>

<p align=center><img src="./images/2.png" width=30%></p>

<br>


<br>
<br>

- PoseConv3D는 위에 언급된 GCN의 문제 해결 
    1. 3D heatmap volume은 up-stream pose estimation 보다 더 robost
        - 경험적으로 PoseConv3D 방식이 다양한 접근법으로 얻은 input skeleton에 대하여 일반화가 잘됨
    2. 발전되는 Convolution Network 적용 가능 다양한 Modality와 함께 사용할 수 있음  
    3. 연상의 overhead 없이 많은 사람들에게 적용 가능
        - 3D heatmap volumne은 사람의 수와 관련 없음

- PoseConv3D 성능을 검증하기 위해 여러개의 데이터 셋 이용
    - FineGYM. NTURGB-D, UCF101, HMDB51, Kinetics400, Volleyball, ..

<br>
<br>

## 2. Related Work
### 3D-CNN for RGB-based action recognition
- 공간의 특징을 학습하는 2D-CNN을 시공간으로 확장한 것이 3D-CNN

<br>

<p align=center><img src="./images/3.png" width=30%></p>

<br>

- Action recognition에서 많이 사용
- 매우 많은 수의 parameter가 있기 때문에 좋은 성능을 내기 위해서는 매우 다양하고 많은 영상이 요구됨
- 이 논문에서는 3D heatmap volume을 input으로 사용하는 3D-CNN 제안

<br>
<br>

### GCN for skeleton-based action recognition
- Skeleton-based action recognition에서 사용하는 가장 대표적인 방법
- 사람의 skeleton sequence를 시공간 그래프로 모델링
- ST-CGCN이 가장 잘 알려진 baseline 모델   

<br>

<p align=center><img src="./images/4.png" width=50%></p>

<br>

- 시공간으로 모델링하기 위해 spatial graph convolutiobn과 interleaving temperal convolution을 결합

<br>
<br>

### CNN for skeleton-based action recognition
- 2D-CNN-based 접근법들은 manually 하게 설계된 변환을 기반으로 skeleton sequence를 psedo image로 먼저 모델링
- 그 중 하나의 방식은 색상 인코딩 또는 학습된 모듈과 함께 heatmap을 시간축으로 합쳐 2D input으로 사용
    - 잘 설계하더라도 heatmap을 합치는 경우 정보가 손실
- 또 다른 방법은 직접적으로 skeleton sequence 좌표를 psedo image로 변환
    - 보통 2D input (shape: K x T)
        - K : 관절의 수 (ex > cocodataset : 17)
        - T : temporal length
    - 이런 input은 convolution 의 지역적 특성을 이용할 수 없기 때문에 GCN보다 효과적이지 않음

    <br>

    - 예전 아주 소수의 연구들이 3D-CNN 방식을 선택
        - 3D input을 만들기 위하여 거리 matrices의 psedo image를 쌓거나 3d skeleton을 요약하여 직육면체로 만듦
        - 이 방식들 역시 정보 손실이 존재하여 낮은 성능을 가짐
- 이 연구는 heatmap을 시간축으로 쌓아 3D heatmap volumne으로 만들어 정보 손실이 없도록 함
- 시공간에 대한 특징 학습을 잘 할 수 있도록 2D가 아닌 3D-CNN 사용

<br>
<br>



## 3. Framework

<br>

<p align=center><img src="./images/5.png" ></p>

<br>


### 3.1 Good Practices for Pose Extraction
- Skeleton-based action recognition의 가장 중요한 pre-processing 과정은 pose 추출을 정확하게 하는 것

<br>

- 일반적으로 2D pose estimation은 3D pose estimation보다 좋은 성능을 가짐
- 이 실험에서는 2D의 Top-down 방식의 pose estimator를 선택
    - Benchmark dataset과 비교하면 2D Bottom-up 방식보다 더 좋은 성능을 얻음
- 여러 사람들 중 몇 명의 사람들에게만 관심이 있을 때, skleton-based recognition에서 좋은 성과를 얻기 위해서는 몇 가지의 사전 지식이 필요
    - 비디오의 첫 프레임에서의 관심 있는 사람에 대한 위치 등
- 예측된 heatmap의 저장 관점에서, 이전 문헌에서는 (x, y, c) 로 저장
    - c : 예측된 heatmap의 최대 score
    - (x, y) : c에 대응되는 좌표
- 위에서 처럼 저장하는 것은 성능 저하가 거의 없이 저장 공간을 줄일 수 있음  


<br>
<br>


### 3.2 From 2D Poses to 3D Heatmap Volumne
- 2D Pose가 추출되고 난 후, PoseConv3D에 적용하기 위해 결과를 3D heatmap volume (K x H x W)으로 재구성
    - K : 관절의 수
    - H, W : frame 의 height, width
- Skeleton 관절에 대한 좌표를 가지고 K 개의 가우시안 맵을 생성

<br>

<p align=center>
<img src="https://latex.codecogs.com/svg.image?J_{kij}=e^{-\frac{{(i-x_{k})}^2&plus;{(j-y_{k})}^2}{2*&space;{\sigma&space;}^2}*c_{k}" title="https://latex.codecogs.com/svg.image?J_{kij}=e^{-\frac{{(i-x_{k})}^2+{(j-y_{k})}^2}{2* {\sigma }^2}*c_{k}" /></p>

<p align=center>
<img src="https://latex.codecogs.com/svg.image?\sigma" title="https://latex.codecogs.com/svg.image?\sigma" />&nbsp; : 가우시안 맵의 분포를 조절 <br>
<img src="https://latex.codecogs.com/svg.image?(x_{k},&space;y_{k})" title="https://latex.codecogs.com/svg.image?(x_{k}, y_{k})" /> &nbsp; : k번째 관절의 좌표 <br>
<img src="https://latex.codecogs.com/svg.image?c_{k}" title="https://latex.codecogs.com/svg.image?c_{k}" /> &nbsp; k 번째 관절의 confidence score
</p> 

<br>

- Limb heatmap 또한 생성이 가능

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?L_{kij}=e^{-\frac{{D((i,j),seg[a_{k},b_{k}])}^2}{2*{\sigma}^2}}*min(c_{a_{k}},c_{b_{k}})" title="https://latex.codecogs.com/svg.image?L_{kij}=e^{-\frac{{D((i,j),seg[a_{k},b_{k}])}^2}{2*{\sigma}^2}}*min(c_{a_{k}},c_{b_{k}})" /></p>

<p align=center>
<img src="https://latex.codecogs.com/svg.image?a_{k},b_{k}" title="https://latex.codecogs.com/svg.image?a_{k},b_{k}" /> &nbsp; : k 번째 두 관절 <br>
<img src="https://latex.codecogs.com/svg.image?seg[a_{k},&space;b_{k}]=segment[(x_{a_{k}},y_{a_{k}}),(x_{b_{k}},y_{b_{k}})]" title="https://latex.codecogs.com/svg.image?seg[a_{k}, b_{k}]=segment[(x_{a_{k}},y_{a_{k}}),(x_{b_{k}},y_{b_{k}})]" /> <br>
<img src="https://latex.codecogs.com/svg.image?D&space;" title="https://latex.codecogs.com/svg.image?D " /> &nbsp; : <img src="https://latex.codecogs.com/svg.image?(i,&space;j)" title="https://latex.codecogs.com/svg.image?(i, j)" /> &nbsp;와 &nbsp; <img src="https://latex.codecogs.com/svg.image?seg[a_{k},&space;b_{k}]" title="https://latex.codecogs.com/svg.image?seg[a_{k}, b_{k}]" />  사이의 거리  
</p>

<br>

- 위의 과정은 한 사람에 대한 결과지만 여러 사람으로 확장되어도 모든 사람에 대한 k번째 가우시안 맵은 확장되지 않음
- 결과적으로 **3D Heatmap Volumne**은 시간 축으로 heatmap (J 또는 L) 을 쌓으며 만들어짐
    -  **K x T x H x W**

<br>

- 추후에 3D heatmap volume의 redundency를 줄이기 위해 2가지 기술을 적용
- **Subjects Centered Cropping**
    - Heatmap을 frame 크기만큼 만드는 것은 비효율적
    - 특히 행동을 분석해야 하는 사람이 전체이미지에서 좁은 영역에 있을 경우
    - 이런 경우 프레임들에 걸쳐 모든 2D pose를 감싸는 가장 작은 bounding box를 찾음
    - 모든 2D Pose와 그들의 행동이 유지가 되면 3D heatmap volume을 공간적으로 줄일 수 있음
- **Uniform Sampling** 
    - 기존의 RGB 기반 action recognition 방식은 짧은 일정 기간의 윈도우로 샘플링을 진행
    - 이 방식은 일정한 시간 간격으로 n개로 나누고 n개의 구간에서 임의로 하나의 frame을 선택 

<br>
<br>


# 추가ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ... 
### 3.3 3D-CNN for Skeleton-based Action Recognition
- **PoseConv3D** 
    - 다양한 3D-CNN backbone으로 인스턴스화 될 수 있음
    - 3D-CNN 초기 단계에서 downsampling 제거 
        - 3D heatmap volume은 RGB 클립만큼 크지 않기 때문 
    - shallower(fewer layers) & thiner (fewer channels)
        - 3D heatmap volumes이 이미 mid-level 특징이기 때문에
    - 이 논문에서 위에 있는 내용을 바탕으로 가장 대표적인 3개의 3D-CNN 알고리즘 선택
        - C3D
        - SlowOnly
        - X3D

<br>

- **RGBPose-Conv3D**
    - PoseConv3D의 interoperability를 보여주기 위해 초기에 human skeleton과 RGB 프레임들을 합친 모델 제시
    - 이 two-stream modality는 Pose modality와 RGB modality를 각각 수행함

## 4. Experiments
### 4.1 Dataset Preparation
- 이 실험에서 6개의 데이터 사용
    - FineGYM
        - 잘 정제된 29,000 개의 체조(운동) 영상
        - 99개의 humam action class
    - NTURGB+D
        - 연구실에서 모은 매우 방대한 양의 human action recognition dataset
        - NTU-60, NTU-120 두 가지 버전 존재
        - NTU-60
            - 57,000 개의 비디오
            - 60개의 human action class
        - NTU-120
            - 114,000개의 비디오
            - 120개의 human action class 
    - Kinetics400, UCF101, HMDB51
        - 이 세개의 dataset은 web에서 얻은 일반적인 action recognition dataset
        - Kinetics400
            - 300,000 개의 비디오
            - 400개의 human action class
        - UFC101
            - 13,000개의 비디오
            - 101개의 human action class
        - HMDB51
            - 6,700개의 비디오
            - 51개의 human action class
    - Volleyball
        - Group activity recognition dataset
        - 4830개의 비디오
        - 8개의 group action class

<br>
<br>

### 4.2 Good properties of PoseConv3D
- 이 실험에서 제시한 모델과 GCN 기반의 모델을 비교하기 위하여 SlowOnly 방식과 MS-G3D 방식 비교
    - SlowOnly (PoseConv3D), MS-G3D (GCN-based)
- 두 모델은 같은 Input을 가짐
    - GCN-based : (x, y, c)
    - PoseConv3D : (x, y, c) 으로부터 생성된 heatmap

<br>

**Performance & Efficiency** <br>

<br>

<p align=center>
<img src="./images/6.png" width=50%>
</p>

<br>

<br>

**Robustness** <br>
- Input의 keypoints를 p의 확률로 없앤 후 이 변화가 얼마나 최종 정확도에 영향을 미치는지 확인 
- 체조 동작에서  몸통이나 얼굴의 keypoint 보다 사지의 keypoint가 더 중요하기 때문에 각 프레임에서 한 개의 limb keypoint를 drop 함

<br>

<p align=center>
<img src="./images/7.png" width=50%>
</p>

<br>
<br>


# 추각ㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱ
**Generalization** <br>
- FineGYM 데이터셋에 대한 교차 모델 검사를 설계
- Pose 추정을 위해 HRNet(Higher-Quality, HQ)과 MobileNet(Lower-Quality, LQ)은 두 가지 모델을 사용하고 그 위에 PoseConv3D 모델을 각각 학습 
- 테스트 과정에서, HQ로 학습된 모델에 LQ input을 주고 그 반대도 수행함

<p align=center>
<img src="./images/8.png" width=50%>
</p>

<br>
<br>

**Scalability**
- GCN 기반의 방식들은 비디오에 있는 사람의 수가 증가하면 scale이 선형적으로 증가하기 때문에 group action recognition 방식에서 성능이 떨어짐
- 이를 증명하기 위해서 Volleyball dataset 사용
- 각 비디오에는 13명의 사람 존재 
