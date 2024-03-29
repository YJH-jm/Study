# 3D Human Pose Estimation
3D Pose Estimation에 대한 기본 내용 정리

<br>
<br>

## 2D 와 3D 차이
- 임의의 **시점(view)** 으로 대상을 표현 할 수 있는지가 근본적인 차이

<br>

<p align=center><img src="./images/2/4.png" width=60%></p>    

<br>

- 2D는 다른 view에서의 pose 볼 수 없음
- 3D는 시점을 변경하여 다른 view에서 pose 보는 것 가능
    - 더 풍부한 정보를 가지고 있음

<br>
<br>

## 3D Human Pose Estimation OverView
입력 이미지로부터 사람의 관절을 3D 공간에 위치화 시키는 알고리즘

<br>


<p align=center><img src="./images/2/18.PNG" width=60%></p>


<br>

### 3D human pose
3D에서의 pose는 관절 좌표, 관절 회전 모두에게 사용

<br>

- 3D 관절 좌표
    
    <br>
    
    <p align=left><img src="./images/2/5.png" width = 20%></p>
    
    <br>

    - 3D 표면 (surface, mesh) 으로 표현 불가능
    - 점 : 관절
    - 선 : Kinematic chain 정보 (관절 연결 정보)

    <br>

- 3D 관절 회전 
    
    <br>

    <p align=left><img src="./images/2/6.png" width=25%></p>

    <br>

    - 3D 표면 (surface, mesh) 으로 표현 가능
        - 관절의 좌표만으로는 회전을 표현할 수 없음
    - 팔꿈치가 움직이면 팔꿈치의 자식 관절들만 움직임
    - 길이가 있다면 엄밀히 말해 6D (각도 3 + 위치 3)

<br>


### 3D Mesh
- 3D object를 표현하는 가장 standard한 자료 구조

<br>

<p align=center><img src="./images/2/7.png" width=30%></p>

<br>

- 3D object를 많은 수의 작은 삼각형의 모음으로 3D 물체의 표면을 표현
    - 내부와 외부는 비어있음
- 꼭짓점(vertex)과 면(face)으로 구성
- 다른 물체 표현법(volume, point cloud) 비해 효율적이고 직관적이기 때문에 가장 많이 사용
    - ex> volumne (메모리 많이 차지하기 때문에 비효율적)
- 삼각형의 **꼭지점의 개수**와 **각 면을 이루는 꼭지점들의 index**는 상수라고 가정하고 꼭지점들의 3D 좌표를 구하는 것이 목표
    - 꼭짓점(vertex)은 삼각형의 각 꼭지점을 의미
    - 각 면을 이루는 꼭지점들의 index는 어떤 삼각형 index가 면을 이루는지를 의미 
    - 예를 들어 두 번째 손가락 손톱은 vertex index 0, 1, 2 로 표현하고 왼쪽 눈은 3, 4, 5 vertex index로 표현
- 3D human mesh estimation의 목표는 꼭짓점들의 좌표를 구하는 것이고 꼭지점의 개수와 꼭지점들의 index를 mesh topology라고 함 

<br>
<br>


### 3D Human Model
- 3D Pose Estimation에는 2개의 orthogonal한 input 존재
    - 3D 관절 회전 (Pose)
        
        <br>

        <p align=left><img src="./images/2/9.png" width=25%></p>

        <br>

        - 같은 shape 일 때 관절 위치 회전
        - Mesh 관점으로 같은 체형일 때 vertex의 위치 변경

        <br>

    - 3D 길이/체형 (Shape)

        <br>

        <p align=left><img src="./images/2/8.png" width=25%></p>

        <br>

        - 같은 t-pose 일 때 길이, 체형 차이


<br>

- 3D Human Model

<br>

<p align=center><img src="./images/2/10.png" width=60%></p>

<br>

- 3D 관절 회전(pose)과 다른 파라미터(Shape)들로부터 3D mesh를 출력하는 함수
- 입력과 출력 사이의 관계를 "model" 이라 함


<br>
<br>

### 3D 관절 회전

<br>

<p align=center><img src="./images/2/1.png" width=30%></p>

<br>

- **해당 관절의 부모 관절에 상대적인 3D 회전**을 하고 회전으로 인해 모든 자식 관절들 이동 
    - Human kinetic chain 존재 (joint의 종류마다 다름)
    - Pelvis는 root joint이고 hip, thorax는 pelvis의 child joint
        - Tree 구조
    - 팔꿈치의 3D 회전은 부모 관절인 어깨의 상대적인 3D 회전
    - 즉, 부모 관절의 회전, 위치에 상관없이 독립적으로 pose 모델링 가능
    - 팔꿈치를 움직이면 child node인 손목만 움직이고 팔꿈치를 포함한 부모 관절은 움직이지 않음
- 따라서 leaf node에 해당하는 관절들은 3D 회전이 정의 되어있지 않음
    - Leaf node는 마지막으로 존재하는 child node이고 이 그림에서는 wrist, ankle, head
    - child node가 없기 때 3D 회전이 정의되지 않음
- Root node에 해당하는 관절의 회전은 global rotation(전신의 3D 회전)에 해당

<br>
<br>


### 3D 길이 및 체형

<br>

<p align=center><img src="./images/2/2.png" width=40%></p>

<br>


- T-pose(zero 3D 관절 회전)를 취한 사람의 길이와 체형에 대한 파라미터
- PCA (Principal Component Analysis)를 통해 사람의 체형과 길이에 대한 latent space를 모델
- PCA의 coefficient를 beta로 사용


<br>
<br>

### 3D 길이 및 체형을 위한 PCA
- T-Pose를 취한 큰 규모의 3D scan dataset에 PCA 알고리즘 적용
- PCA를 통해 얻은 PC (Principal Component) 들은 사람 체형을 구분하는 가장 주된 기준 제시
- PCA components에 beta (weights)를 곱한 후 더해서 최종 t-posed 3D mesh를 획득

<br>

<p align=center><img src="./images/2/11.png"></p>

<br>

- PC1, 2가 가장 큰 영향을 줌
- PC 1
    - 기준이 키인 component
    - beta 값이 달라지면 키가 달라짐
- PC 2
    - 기준이 체형인 component
    - beta 값이 달라지면 체형이 달라짐

<br>
<br>


### 3D Human Model Bssic Process

<br>

<p align=center><img src="./images/2/12.png" width=60%></p>

<br>

- PCA를 통해 3D 사람의 길이 및 체형을 결정하고 T-posed 3D mesh 생성
- T-posed 3D mesh와 3D 관절 회전로부터 skining function을 적용하여 pose를 취한 특정 체형의 3D mesh 획득
- Skinning Function
    - 예를들어 10,000개의 vertex가 존재해도 관절들의 움직임, 즉 관절 좌표는 20개만 존재하기 때문에 이 관절들의 위치로부터 피부를 알아내야 함!
    - Skinning function은 관절들의 위치로부터 피부를 입히는 기능
    - 대표적인 알고리즘으로 **LBS(Linear Blend Skinning)**
    - Skinning function을 이해하기 위해서는 Skinning weight를 알아야 함

<br>
<br>

### Skinning Weight


<br>

<p align=center><img src="./images/2/3.png" width=20%></p>

<br>

- 각 mesh vertex가 각 관절에 영향을 받는 정도를 의미
    - 새끼손가락의 피부들은 대부분 새끼 손가락의 관절 부분 회전엔 가장 크게 영향을 받음
    - 팔꿈치 근처에 있는 vertex는 팔꿈치의 rotation에 가장 큰 영향을 받음

<br>

<p align=center><img src="./images/2/13.png" width=20%></p>

<br>

- V는 mesh의 수, J는 관절의 수를 의미함
- 3D human model 마다 3D artist가 미리 만들어 놓는 weight 존재
    - 그대로 쓰기도 하지만 데이터에 맞춰 optimize 하는 경우도 많음

<br>
<br>

### LBS (Linear Blend Skinning)
- Skinning function도 종류가 많고 그 중 가장 단순하고, 많이 사용되는 알고리즘은 LBS
- LBS는 모든 관절의 변형을 선형(Linear)으로 합쳐서(blend) 3D mesh를 얻는 알고리즘
<br>

<p align=center><img src="./images/2/14.png" width = 40%></p>

<br>

- 아주 단순해서 좋은 결과를 내기 힘들기 때문에 다양한 correction 방법들이 존재

<br>
<br>

### **LBS, Correction** 을 적용한 3D Human Model

<br>

<p align=center><img src="./images/2/15.png" width = 60%></p>

<br>

- LBS의 단점을 보완하기 위한 다양한 correction 방법 적용
- 대표적인 방법으로는 **pose-dependent correctives**
    - Pose를 취할 때 사람의 살이 중력 등의 여러 요소로 인해 변화하는데 이를 모델링해주어야 함
    - Skin이 조금 더 realistic 하게 표현 가능 
    - LBS는 선형 function 이고 자세를 바꿔주는 알고리즘이기 때문에 중력에 의한 변화를 반영 할 수 없음

<br>
<br>

### 3D 모델을 만들기 위해 필요한 요소
1. 3D 관절 회전을 정의하기 위한 **human Kinematic Chain** 
2. PCA를 통해 3D 길이 및 체형 space model을 만들기 위한 **여러 체형과 키를 가진 사람의 3D scan 필요**
    - PCA 알고리즘을 적용할 큰 규모의, 다양한 체형의 T-Pose 3D scan 데이터 셋 필요
    - 자세가 아닌 다양한 체형의 데이터가 필요
3. Pose-dependent corrective 를 하기 위한 **여러 포즈를 취한 사람의 3D scan 데이터** 필요
    - 그냥 3D scan 데이터를 얻는 것도 어렵지만 어려운 포즈 데이터를 얻는 것은 더 어려움
4. **Skinning weight**
    - 주로 3D artist가 만든 후 약간의 fine-tuning 

<br>
<br>

### 3D human model의 역할
- 3D 관절 회전과 3D 길이 및 체형 파라미터가 있다면 인간의 3D mesh 획득 가능
    - 정해진 캐릭터가 존재한다면 길이 및 체형에 대한 정보는 필요하지 않으며, 3D 관절 회전에 대한 정보만 필요
- 3D pose estimation 가능
    - 입력 이미지로부터 3D 관절 회전과 다른 파라미터들을 추정하는 것을 목표
    - 이 때의 입력 이미지는 2가지 종류가 존재
        - Marker-less MoCap
            - 스마트폰이나 카메라로 얻은 이미지를 이용하여 추정
        - Marker-based MoCap
            - 게임이나 영화에서는 고퀄리티의 영상이 필요하기 때문에 안전하게 고퀄리티의 정보를 얻기 위해서 사용
            - 카메라가 여러대 필요하고 marker를 부착해야 함
            - 부착한 marker의 위치를 추정하고 각도 얻어낸 다음 케릭터를 움직이게 함
            - 실생활에서 적용하여 사용할 수는 없음
    - 3D 관절 회전과 다른 파라미터들을 추정하여 human model에 적용하면 human mesh 획득도 가능 

<br>
<br>

## 3D Human Pose Estimation 분류
Forward 과정에서 3D human model이 있는지 없는지에 따라 달라짐
- Model-base

    <br>

    <p align=left><img src="./images/2/19.png" width=40%></p>

    <br>

    

    - 입력 이미지로부터 3D human model parameter($\theta$, $\beta$) 추정하는 방식
    - 추정된 parameter들을 3D human model의 입력으로 넣어 3D human mesh를 얻음

    <br>
    <br>

- Model-free

    <br>

    <p align=cetner><img src="./images/2/20.png" width=30%></p>

    <br>

    - 입력 이미지로부터 3D human mesh vertex들의 좌표들을 직접적으로 추정
    - 추정한 3D human mesh는 3D human model의 topology와 같다고 가정 
        - 좌표만 추정하고 mesh의 topology는 3D human model과 같다는 의미
        - 이로 인해 완전한 model-free는 아님

<br>
<br>

### Model-base 

<br>

<p align=center><img src="./images/2/19.png" width=60%></p>

<br>

- 3D 관절 회전을 통해 3D mesh를 얻는 과정은 human kinematic chain을 따라 3D relative rotation이 누적되어 적용 (LBS)
- 부모 joint의 3D rotation 에러가 자식 joint에 영향을 끼칠 수 있음
    - 즉, end point, leaf node에 error가 쌓여가고 이를 error accumulation이라고 함
- Error Accumulation 현상으로 인해 model-free에 의해 정확도가 낮을 수 있음
- 3D human body parameter를 얻으므로 다른 캐릭터로의 포즈 전송 등 여러 application에서 유용하게 사용될 수 있음 
- Model-based 에시
    - HMR (Human Mesh Recovery), Pose2Pose, HybrIK, ..


<br>
<br>

#### Model-based 학습
- In-the-wild image dataset 
    - 2D pose annotation만 존재, 3D pose annotation 존재하지 않음
    - 추정된 3D mesh를 입력 2d 이미지 공간으로 project 시킨 후, GT 2D pose와 loss를 계산
    - 관절에만 loss를 주어도 skinning function 덕분에 관절로부터 떨어져 있는 mesh vertex들도 loss에 영향을 받게 됨


    <br>

    <p align=left><img src="./images/2/21.png" width=60%></p>

    <br>


    1. 추정된 3D mesh에 joint regression matrix (3D human model에 정의)를 곱해서 3D 관절 좌표 획득
    2. 사람을 담고있는 입력 이미지를 촬영한 가상의 카메라를 설치 (즉, 사람 crop 이미지를 획득)하고 사람의 3D translation vector를  추정 
        - 3D translation vector
            - Regressor가 추정
            - 사람이 x, y, z축으로부터 어디에 있는지 
    3. 가상 카메라의 intrinsic parameter와 추정된 3D translation vector를 이용하여 3D 관절 좌표를 입력 2D 이미지 공간으로 project
        - Intrinsic parameter는 상수로 가정
            - 이 parameter 추정하면 가능한 경우의 수가 많아짐
    4. project된 2D 관절 좌표와 GT 2D pose 사이의 L1 loss 계산


<br>
<br>

- MoCap image dataset
    - 3D pose annotaation 제공
    - 추정된 3D mesh로부터 3D 관절 좌표를 얻고 GT 3D pose와 loss 계산
    - 관절에만 loss를 주어도 skinning function 덕분에 관절로부터 떨어져 있는 mesh vertex들도 loss에 영향을 받게 됨


    <br>

    <p align=left><img src="./images/2/22.png" width=60%></p>

    <br>

    1. 추정된 3D mesh에 joint regression matrix (3D human model에 정의)를 곱해서 3D 관절 좌표 획득
    2. 얻은 3D 관절 좌표와 GT 3D oise 사이의 L1 loss 계산


<br>
<br>

### Model-free

<br>

<p align=center><img src="./images/2/20.png" width=50%></p>

<br>

- 예를 들어, SMPL은 6980개의 vertex로 구성되어 있어 6980개의 vertex의 3D 좌표를 모두 추정해야 함 (6980 x  3)
- 관절과 다르게 vertex는 개수가 매우 많기 때문에 computational overhead 발생
- 그래서 추정된 mesh vertex들이 3D human model에서 얻는 것처럼 자연스럽게 이어지지 않을 수 있음

    <br>

    <p align=center><img src="./images/2/23.png" width=40%></p>

    <br>


- 3D human body parameter를 얻을 수 없기 때문에 다른 application에 적용할 수 없음
- Error accumulation 현상이 없기 때문에 정확도는 model-base 보다 높을 수 있음
- Model-free의 예시
    - I2L-MeshNet, Pose2Mesh, METRO, .. 


<br>
<br>


#### Model-free 학습
- In-the-wild image dataset, MoCap datset들에서 얻는 GT는 2D/3D의 관절 좌표
- Model-free 접근법들은 Skinning function으로 인해 관절에만 loss를 계산하면 되지만 model-free는 모든 vertex에 loss를 적용 

    <br>

    <p align=center><img src="./images/2/20.png" width=50%></p>

    <br>

    1. Pre-processing stage 에서 GT 2D/3D pose로부터 얻은 3D pseudo-GT mesh를 얻음
        - Simplify-X 혹은 NeuralAnnot
        - 3D pseudo-GT mesh는 model-based 에서 사용하여 정확도를 많이 올릴 수 있음 (Optional)
    2. 얻은 3D mesh와 pre-processing stage 단계에서 얻은 3D pseudo-GT mesh와 L1 loss를 계산하여 학습
        - 3D pseudo-GT는 model-based 학습때도 동일하게 사용
