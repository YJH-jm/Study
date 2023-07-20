# Generative Adversarial Networks
[논문](https://arxiv.org/pdf/1406.2661.pdf)

## prerequisite
### 확률 분포의 추정

<br>

<p align=center><img src="./images/1/1.jpg" width=50%></p>

<br>

- $x$가 학습 데이터의 내의 실제 이미지면 이는 (64 x 64 x 3) 과 같은 차원을 걎는 고차원 벡터로 표현 가능

<br>

<hr>


<br>

<p align=center><img src="./images/1/2.png" width=40%></p>

<br>

- $x$ 값들은 하나하나가 각 이미지를 의미
- 우리는 실제 이미지들의 정확한 분포 필요

<br>

- 사람 얼굴 이미지 예시

    <br>

    <p align=left><img src="./images/1/3.png" width=40%></p>

    <br>

    - 이 데이터 셋은 안경을 쓴 남성 이미지들을 몇몇 포함하고 있을 수 있음
    
    <br>
    
    <p align=left><img src="./images/1/4.png" width=40%></p>

    <br>
    
    - 이 데이터 셋은 검은 머리 여성의 이미지들을 포함할 수 있음

    <br>
    
    <p align=left><img src="./images/1/5.png" width=40%></p>

    <br>
    
    - 이 데이터 셋은 금발 머리 여성의 이미지들을 다수 포함할 수 있음

    <br>
    
    <p align=left><img src="./images/1/6.png" width=40%></p>

    <br>
    
    - 이 데이터 셋은 이상한 이미지들을 포함하고 있을 수 있음

<br>

<hr>

<br>

- 생성 모델은 $p_{data}(x)$를 잘 근사하는 $p_{model}$ 을 찾는 것이 목표
    - $p_{data}(x)$ : 실제 이미지들의 분포
    - $p_{model}$ : 모델이 생성한 이미지들의 분포

    <br>

    <p align=center><img src="./images/1/7.png" width=40%></p>

    <br>

    - $p_{model}$를 실제 이미지들의 분포인 $p_{data}$와 비슷하게 추정해야 다양하고 정확한 sample 이미지를 얻을 수 있음
    - 잘못 추정을 하면 실제로 존재하지 않는 이미지나 이상한한 이미지가 나올 확률이 높음
    - 즉, 확률 분포 모델도 선정을 잘 해야함

<br>
<br>

## VAE와의 차이
- 복잡하고 고차원인 학습 분포로부터 데이터를 샘플링하고자 하는데 이를 직접적으로 하는 것은 불가능
    - 확률 분폰의 추정 자체가 고차원 공간에서는 불안정하고 힘듦
- 이를 해결하기 위해 쉽게 데이터를 샘플링하는 것이 간단한 분포를 이용
    - Gaussian distribution 과 같은
- 이 간단한 분포를 학습 분포로 변형 (Transfromation) 하는 방법을 학습

<br>

- VAE
    - 이미지를 학습한 encoder로부터 얻어진 $\mu, \sigma$ 를 통하여 latent vector $z$를 결정
    - latent vector가 decoder를 통과하여 입력으로 넣어준 이미지 데이터를 잘 복원하는 형태로 학습
- GAN
    - 학습 때 부터 표준정규분포에서 랜덤하게 얻은 vector $z$ 를 자유롭게 입력으로 넣어줌
    - $z$ 가 **Generator (Decoder)** 를 통과하여 나온 데이터는 학습 데이터에 한 이미지에 대응된다고 보기 어려움
    - 학습 이미지를 복원하는 방법으로 학습이 불가능
    - 이를 해결하기 위해 **Discriminator** 라는 네트워크를 하나 더 도입하여  생성된 이미지가 데이터 분포 내에 속하는지 판단하도록 함


<br>

## GAN 학습 

<br>

<p align=center><img src="./images/1/8.png" width=40%></p>

<br>

- **Discriminator**
    - 진짜 이미지와 가짜 이미지를 구분
    - 즉, Generator로 생성된 데이터가 실제 데이터 (학습 데이터) 가 가지고있는 참값의 데이터 분포에 속하고 있는지 판단
- **Generator**
    - 진짜처럼 보이는 이미지를 생성하여 discriminator를 속임
    - 표준 정규분포로 random 하게 얻은 $z$ 값이 입력으로 들어감


<br>

- VAE와 다르게 생성된 이미지 각각의 pixel이 어떤 값이어야 한다는 기준 없기 때문에 loss를 구할 수 없음
- 이 문제를 해결하기 위해서 discriminator 통과하여 간접적으로 generator를 학습
- 학습이 끝나면 generator의 network를 이용하여 새로운 데이터 생성


<br>
<br>

### Discriminator 학습
Generator가 생성한 이미지와 real 이미지를 잘 분류하기 위해 학습



<br>

<p align=center><img src="./images/1/9.png" width=20%></p>

<br>

- 진짜 이미지 $x$ 가 Discriminator의 입력으로 들어가면 이 이미지가 진짜인지 가짜인지 판단하는 binary classification 진행
    - Sigmoid 함수를 통과하여 하나의 scalar 값 생성
- 출력값이 1에 가까울 수록 real image, 0에 가까울수록 generator로 생성된 fake image 일 확률 높음
- 따라서 $D(x)$ 값이 최대한 1과 가까워야 함
    - 이렇게 학습

<br>
<br>

### Generator 학습
Discriminator를 속이기 위해 진짜와 구분하기 어려운 이미지를 생성하기 위해 학습

<br>

<p align=center><img src="./images/1/10.png" width=40%></p>

<br>

- 표준 정규 분포에서 sampling 하여 latent vector (code)를 생성하여 generator의 입력으로 사용
- Generator는 이 값으로 입력 이미지와 같은 사이즈의 가짜 이미지 생성
- 생성된 이미지를 Discrimator의 입력으로 사용하여 진짜 이미지인지 가짜 이미지인지 판단
- Generator의 입장에서는 Discriminator가 1에 가깝게, 즉 진짜이미지로 판단하도록 학습

<br>
<br>

### Loss Function
