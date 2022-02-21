# Autoencoder 개요
1. Unsupervised learning
    - 오토인코더 학습 할 때 unsupervised learning 
2. Manifold learning
    - 학습된 오토인코더에서 인코더는 차원 축소 역할 수행
3. Generative model learning
    - 학습된 오토인코더에서 디코더는 생성 모델의 역할 수행
4. ML density estimation
    - 오토인코더 학습 할 때 losss는 negative ML

<br>
<br>

# Autoencoder를 위한 기본 내용 학습
## Deep Learning 

<br>

<p align=center><img src="images/image141.PNG" width=40%></p>
<p align=center><a href="https://www.slideshare.net/NaverEngineering/ss-96581209">출처</a></p>

### 1. 데이터 수집

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?\begin{matrix}x={x_{1},&space;x_{2},&space;...,&space;x_{n}}&space;\\y={y_{1},&space;y_{2},&space;...,&space;y_{n}}&space;\\D=\{(x_{1},y_{1}),&space;(x_{2},y_{2}),&space;(x_{3},y_{3})\}\end{matrix}" title="\begin{matrix}x={x_{1}, x_{2}, ..., x_{n}} \\y={y_{1}, y_{2}, ..., y_{n}} \\D=\{(x_{1},y_{1}), (x_{2},y_{2}), (x_{3},y_{3})\}\end{matrix}" /></p>

<br>

### 2. 모델의 종류와 손실 함수 정의
- 학습 모델

    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?f_{\theta}(x)" title="f_{\theta}(x)" /></p>
    <p align=center><img src="https://latex.codecogs.com/svg.image?\inline&space;\theta" title="\inline \theta" /> &nbsp;: 학습 파라미터 ( <img src="https://latex.codecogs.com/svg.image?\inline&space;W,b" title="\inline W,b" /> )</p>

    <br>

- 손실 함수
    - Backpropagation을 사용하기 위한 2가지 가정
        1. 전체 training data에 대한 loss의 합은 각 edata에 대한 loss의 합과 같음
        2. 손실함수를 구성할 때 network의 출력값과 정답값만 사용
    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?L(f_{\theta}(x),y)&space;" title="L(f_{\theta}(x),y) " /></p>

    <br>


### 3. 학습
- 최적의 파라미터 찾아 주어진 데이터를 가장 잘 설명하는 모델 찾기

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?\theta^{*}=\underset{\theta}{argmin}L(f_{\theta}(x),y)" title="\theta^{*}=\underset{\theta}{argmin}L(f_{\theta}(x),y)" /></p>

<br>

- Gradient Descent 방식을 기본으로 하는 알고리즘 사용

    <br>

    <p align=center><img src="images/image142.PNG" width=40%></p>
    <p align=center><a href="https://www.slideshare.net/NaverEngineering/ss-96581209">출처</a></p>


### 4. 예측

<br>
<br>

## Backpropagation 관점 해석

<br>
<br>

## Maximum likelihood 관점 해석

<br>
<br>

# Manifold Learning
## Manifold
- 고차원 데이터가 있을 때, 이 데이터를 데이터 공간에 배치하면 이 데이터들을 잘 아우르는 subspace를 **Manifold**  라고 함

<br>

<p align=center><img src="images/image143.PNG" width=40%></p>
<p align=center><a href="https://www.slideshare.net/NaverEngineering/ss-96581209">출처</a></p>

<br>
<br>

## Manifold Hypothesis
1. 고차원의 데이터 밀도는 낮지만, 이들의 집합을 포함하는 저차원 manifold 존재
2. 저차원 manifold를 벗어나면 급격하게 데이터의 밀도 낮아짐

<br>
<br>


## Manifold 역할
- Data compression
- Data visualization
    - Data intuition, 해석, ...
- Curse of dimensionality 극복
    - Curse of dimensionality
        
        <br>

        <p align=center><img src="images/image144.jpg" width=40%></p>
        <p align=center><a href="https://www.slideshare.net/NaverEngineering/ss-96581209">출처</a></p>
        
        <br>
        
        - 1차원 10개의 공간에 8개의 데이터가 있다고 할 때, 2차원으로 늘리면 100개의 공간에 8개의 데이터가 존재하고 3차원으로 늘리면 1000개의 공간에 8개의 데이터가 존재함
        - 즉, 동일한 개수의 데이터의 밀도가 감소
        - 차원을 늘리면 동일한 데이터의 밀도가 떨어지고 모델 prediction이 제대로 되지 않음
        - 고차원에서 제대로 prediction 하기 위해서는 매우 많은 수의 데이터 필요
- Discovering most import features
    - 고차원의 데이터를 잘 표현하는 manifold를 이용해 데이터의 특징 파악 가능


<br>
<br>


# Autoencoder
## Basic Autoencoder (AE) 

<br>

<p align=center><img src="images/image145.PNG" width=40%></p>
<p align=center><a href="https://www.slideshare.net/NaverEngineering/ss-96581209">출처</a></p>

<br>

- input과 output이 같은 구조 
- 보통은 가운데 차원이 줄어드는 형태
    - 초반에는 차원이 늘어나는 sparse autoencoder, 지금은 거의 사용하지 않음
- Bottleneck Hidden layer
    - Latent Variable, Feature, Hidden representation, .. 등과 같은 표현

<br>
<br>

### Autoencoder 수식

<br>

<p align=center><img src="images/image146.PNG" width=40%></p>

<br>

- Input은 같은 크기의 output을 생성
    
    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?x,y\in\mathbb{R}^{d}" title="x,y\in\mathbb{R}^{d}" /></p>
    <p align=center><img src="https://latex.codecogs.com/svg.image?\inline&space;x" title="\inline x" /> &nbsp;: input</p>
    <p align=center><img src="https://latex.codecogs.com/svg.image?\inline&space;y" title="\inline y" /> &nbsp;: ourput</p>


    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?z=h(x)\in\mathbb{R}^{d_{z}}" title="z=h(x)\in\mathbb{R}^{d_{z}}" /></p>
    <p align=center><img src="https://latex.codecogs.com/svg.image?\inline&space;z" title="\inline z" /> &nbsp;: latent variable</p>
    <p align=center><img src="https://latex.codecogs.com/svg.image?y=g(z)=g(h(x))" title="y=g(z)=g(h(x))" /></p>


    <br>


- AE의 Loss를 Reconstruction error 라고도 함
    - 네트워크 input, output 값 이용

    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?L(x,y)" title="L(x,y)" /></p>
    <p align=center><img src="https://latex.codecogs.com/svg.image?L_{AE}=\sum_{x\in&space;D}L(x,&space;y)" title="L_{AE}=\sum_{x\in D}L(x, y)" /></p>
    <br>

    - 이미 정답인 값을 알고있기 때문에 unsupervised learning에서 **sunpervised learning, self learning**으로 문제를 바꾸어 해결 가능
        - 차원 축소가 얼마나 잘 이루어졌는지 확인이 가능해짐


<be>

- 보통은 학습이 끝나면 encoder와 decoder를 분리하여 사용
- Decoder가 최소한 학습 데이터는 생성해 낼 수 있음 
    - 생성된 데이터가 학습 데이터와 비슷한 양상을 가짐
    - 최소한의 성능 보장
- Encoder가 최소한 학습 데이터는 latent vector로 표현을 잘 할 수 있음
    - 데이터 추상화 가능

<br>
<br>

### Linear Autoencoder
- Hidden layer를 activation function 없이 사용

<br>

<p align=center><img src="images/image147.PNG" width=40%></p>

<br>

- Hidden layer 1개, layer간 fully-connected로 연결 

    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?h(x)=W_{e}x&plus;b_{e}" title="h(x)=W_{e}x+b_{e}" /></p>
    <p align=center><img src="https://latex.codecogs.com/svg.image?g(h(x))=W_{d}z&plus;b_{d}" title="g(h(x))=W_{d}z+b_{d}" /></p>
    <p align=center><img src="https://latex.codecogs.com/svg.image?L(x,y)=||x-y||^{2}&space;\;\;&space;or&space;\;\;&space;cross-entropy&space;" title="L(x,y)=||x-y||^{2} \;\; or \;\; cross-entropy " /></p>

    <br>

    - Loss function으로 MSE 사용하는 경우 PCA와 같은 manifold 학습

<br>
<br>

### SAE (Stacking AutoEncoder)
- 초기의 autoencoder는 네트워크 파라미터 초기화에도 많이 사용
    - 즉, pretraining 하는데 많이 사용
    - 지금은 거의 사용하지 않음

<br>

1. MNIST 데이터를 분류하기 위한 네트워크 구성

    <br>

    <p align=center><img src="images/image148.png" width=20%></p>
    <p align=center><a href="http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/auto.pdf">출처</a></p>

    <br>

2. 1000개의 weight를 가진 layer를 지나 다시 input 복원하는 과정에서 데이터의 특징을 가지고 있는 weight를 학습


    <br>

    <p align=center><img src="images/image149.png" width=40%></p>
    <p align=center><a href="http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/auto.pdf">출처</a></p>


    <br>

3. 이 weight들을 이용하여 초기 파라미터 설정하고 다른 layer들도 같은 방식으로 반복

    <br>

    <p align=center><img src="images/image150.png" width=40%></p>
    <p align=center><a href="http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/auto.pdf">출처</a></p>


    <br>

    <p align=center><img src="images/image151.png" width=40%></p>
    <p align=center><a href="http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/auto.pdf">출처</a></p>


4. 마지막 layer의 weight들은 렌덤하게 초기화 한 후 backpropagation을 통해 파라미터 학습

    <br>

    <p align=center><img src="images/image152.png" width=40%></p>
    <p align=center><a href="http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/auto.pdf">출처</a></p>

    <br>

## DAE (Denosing AutoEncoder)

<br>

<p align=center><img src="images/image154.PNG" width=40%></p>

<br>

- 기존의 AE의 input에 확률적 맵핑을 시켜 noise를 추가한 새로운 input 생성

    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?\widetilde{x}&space;\sim&space;q_{D}(\widetilde{x}|x)&space;" title="\widetilde{x} \sim q_{D}(\widetilde{x}|x) " /></p>
    
    <br>

    - 사람이 봤을 때 의미적으로 벗어나지 않을 만큼의 noise 추가
    - 다양한 방법으로 noise를 추가하나 이 논문에서는 noise에 해당하는 위치의 pixel울 0으로 바꿈
        - Extracting and Composing Robust Features with Denoising Autoencoders(2018)
    
    <br>
    
    <p align=center><img src="images/image153.PNG" width=40%></p>
    <p align=center><a href="http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/auto.pdf">출처</a></p>

    <br>

- Loss는 **noise가 추가되기 전의 데이터**와 **noise를 추가한 데이터가 DAE를 통과한 후의 output**을 이용

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?L_{DAE}(x,y)=\sum_{x\in&space;D}&space;E_{q(\widetilde{x}|x)}[L(x,g(h(\widetilde{x})))]\approx&space;\sum_{x\in&space;D}\frac{1}{L}\sum&space;_{i=1}^{L}L(x,g(h(\widetilde{x}_{i})))" title="L_{DAE}(x,y)=\sum_{x\in D} E_{q(\widetilde{x}|x)}[L(x,g(h(\widetilde{x})))]\approx \sum_{x\in D}\frac{1}{L}\sum _{i=1}^{L}L(x,g(h(\widetilde{x}_{i})))" /></p>

<br>

- 즉, 학습된 Network는 noise가 추가된 데이터를 넣으면 noise가 제거된 데이터가 output으로 나오므로 denoise 됨

<br>

### Manifold Learning 관점

<br>

<p align=center><img src="images/image155.PNG" width=40%></p>
<p align=center><a href="https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf">출처</a></p>

<br>

- 하나의 데이터에 여러개의 noise를 추가한 데이터들은 의미적으로는 같은 sample
- 이 모든 sample들은 같은 manifold 공간에 맵핑이 되어야 함
- 그러므로, Decoder를 통해 복원되는 데이터는 noise가 제거된, 즉 noise를 추가하기 전의 데이터 하나


### SDAE ((Stacking Denoising AutoEncoder))
- Weight를 초기화하기 위해 pretrain하는 과정을 SAE대신 SDAE를 사용한 방법 
    - DAE를 제외하고는 위의 설명과 동일

<br>

<p align=center><img src="images/image156.PNG" width=40%></p>
<p align=center><a href="https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf">출처</a></p>

<br>

- Noise를 추가했더니 점점 더 강한 edge detector가 나옴

<br>
<br>

## SCAE (Stochastic Contractive AutoEncoder)
- AE의 loss 에 regularization을 추가하면 DAE와 비슷하거나 더 좋은 효과를 낼 수 있음
- DAE의 의미를 수식적으로 표현했다고 볼 수 있음

<br>

- Loss Function

    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?L_{SCAE}=\sum_{x\in&space;D}L(x,g(h(x)))&plus;\lambda&space;E_{q(\tilde{x}|x)}[{||h(x)-h(\tilde{x})||}^2]" title="L_{SCAE}=\sum_{x\in D}L(x,g(h(x)))+\lambda E_{q(\tilde{x}|x)}[{||h(x)-h(\tilde{x})||}^2]" /></p>

    <br>

    - AE의 reconstruction error

        <br>

        <p align=center><img src="https://latex.codecogs.com/svg.image?\sum_{x\in&space;D}L(x,g(h(x)))" title="\sum_{x\in D}L(x,g(h(x)))" /></p>
        
        <br>
    
    - Stochastic regularization
        - Manifold 상에서 위치를 같게 만들어주고 싶기 때문에 이 식 반영

        <br>

        <p align=center><img src="https://latex.codecogs.com/svg.image?E_{q(\tilde{x}|x)}[{||h(x)-h(\tilde{x})||}^2]" title="E_{q(\tilde{x}|x)}[{||h(x)-h(\tilde{x})||}^2]" /></p>
        
        <br>

<br>
<br>

## CAE ( Contractive AutoEncoder )
- SCAE의 stochastic 한 식을 deterministic 한 형태로 바꿈
- Noise
    
    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?E_{q(\tilde{x}|x)}[{||h(x)-h(\tilde{x})||}^2]" title="E_{q(\tilde{x}|x)}[{||h(x)-h(\tilde{x})||}^2]" /></p>
        
    <br>

<br>
<br>

## VAE (Variational AutoEncoder)
- AutoEncoder는 manifold learning이 목적
    - Encoder를 self supervised learning으로 학습하기 위해 decoder를 이용
    - 주 목적은 encoder
- VAE는 generative model로 데이터 생성이 목적
    - Decoder로 데이터를 만들기 위해 앞단인 encoder가 붙인 것

<br>

### VAE 전체 흐름
- 이 부분 먼저 봐야 나머지 흐름 이해감

<br>

<p align=center><img src="images/image158.png" width=40%></p>
<p align=center><a href="https://taeu.github.io/paper/deeplearning-paper-vae/">출처</a></p>

<br>

- AE는 encoder를 통과한 후 바로 latent space 생성
- VAE는 encoder를 거치면 latent variable을 생성하기 전에 output으로 2개의 vector 생성
    - 평균, 표준편차
- 평균과 표준편차로 normal distribution을 생성하고 여기서 값을 sampling 하여 latenet variable 만듦
- Sampling 하는 과정에서 reparameterization trick 사용
    - 이 과정이 있어야 backpropagation이 가능
    - 미분이 가능하게 바꿔주는 과정

<br>

### VAE 학습

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?-\sum_{j=1}^{D}x_{i,j}log{p_{i,j}}&plus;(1-x_{i,j})log(1-p_{i,j})" title="-\sum_{j=1}^{D}x_{i,j}log{p_{i,j}}+(1-x_{i,j})log(1-p_{i,j})" /></p>

<br>

- 우리가 궁극적으로 알고싶은 것은 &nbsp; <img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}(x)" title="\inline p_{\theta}(x)" />
    -  x를 우리가 가지고 있는 데이터라고 한다면 그 trainng 데이터의 likelihood를 최대화하고 싶음
        - 내가 가지고 있는 x가 나올 확률이 가장 큰 distribution을 찾아야 함
    - 아래 식을 최대화


<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?\frac{p_{\theta}(x,z)}{p_{\theta}(z)}=&space;p_{\theta}(x|z)dz" title="\frac{p_{\theta}(x,z)}{p_{\theta}(z)}= p_{\theta}(x|z)dz" /></p>
<p align=center><img src="https://latex.codecogs.com/svg.image?\int&space;p_{\theta}(x,z)dz=p_{\theta}(z)" title="\int p_{\theta}(x,z)dz=p_{\theta}(z)" /></p>
<p align=center><img src="https://latex.codecogs.com/svg.image?p_{\theta}(x)=\int&space;p_{\theta}(z)p_{\theta}(x|z)dz" title="p_{\theta}(x)=\int p_{\theta}(z)p_{\theta}(x|z)dz" /></p>
<p align=center><img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}(z)" title="\inline p_{\theta}(z)" /> &nbsp;: Simple gaussian prior</p>
<p align=center> <img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}(x|z)" title="\inline p_{\theta}(x|z)" /> &nbsp;: Decoder neural network</p>

<br>

- <img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}(z)" title="\inline p_{\theta}(z)" /> 는 gaussian 분포를 따른다고 가정하므로 알 수 있음
- <img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}(x|z)" title="\inline p_{\theta}(x|z)" /> 는 decoder 이기 때문에 신경망으로 구성할 수 있음
- 하지만 모든 &nbsp;<img src="https://latex.codecogs.com/svg.image?\inline&space;z" title="\inline z" /> 에 대해서 <img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}(x|z)" title="\inline p_{\theta}(x|z)" /> 를 적분하는 것은 어려움

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?\begin{matrix}p_{\theta}(z|x)p_{\theta}(x)=p_{\theta}(x|z)p_{\theta}(z)\\p_{\theta}(z|x)=p_{\theta}(x|z)p_{\theta}(z)/p_{\theta}(x)\end{matrix}&space;" title="\begin{matrix}p_{\theta}(z|x)p_{\theta}(x)=p_{\theta}(x|z)p_{\theta}(z)\\p_{\theta}(z|x)=p_{\theta}(x|z)p_{\theta}(z)/p_{\theta}(x)\end{matrix} " /></p>

<br>

- 반대의 경우를 생각해보아도 &nbsp;<img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}(x)" title="\inline p_{\theta}(x)" />가 존재하기 때문에 불가능

<br>

- <img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}(x|z)" title="\inline p_{\theta}(x|z)" />&nbsp; 모델링 문제를 해결하기 위해서 **encoder**를 구성

<br>

<p align=center><img src="images/image159.PNG" width=20%></p>

<br>

- <img src="https://latex.codecogs.com/svg.image?\inline&space;q_{\phi}(z|x)" title="\inline q_{\phi}(z|x)" />는 &nbsp;<img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}(z|x)" title="\inline p_{\theta}(z|x)" /> 를 가장 근사화하는 네트워크 

<br>

- Encoder를 덧붙여 학습

<p align=center><img src="https://latex.codecogs.com/svg.image?log{p_{\theta}(x^{i})}=E_{z\sim&space;q_{\phi}(z|x^{i})}[logp_{\theta}(x^{i})]" title="log{p_{\theta}(x^{i})}=E_{z\sim q_{\phi}(z|x^{i})}[logp_{\theta}(x^{i})]" /></p>

<br>

- <img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}(x)" title="\inline p_{\theta}(x)" /> 를 최대화 시키는 것이 목적이기 때문에 이 값에 log를 씌움
- 그리고 기댓값의 형태로 나타냄

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?\begin{matrix}log{p_{\theta}(x^{(i)})}=E_{z\sim&space;q_{\phi}(z|x^{(i)})}[logp_{\theta}(x^{(i)})]&space;\\=&space;E_{z}[log\frac{p_{\theta}(x^{(i)}|z)p_{\theta}{(z)}}{p_{\theta}(z|x^{(i)})}]&space;\\=&space;E_{z}[log\frac{p_{\theta}(x^{(i)}|z)p_{\theta}{(z)}}{p_{\theta}(z|x^{(i)})}&space;\frac{q_{\phi}(z|x^{(i)})}{q_{\phi}(z|x^{(i)})}]&space;\\=E_{z}[log{p_{\theta}(x^{(i)}|z)}]-E_{z}[log\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}{(z)}})]&plus;E_{z}[log\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z|x^{(i)})}]\\\end{matrix}&space;" title="\begin{matrix}log{p_{\theta}(x^{(i)})}=E_{z\sim q_{\phi}(z|x^{(i)})}[logp_{\theta}(x^{(i)})] \\= E_{z}[log\frac{p_{\theta}(x^{(i)}|z)p_{\theta}{(z)}}{p_{\theta}(z|x^{(i)})}] \\= E_{z}[log\frac{p_{\theta}(x^{(i)}|z)p_{\theta}{(z)}}{p_{\theta}(z|x^{(i)})} \frac{q_{\phi}(z|x^{(i)})}{q_{\phi}(z|x^{(i)})}] \\=E_{z}[log{p_{\theta}(x^{(i)}|z)}]-E_{z}[log\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}{(z)}})]+E_{z}[log\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z|x^{(i)})}]\\\end{matrix} " /></p>

<br>

- Bayes' rule과 log 공식을 이용하여 식을 정리

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?\begin{matrix}log{p_{\theta}(x^{(i)})}=E_{z}\begin{bmatrix}log{p_{\theta}(x^{(i)}|z)}\end{bmatrix}-D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}{(z)}\end{pmatrix}&plus;D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})\end{pmatrix}\end{matrix}" title="\begin{matrix}log{p_{\theta}(x^{(i)})}=E_{z}\begin{bmatrix}log{p_{\theta}(x^{(i)}|z)}\end{bmatrix}-D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}{(z)}\end{pmatrix}+D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})\end{pmatrix}\end{matrix}" /></p>

<br>

- Expectation 개념을 이용해 적분으로 변환
    - <img src="https://latex.codecogs.com/svg.image?\inline&space;E_{z\sim&space;q_{\phi}(z|x^{i})}\begin{bmatrix}{log\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z)}}\end{bmatrix}=\int_{z}log\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z)}q_{\phi}(z|x^{i})dz" title="\inline E_{z\sim q_{\phi}(z|x^{i})}\begin{bmatrix}{log\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z)}}\end{bmatrix}=\int_{z}log\frac{q_{\phi}(z|x^{(i)})}{p_{\theta}(z)}q_{\phi}(z|x^{i})dz" />
- KL divergence를 이용하여 변환
    - <img src="https://latex.codecogs.com/svg.image?\inline&space;KL(P||Q)=\sum_{x}&space;P(x)log\frac{P(x)}{Q(x)}" title="\inline KL(P||Q)=\sum_{x} P(x)log\frac{P(x)}{Q(x)}" />
    - KL divergence를 이용하면 두 확률분포의 차이(거리)를 계산
- 즉, 변형된 위의 식을 최대화 해야함

<br>

- <img src="https://latex.codecogs.com/svg.image?\inline&space;D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}{(z)}\end{pmatrix}" title="\inline D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}{(z)}\end{pmatrix}" />
    - Encoder를 통과한 확률분포가 &nbsp;<img src="https://latex.codecogs.com/svg.image?\inline&space;z" title="\inline z" /> 의 확률분포와 같아야 함
- <img src="https://latex.codecogs.com/svg.image?\inline&space;D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})\end{pmatrix}" title="\inline D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})\end{pmatrix}" />
    - <img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}(z|x^{(i)})" title="\inline p_{\theta}(z|x^{(i)})" /> 는 우리가 알 수 없으므로 계산을 할 수 없음
    - 다만 KL divergence는 차이이기 때문에 항상 0 보다 크거나 같음을 알 수 있음

    <br>
    <p align=center><img src="https://latex.codecogs.com/svg.image?D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})\end{pmatrix}\geq&space;0" title="D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})\end{pmatrix}\geq 0" /></p>

<br>

- Tractable lower bound
    <p align=center><img src="https://latex.codecogs.com/svg.image?L(x^{(i)},\theta,\phi)=E_{z}\begin{bmatrix}log{p_{\theta}(x^{(i)}|z)}\end{bmatrix}-D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}{(z)}\end{pmatrix}" title="L(x^{(i)},\theta,\phi)=E_{z}\begin{bmatrix}log{p_{\theta}(x^{(i)}|z)}\end{bmatrix}-D_{kL}\begin{pmatrix}q_{\phi}(z|x^{(i)})||p_{\theta}{(z)}\end{pmatrix}" /></p>

- **ELBO (Evidence LowerBOund)**
    - Variational lower bound
    - 우리가 최적화 시켜야 하는 부분
        - 
    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?log{p_{\theta}\begin{pmatrix}x^{(i)}\end{pmatrix}}\geq&space;L(x^{(i)},\theta,\phi)&space;" title="log{p_{\theta}\begin{pmatrix}x^{(i)}\end{pmatrix}}\geq L(x^{(i)},\theta,\phi) " /></p>

    <p align=center><img src="https://latex.codecogs.com/svg.image?\theta^{*},&space;\phi^{*}=\underset{\theta,\phi}{argmax}\sum_{i=1}^{N}L(x^{(i)},\theta,\phi)" title="\theta^{*}, \phi^{*}=\underset{\theta,\phi}{argmax}\sum_{i=1}^{N}L(x^{(i)},\theta,\phi)" /></p>

<br>


### VAE loss function

<br>

<p align=center><img src="images/image160.PNG" width=40%></p>

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?\underset{\theta,\phi}{argmin}\sum_{i=1}-E_{q_{\phi}(z|x_{i})}\begin{bmatrix}log\begin{pmatrix}{p(x_{i}|g_{\theta}(z)}\end{bmatrix}\end{bmatrix}&plus;D_{kL}\begin{pmatrix}q_{\phi}(z|x_{i})||p{(z)}\end{pmatrix}" title="\underset{\theta,\phi}{argmin}\sum_{i=1}-E_{q_{\phi}(z|x_{i})}\begin{bmatrix}log\begin{pmatrix}{p(x_{i}|g_{\theta}(z)}\end{bmatrix}\end{bmatrix}+D_{kL}\begin{pmatrix}q_{\phi}(z|x_{i})||p{(z)}\end{pmatrix}" /></p>
<p align=center><img src="https://latex.codecogs.com/svg.image?\inline&space;cf:p\begin{pmatrix}x|g_{\theta}(z)\end{pmatrix}=p_{\theta}(x|z)" title="\inline cf:p\begin{pmatrix}x|g_{\theta}(z)\end{pmatrix}=p_{\theta}(x|z)" /></p>

<br>

 <br>

#### Regularization
- Assumption
    1. Encoder를 통과해서 나오는 distribution이 diagonal covariance를 가진다고 가정
    
    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?q_{\phi}(z|x_{i})\sim&space;N(\mu_{i},\sigma_{i}^{2}I)" title="q_{\phi}(z|x_{i})\sim N(\mu_{i},\sigma_{i}^{2}I)" /></p>

    <br>

    2. 실제 z에 대한 distribution은 normal distribution을 따름
    
    <br>

    <p align=center><img src="https://latex.codecogs.com/svg.image?p(z)\sim&space;N(0,I)" title="p(z)\sim N(0,I)" /></p>

    <br>

- 둘을 같게 만들어주어여 함
    - 즉, Encoder를 통과한 값이 항상 normal distribution을 따르도록 만듦 
    - Encoder를 통과하는 확률 분포와 정규분포와의 거리가 최소화되도록 함
    - KL divergence 를 최소화

<br>

- 식을 정리하면 다음과 같은 식으로 정리가 됨

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?\frac{1}{2}\sum_{j=1}^{J}(\mu_{i,j}^{2}&plus;\sigma_{i,j}^2-ln(\sigma_{i,j}^2)-1)" title="\frac{1}{2}\sum_{j=1}^{J}(\mu_{i,j}^{2}+\sigma_{i,j}^2-ln(\sigma_{i,j}^2)-1)" /></p>

<br>


#### Reconstruction error
- Input이 그대로 복원될 수  있도록 하는 역할
- 현재 sampling 용 함수에 대한 negative log likelihood

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?\begin{matrix}E_{q_{\phi}(z|x_{i})}\begin{bmatrix}log\begin{pmatrix}{p(x_{i}|z}\end{pmatrix}\end{bmatrix}=\int&space;log\begin{pmatrix}{p(x_{i}|z}\end{pmatrix}q_{\phi}(z|x_{i})dz\\Monte-carlo&space;technique&space;\approx&space;\frac{1}{L}\sum_{z^{i,l}}log\begin{pmatrix}{p(x_{i}|z^{i,l})}\end{pmatrix}\end{matrix}" title="\begin{matrix}E_{q_{\phi}(z|x_{i})}\begin{bmatrix}log\begin{pmatrix}{p(x_{i}|z}\end{pmatrix}\end{bmatrix}=\int log\begin{pmatrix}{p(x_{i}|z}\end{pmatrix}q_{\phi}(z|x_{i})dz\\Monte-carlo technique \approx \frac{1}{L}\sum_{z^{i,l}}log\begin{pmatrix}{p(x_{i}|z^{i,l})}\end{pmatrix}\end{matrix}" /></p>

<br>

- Monte-carlo technique 이용
    - 무한개, 혹은 무수히 많은 수의 sampling을 해서 평균을 내면 전체에 대한 기댓값과 거의 동일해짐

    
<br>

<p align=center><img src="images/image161.PNG" width=40% /></p>

<br>

- Deep learning에서 이 방법을 쓰기에는 계산량이 너무 많음
    - 그래서 L을 1로 가정
    - 또한, sampling을 reparameterization trick 방식으로 진행

<p align=center><img src="https://latex.codecogs.com/svg.image?\begin{matrix}\frac{1}{L}\sum_{z^{i,l}}log\begin{pmatrix}{p(x_{i}|z^{i,l})}\end{pmatrix}\approx&space;log\begin{pmatrix}{p(x_{i}|z^{i})}\end{pmatrix}\end{matrix}" title="\begin{matrix}\frac{1}{L}\sum_{z^{i,l}}log\begin{pmatrix}{p(x_{i}|z^{i,l})}\end{pmatrix}\approx log\begin{pmatrix}{p(x_{i}|z^{i})}\end{pmatrix}\end{matrix}" /></p>

 <br>

- Assumption 
    3. <img src="https://latex.codecogs.com/svg.image?\inline&space;p_{\theta}" title="\inline p_{\theta}" /> 가 bernoulli distribution나 gaussian distribution를 따른다고 가정
    - Bernoulli로 가정하면 Cross entropy 식으로 바뀜

        <br>

        <p align=center><img src="https://latex.codecogs.com/svg.image?\begin{matrix}log\begin{pmatrix}{p_{\theta}(x_{i}|z^{i})}\end{pmatrix}=log\prod_{j=1}^{D}p_{\theta}(x_{i,j}|z^{i})\\=\sum_{j=1}^{D}logp_{\theta}(x_{i,j}|z^{i})\\=\sum_{j=1}^{D}logp_{i,j}^{x_i,j}(q-p_{i,j})^{1-x_{i,j}}\\=\sum_{j=1}^{D}x_{i,j}logp_{i,j}&plus;(1-x_{i,j})log(1-p_{i,j})\end{matrix}" title="\begin{matrix}log\begin{pmatrix}{p_{\theta}(x_{i}|z^{i})}\end{pmatrix}=log\prod_{j=1}^{D}p_{\theta}(x_{i,j}|z^{i})\\=\sum_{j=1}^{D}logp_{\theta}(x_{i,j}|z^{i})\\=\sum_{j=1}^{D}logp_{i,j}^{x_i,j}(q-p_{i,j})^{1-x_{i,j}}\\=\sum_{j=1}^{D}x_{i,j}logp_{i,j}+(1-x_{i,j})log(1-p_{i,j})\end{matrix}" /></p>

        <br>

        - Gaussian으로 가정하면 MSE로 바뀜

    

<br>

### Reparameterization Trick

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?z^{i,l}\sim&space;N(\mu_{i},&space;\sigma_{i}^{2}I)" title="z^{i,l}\sim N(\mu_{i}, \sigma_{i}^{2}I)" /></p>

<br>

- 단순히 평균과 표준편차만 이용하면 미분을 할 수 없어 backpropagation이 불가능


<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?\begin{matrix}z^{i,l}=\mu_{i}&plus;\sigma_{i}\odot&space;\epsilon&space;\\\epsilon\sim&space;N(0,I)\end{matrix}" title="\begin{matrix}z^{i,l}=\mu_{i}+\sigma_{i}\odot \epsilon \\\epsilon\sim N(0,I)\end{matrix}" /></p>

<br>

- Normal distribution에서 sampling 한 후 표준편차에 더한 후 평균을 더하면 z에 관한 식이 나오고 위와 같은 결과를 얻게 됨

<br>

### Latent variable 차원 특징

<br>

<p align=center><img src="images/image162.PNG" width=40%/></p>
<p align=center><a href="https://arxiv.org/pdf/1312.6114.pdf">출처</a></p>
<br>

- 너무 작은 차원으로 축소를 한 것보다 큰 차원으로 축소한 것이 의미는 있음 복원은 잘되긴 함

<br>
<br>

## CVAE (Conditional Variational AutoEncoder)
- VAE를 기반으로 한 방법
- VAE 에서는 label 정보를 사용하지 않음
- Condition
    - Label 정보를 알고있으니 encoder에도 사용하고 decoder에도 사용하겠다는 의미

<br>

### CVAE(M2) : Supervised version
- 모든 데이터의 label 정보를 다 알고있고 이를 이용하는 경우

<br>

<p align=center><img src="images/image163.PNG" width=40%/></p>
<p align=center><a href="https://www.slideshare.net/NaverEngineering/ss-96581209">출처</a></p>

<br>

### ELBO
- VAE와 같은 방식으로 유도

<br>

<p align=center><img src="https://latex.codecogs.com/svg.image?\begin{matrix}log\begin{pmatrix}p_{\theta}(x,y)\end{matrix}=log\int&space;p_{\theta}(x,y|z)\frac{p(z)}{q_{\phi}(z|x,y)}q_{\phi}(z|x,y)dz\\\geq&space;\int&space;log&space;\begin{pmatrix}p_{\theta}(x,y|z)\frac{p(z)}{q_{\phi}(z|x,y)}\end{pmatrix}q_{\phi}(z|x,y)dz\\=\int&space;log&space;\begin{pmatrix}p_{\theta}(x|y,z)\frac{p(y)p(z)}{q_{\phi}(z|x,y)}\end{pmatrix}q_{\phi}(z|x,y)dz\\=E_{q_{\phi}(z|x,y)}\begin{bmatrix}log\begin{pmatrix}p_{\theta}(x|y,z)\end{bmatrix}&plus;log(p(y))\end{bmatrix}\\=-L(x,y)\end{matrix}&space;" title="\begin{matrix}log\begin{pmatrix}p_{\theta}(x,y)\end{matrix}=log\int p_{\theta}(x,y|z)\frac{p(z)}{q_{\phi}(z|x,y)}q_{\phi}(z|x,y)dz\\\geq \int log \begin{pmatrix}p_{\theta}(x,y|z)\frac{p(z)}{q_{\phi}(z|x,y)}\end{pmatrix}q_{\phi}(z|x,y)dz\\=\int log \begin{pmatrix}p_{\theta}(x|y,z)\frac{p(y)p(z)}{q_{\phi}(z|x,y)}\end{pmatrix}q_{\phi}(z|x,y)dz\\=E_{q_{\phi}(z|x,y)}\begin{bmatrix}log\begin{pmatrix}p_{\theta}(x|y,z)\end{bmatrix}+log(p(y))\end{bmatrix}\\=-L(x,y)\end{matrix} " /></p>

<br>

### CVAE(M2) : Unsupervised version (or Semi supervised version)
- 일부 데이터의 label만 알고있는 경우
- Label을 알고있을 경우는 CVAE를 이용
- Label을 모르는 경우는 그 모르는 데이터에 대한 condition(y)를 추정하는 별도의 network를 이용
- 추정한 y값으로 CVAE 이용

<br>

<p align=center><img src="images/image164.PNG" width=40%/></p>
<p align=center><a href="https://www.slideshare.net/NaverEngineering/ss-96581209">출처</a></p>

<br>

### CVAE(M3) : Unsupervised version (or Semi supervised version)
- 일부 데이터의 label만 알고있는 경우


<br>

<p align=center><img src="images/image165.PNG" width=40%/></p>
<p align=center><a href="https://www.slideshare.net/NaverEngineering/ss-96581209">출처</a></p>

<br>

- 기존 VAE와 같은 M1으로 학습
- Latent variable z를 생성하는 과정에서 label y를 추정하는 network 추가
- label 정보 추정하여 z 생성
- 성능이 더 좋음 


<br>

### MNIST Result

<br>

<p align=center><img src="images/image166.PNG" width=40%/></p>
<p align=center><a href="https://arxiv.org/pdf/1406.5298.pdf">출처</a></p>

<br>

- Label을 제외한 주된 feature를 latent variable이 학습
- (a) : label인 y 값을 고정하고 style을 바꾸는 경우
- (b) : z 값, style을 고정하고 condition만 바꾸면 같은 style인데 숫자만 다르게 나옴  
 
 <br>
 <br>

<!-- ## AAE (Adversarial AutoEncoder)
- VAE 학습 할 때 KL divergence term을 이ㅛㅇㅇ하여 prior과 sampling 함수의 차이 조절  -->
