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
## Autoencoder 기본적인 구조 

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

## Autoencoder 수식

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


- Loss function에 네트워크 input, output 값 이용

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

## Linear Autoencoder
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

## Stacking Autoencoder
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

- 기존의 AE의 input에 rando mnoise 추가
    - 사람이 봤을 때 의미적으로 벗어나지 않을 만큼의 noise 추가
    - 다양한 방법으로 noise를 추가하나 이 논문에서는 noise에 해당하는 위치의 pixel울 0으로 바꿈
        - Extracting and Composing Robust Features with Denoising Autoencoders(2018)
    
    <br>
    
    <p align=center><img src="images/image153.PNG" width=40%></p>
    <p align=center><a href="http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/auto.pdf">출처</a></p>

    <br>

- Loss은 noise가 추가되기 전의 데이터와 DAE를 통과한 후의 output을 비교
<p align=center><img src="https://latex.codecogs.com/svg.image?L_{DAE}(x,y)=\sum_{x\in&space;D}&space;E_{q(\widetilde{x}|x)}[L(x,g(h(\widetilde{x})))]" title="L_{DAE}(x,y)=\sum_{x\in D} E_{q(\widetilde{x}|x)}[L(x,g(h(\widetilde{x})))]" /></p>

    - Manifold 상에서는 똑같지만 원공간에서는 다른 데이터를 학습시
