## Basic
Fixed Point (고정 소수점)
-  정수를 표현하는 비트와 소수를 표현하는 비트수를 미리 정하고 해당 비트만을 활용하여 실수 표현

<br>

<p align=center><img src="./images/4/1.png" width=50%></p>

<br>

- 예시

<br>

$${7.625}_{10}={111.101}_{2}$$

<br>

<p align=center><img src="./images/4/3.png" width=50%></p>

<br>

Floating Point (부동 소수점)
- 고정 소수점 방식과 비트를 사용하는 체계가 다르며 이를 표현하는 다양한 체계 존재
    - 일반적으로 IEEE 754 방식을 많이 사용 

<br>

<p align=center><img src="./images/4/2.png" width=50%></p>

<br>

- 예시
    - 이진수 변환 : ${7.625}_{10}={111.101}_{2}$
    - 정규화 : ${1.11101}_{2}\times2^{2} $
    - Exponent : $2_{10}+127_{10} (bias)=129_{10}=10000001_{2}$
    - Mantissa : $11101_{2}$

<br>

<p align=center><img src="./images/4/4.png" width=50%></p>

<br>

Dynamic Range
-  숫자의 표현 범위를 의미

Precision / Resolution
- 범위 내에서 얼마나 세밀하게 숫자를 나눠서 표현하는지 


<br>
<br>

# A A Survey of Quantization Methods for Efficient Neural Network Inference
## Ⅲ Basic Conceopts of Quantization
### A. Problem Setup and Notations

<br>
<br>

### B. Uniform Quantization
- NN (Neural Network)의 가중치와 activation(활성화 출력)을 유한한 값으로 바꾸기 위해서 양자화 할 함수를 정의하는 것이 우선
- 이 함수는 floating point의 실제의 값을 작은 precision 범위로 바꿔줌 


<br>

<p align=center><img src="./images/4/5.png" width=50%></p>

<br>

$$Q(r)=Int(r/S)-Z$$
$$Q : Quantization \ operator$$
$$r : real \ valued \ input$$
$$S : real \ valued \ scaling \ factor$$
$$X : integer \ zero \ point$$

<br>

- $Int$ 함수는 rounding operation을 통해 real value를 int 값으로 변환
- 이 방법은 **uniform quantization**
- 이 방법은 양자화 된 값 $Q(r)$에서  실수값 $r$로 값을 다시 뱐환 가능하고 이를 **dequantization** 이라 함

<br>

$$\widetilde{r}=S\left (Q(r)+Z\right )$$

<br>

- $\widetilde{r}은$ rounding operation 때문에 $r$과 같지 않을 수도 있음

<br>
<br>

### C. Symmetric and Asymmetric Quantization
- Uniform quantization의 가장 중요한 요소는 scaling factor인 $S$를 선택하는 것
- Scaling factor는 실수 $r$을 

$$S=\frac{\beta-\alpha}{2^{b}-1}$$
$$[\alpha, \beta] : clapping \ range$$
$$b : quantization \ bit \ width$$

<br>

- 가장 먼저 $[\alpha, \beta]$ 의 범위를 결정해야 하는데 이 과정을 *calibration* 이라고 하기도 함
- $[\alpha, \beta]$은 칩이 ARM인지 Intel 계열인지에 따라 다름
    - ARM : MinMax 이용
    - Intel : Histogram 이용 
- MinMax에서 $[\alpha, \beta]=[r_{min}, r_{max}]$ 이고 이는 **asymmetric quantization** 영역
    - **Asymmetric quantization**는 $-\alpha\neq\beta$ 인 경우
    - **Symmetric quantization**은 $\alpha=\beta$ 인 경우


<br>
<p align=center><img src="./images/4/6.png" width=50%></p>
<br>

- MinMax를 이용하여 symmetric quantization 적용 가능
    - $-\alpha=\beta=\max(\left| r_{max}\right|, \left| r_{min}\right|)$ 
- Asymmetric quantization은 symmetric 과 비교하여 더 타이트한 clipping range를 가지게 됨
- 이는 양자화하고자 하는 가중치나 activation들이 불균형 할 때 중요함
    - activation ReLU를 통과한 값은 언제나 양수의 값을 가짐 
- 하지만 $Z=0$가 되므로 symmetric을 이용할 때는 식이 간단해 질 수 있음

<br>

$$Q(r)=Int(\frac{r}{S})$$

<br>

- Scaling factor를 결정하는 2가지 선택
    - **Full range** 
        - floor rounding mode : $S=\frac{2max(|r|)}{2^{n}-1}$
        - INT8 range  : $[-128, 127]$

    - **Restricted range**
        - $S=\frac{max(|r)}{2^{n-1}-1}$
        - INT8 range : $[-127, 127]$

- Full range가 더 정확

<br>

- Symmetric quantization가 실제로 더 많이 사용됨
    - $Z=0$이 되어서 추론하는 동동안 계산 비용이 줄어듦
    - 더 직관적으로 적용이 가능


<br>

- MinMax를 이용하여 symmetric, asymmetric quantization을 진행하는 것은 매우 많이 사용하는 방법
- 하지만 이는 데이터의 이상치에 매우 민감
    - 불필요하게 범위를 늘리고 그 결과로 quantization의 resolution이 감소
- 이를 해결할 방법은 MinMax 대신 percentile을 사용하는 것 
    - 즉, 가장 큰 수 대신 i번째로 큰/작은 수를 $\beta, \alpha$ 로 사용 
- 또는 실수 값과 양자화된 값 사이의 information loss 등의 KL divergence를 최소화하는 $\alpha$와 $\beta$를 선택하는 방법 이용

<br>

**Summary (Symmetric vs Asymmetric Quantization)**
- Symmetric quantization은 symmetric range를 사용하여 clipping 분할
- $Z=0$ 이기 때문에 쉽게 계산과 적용 가능
- 범위가 왜곡되거나 symmetric 하지 않은 경우에서는 차선의 선택
- 이런 경우에는 asymmetric quantization 사용

<br>
<br>

### D. Range Calibration Algorithms : Static vs Dynamic Quantization
- Clipping range인 $[\alpha, \beta]$ 을 결정하는 다른 calibration 방법들에 대해서 이야기를 함
- Quantization 방법을 나누는 다른 방법은 **언제** clipping range를 결정하는지
    - **Static quantization**
    - **Dynamic quantization**
- 이 range는 가중치에 대해서는 정적으로 계산이 되고 파라미터들은 추론하는 동안 보통 고정됨
- 아...모르겠다...ㅠㅠ

<br>

- Dynamic quantization에서 런타임동안 각 activation map의 clipping range는 
- 이 방식은 실시간 신호 통계의 계산이 필요하며 이는 매우 큰 오버헤드를 가짐 
- 하지만 dynamic quantization은 각 input마다 정확한 signal range를 계산하므로 더 높은 정확도를

<br>

- d이

<br>

**Summary (Dynamic vs Static Quantization)**
- Dynamic quantization은 동적으로 각 activation의 clipping range를 계산하여 대체로 높은 정확도를 얻음
- 하지만 신호를 동적으로 계산하는 것은 매우 비용이 비싸기 때문에 주로 clipping range가 모든 입력에서 고정된 static quantization을 사용 

<br>
<br>

### E. Quantization Granularity
- 대부분의 computer vision에서, 하나의 layer로 들어가는 activation input은 많은 다양한 필터들과 convolution 연산을 진행

<br>

<p align=center><img src="./images/4/7.png" width=30%></p>

<br>

- 각 convolution filter 다른 범위의 값들을 가지고 있음
- 가중치 (weights)에 대해 clipping 범위를 계산을 어느 세부 수준에서 계산할 것인가에 대하여 quantization 방법을 나눌 수 있음
    - **Layerwise Quantization** 
    - **Groupwise Quantization**

<br>

<p align=center><img src="./images/4/8.png" width=50%></p>

<br>

#### a) Layerwise Quantization
- 한 layer의 모든 convolution filter들의 가중치를 고려하여 cliiping range를 고려
- 한 layer의 모든 filter들에 같은 clipping range 적용
- 이 방법은 적용하기에는 매우 쉽지만, 각 filter들의 분포가 다양하기 때문에 정확도가 높지 않음
- 한 convolution filter가 상대적으로 작은 범위의 파라미터를 가진다면, quantization resolution을 손실 할 수 있음 (다른 filter는 상대적으로 큰 값을 가짐)

<br>

#### b) Groupwise Quantization
- 한 layer 안에서 여러 개의 서로 다른 channel들을 그룹화하여 clipping range를 계산 가능 
- 이 방법은 하나의 convolution/activation에 걸쳐 파라미터의 분포가 많이 달라지는 경우에 유용
- 하지만 다른 scaling factor를 계산해야 한다는 단점 존재

<br>

#### c) Channelwise Quantization
- 보편적으로 가장 많이 사용하는 방법
- 다른 채널들에 독립적으로 각 convolution filter가 고정된 clipping range를 가짐
    - 즉, 각 channel이 섬세한 scaling factor를 가지게 됨
- 이로 인해 quantization resolution이 더 좋아지고 높은 정확도 얻는 것이 가능

<br>

#### d) Sub-channelwise Quantization
- 이전 방법들은 convolution 또는 fully-connected layer의 파라미터들의 그룹 단위로 clipping range가 정해짐
- 하나의 convolution 이나 fully-connected layer들을 처리할 때 다른 scaling factor들을 고려해야하므로 상당한 오버헤드 존재
- 그러므로 groupwise 가 quantization resolution과 computation overhead 사이의 좋은 타협점을 제시함

<br>

**Summary(Quantization Granularity)**
- Channelwise Quantization는 convolution filter에 가장 많이 사용되는 표준 방법
- 이는 각 convolution filter 마다 다른 clipping range를 가지는데 이 때 발생하는 overhead는 무시할 만 함
- Sub-channelwise quantization은 상당히 많은 overhead가 발생하기 때문에 이는 표준의 방법이라고 볼 수 없음 

<br>


#### F. Non-Uniform Quantization

<br>

<p align=center><img src="./images/4/5.png" width=50%></p>

<br>

- Quantization step이나 quantization level이 균등하지 않게 존재



<br>

$$Q(r)=X_{i}, \ \  if \ r \in [\Delta_{i}, \Delta_{i+1})$$
$$r : 실수$$
$$X_{i} : discrete\ quantization \ level$$
$$\Delta_{i} : quantization \ steps$$
$$Q : quantizer$$

<br>

- $X_{i}$와 $\Delta_{i}$ 둘 다 일정하지 않은 간격으로 존재

<br>

- Non-uniform quantization은 고정된 bit 길이에서 더 좋은 정확도를 얻음 
    - 중요한 가치가 있는 영역에 집중하거나 적절한 동적 범위를 찾아 더 좋은 분포를 찾게 해줌 
- 전형적인 규칙기반의 non-uniform quantization은 logarithmic 분포를 이용 
    - Quantization step과 level이 선형적이 아닌 지수적으로 증가 
- 또다른 방법은 binary-code-base quantization  


**Summary (Uniform vs Non-uniform Quantization)**
- 일반적으로 non-uniform quantizatoin이 많은 신호 정보를 알 수 있음 
- 하지만 non-uniform quantization을 GPU와 CPU와 같은 하드웨어에 효과적으로 적용 할 수 없음
- Uniform quantization이 간단하고 효과적으로 하드웨어에 맵핑되기 때문에 현재 훨씬 더 잘 사용됨

<br>

#### G.Fine-tuning Method
- NN의 파라미터들을 quantization 한 후 조정한는 경우도 존재
    - **QAT(Quantization Aware Training)**
        - 모델 재학습
    - **PTA(Post-Training Quantization)**
        - 모델 재학습 없음

