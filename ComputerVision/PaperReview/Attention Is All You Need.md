## Basic
참고
- https://wikidocs.net/24996
- https://www.youtube.com/watch?v=AA621UofTUA

<br>
<br>

### 자연어 처리 (기계 번역) 개요
- Seq2Seq는 소스 문장을 고정된 크기의 context vector로 만들어 사용하기 때문에 성능의 제한이 존재
- 2015년 Attention 기법이 나오기 시작하면서 입력 시퀀스 전체에서 정보를 추출하는 방법
- 2017년에 Transformer가 나온 후 자연어 관련 작업에서 RNN 기반의 방식보다는 attention 기반의 방식이 주가 됨
- 그 이후에 나온 GPT, BERT는 각각 Transformer의 디코더(Decoder), 인코더 (Enocder) 구조를 활용

<br>
<br>

### Seq2Seq 모델 구조
- 입력 문장을 받는 RNN 셀을 인코더, 출력 문장을 출력하는 RNN 셀을 디코더라고 함
- 실제로는 성능 문제로 RNN 이 아닌 LSTM 이나 GRU 셀로 구성
- 입력 문장은 단어 토큰화를 통해 단어 단위로 쪼개짐
- 단어 토큰은 각각 RNN 셀의 입력이 됨
- 인코더 RNN 셀은 모든 단어를 입력 받은 뒤의 인코더 RNN 셀의 마지막 시점의 hidden state 값을 디코더에 넘겨주는데 이를 context vector라 함 

<br>
<br>

### Seq2Seq 모델의 한게
- 소스 문장을 압축하여 context vetor (문맥 벡터)에 압축
    - 병목현상이 일어나 성능 하락의 원인
    - 소스 문장의 길이와 상관없이 항상 고정된 크기의 context vector를 만들어야 하므로 성능 하락
- 매번 새로운 단어가 들어올 때마다 hidden state 값이 갱신
- 마지막 단어가 들어왔을 때의 hidden state vector는 source 문장 전체의 정보가 들어있음  
- 그 고정된 context vector는 디코더의 입력으로 사용

<br>

- 하지만 출력되는 문장이 길어지면 contect vector의 정보가 손실 될 수 있음 
- 성능을 개선하기 위해서 context vector를 RNN에서 항상 참고하도록해서 성능을 개선할 수 있음
- 하지만 이렇게 개선한다고 하더라도 입력 소스 문장을 고정된 크기의 vector로 압축해야 하기 때문에 병목 현상은 여전히 발생 
- 즉, 하나의 context vector가 소스 문장 전체를 포함하고 있어야한다는 점이 문제

<br>
<br>

### Seq2Seq with Attention
- 출력 단어를 만들 때 마다 소스 문장에서의 출력 전부를 입력을 받는 방식으로 Seq2Seq 문제 해결
    - 최신 GPU의 빠른 병렬처리와 메모리 증가로 가능
- Seq2Seq 모델에 attention 적용
    - 디코더는 인코더의 모든 출력을 참고할 수 있음 
- 매 단어가 입력될 때마다 얻는 hidden state h값을 전부 저장하고 이를 디코더에서 단어를 출력할 때마다 이용하여 소스 문장 전체를 반영하겠다는 의미 
- 디코더에서 현재의 hidden state 값을 만든다고 하면 이전 단어의 hidden state (s_{t-1}) 값과 인코더의 소스 문장 단의 hidden state 값을 묶어 별도의 행렬 곱을 수행하여 Energy 값을 만듦 
- Energy
    - 현재 어떤 단어나 값을 출력하기 위해서 소스 문장에서 어떤 단어에 초점을 둘 필요가 있는지 수치화해서 나타낸 값
- Energy 값에 softmax를 취하여 확률값을 구한 뒤 소스 문장의 각각의 hidden state 값에 대해서 어떤 vector에 더 가중치를 두어서 참고하면 좋을지를 반영하여 가중치 값을 hidden state에 곱한 것을 각각의 비율에 맞게 더해준 후 그러한 weighted sum 값을 매번 출력 단어를 만들기 위해서 반영 


<br>
<br>


### Seq2Seq with Attention : 디코더
- 디코더는 매번 인코더의 모든 출력 중에서 어떤 정보가 중요한지 계산


<br>

- 에너지 (Energy) 
    - 소스 문장에서 나왔던 모든 출력값들 중에서 어떤 값과 가장 연관성이 있는지를 알기 위해 수치로 구한 것 

    <br>

    <p align=center><img src="./images/3/1.png"><br> 
                    i : 현재 디코더가 처리 중인 인덱스 <br>
                    j : 각각의 인코더 출력 인덱스</p>

    <br>

    - 매번 출력 단어를 만들 때마다 모든 j를 고려, 즉 인코더의 모든 출력 고려하겠다는 의미

<br>

- 가중치 (Weight)
    - 에너지 값에 softmax를 취하여 확률 값 얻음
   
    <br>

    <p align=center><img src="./images/3/2.png"></p>

    <br>
<br>

- 가중치 합 (Weighted sum)
    - 가중치 값들을 소스 문장의 각각의 hidden state와 곱하여 전부 더해준 값
    - 디코더의 이전 hidden state와 함께 들어가는 입력


<br>

- Attention 가중치를 사용하여 각 출력이 어떤 입력 정보를 참고했는지 알 수 있음 

<br>

# Attention is All You Need
## Abstract 
- 시퀀스 변역 모델은 복잡한 인코더와 디코더를 포함한 RNN이나 CNN모델 사용
- 가장 좋은 성능을 내는 모델은 인코더와 디코더를 연결하는 attention 매커니즘 사용
- 이 논문에서는 간단한 매커니즘은 **Transformer** 를 제안
- 두 가지 기계 번역 task를 실행한 결과 다른 알고리즘보다 Transformer가 훨씬 더 좋은 성능을 보이는 것을 확인
- 하지만 학습을 시킬 때 요구되는 시간은 현저하게 적음
- 이 논문에서는 Transformer가 다른 작업에서도 일반화가 잘 되는 것을 확인 
    - 데이터 양이 많고 제한된 학습 데이터에 모두에서 English constituency에 성공적으로 적용됨

<br>
<br>

## Introduction
- Recurrent neural networks(RNN), long-short-term memory(LSTM), gated recurrent neural network은 특히 시퀀스 모델링과 변환 문제에 대해서 최신의 접근법을 제시
    - 언어 모델링, 기계 번역 등
- Recurrent 언어 모델과 인코더-디코더 구조를 벗어나기 위해 많은 시도를 함

<br>

- Recurrent 모델은 전형적으로 입력과 출력 시퀀스의 심볼 자리를 통한 요소 계산

$ \Alpha \rightarrow \Omega $

$ \Alpha \rightarrow  $
