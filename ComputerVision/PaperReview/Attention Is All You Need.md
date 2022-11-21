## Basic
### 자연어 처리 (기계 번역) 개요
- Seq2Seq는 소스 문장을 고정된 크기의 context vector로 만들어 사용하기 때문에 성능의 제한이 존재
- 2015년 Attention 기법이 나오기 시작하면서 입력 시퀀스 전체에서 정보를 추출하는 방법
- 2017년에 Transformer가 나온 후 자연어 관련 tast에서 RNN 기반의 방식보다는 attention 기반의 방식이 주가 됨
- 그 이후에 나온 GPT, BERT는 각각 Transformer의 디코더(Decoder), 인코더 (Enocder) 구조를 활용

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
<br>

# Attention is All You Need
## Abstract 
