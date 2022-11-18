## Basic
### 자연어 처리 (기계 번역) 개요
- Seq2Seq는 소스 문장을 고정된 크기의 context vector로 만들어 사용하기 때문에 성능의 제한이 존재
- 2015년 Attention 기법이 나오기 시작하면서 입력 시퀀스 전체에서 정보를 추출하는 방법
- 2017년에 Transformer가 나온 후 자연어 관련 tast에서 RNN 기반의 방식보다는 attention 기반의 방식이 주가 됨
- 그 이후에 나온 GPT, BERT는 각각 Transformer의 디코더(Decoder), 인코더 (Enocder) 구조를 활용

<br>
<br>

### Seq2Seq 모델의 한게
- 소스 문장을 압축하여 contect vetor에 압축
    - 병목현상이 일어나 성능 하락의 원인이 더;ㅁ

<br>
<br>

## Abstract 
