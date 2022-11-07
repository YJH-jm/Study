# Basic
- 모바일 환경에서의 딥러닝 기술은 이점 
    - 개인정보 보호
        - 서버까지 데이터를 전송하지 않아도 됨
    - 실시간 처리
        - On device 환경에서 GPU가 탑재되어있다면 서버와 데이터를 주고받을 필요 없음
    - 네트워크 대역폭 
- 딥러닝 모델은 대부분 큰 용량을 가지고 있기 때문에 모바일 앱에 딥러닝 모델 직접적으로 탑재하기 어려움
- 딥러닝 모델의 크기가 크면 에너지 소비의 문제가 생김
    - 큰 네트워크를 사용할 때 가중치를 fetch 하는 과정에서 많은 대역폭 요구
    - 포워딩을을 위해 많은 내적 연산 수행
    - 특히 메모리 접근 과정에서 많은 에너지 소비 발생 가능 

    <br>
    <p align=center><img src="./images/2/1.jpeg" width=50%></p>
    <br>

- 뉴럴 네크워크 용량과 에너지 소비량을 줄일수 있는 Deep Compression 제안 

<br>
<br>


# DEEP COMPRESSION: COMPRESSING DEEP NEURAL NETWORKS WITH PRUNING, TRAINED  QUANTIZATION AND HUFFMAN CODING
**2016년 ICLR에서 best paper로 선정된 우수 논문**

<br>
<br>

## Abstract
- Neural Network는 계산 집약적이고 메모리 집약적이기 때문에 제한된 하드웨어 리소스를 가진 임베디드 시스템에 적요하기에 어려움
- 이를 해결하기 위해 세 단계의 파이프라인으로 구성된 "**deep compressin**" 을 도입
    - Pruning (가지치기)
    - trained Quantization (양자화)
    - Huffman coding
- 세 방법을 모두 적용하여 정확도에 영향을 미치지 않고 용량을 35 ~ 49 배를 줄임
- Pruning은 중요한 네트워크 연결만 학습할 수 있도록 네트워크를 가지치기 하여 중요한 연결만 학습하는 것
- Quantization은 가중치를 양자화하여 가중치 공유를 잘 할 수 있도록 만들고 Huffman coding을 적용할 수 있도록 만듦
- 두 과정에서 남은 중요한 연결에 대한 finetuning과 양자화된 centroids 위한 재학습을 진행
- 이 방법은 off-chip DRAM이 아닌 on-chip SRAM cache에 저장 할 수 있게 함
- Application size와 다운로드 대역폭이 제한된 모바일 어플리케이션에서 복잡한 뉴럴 네트워크를 사용 할 수 있게 함

<br>
<br>

## 1. Introduction
- Computer vision에서 딥러닝은 매우 강력한 기술이지 많은 수의 가중치들은 상당히 많은 저장 공간과 메모라 대역폭이 필요함
- 이는 딥러닝 네크워크를 모바일 시스템에 탑재하기 어렵게 만듦

<br>

- 많은 mobile-first 회사들의 앱들은 다양한 앱스토어를 통하여 업데이트가 되고 이 회사들은 binary 파일의 크기에 민감
    - App store에서 "100MB 가 넘는 app은 Wi-Fi 연결이 될 때까지 다운로드 할 수 없음" 등의 제약이 있기 때문 

<br>

- Memory access 과정에서 에너지 소비가 많이 발생하게 됨 
- 45nm 이하 CMOS기술에서 , 32 bit floating point 덧셈은 **0.9pJ**, 32 bit SRAM chache는 **5pJ**, 반면에 32 bit DRAM은 **640pJ** 요구함 
- 큰 네트워크는 on-chip 저장 공간에 맞지 않기 때문에 DRAM access가 필요

<br>

- 이 연구의 목표는 모바일 장치에 딥러닝 모델을 탑재시켜 추론이 가능할 수 있도록 저장 공간과 에너지 소비를 줄이는 것
- 이를 위하여 "deep compression" 제안
    - 정확도를 보존하고 뉴럴 네트워크의 저장 공간을 줄이기 위한 3 단계의 파이프라인 방식
- 불필요한 연결을 제거하고 주요한 연결만 남기는 prun
- 가중치를 양자화하여 많은 연결들이 같은 가중치를 공유하고, 그로 인해서 codebook (effective weights)와 indice 만 저장하면 됨

<br>

- 이 논문에서 주장하는 이 실험의 가장 중요한 insight는 pruning과 quantization을 통해서 ~ 없이 네트워크를 압축할 수 있다는 것
- 이 과정을 통해 네트워크를 압축하면 모든 가중치들을 on-chip cache에서 이용가능

<br>
<br>

## 2. Network Pruning
- 초기 연구에서부터, pruning을 사용하면 네트워크 복잡도를 감소시키고 overfitting을 막을 수 있다는 것이 증명됨
- 2015년 연구에서 최신의 CNN 모델에서 pruning 기법을 사용하더라도 정확도의 손실이 없음을 확인
- 첫 번째로 일반적인 네트워크 학습을 진행
- 모든 연결(가중치 값)에서 가중치 값이 작은 연결들을 가지치기
    - 특정 threshold 값보다 작은 경우 
        - 3보다 작은 경우 

    <br>
    <p align=center><img src="./images/2/2.png" width=50%><p>
    <br>

- 남아있는 sparse한 연결들의 가중치 값들을 얻기 위해 네트워크 재학습
- Pruning은 AlexNet의 파라미터를 9배, VGG16의 파라미터를 13배 감소시킴

<br>

- Pruning의 결과로 나온 sparse 구조를 CSR(Compressed Sparse Row) 또는 CSC(Compressed Sparse Column) 형식으로 저장 
- CSR, CSC는 2a+n+1 개의 원소 필요
    - a : 0 이 아닌 요소의 수
    - n : column 또는 row의 수 

    - COO (Coordinate Format)
        - 행렬에 포함된 0이 아닌 값을 가진 데이터에 대하여 행과 열의 위치 정보를 기록 
        - 0 이 아닌 원소의 수가 a개 일 때 3a 만큼의 원소 필요

    <br>
    <p align=center><img src="./images/2/3.png" width=50%><p>
    <br>

    - CSR (Compressed Sparse Row)
        - 행의 압축 정보인 Row Pointer 를 이용하여 표현
        - 2a + (n+1) 만큼의 원소 필요, a는 0이 아닌 원소 수, n은 행의 길이
        - 일반적으로 COO 보다 메모리가 적게 사용 됨
        - Row Pointer에 접근해서 각 행에 0이 아닌 원소의 수가 몇 개 인지 알 수 있음

    <br>
    <p align=center><img src="./images/2/4.png" width=50%><p>
    <br>
    

<br>

- 더 압축하기 위해서, 절대적인 위치를 저장하는 대신 index의 차이를 저장
- 이 차이를 convolution layer에 8비트, fc layer에 5비트로 인코딩
- 인코딩 범위보다 큰 index 차이가 생기는 경우 zero padding solution을 이용
    - 차이 값을 저장하기 위해 3bits만을 사용할 때, 그 차이가 3bits 보다 크면 패딩 삽입

    <br>
    <p align=center><img src="./images/2/5.png" width=50%></p>
    <br>

    - index 4에 위치한 원소와 index 15에 위치한 원소의 거리가 8보다 크기 때문에 중간에 0을 삽입하여 3bit로 표현 가능하게 만들어줌

<br>
<br>

## 3. Trained Quantization and Weight Sharing
- Pruning 한 네트워크는 각 가중치를 표현하기 위한 bit수를 줄이는 Quantization 과정과 가중치 공유를 통하여 더 압축 가능


<br>
<p align=center><img src="./images/2/6.png" width=50%></p>
<br>

- 4개의 input 4개의 output이 존재하는 경우
- 16(n = 4x4) 개의 가중치 존재
- k = 4로 설정, 즉 4개의 cluster를 사용한다는 의미
- 압축률의 계산은 아래와 같은 식으로 진행
    - k = 4로 설정한 경우 4개의 클러스터를 이


    <br>

    <p align=center><img src="./images/2/7.png" > <br>
    n : 총 가중치의 수 <br>
    k : 클러스터의 수 

    </p>

    <br>
    
    <p align=center><img src="./images/2/8.png"></p>

<br>

- 각 cluster들의 평균값을 구하여 centroids 라고 이 값들을 해당 클러스터에서 이 값을 공유하여 사용
- centroids 들은 계속 사용하는 것이 아니라 fine-tuning을 통해 업데이트


<!-- 1. 실제로 사용할 가중치의 개수 k를 설정
2. 해당 k개의 가중치를 별도의 메모리에 저장한 뒤에 이를 공유 (sharing)
3. 해당 k개의 가중치를 fine-tuning을 통하여 학습시켜 정확도를 올림 -->


<br>

### 3.1 Weight Sharing
- 


### 3.2 Initialization of Shared Weights
### 3.3 Feed-Forward and Back-Propagation

<br>
<br>

## 4. Huffman Coding 


<br>
<br>

## 5. Experiments
### 5.1 LeNet-300-100 and LeNet-5 on MNIST
### 5.2 AlexNet on ImageNet
### 5.3 VGG-16 on ImageNet

## 6. Discussions
### 6.1 Pruning and Quantization Working Together
### 6.2 Centeroid Initialization
### 6.3 Speedup and Energy Efficiency
### 6.4 Ratio of Weights, Index and Codebook

<br>
<br>

## 7. Related Work

<br>
<br>

## 8. Future Work

<br>
<br>

## 9. Conclusion
