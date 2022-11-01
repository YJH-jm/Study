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
- Computer vision에서 딥러닝은 매우 강력한 기술이지 맘ㅎ은 수의 가중치들은 상당히 많은 저장 공간과 메모라 대역폭이 필요함
- 이는 딥러닝 네크워크를 모바일 시스템에 탑재하기 어렵게 만듦

<br>

- 많은 mobile-first 회사들의 앱들은 다양한 앱스토어를 통하여 업데이트가 되고 이 회사들은 binary 파일의 크기에 민감
    - App store에서 "100MB 가 넘는 app은 Wi-Fi 연결이 될 때까지 다운로드 할 수 없음" 등의 제약이 있기 때문 

<br>

- d


<br>
<br>

## 2. Network Pruning
- 초기 연구에서부터, pruning을 사용하면 네트워크 복잡도를 감소시키고 overfitting을 막을 수 있다는 것이 증명됨
- 2015년 연구에서 최신의 CNN 모델에서 pruning 기법을 사용하더라도 정확도의 손실이 없음을 확인
- 첫 번째로 일반적인 네트워크 학습을 진행
- 모든 연결(가중치 값)에서 