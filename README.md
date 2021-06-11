# tensorflow
Tensorflow and Keras 학습 저장소
```txt
Keras API를 기반의 딥러닝 모델 네트워크를 정의하고 학습 및 테스트를 목표할 때 필요한 내용을 포함하며,
Unet 논문을 바탕으로 네트워크를 정의하는 것을 목표로 설정하여 코드를 작성했습니다.
```

<img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" width="70%">

Unet에 대한 자세한 내용은 [논문](https://arxiv.org/pdf/1505.04597.pdf) 을 참조하면 됩니다.
***

## 코드 실행 환경

- Ubuntu 18.04
- Anaconda 4.10.1

## 아나콘다 설치

- [설치 가이드](https://docs.anaconda.com/anaconda/install/linux/)
- [Archive 파일 url](https://repo.anaconda.com/archive/)
```bash
    wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh -O ~/Anaconda3-5.3.1-Linux-x86_64.sh
    bash ~/Anaconda3-5.3.1-Linux-x86_64.sh
```

## 아나콘다 환경 생성

```bash
    conda create -n tensorflow python=3.7
    conda activate tensorflow
```

## 파이썬 라이브러리 설치

```bash
    # OpenCV
    pip install tensorflow==1.15.0
    # Pillow
    pip install pillow
    # Matplotlib
    pip install matplotlib
```

## 스크립트에서 확인할 수 있는 함수들


