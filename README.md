# tensorflow
Tensorflow and Keras 학습 저장소
```txt
Keras API를 기반의 딥러닝 모델 네트워크를 정의하고 학습 및 테스트를 목표할 때 필요한 내용을 작성했습니다.
```


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
    pip install tensorflow
    # Pillow
    pip install Pillow
    # Matplotlib
    pip install matplotlib
```

## 함수 예제 스크립트

1. 01-tensorflow_base.py
    - check_tensor_contant() : 텐서 상수
    - check_random_tensor() : 랜덤 텐서
    - check_eager_execution() : 즉시실행 모드
    - check_tensor_and_numpy() : Numpy 호환성
    - check_slicing_tensor() : 텐서 슬라이싱
    - check_tensor_function() : 텐서 지원 기본 함수
    - check_function_with_tf_function_annot() : @tf.function + 함수
    - check_model_with_tf_function_annot() : @tf.function + 클래스
    - check_graph_gradient() : gradient(미분) 값 계산


## 모델 예제 스크립트 (implementation)

1. 01-tensorflow_linear_regression.py
    - Linear regression 모델를 optimizer(Adam)의 apply_gradients() 와 minimize()함수를 각각 사용해서 학습하는 코드