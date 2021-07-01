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

## (Optional) Keras 모델 시각화

```bash
    # tensorflow.keras.utils.plot_model() 호출 시, 필요한 라이브러리
    sudo apt update
    sudo apt install graphviz
```

## 예제 스크립트

1. 01-tensorflow_base.py

```txt
    usage: 01-tensorflow_base.py [-h]
                             {tensor_contant,random_tensor,eager_execution,tensor_and_numpy,slicing_tensor,tensor_function,function_with_tf_function_annot,model_with_tf_function,graph_gradient}
                             ...

    Tensorflow 기본 예제 코드

    positional arguments:
    {tensor_contant,random_tensor,eager_execution,tensor_and_numpy,slicing_tensor,tensor_function,function_with_tf_function_annot,model_with_tf_function,graph_gradient}
        tensor_contant      tf.constant Tensor
        random_tensor       tf.random Tensor
        eager_execution     Eager execution : 세션을 초기화 없이 그래프를 즉시 실행
        tensor_and_numpy    Tensor와 Numpy 배열 호환성
        slicing_tensor      Tensor 슬라이싱
        tensor_function     Tensor 지원 함수 예제
        function_with_tf_function_annot
                            tf.function 유무에 따른 함수 수행시간 비교
        model_with_tf_function
                            tf.function 유무에 따른 Keras 모델 학습 시간 비교
        graph_gradient      미분 연산

    optional arguments:
    -h, --help            show this help message and exit
```

2. 02-keras_base.py

```
    usage: 02-keras_base.py [-h] {sequential_model,functional_model,multiple_input_model} ...

    Tensorflow-Keras 예제 코드

    positional arguments:
    {sequential_model,functional_model,multiple_input_model}
        sequential_model    Sequential API 모델 예제 : 다중 입/출력 불가
        functional_model    Functional API 모델 예제 (Recommended)
        multiple_input_model
                            다중 입/출력을 사용하는 Functional API 모델 예제

    optional arguments:
    -h, --help            show this help message and exit
```
3. 03-tfrecord.py

```txt
    usage: .\tfrecord.py [-h] {convert,tf_example,tf_record,tf_record_with_img} ...

    Tool for converting raw data to TF Record.

    positional arguments:
    {convert,tf_example,tf_record,tf_record_with_img}
        convert             각 타입 별 value를 tf.train.Feature 클래스 변수로 변환
        tf_example          각 타입 별 value를 tf.train.Example message를 생성
        tf_record           Numpy array들을 기반으로 tf.train.Example을 생성하고, TFRecord 파일에 저장한 뒤 결과 확인
        tf_record_with_img  Image 파일을 기반으로 tf.train.Example을 생성하고, TFRecord 파일에 저장한 뒤 결과 확인

    optional arguments:
    -h, --help            show this help message and exit
```

## 모델 예제 스크립트 (model)

1. tensorflow_01_linear_regression.py
    - Linear regression 모델를 optimizer(Adam)의 apply_gradients() 와 minimize()함수를 각각 사용해서 학습하는 코드
    
2. tensorflow_02_logistic_regression_mnist.py
    - MNIST 데이터를 tf.data.Dataset으로 전처리하여 Logisitic regression 모델을 학습하는 코드
