import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 하이퍼파라미터
num_classes = 10
num_features = 784
learning_rate = 1e-2
training_steps = 1000
batch_size = 256
shuffle_buffer_size = 5000

# Optimizer
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

def preprocess_features(train_features, test_features):
    """이미지 데이터 전처리
    """
    
    # Numpy 배열로 전환
    train_features, test_features = np.array(train_features, dtype=np.float32), np.array(test_features, dtype=np.float32)

    # (784,) shape의 1차원 배열로 flatten
    train_features, test_features = train_features.reshape([-1, num_features]), test_features.reshape([-1, num_features])

    # 이미지 픽셀 값을 0~1 사이로 Normalization
    train_features, test_features = train_features / 255., test_features / 255.

    return train_features, test_features

def convert_numpy_to_dataset(features, labels, is_train=False):
    """Numpy 배열을 tf.data.Dataset으로 변환

    Args:
        features ([type]): [description]
        labels ([type]): [description]
        is_train (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    # (features, labels)로 학습 데이터 셋 생성
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    if is_train:
        repeat_count = -1 # indefinitely
    else:
        repeat_count = 1
    
    # 데이터 augmentation을 위한 반복횟수 설정 
    dataset = dataset.repeat(repeat_count)

    if is_train:
        # 데이터 shuffle에 사용할 buffer 크기 설정
        dataset = dataset.shuffle(shuffle_buffer_size)

    # 미니 배치 사이즈 설정
    dataset = dataset.batch(batch_size)

    # 현재 작업을 처리하면서 미리 준비할 다음 작업량 설정
    dataset = dataset.prefetch(1)

    return dataset

def initialize_variables():
    """W, B 초기화
    """
    # Weight
    W = tf.Variable(tf.random.normal((num_features, num_classes)), name='weight')
    # Bias
    B = tf.Variable(tf.zeros(num_classes), name='bias')

    return W, B

def logistic_regression(features, w, b):
    """Logistic regression 모델 텐서 그래프
    """
    return tf.nn.softmax(tf.matmul(features, w)+ b)

def loss_fun(preds, labels):
    """Loss 텐서 그래프
    """
    labels = tf.one_hot(labels, depth=num_classes)
    preds = tf.clip_by_value(preds, clip_value_min=1e-9, clip_value_max=1.)
    return tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(preds), axis=1))

def accuracy(preds, labels):
    """Accuracy 텐서 그래프
    """
    correction = tf.equal(tf.argmax(preds,1), tf.cast(labels, dtype=tf.int64))
    return tf.reduce_mean(tf.cast(correction, dtype=tf.float32))


def run_optimization(features:tf.Tensor, labels:tf.Tensor, w:tf.Variable, b:tf.Variable):
    """tf.GradientTape()로 w, b에 gradients 반영

    Args:
        features (tf.Tensor): feature 텐서
        labels (tf.Tensor): label 텐서
        w (tf.Variable): Weight
        b (tf.Variable): Bias
    """
    with tf.GradientTape() as tape:
        preds = logistic_regression(features, w, b)
        loss = loss_fun(preds, labels)
    
    gradients = tape.gradient(loss, [w, b])

    optimizer.apply_gradients(zip(gradients, [w, b]))


def training(train_dataset:tf.data.Dataset, w:tf.Variable, b:tf.Variable):
    """학습 데이터셋으로 w, b를 학습

    Args:
        train_dataset (tf.data.Dataset): MNIST 학습데이터를 담고있는 dataset
        w (tf.Variable): Weight
        b (tf.Variable): Bias
    """
    for step, (batch_features, batch_labels) in enumerate(train_dataset.take(training_steps), 1):
        run_optimization(batch_features, batch_labels, w, b)

        if step % 50 == 0:
            batch_preds = logistic_regression(batch_features, w, b)
            loss = loss_fun(batch_preds, batch_labels)
            acc = accuracy(batch_preds, batch_labels)
            print('TRAINING | step: {:4d}\t loss: {:.4f}\t acc: {:.4f}'.format(step, loss, acc))
    
    print('-'*100)


def test(test_dataset:tf.data.Dataset, w:tf.Variable, b:tf.Variable):
    """학습된 w, b의 테스트 데이터셋에 대한 loss 값과 accuracy 측정

    Args:
        test_dataset (tf.data.Dataset): MNIST 테스트 데이터를 담고있는 dataset
        w (tf.Variable): Weight
        b (tf.Variable): Bias
    """
    for step, (batch_features, batch_labels) in enumerate(test_dataset, 1):
        batch_preds = logistic_regression(batch_features, w, b)
        loss = loss_fun(batch_preds, batch_labels)
        acc = accuracy(batch_preds, batch_labels)
        print('TEST | step: {:4d}\t loss: {:.4f}\t acc: {:.4f}'.format(step, loss, acc))
    print('-'*100)

def main():
    
    # (28, 28, 1) shape의 grayscale MNIST 이미지 데이터 로드
    (train_features, train_labels), (test_features, test_labels) = mnist.load_data()

    # MNIST 이미지 데이터 전처리
    train_features, test_features = preprocess_features(train_features, test_features)

    # 학습용 데이터 셋 생성
    train_dataset = convert_numpy_to_dataset(features=train_features, labels=train_labels, is_train=True)

    # 테스트용 데이터 셋 생성
    test_dataset = convert_numpy_to_dataset(features=test_features, labels=test_labels, is_train=False)

    # 학습할 parameter 초기화
    w, b = initialize_variables()

    # 모델 학습
    training(train_dataset, w, b)

    # 테스트
    test(test_dataset, w, b)

if __name__ == '__main__':
    main()
# %%
