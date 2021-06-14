import timeit
import numpy as np
import tensorflow as tf
import timeit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Eager execution
if tf.__version__ < "2":
    tf.enable_eager_execution()
    print('Eager exeuction enabled')
else:
    print('Tensorflow version 2.x supports Eager exeuction')

def check_tensor_contant():
    """텐서 상수 확인
    """
    # 텐서 상수
    a = tf.constant(2)
    print('Rank(차원) : ', tf.rank(a))
    print('Value(텐서) : ', a)

    b = tf.constant([0,1])
    print('Rank(차원) : ', tf.rank(b))
    print('Value(텐서) : ', b)

    c = tf.constant([[0,1], [2,3]])
    print('Rank(차원) : ', tf.rank(c))
    print('Value(텐서) : ', c)

    d = tf.constant(['Tensorflow'])
    print('Rank(차원) : ', tf.rank(d))
    print('Value(텐서) : ', d)

def check_random_tensor():
    """random 텐서 확인
    """
    # uniform 난수 생성
    rand_uniform = tf.random.uniform(shape=(10,), minval=0, maxval=1)
    print('rand_uniform.shape : ',rand_uniform.shape)
    print('rand_uniform : ', rand_uniform)

    # 정규분포 난수 생성
    rand_normal = tf.random.normal(shape=(10, 2), mean=0., stddev=1.)
    print('rand_normal.shape : ',rand_normal.shape)
    print('rand_normal : ', rand_normal)

def check_eager_execution():
    """Eager execution 확인 : 세션을 초기화 없이 그래프를 즉시 실행
    """
    a = tf.constant(5)
    b = tf.constant(2)
    print('tf.add(a,b) : ', tf.add(a,b))
    print('a+b : ', a+b)
    print('tf.subtract(a,b) : ', tf.subtract(a,b))
    print('a-b :',a-b)
    print('tf.multiply(a,b) : ',tf.multiply(a,b))
    print('a*b : ',a*b)

def check_tensor_and_numpy():
    """Tensor와 Numpy 배열 호환성 확인
    """
    a = tf.constant(5)
    b = tf.constant(2)
    np_value = tf.add(a, b)
    print(type(np_value))
    print(np_value)

    square = np.square(np_value, dtype=np.float32)
    print(type(square))
    print(square)

    square_tensor = tf.convert_to_tensor(square)
    print(type(square_tensor))
    print(square_tensor)

def check_slicing_tensor():
    """텐서 슬라이싱 확인
    """
    tensor = tf.constant([[1., 3., 5.],[7., 9., 11.]])
    print(tensor)
    print(tensor[:, 1:])
    print(tensor[..., 1, tf.newaxis])

def check_tensor_function():
    """텐서 지원 함수 확인
    """
    tensor = tf.constant([[1., 3., 5., 7.],[7., 9., 11., 13.]])
    print(tensor + 1)
    print(tf.square(tensor))
    print(tf.transpose(tensor))
    print(tf.matmul(tensor, tf.transpose(tensor)))
    print(tf.cast(tensor, dtype=tf.int32))

def check_function_with_tf_function_annot():
    """tf.function 적용 유무에 따른 함수 수행시간 비교
    """

    @tf.function
    def tf_fun(x):
        return x**2 - 10*x +3
    print(tf_fun(2))
    print(tf_fun(tf.constant(2)))

    def fun(x):
        return x**2 - 10*x +3
    print(fun(2))
    print(fun(tf.constant(2)))

    inputs = tf.constant(2)
    print('함수 Eager time ', timeit.timeit(lambda: tf_fun(inputs), number=10000))
    print('함수 Graph time ', timeit.timeit(lambda: fun(inputs), number=10000))


def check_model_with_tf_function_annot():
    """tf.function annotation 적용 유무에 따른 Subclassing 모델 학습 시간 비교
    """
    class SubclassModel(Model):
        """Subclassing 케라스 모델 클래스

        Args:
            Model (tensorflow.keras.models.Model): 케라스 모델 클래스
        """
        def __init__(self, *args, **kwargs):
            super(SubclassModel, self).__init__(*args, **kwargs)
            self.flatten = Flatten(input_shape=(28,28))
            self.dense_1 = Dense(256, activation='relu')
            self.dropout = Dropout(0.5)
            self.dense_2 = Dense(10)

        def call(self, x):
            x = self.flatten(x)
            x = self.dense_1(x)
            x = self.dropout(x)
            x = self.dense_2(x)
            return x

    inputs = tf.random.normal((120, 28, 28))

    eager_model = SubclassModel()
    graph_model = tf.function(eager_model)

    print('클래스 Eager time ', timeit.timeit(lambda: eager_model(inputs), number=10000))
    print('클래스 Graph time ', timeit.timeit(lambda: graph_model(inputs), number=10000))


def check_graph_gradient():
    """미분 연산 확인
    """
    W = tf.Variable(tf.random.normal((3,2)), name='weights')
    B = tf.Variable(tf.zeros(2, dtype=tf.float32), name='biases')

    x = [[1., 2., 3.]]

    with tf.GradientTape() as tape:
        y = x @ W + B
        loss = tf.reduce_mean(tf.square(y))
        [dl_dw, dl_db] = tape.gradient(loss, [W, B])
        print(W.shape)
        print(dl_dw.shape)
        print(B.shape)
        print(dl_db.shape)

def main():

    # 텐서 상수
    check_tensor_contant()

    # 랜덤 텐서
    check_random_tensor()

    # 즉시실행 모드
    check_eager_execution()

    # Numpy 호환성 확인
    check_tensor_and_numpy()

    # 텐서 슬라이싱
    check_slicing_tensor()

    # 텐서 지원 함수 
    check_tensor_function()

    # tf.function 어노테이션
    check_function_with_tf_function_annot()
    check_model_with_tf_function_annot()

    # gradient(미분) 값
    check_graph_gradient()

if __name__ == '__main__':
    main()