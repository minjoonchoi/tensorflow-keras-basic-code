import sys
import argparse

import timeit
import numpy as np
import tensorflow as tf
import timeit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Concatenate, Dropout
from tensorflow.keras.models import Sequential
# Eager execution
if tf.__version__ < "2":
    tf.enable_eager_execution()
    print('Eager exeuction enabled')
else:
    print('Tensorflow version 2.x supports Eager exeuction')

def tensor_contant():
    # Tensor 상수
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

def random_tensor():
    # uniform 난수 생성
    rand_uniform = tf.random.uniform(shape=(10,), minval=0, maxval=1)
    print('rand_uniform.shape : ',rand_uniform.shape)
    print('rand_uniform : ', rand_uniform)

    # 정규분포 난수 생성
    rand_normal = tf.random.normal(shape=(10, 2), mean=0., stddev=1.)
    print('rand_normal.shape : ',rand_normal.shape)
    print('rand_normal : ', rand_normal)

def eager_execution():
    a = tf.constant(5)
    b = tf.constant(2)
    print('tf.add(a,b) : ', tf.add(a,b))
    print('a+b : ', a+b)
    print('tf.subtract(a,b) : ', tf.subtract(a,b))
    print('a-b :',a-b)
    print('tf.multiply(a,b) : ',tf.multiply(a,b))
    print('a*b : ',a*b)

def tensor_and_numpy():
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

def slicing_tensor():
    tensor = tf.constant([[1., 3., 5.],[7., 9., 11.]])
    print(tensor)
    print(tensor[:, 1:])
    print(tensor[..., 1, tf.newaxis])

def tensor_function():
    tensor = tf.constant([[1., 3., 5., 7.],[7., 9., 11., 13.]])
    print(tensor + 1)
    print(tf.square(tensor))
    print(tf.transpose(tensor))
    print(tf.matmul(tensor, tf.transpose(tensor)))
    print(tf.cast(tensor, dtype=tf.int32))

def function_with_tf_function_annot():

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


def model_with_tf_function():
    inputs = tf.random.normal((120, 28, 28))

    eager_model = Sequential()
    eager_model.add(Input(shape=(28,28)))
    eager_model.add(Dense(units=50, activation='relu'))
    eager_model.add(Dense(units=100, activation='relu'))
    eager_model.add(Dense(units=10, activation='softmax'))

    graph_model = tf.function(eager_model)

    print('클래스 Eager time ', timeit.timeit(lambda: eager_model(inputs), number=10000))
    print('클래스 Graph time ', timeit.timeit(lambda: graph_model(inputs), number=10000))


def graph_gradient():
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


#----------------------------------------------------------------------------

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tensorflow 기본 예제 코드',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(cmd='tensor_contant', desc='tf.constant Tensor')
    
    p = add_command(cmd='random_tensor',  desc='tf.random Tensor')
    
    p = add_command(cmd='eager_execution',  desc='Eager execution : 세션을 초기화 없이 그래프를 즉시 실행')
    
    p = add_command(cmd='tensor_and_numpy',  desc='Tensor와 Numpy 배열 호환성')

    p = add_command(cmd='slicing_tensor',  desc='Tensor 슬라이싱')

    p = add_command(cmd='tensor_function',  desc='Tensor 지원 함수 예제')

    p = add_command(cmd='function_with_tf_function_annot',  desc='tf.function 유무에 따른 함수 수행시간 비교')

    p = add_command(cmd='model_with_tf_function',  desc='tf.function 유무에 따른 Keras 모델 학습 시간 비교')

    p = add_command(cmd='graph_gradient',  desc='미분 연산')
    
    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)