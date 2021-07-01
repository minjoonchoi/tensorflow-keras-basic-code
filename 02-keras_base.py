import sys
import argparse

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Concatenate, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

def sequential_model():
    model = Sequential()
    model.add(Input(shape=(28,28)))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    # 모델 구조 출력
    model.summary()

    # 모델 구조 시각화
    plot_model(model, to_file='./keras_base_sequential_model.png', show_shapes=True)

    return model

def functional_model():
    inputs = Input(shape=(28, 28, 1))
    x = Flatten(input_shape=(28, 28, 1))(inputs)
    x = Dense(units=50, activation='relu')(x)
    x = Dense(units=100, activation='relu')(x)
    outputs = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    # 모델 구조 출력
    model.summary()

    # 모델 구조 시각화
    plot_model(model, to_file='./keras_base_functional_model.png', show_shapes=True)

    return model

def multiple_input_model():
    input_1 = Input(shape=(10,3))
    hidden_1 = Dense(units=50, activation='relu')(input_1)
    hidden_2 = Dense(units=100, activation='relu')(hidden_1)

    input_2 = Input(shape=(10,5))
    hidden_3 = Dense(units=100, activation='relu')(input_2)

    concat = Concatenate()([hidden_2, hidden_3])
    outputs = Dense(units=10, activation='softmax')(concat)

    model = Model(inputs=[input_1, input_2], outputs=[outputs])
    # 모델 구조 출력
    model.summary()

    # 모델 구조 시각화
    plot_model(model, to_file='./keras_base_multiple_input.png', show_shapes=True)

    return model


#----------------------------------------------------------------------------

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tensorflow-Keras 모델 예제 코드',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(cmd='sequential_model', desc='Sequential API 모델 예제 : 다중 입/출력 불가')
    
    p = add_command(cmd='functional_model',  desc='Functional API 모델 예제 (Recommended)')
    
    p = add_command(cmd='multiple_input_model',  desc='다중 입/출력을 사용하는 Functional API 모델 예제')
    
    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)