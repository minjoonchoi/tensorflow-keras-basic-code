import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Concatenate, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

def check_sequential_model():
    """Sequential API 모델 예제로 
    다중 입/출력 모델은 불가능
    """
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

def check_functional_model():
    """Functional API 모델 예제로 
    다중 입/출력이 가능해 네트워크 구성시 가장 권장되는 방법
    """
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

def check_multiple_input_model():
    """다중 입/출력을 사용하는 Functional API 모델
    """
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


def main():
    # Sequential API 사용 네트워크
    check_sequential_model()

    # Functional API 사용 네트워크
    check_functional_model()

    # Functional API + 다중 입력
    check_multiple_input_model()
    
if __name__=='__main__':
    main()
