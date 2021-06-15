#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

learning_rate = 1e-3
training_steps = 1000

# Optimizer
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# Input
X = np.random.randn(50)
# Label
Y = 2*X + np.random.randn(50)

# Weight
W = tf.Variable(np.random.randn(), name='weights')
# Bias
B = tf.Variable(np.random.randn(), name='biases')

# 회귀 모델 함수
def linear_regression(x):
    return W * x + B

def logistic_regression(x):
    return W * x + B

# 손실 함수
def loss_fun_v1(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

def optimize_gradients():
    with tf.GradientTape() as tape:
        pred = linear_regression(X)
        loss = loss_fun_v1(pred, Y)
    
    gradients = tape.gradient(loss, [W, B])

    optimizer.apply_gradients(zip(gradients, [W, B]))

def use_apply_gradients():
    for step in range(1, training_steps + 1):
        optimize_gradients()

        if step % 50 == 0:
            pred = linear_regression(X)
            loss = loss_fun_v1(pred, Y)
            print('step: {:4d}\t loss: {:.4f}\t W: {:.4f}\t B: {:.4f}'.format(step, loss, W.numpy(), B.numpy()))

def loss_fun_v2():
    pred_y = linear_regression(X)
    return tf.reduce_mean(tf.square(Y-pred_y))

def use_minimize(epochs=1000):
    for epoch in range(1, epochs + 1, 1):
        optimizer.minimize(loss_fun_v2, var_list=[W,B])
        
        if epoch % 100 == 0:
            print('epoch: {:4d}\t W: {:.4f}\t B: {:.4f}'.format(epoch, W.numpy(), B.numpy()))

def plot_model():
    plt.plot(X,Y, 'ro', label='Data')
    plt.plot(X, np.array(X*W + B), label='Fitted Line')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # 모델 학습
    # use_apply_gradients()
    use_minimize()

    # 회귀모델 그래프 확인
    plot_model()


if __name__ == '__main__':
    main()