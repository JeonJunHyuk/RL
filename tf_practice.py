import tensorflow as tf
import numpy as np

x = [1,2,3,4,5]
y = [1,2,3,4,5]

W = tf.Variable(2.9)
b = tf.Variable(0.5)

hypo = W*x + b
cost = tf.reduce_mean(tf.square(hypo - y))

learning_rate = 0.01

for i in range(101):
    with tf.GradientTape() as tape:
        hypo = W*x + b
        cost = tf.reduce_mean(tf.square(hypo - y))

    W_grad, b_grad = tape.gradient(cost, [W,b])

    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print("{:5}|{:10.4}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

print(W*5+b)
print(W*2.5+b)


