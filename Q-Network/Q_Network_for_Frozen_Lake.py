#입력의 경우 state를 onehot을 이용하여 16가지 경우의 array를 만든다.
import gym
import numpy as np
import tensorflow as tf

def one_hot(x):
    return np.identity(16)[x:x + 1] #np.identity(16)의 table에서 x가 7일경우 [7:8] (slice한것) 즉, 7번째 것만 뽑아서 사용한다.
#np.identity(16)은 0,0 / 1,1 ~ 이렇게 대각선의 값이 1인 16 x 16 table이 형성된다.
#[x:x + 1]은

env = gym.make('FrozenLake-v0')

input_size = env.observation_space.n #16 (16개의 숫자)
output_size = env.action_space.n #4 (4방향에 해당한다.)
learning_rate = 0.1

X = tf.placeholder(shape=[1,input_size], dtype=tf.float32) #one_hot이니까 shape가 1, 16이다.
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01)) #Weight의 크기는 4 x 16이다.

Qpred = tf.matmul(X, W) #Q_hat이며, Ws이며, Qpred이다. (예측하는값)
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32) #Y label

loss = tf.reduce_sum(tf.square(Y - Qpred)) #제곱을 할 것이기때문에 Y와 Qpred의 순서가 바뀌어도 상관이 없다.

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

dis = .99
num_episodes = 2000

rList = []
init = tf.global_variables_initializer() #초기화
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        s = env.reset()
        e = 1. / ((i / 50) + 10)
        rAll = 0
        done = False
        local_loss = []

        while not done:
            Qs = sess.run(Qpred, feed_dict={X: one_hot(s)}) #예측값을 출력받아야한다.
            # ↓ action을 어떻게 시킬 것인가?
            if np.random.rand(1) < e: #random하게 가거나 아니면 argmax로 가장 큰 값쪽으로 가거나
                a = env.action_space.sample() #a는 action에 해당
            else:
                a = np.argmax(Qs)

            s1, reward, done, _ = env.step(a)
            if done: #done이 True면 게임이 끝났을 경우이다. reward를 넣게된다.
                Qs[0, a] = reward
            else:
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)}) #다음 상태를 one_hot으로 넣게되는 것
                Qs[0, a] = reward + dis * np.max(Qs1) #Q_learning을 계산하는 공식이다.

            #★왜 Qs[0, a]인가? input이 1 x 16 이며, W가 4 x 16이다. 그렇게 된다면 출력은 1 x 4로 나타나게 된다.
            #42번째줄에서 보면 왜 Qs가 1 x 4로 출력이 되는지 알 수 있다. [[a1, a2, a3, a4]]이렇게 되기 때문에 Qs[a]가 아닌 Qs[0, a]인 것이다.

            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})
            #train을 실행을 시키므로, 학습이 일어난다.

            rAll += reward
            s = s1
        rList.append(rAll)

print("Percent of successful episodes: ", str(sum(rList)/num_episodes) + "%")