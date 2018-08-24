#이제 Network를 통해서 학습을 제대로 시켜서 진행을 한다.
import numpy as np
import tensorflow as tf
import gym
env = gym.make('CartPole-v0')

learning_rate = 1e-1
input_size = env.observation_space.shape[0] #4개의 input => Q_Network_for_Cart_Pole1.py를 실행해보면 observation을 출력해 놓은 부분이 있다.
output_size = env.action_space.n #출력은 오른쪽, 왼쪽 2가지 뿐이다. 막대를 떨어뜨리지 않기 위해서

X = tf.placeholder(tf.float32, [None, input_size], name="input_x")
#None이지만 어떠한 값이든 들어올 수 있게 만들었을 뿐, 1이라고 알면 된다.

W1 = tf.get_variable("W1", shape=[input_size, output_size],
                     initializer=tf.contrib.layers.xavier_initializer())
#그냥 variable를 사용하는 것보다 get_variable를 사용하면 좋은점 : "W1"이라는 이름을 정할 수 있고, xavier이라는 좀 더 좋은 초기화를 진행할 수 있다.
#결론적으로 하나의 형태이기 때문에 그렇다고 알아둘 것.
Qpred = tf.matmul(X, W1)

Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)
#이곳의 None는 4이다.

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#Adam으로 loss를 학습시킨다.

num_episodes = 2000
dis = 0.9
rList = []

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(num_episodes):
    e = 1. / ((i / 10) + 1)
    rAll = 0
    step_count = 0
    s = env.reset()
    done = False

    while not done:
        step_count += 1
        x = np.reshape(s, [1, input_size]) #입력받은 s를 우리의 array에 맞게 [1, input_size]로 reshape해준다.
        Qs = sess.run(Qpred, feed_dict={X: x})
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0, a] = -100 #이 경우 막대가 넘어진 것이기때문에 경고의 의미로 -100이라는 큰 값을 준다.
        else:
            x1 = np.reshape(s1, [1, input_size])
            Qs1 = sess.run(Qpred, feed_dict={X: x1}) #다음 상태의 예측값
            Qs[0, a] = reward + dis * np.max(Qs1)

        sess.run(train, feed_dict={X: x, Y: Qs}) #train학습, Y의 값으로 y_label이 들어간다.
        s = s1

    rList.append(step_count)
    print("Episode: {} steps: {}".format(i, step_count))
    if len(rList) > 10 and np.mean(rList[-10:]) > 500: #500이상 성공을 하면 빠져나가게 설정
        break

    #Q_Network가 학습이 잘 되었으므로 이제 실험을 해본다.
    observation = env.reset() #상태 초기화
    reward_sum = 0
    while True:
        env.render()

        x = np.reshape(observation, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X: x}) #x값을 만들고 Q에게 물어본다.
        a = np.argmax(Qs)

        observation, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break

#해당 network는 학습이 잘 되지 않는다.
#W가 하나밖에 없으며, 4개의 input밖에 없어서 학습을 매우 조금씩 할 수 밖에 없다고 생각해 볼만 하다.