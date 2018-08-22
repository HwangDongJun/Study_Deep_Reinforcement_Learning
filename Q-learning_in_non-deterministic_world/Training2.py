import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
#Q-learning_exploit&exploration_and_discounted_reward 에서 제공하는 코드에서와의 차이점
'''
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery' : False} #is_slippery : False는 내가 이 상황이 주어졌을때, 이동시 미끄럽지 않게 하라. 즉, 한칸씩만 이동이 가능하게 한 것이다.
)

env = gym.make('FrozenLake-v3') 이것과 차이가 있다.
'''

Q = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = .85
dis = .99
num_episodes = 2000

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
        new_state, reward, done, _ = env.step(action)

        Q[state, action] = reward + dis * np.max(Q[new_state,:])
        state = new_state

        rAll += reward

    rList.append(rAll)

print("Score over time: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()