#전체적인 표를 0으로 초기화하라는 것은 Table.PNG참고
import gym
import numpy as np
import random as pr

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

env = gym.make("FrozenLake-v3")
env.render()

Q = np.zeros([env.observation_space.n, env.action_space.n])
#해당 표를 0으로 초기화를 시키는데, state는 16, action은 4인 것을 알고 있지만 해당 방법으로 불러올 수 있다.
num_episodes = 2000 #몇번정도 수행을 할것인가?

rList = []
for i in range(num_episodes):
    state = env.reset() #첫번째 state를 가져온다.
    rAll = 0
    done = False

    while not done: #게임이 끝날때까지 반복
        action = rargmax(Q[state, :]) #초반에 값이 똑같다면 랜덤한 방향으로 간다. (초기에는 전부 0)

        new_state, reward, done, _ = env.step(action)

        Q[state, action] = reward + np.max(Q[new_state,:]) #:는 모든 경우를 의미한다.
        #Q를 업데이트하는 가장 중요한 공식

        rAll += reward #reward에 대한 값을 더하게 된다.
        state = new_state
    rList.append(rAll) #각 에피소드의 결과를 기록하는 list이다. 전부 더해서 2000으로 나누면 평균을 구할 수 있다.

#해당 코드는 알고리즘 부분만 구현을 했으며, 직접적인 코드는 돌아가지 않습니다.
# 전체적인 code의 경우 해당 강좌의 강의 슬라이드를 참고