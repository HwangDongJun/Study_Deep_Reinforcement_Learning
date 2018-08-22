#OpenAI_GYM에서와 같이 내가 직접 움직이는 방법

import gym
import readchar #나의 key를 입력받기 위한 import이다.

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT
}
#해당 코드는 그냥 화살표를 입력받기 위한 방법이라고만 알아두면 된다.

# is_slippery True
env = gym.make('FrozenLake-v0')
env.render()

while True:
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break