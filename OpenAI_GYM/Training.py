import gym
from gym.envs.registration import register
# import sys,tty,termios

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
) #map의 크기는 4 x 4이다.

env = gym.make("FrozenLake-v3")
env.render()
#환경 생성 후, render()

# class _Getch:
#    def __call__(self):
#        fd=sys.stdin.fileno()
#        old_settings=termios.tcgetattr(fd)
#        try:
#            tty.setraw(sys.stdin.fileno())
#            ch=sys.stdin.read(3)
#        finally:
#            termios.tcsetattr(fd,termios.TCSADRAIN,old_settings)
#
#        return ch
#해당 과정은 terminal에서 key를 입력받기 위해 사용하는 방법이다. pycharm에서는 쓰이지 않는다.

from msvcrt import getch
# inkey=msvcrt.getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {72: UP,
              80: DOWN,
              77: RIGHT,
              75: LEFT}
#key가 눌러지면 들어오는 코드에 해당한다. 결국에는 컴퓨터로 해야하기 때문에 key를 설정하여 나 자신이 action의 주체가 되는 것이다.
while True: #무한 반복을 통해 break의 결과가 존재할때까지 진행한다.
    key = getch()
    # print(ord(key))
    if ord(key) == 224:
        key = ord(getch())
        # print(key)
    if key not in arrow_keys.keys(): #설정한 key를 제외하고 입력을 하면 break된다.
        print("Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)

    env.render()

    print("State ", state, "Action ", action, "Reward: ", reward, "Info: ", info)
    #최종적으로 현재 위치나 상태등을 알 수 있는 print이다.
    if done: #원하는 목표에 도착하였을 때
        print("Finished with reward", reward)
        break