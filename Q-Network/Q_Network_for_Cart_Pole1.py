import gym
#CartPole은 손바닥 위에 막대를 올려두고 넘어뜨리지 않고 세워두는 게임
env = gym.make('CartPole-v0')
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    env.render() #rendering한다는 것은 화면에 상황을 보여준다는 것이다.
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    print(observation, reward, done)
    reward_sum += reward
    if done: #True로 성공했을 경우
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()

#해당방법은 random하게 살아남는 방법으로 학습은 전혀 하지 않는다. 말 그대로 random하다.