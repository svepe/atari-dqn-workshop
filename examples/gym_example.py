from time import sleep
from gym import envs, make

# List all available environments
# print([env.id for env in envs.registry.all()])

# List all ATARI environments
# print([env.id for env in envs.registry.all()
#        if env.id.endswith("NoFrameskip-v4")])

env = make("PongNoFrameskip-v4")
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Get the initial observation (s0)
observation = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation.shape, reward, done)
    env.render()
    sleep(0.01)

env.close()
