from get_env import get_env

env = get_env()

for i in range(100):
    obs, r, done, info = env.step(1)
    print(r)
    env.render()
