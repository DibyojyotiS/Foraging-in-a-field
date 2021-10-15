from PIL import Image
import numpy as np
import get_env

env = get_env.get_env()
acs = [[0,100],[5, 70],[4,800]]
gif_images = []
for a,steps in acs:
    for step in range(steps):
        obs, r, done, info = env.step(a)
        env.render()
        # if step%4: gif_images.append(Image.fromarray(env.render(returnRGB=True)))
# gif_images[0].save('imagedraw.gif', save_all=True, append_images=gif_images[1:], optimize=False, duration=0.2, loop=0)
env.viewer.close()
