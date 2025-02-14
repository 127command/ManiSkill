import mani_skill.envs
import gymnasium as gym
from PIL import Image
import numpy as np
nu=19
env = gym.make('final', render_mode="rgb_array", num_envs=nu, reconfiguration_freq=1)
env.reset()
for i in range(nu):
    img = env.render()[i].cpu()
    img = np.array(img)
    image = Image.fromarray(img)
    image.save(f'/home/changruinian/bowen/EAI/ManiSkill/mtask/{i+1}.png')

# env.reset()
# img = env.render()[0].cpu()
# img = np.array(img)
# image = Image.fromarray(img)
# image.save('/home/changruinian/bowen/EAI/ManiSkill/mtask/2.png')

# env.reset()
# img = env.render()[0].cpu()
# img = np.array(img)
# image = Image.fromarray(img)
# image.save('/home/changruinian/bowen/EAI/ManiSkill/mtask/3.png')

# env.reset()
# img = env.render()[0].cpu()
# img = np.array(img)
# image = Image.fromarray(img)
# image.save('/home/changruinian/bowen/EAI/ManiSkill/mtask/4.png')