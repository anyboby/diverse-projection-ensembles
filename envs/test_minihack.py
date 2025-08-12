import gym
import minihack

if __name__ == "__main__":
    env = gym.make("MiniHack-ExploreMaze-Easy-v0", observation_keys=("pixel",),)

    # Rendering random rollouts for ten episodes
    for _ in range(10):
        done = False
        ret = 0
        obs = env.reset()
        ret = 0
        while not done:
            obs, rew, terminated, info = env.step(env.action_space.sample())
            # env.render()
            ret += rew
            done = terminated
            if done:
                print("return: ", ret)
