import gym

# from vizdoom import gym_wrapper  # noqa
import vizdoomgym

if __name__ == "__main__":
    env = gym.make("VizdoomMyWayHomeVerySparse-v0", render_mode="human")


    # Rendering random rollouts for ten episodes
    for _ in range(5):
        done = False
        ret = 0
        obs = env.reset()
        ret = 0
        while not done:
            obs, rew, terminated, info = env.step(env.action_space.sample())
            ret += rew
            done = terminated
            if done:
                print("return: ", ret)
