import argparse
from stable_baselines import PPO2

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.cmd_util import make_atari_env

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--load_path",
        help="Load Path of policy",
        required=True)
    args = parser.parse_args()
    load_path = args.load_path

    # env = VecFrameStack(make_atari_env(env_id, n_envs, seed), 4)
    env = VecFrameStack(make_atari_env("Phoenix-v4", 1, 0), 4)

    # Load the trained agent
    model = PPO2.load(load_path, env=env)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()