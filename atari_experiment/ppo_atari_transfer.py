import os
import argparse

from stable_baselines import PPO2, logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy

from atari_experiment.utils import gen_exp_log_dir_name


def train(num_timesteps,
          seed,
          base_policy_path,
          base_task_id,
          target_task_id,
          n_envs=8,
          nminibatches=4,
          n_steps=128):
    """
    Train PPO2 model for atari environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    :param base_policy_path: (str) Path to the base policy log file.
    :param n_envs: (int) Number of parallel environments
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    """
    exp_log_path, tensorboard_log, model_save_path = \
        gen_exp_log_dir_name(target_task_id, base_task_id)
    env = VecFrameStack(make_atari_env(target_task_id, n_envs, seed), 4)

    model = PPO2.load(base_policy_path, env=env)
    model.pre_train_vf = True
    model.tensorboard_log = tensorboard_log
    model.full_tensorboard_log = True

    model.learn(total_timesteps=num_timesteps)

    model.save(model_save_path)

    env.close()
    # Free memory
    del model


def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument("-l",
                        "--base_policy_path",
                        help="Load Path of base policy",
                        required=True)
    parser.add_argument("-b",
                        "--base_task_id",
                        help="Base task to transfer from",
                        required=True)
    parser.add_argument("-t",
                        "--target_task_id",
                        help="Target task to transfer to",
                        required=True)
    args = parser.parse_args()
    logger.configure()
    train(num_timesteps=args.num_timesteps,
          seed=args.seed,
          base_policy_path=args.base_policy_path,
          base_task_id=args.base_task_id,
          target_task_id=args.target_task_id)


if __name__ == '__main__':
    main()
