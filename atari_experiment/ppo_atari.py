import os

from stable_baselines import PPO2, logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy

from atari_experiment.utils import gen_exp_log_dir_name, exp_log


def train(env_id,
          num_timesteps,
          seed,
          policy,
          n_envs=8,
          nminibatches=4,
          n_steps=128,
          frameskip=4,
          mode=0,
          difficulty=0):
    """
    Train PPO2 model for atari environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    :param n_envs: (int) Number of parallel environments
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    """

    env = VecFrameStack(
        make_atari_env(env_id,
                       n_envs,
                       seed,
                       frameskip=frameskip,
                       mode=mode,
                       difficulty=difficulty), 4)
    policy = {
        'cnn': CnnPolicy,
        'lstm': CnnLstmPolicy,
        'lnlstm': CnnLnLstmPolicy,
        'mlp': MlpPolicy
    }[policy]
    exp_log_path, tensorboard_log, model_save_path = \
        gen_exp_log_dir_name(env_id)
    model = PPO2(policy=policy,
                 env=env,
                 n_steps=n_steps,
                 nminibatches=nminibatches,
                 lam=0.95,
                 gamma=0.99,
                 noptepochs=4,
                 ent_coef=.01,
                 learning_rate=lambda f: f * 2.5e-4,
                 cliprange=lambda f: f * 0.1,
                 verbose=1,
                 tensorboard_log=tensorboard_log,
                 full_tensorboard_log=False)
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
    parser.add_argument('--policy',
                        help='Policy architecture',
                        choices=['cnn', 'lstm', 'lnlstm', 'mlp'],
                        default='cnn')
    parser.add_argument('--frameskip',
                        help='Frame skipping',
                        default=4)
    parser.add_argument('--mode',
                        help='Mode of atari game',
                        default=0)
    parser.add_argument('--difficulty',
                        help='Difficulty of atari.',
                        default=0)
    args = parser.parse_args()
    logger.configure()
    train(args.env,
          num_timesteps=args.num_timesteps,
          seed=args.seed,
          policy=args.policy,
          frameskip=args.frameskip,
          mode=args.mode,
          difficulty=args.difficulty)


if __name__ == '__main__':
    main()
