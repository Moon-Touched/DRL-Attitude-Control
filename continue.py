from stable_baselines3 import PPO
from env import MyEnv


def continue_train(path: str, filename: str = "", num_timestep: int = 6000, num_episode: int = 2000, learning_rate=0.0003):
    models_dir = f"{path}/model"
    logdir = f"{path}/logs"
    name = f"LR{learning_rate}"

    env = MyEnv()
    env.reset()

    model = PPO.load(path=f"{models_dir}/{filename}", env=env, tensorboard_log=logdir)
    for i in range(num_episode):
        model.learn(total_timesteps=num_timestep, reset_num_timesteps=False, tb_log_name=name)
        model.save(f"{models_dir}/{model.num_timesteps}")


continue_train("C:/training/LR0.0003", "2457600.zip", num_timestep=6000, num_episode=2000, learning_rate=0.0003)
