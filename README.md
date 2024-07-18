# DRL-Attitude_control

# 需要的库

* 动力学仿真：Basilisk

  [Welcome to Basilisk: an Astrodynamics Simulation Framework — Basilisk 2.3.0 documentation (hanspeterschaub.info)](https://hanspeterschaub.info/basilisk/index.html)

  * [ ] 自己的四元数模型与 Basilisk 的误差大约在 10^-4，训练效果不好，待验证是误差引起的还是训练环境本身有问题。
* pip

  ```python
  pip install gymnasium
  pip install stable-baselines3
  pip install numpy
  pip install matplotlib
  pip install tensorboard
  ```

# 动力学模型(dynamic.py)

* ​`BasiliskModel`​

  * `def __init__(self, I: list, ref_MRP: np.ndarray, torque_mode: str = "wheel")`​

    ​`I`​：转动惯量（展开成一行，貌似 `np.array` ​与 Basilisk 使用的 `Eigen` ​库有冲突，传入后自动使用 Basilisk 的函数转换为矩阵）

    ​`ref_MRP`​：MRP 目标姿态

    ​`torque_mode`​：`"wheel"`​，输入四轮力矩，映射到三轴。`"axis"` ​直接输入三轴力矩。

  * ​`def step(self, cur_nano_second: int, torque: np.ndarray):`​

    ​`cur_nano_second`​：绝对时间，等于仿真步数乘以步长

    返回 `return self.cur_MRP`​，`self.cur_error_MRP`​，`self.cur_error_angle`​，`self.cur_omega`​，`self.cur_omega_dot`​

## 验证

* 使用 PD 控制器验证动力学模型

  将 `cur_error_MRP` ​和 `cur_omega` ​输入:

  $$
  \mathbf{T}=K_P\mathbf{e}-K_d\boldsymbol{\omega}
  $$

  对于姿态控制，角速度最终会稳定的到 0，因此直接将角速度取反输入即为误差。

# 环境

* 动作空间

  四维向量，表示反作用轮力矩，顺序是 x，y，z 和斜向轮。力矩范围[-0.01, 0.01]
* 观测空间

  6 维向量，`cur_error_MRP` ​和 `cur_omega`​
* 奖励函数

  ​`BasiliskModel` ​的 `calculate_reward` ​留空，方便重载。

  ```python
  from Envs import BasiliskEnv


  class TrainEnv(BasiliskEnv):
      def calculate_reward(self):
  	......
  	return reward
  ```

# 训练

* Train.py

  ```python
  def train(
      path: str,
      env_name: str,
      num_timestep: int,
      num_episode: int = 2000,
      env_num=4,
      vec_env_cls=DummyVecEnv,
      faulty=False,
      torque_mode: str = "wheel",
      device="cuda",
  ):
  ```

  * ​`path`​(str): The path to save the trained models and logs.
  * ​`env_name`​(str): The name of the environment to train on.
  * ​`num_timestep`​ (int): The total number of timesteps to train for.
  * ​`num_episode`​(int, optional): The number of episodes to train for. Defaults to 2000.
  * ​`env_num`​(int, optional): The number of parallel environments. Defaults to 4.
  * ​`vec_env_cls`​(class, optional): The class for creating vectorized environments. Defaults to DummyVecEnv.
  * ​`faulty`​(bool, optional): Whether to use a faulty environment. Defaults to False.
  * ​`torque_mode`​(str, optional): The torque mode to use. Defaults to "wheel".
  * ​`device`​(str, optional): The device to use for training. Defaults to "cuda".

> 将board.bat执行 `tensorboard --logdir ./`​ 放在log目录运行即可使用TensorBoard观测训练过程。

# 加载并应用模型

* load_model.py

  ```python
  import os

  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
  from stable_baselines3 import PPO
  from Envs import BasiliskEnv
  import Tools


  def load(path: str):
      model = PPO.load(path)
      env = BasiliskEnv(faulty=False, torque_mode="wheel")
      obs, _ = env.reset()
      for i in range(6000):
          action, _states = model.predict(obs)
          obs, reward, terminated, truncated, info = env.step(action)
      Tools.plot_history(env.model)


  load("E:\\training\\train_Basilisk\\env01_False_wheel\\model\\3473408.zip")
  ```

  运行后输出如下图像：

  ![Figure_1](https://github.com/user-attachments/assets/385454d3-3993-4d42-aff4-a187b63cddf9)
​

# 预设奖励函数：

配合`env_num`​使用

* Vec_Env_01

  $$
  \begin{align*}
  &r_1=\left (\theta_e(k-1)-\theta_e(k)\right)/\pi
  \\
  &r_2=\begin{cases}
  	-1 & if |\omega|>1rad/s
  	\\
  	0  & otherwise
      \end{cases}
  \\
  &r_3=\begin{cases}
  	1 & \text{if step == 6000 and }\theta_e(k)<0.25\degree
  	\\
  	0  & otherwise
      \end{cases}
  \end{align*}
  $$

* Vec_Env_02

  $$
  \begin{align*}
  &r_1=(10-\theta_e(k))(\theta_e(k-1)-\theta_e(k))/\pi

  \\
  &r_2=\begin{cases}
  	-1 & if\ |\omega|>1rad/s
  	\\
  	0  & otherwise
      \end{cases}
  \\
  &r_3=\begin{cases}
  	1 & if \ step == 6000\ and\ \theta_e(k)<0.25\degree
  	\\
  	0  & otherwise
      \end{cases}
  \end{align*}
  $$

* Vec_Env_03

  $$
  \begin{align*}
  &r_1=(10-\theta_e(k))(\theta_e(k-1)-\theta_e(k))/\pi

  \\
  &r_2=\begin{cases}
  	-1 & if\ |\omega|>1rad/s
  	\\
  	0  & otherwise
      \end{cases}
  \\
  &r_3=\frac{1}{1+\theta_e(k)}
  \end{align*}
  $$

* Vec_Env_04

  $$
  \begin{align*}
  &r_1=(\theta_e(k-1)-\theta_e(k))/\theta_e(k-1)

  \\
  &r_2=\begin{cases}
  	-1 & if\ |\omega|>1rad/s
  	\\
  	0  & otherwise
      \end{cases}
  \\
  &r_3=\begin{cases}
  	1 & if \ \theta_e(k)<0.25\degree
  	\\
  	0  & otherwise
      \end{cases}
  \end{align*}
  $$
