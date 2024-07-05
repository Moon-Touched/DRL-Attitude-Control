# DRL-Attitude_control

## 需要的库

```python
pip install gymnasium
pip install stable-baselines3
pip install numpy
```

如果需要画图还需要`matplotlib`​

## 动力学模型(dynamic.py)

数学模型参考姿态控制

* `model = DynamicModel(T_s: float, ref_quat, mode):`​

  `T_s`​：采样时间

  `ref_quat`​：目标姿态四元数

  `mode`​：0代表输入三轴力矩，1代表输入四个反作用轮力矩。

  ‍
* `step(torque_in: np.ndarray[np.float32]):`​

  每步迭代记录当前四元数，当前误差四元数，当前角速度（体坐标系），当前输入力矩。

### 控制律

* 使用PD控制器验证动力学模型

  将误差四元数 $\mathbf{e}=[e_w\ \mathbf{e}_v]^T$ 的虚部 $\mathbf{e}_v$ 作为控制器的输入，则:

  $$\mathbf{T}=K_P\mathbf{e}_v-K_d\boldsymbol{\omega}_b$$

  其中 $T$ 为三维向量，对应三轴控制力矩， $\mathbf{e_v}$ 为输入误差， $\boldsymbol{\omega}_b$ 为相对于体坐标系的三轴角速度。

  对于姿态控制，角速度最终会稳定的到0，因此直接将角速度取反输入即可。

## 环境

* 动作空间

  四维向量，表示反作用轮力矩，顺序是x，y，z和斜向轮。力矩范围[-0.01, 0.01]
* 观测空间

  6维向量，将 $\mathbf{e_v}$ 和 $\boldsymbol{\omega}_b$ 拼接得到。
* 奖励函数

  [ ] 效果不太好，调整后再训练试试

  $$\begin{array}R = r_1 + r_2 + r_3 & \\\\r_1 = \begin{cases}1 & \text{if } \theta_e < 0.025^\circ \\0 & \text{otherwise}\end{cases} & \\\\r_2 = \frac{\theta_e(t - \Delta t) - \theta_e(t)}{\pi} & \\\\r_3 = \begin{cases}-1 & \text{if any } |\omega_i| > 1 \text{ rad/s} \\0 & \text{otherwise}\end{cases} & \\\end{array}$$

  其中 $\theta_e=2\arccos({e_w})$ 

## 训练

```python
 model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
    for i in range(num_episode):
        print(f"num:{i}")
        model.learn(total_timesteps=num_timestep, reset_num_timesteps=False, tb_log_name=name)
        model.save(f"{models_dir}/{model.num_timesteps}")
```
