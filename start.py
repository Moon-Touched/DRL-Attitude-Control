import os, multiprocessing

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from Train import train

if __name__ == "__main__":
    processes = []
    args = [
        {
            "path": "C:\\training\\train_Basilisk",
            "env_name": "menv01",
            "num_timestep": 10000,
            "num_episode": 1000,
            "env_num": 4,
            "faulty": False,
            "torque_mode": "wheel",
            "device": "cpu",
        },
        {
            "path": "C:\\training\\train_Basilisk",
            "env_name": "menv01",
            "num_timestep": 10000,
            "num_episode": 1000,
            "env_num": 4,
            "faulty": True,
            "torque_mode": "wheel",
            "device": "cpu",
        },
    ]
    for a in args:
        process = multiprocessing.Process(target=train, kwargs=a)
        processes.append(process)
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()
