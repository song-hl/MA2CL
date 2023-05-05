import datetime
import os
import socket
import warnings

import hydra
import numpy as np
import setproctitle

# import termcolor
import wandb
from omegaconf import DictConfig, OmegaConf

# from envs.dexteroushandenvs.utils.config import get_args, load_env_cfg, parse_sim_params
# from envs.dexteroushandenvs.utils.parse_task import parse_task
# from envs.dexteroushandenvs.utils.process_marl import get_AgentIndex
from envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from envs.starcraft2.smac_maps import get_map_params
from envs.starcraft2.StarCraft2_Env import StarCraft2Env

import runners
import torch

warnings.filterwarnings("ignore")


def get_runner(env_name, algorithm_name):
    # assert env_name in ["football", "hands", "mujoco", "StarCraft2"]
    runner_dict = {
        "seperated": {
            "football": runners.separated.football_runner.FootballRunner,
            "pixel_football": runners.separated.football_runner.FootballRunner,
            # "hands": runners.separated.hands_runner.HandsRunner,
            "mujoco": runners.separated.mujoco_runner.MujocoRunner,
            "StarCraft2": runners.separated.smac_runner.SMACRunner,
            "butterfly": runners.separated.butterfly_runner.ButterflyRunner,
            "drone": runners.separated.drone_runner.DroneRunner,
        },
        "shared": {
            "football": runners.shared.football_runner.FootballRunner,
            "pixel_football": runners.shared.football_runner.FootballRunner,
            # "hands": runners.shared.hands_runner.HandsRunner,
            "mujoco": runners.shared.mujoco_runner.MujocoRunner,
            "StarCraft2": runners.shared.smac_runner.SMACRunner,
            "butterfly": runners.shared.butterfly_runner.ButterflyRunner,
            "drone": runners.shared.drone_runner.DroneRunner,
        },
    }
    runner_type = "seperated" if algorithm_name in ["happo", "hatrpo","happo_maska"] else "shared"
    return runner_dict[runner_type][env_name]


def make_env(all_args: DictConfig, eval=False):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "football":
                from envs.football.football_env import FootballEnv

                env_args = {
                    "scenario": all_args.scenario,
                    "n_agent": all_args.n_agent,
                    "reward": "scoring",
                    "use_sight_range": all_args.use_sight_range,
                    "sight_range": all_args.sight_range
                }
                env = FootballEnv(env_args=env_args)
            elif all_args.env_name == "pixel_football":
                from envs.football.pixel_football_env import PixelFootballEnv

                env_args = {
                    "scenario": all_args.scenario,
                    "n_agent": all_args.n_agent,
                    "reward": "scoring",
                    # "pre_transform_image_size": all_args.pre_transform_image_size,
                }
                env = PixelFootballEnv(env_args=env_args)
            elif all_args.env_name == "mujoco":
                from envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

                env_args = {
                    "scenario": all_args.scenario,
                    "agent_conf": all_args.agent_conf,
                    "agent_obsk": all_args.agent_obsk,
                    "episode_limit": 1000,
                }
                env = MujocoMulti(env_args=env_args)
            elif all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            elif all_args.env_name == "butterfly":
                from envs.butterfly.butterfly_env import ButterflyEnv
                env = ButterflyEnv(all_args)
            elif all_args.env_name == "drone":
                env_args = {
                    "map_name": all_args.map_name,
                    "num_agents": all_args.num_agents,
                    "is_initial_xyz": all_args.is_initial_xyz,
                    "is_initial_rpy": all_args.is_initial_rpy,
                    "action_type": all_args.action_type,
                    "observation_type": all_args.observation_type,
                    "use_camera_state": all_args.use_camera_state,
                }
                from envs.gym_pybullet_drones.drone_env import DroneEnv
                env = DroneEnv(env_args)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError

            if eval:
                env.seed(all_args.seed * 50000 + rank * 10000)
            else:
                env.seed(all_args.seed + rank * 1000)

            return env

        return init_env
    if all_args.env_name == "drone":
        if all_args.use_camera_state:
            all_args.use_share_obs = True
        else:
            all_args.use_share_obs = False

    if all_args.env_name == "hands":
        pass
        # args = get_args(all_args=all_args)
        # cfg = load_env_cfg(args)
        # cfg["env"]["numEnvs"] = all_args.n_rollout_threads
        # all_args.episode_length = cfg["env"]["episodeLength"]
        # sim_params = parse_sim_params(args, cfg)
        # agent_index = get_AgentIndex(cfg)
        # if not os.path.exists(cfg["env"]["asset"]["assetRoot"]):
        #     cfg["env"]["asset"]["assetRoot"] = cfg["env"]["asset"]["assetRoot"][1:]
        # env = parse_task(args, cfg, sim_params, agent_index)
        # return env

    elif not eval:
        if all_args.n_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
            )
    else:
        n_envs = (
            all_args.n_eval_rollout_threads
            if all_args.env_name == "StarCraft2"
            else all_args.eval_episodes
        )
        if n_envs == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_envs)])


@hydra.main(
    config_name="ant",
    config_path="./configs/mappo_maska/mujoco",
    version_base="1.2",
)
def main(all_args: DictConfig):
    OmegaConf.set_struct(all_args, False)

    if all_args.seed < 0:
        all_args.seed = np.random.randint(1000, 10000)
    print("seed is :", all_args.seed)

    # if all_args.env_name == "hands":
    #     print(
    #         termcolor.colored(
    #             "WARNING: You can only use the following command when using 'train.py'",
    #             "yellow",
    #         )
    #     )
    #     print(
    #         termcolor.colored(
    #             "\tpython train.py --config-path ./configs/<ALGO>/hands --config-name <TASK>",
    #             "green",
    #         )
    #     )
    #     print(
    #         termcolor.colored(
    #             "No extra args can be added in this case. Please modify items in yaml files beforing running the command",
    #             "yellow",
    #         )
    #     )

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        assert (
            all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy
        ), "check recurrent policy!"
    elif "mat" in all_args.algorithm_name:
        assert (
            all_args.use_recurrent_policy == False
            and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    elif "mat_dec" in all_args.algorithm_name:
        all_args.dec_actor = True
        all_args.share_actor = True
    elif "major" in all_args.algorithm_name:
        all_args.use_jpr = True
    if "mask" in all_args.algorithm_name:
        all_args.mask_agent = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    if all_args.work_dir is None:
        run_dir = os.getcwd()
    elif os.path.isabs(all_args.work_dir):
        run_dir = all_args.work_dir
    else:
        run_dir = os.path.join(hydra.utils.get_original_cwd(), all_args.work_dir)
    run_dir = os.path.join(
        run_dir,
        all_args.env_name,
        all_args.get("scenario", all_args.get("map_name", all_args.get("task", ""))),
        all_args.algorithm_name,
        f"{datetime.datetime.now():%m%d-%H%M}",
    )
    os.makedirs(run_dir, exist_ok=True)

    setproctitle.setproctitle(
        str(all_args.seed)
        + "-"
        + str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + socket.gethostname()
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    all_args["run_dir"] = run_dir
    envs = make_env(all_args)
    eval_envs = None
    if all_args.use_eval:
        if all_args.env_name == "hands" or all_args.env_name == "pixel_football":
            eval_envs = envs
        else:
            eval_envs = make_env(all_args, eval=True)

    if all_args.env_name == "StarCraft2_transfer":
        if all_args.map_name == "3s_vs_3z" or all_args.map_name == "3s_vs_4z":
            num_agents = get_map_params("3s_vs_4z")["n_agents"]
        elif all_args.map_name == "8m_vs_9m" or all_args.map_name == "10m_vs_11m":
            num_agents = get_map_params("10m_vs_11m")["n_agents"]
        elif all_args.map_name == "3s5z" or all_args.map_name == "1c3s5z":
            num_agents = get_map_params("1c3s5z")["n_agents"]
        elif all_args.map_name == "25m" or all_args.map_name == "27m_vs_30m":
            num_agents = get_map_params("27m_vs_30m")["n_agents"]
        else:
            print("%s is not supported yet" % all_args.map_name)
            raise NotImplementedError
    elif all_args.env_name == "StarCraft2":
        num_agents = get_map_params(all_args.map_name)["n_agents"]
    else:
        num_agents = getattr(envs, "num_agents", envs.n_agents)
    all_args.num_agents = num_agents

    if all_args.use_wandb:
        run = wandb.init(
            config=OmegaConf.to_object(all_args),
            project=all_args.wandb_project_name,
            notes=all_args.get("wandb_notes", None),
            tags=all_args.get("wandb_tags", None),
            name=f"{all_args.algorithm_name}_"
            f"{all_args.env_name}_"
            f"{all_args.get('scenario', all_args.get('map_name', ''))}_"
            f"seed{all_args.seed}_"
            f"{datetime.datetime.now():%m%d-%H%M}",
            dir=run_dir,
            reinit=True,
        )
        wandb.config.update({"Trueseed": all_args.seed}, allow_val_change=True)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    runner = get_runner(all_args.env_name, all_args.algorithm_name)(config)
    runner.run()

    # post process
    if all_args.env_name != "hands":
        envs.close()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main()
