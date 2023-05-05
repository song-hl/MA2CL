## Installation
### common dependences
```
pip install torch wandb hydra-core kornia pysc2 gfootball mujoco mujoco-py \
    scikit-image scikit-learn scikit-video \
    tensorboard tensorboardX pandas seaborn matplotlib opencv-python==4.5.5.64
```
An alternative way is to use the provided dockerfile to build docker, and before building docker image, you need to download `IsaacGym_Preview_3_Package.tar.gz`, `mujoco210-linux-x86_64.tar.gz`, `SC2.4.10.zip` and `SMAC_Maps.zip` in `docker` folder

### Multi-Agent Quadcopter Control
Folling the instrctions in https://github.com/utiasDSL/gym-pybullet-drones to setup the environment.
Note that MAQC need `gym==0.21.0`

### Multi-agent MuJoCo
Following the instructios in https://github.com/openai/mujoco-py and https://github.com/schroederdewitt/multiagent_mujoco to setup a mujoco environment. In the end, remember to set the following environment variables:
``` Bash
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

### StarCraft II & SMAC
Run the script
``` Bash
bash install_sc2.sh
```
Or you could install them manually to other path you like, just follow here: https://github.com/oxwhirl/smac.

### Google Research Football
Please following the instructios in https://github.com/google-research/football. 

## How to run
When your environment is ready, you could run `train.py` provided configs. For example:
``` Bash
python train.py --config-path ./configs/<algo_name>/<env_name> --config-name <scenario> 
```
Please check the `configs` folder to varify current supported envs, algos and scenarios.

If you would like to change the configs of experiments, you could modify yaml files or directly modify it in cmd. For more details, please read [hydra's doc](https://hydra.cc).
