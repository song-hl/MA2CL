# multi-agent mujoco
pip install gym==0.10.8
mkdir -p $HOME/.mujoco
wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz
tar -xf mujoco.tar.gz -C $HOME/.mujoco
rm mujoco.tar.gz
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so