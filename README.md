This repo contains implementation of RL algorithm PPO, PAIR.

### Installation
1. Clone the environment and algorithm repos
```
git clone git@github.com:IrisLi17/stacking_env.git
git clone git@github.com:IrisLi17/onpolicy_algorithm.git
```
2. Download docker image
```
docker pull irisli20/bullet_torch
```
3. Create a container
```
docker run -it --gpus all -e NVIDIDA_DRIVER_CAPABILITIES=all -v /absolute/path/to/onpolicy_algorithm:/projects/onpolicy_algorithm -v /absolute/path/to/stacking_env:/projects/stacking_env --name your_container_name irisli20/bullet_torch bash
```
4. Run experiments inside the container. 
```
cd /projects/onpolicy_algorithm
tmux new -s test
# Train a pick-and-place policy with PPO
python train.py --config config.pick_and_place
# Record video of a trained agent. Images will be saved to tmp/
python train.py --config config.pick_and_place --load_path pretrained_model/pick_and_place.pt --play
# Train stack-6 from a pretrained stack-1 model
python train.py --config config.stacking_pair --load_path pretrained_model/stack1_model.pt
```