# Project 3: Collaboration and Competition

This project covers the solution for Project 3 of the Udacity Deep Reinforcement Learning Nanodegree. The goal of the project was to train many agents to cooperate or compete in a continuous action space.

## Project Details

![Trained Agent](assets/multi_agent.gif)

This project required training two agents to keep a ball aloft in a ping pong environment. A reward of 0.1 is given to any agent that successfully hits the ball over the net and a reward of -0.01 is given to the agent that lets the ball hit the ground or hits it out of bounds.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions between -1 and 1 are available, corresponding to movement toward (or away from) the net, and jumping.

## Getting Started

### Prerequisites (Conda)

1. Setup conda environment `conda create -n tennis python=3.6` and `conda activate tennis`.
1. Install [PyTorch version 0.4.1](https://pytorch.org/get-started/previous-versions/) for the version of CuDA you have installed.
2. Run `pip -q install ./python`

### Unity Environment Setup
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
2. Place the file in the `environments/` folder, and unzip (or decompress) the file.

### Instructions

Run `jupyter notebook` from this directory.

Open `Tennis.ipynb` and run the cells to see how the agent was trained!
