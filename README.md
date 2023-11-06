# LearningAgileFlight_SE3
Learning Agile Flights on SE(3): a novel deep SE(3) motion planning and control method for quadrotors. It learns an MPC's adaptive SE(3) decision variables parameterized by a portable DNN, encouraging the quadrotor to fly through the gate with maximum safety margins under diverse settings.

Animation  |      Method   |      Animation
:-------------------:|:--------------------:|:--------------------:
![animation_demo](https://github.com/BinghengNUS/LearningAgileFlight_SE3/assets/70559054/b3347e01-49db-4eae-a3e6-19d3b96c6942) | ![Graphic Abstract_updated](https://github.com/BinghengNUS/LearningAgileFlight_SE3/assets/70559054/21deda8e-eb70-49bf-b496-cdf2d45953c4) | ![animation_demo2](https://github.com/BinghengNUS/LearningAgileFlight_SE3/assets/70559054/e405d6a8-988e-4e63-aa7c-6d54f6e1f7ed)


Please find out more details in our paper (ICRA 2023): https://ieeexplore-ieee-org.libproxy1.nus.edu.sg/document/10160712

## Table of contents
1. [Project Overview](#project-Overview)
2. [Dependency Packages](#Dependency-Packages)
3. [How to Use](#How-to-Use)
4. [Contact Us](#Contact-Us)

## 1. Project Overview
Agile flights of autonomous quadrotors in cluttered environments require constrained motion planning and control subject to translational and rotational dynamics. Traditional model-based methods typically demand complicated design and heavy computation. In this paper, we develop a novel deep reinforcement learning-based method that tackles the challenging task of flying through a dynamic narrow gate. We design a model predictive controller with its adaptive tracking references parameterized by a deep neural network (DNN). These references include the traversal time and the quadrotor SE(3) traversal pose that encourage the robot to fly through the gate with maximum safety margins from various initial conditions.

## 2. Dependency Packages
Please make sure that the following packages have already been installed before running the source code.
* CasADi: version 3.5.5 Info: https://web.casadi.org/
* Numpy: version 1.23.0 Info: https://numpy.org/
* Pytorch: version 1.12.0+cu116 Info: https://pytorch.org/
* Matplotlib: version 3.3.0 Info: https://matplotlib.org/
* Python: version 3.9.12 Info: https://www.python.org/

## 3. How to Use
First and foremost, the training process is both efficient and straightforward to setup. The source code has been comprehensively annotated to facilitate ease of use. To reproduce the simulation results presented in the paper, simply follow the steps outlined below.

1.  Run the Python file '**nn_train.py**' to pre-train the 1st DNN via supervised learning.
2.  Run the Python file '**deep_learning.py**' to train the 1st DNN (pretained in Step 1) via reinforcement learning.
3.  Run the Python file '**nn_train_2**' to train the 2nd DNN via imitation learning (i.e., imitating the outputs of the 1st DNN).
4.  Run the Python file '**main.py**' to evaluate the trained 2nd DNN in the challenging task of flying through a dynamic narrow gate.
5.  Run the Python file '**Pybullet_simulation.py**' in the folder '**gym_pybullet_drone**' to evaluate the trained 2d DNN in the pybullet-drone environment. Please ensure that the simulator is properly installed before running the Python file. An installation tutorial can be found at https://github.com/utiasDSL/gym-pybullet-drones

## 4. Contact Us
If you encounter a bug in your implementation of the code, please do not hesitate to inform me.
* Name: Mr. Bingheng Wang
* Email: wangbingheng@u.nus.edu
