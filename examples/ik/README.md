# Inverse kinematics

The inverse kinematics algorithm for the 6-axis articulated robot.


## Problem formulation

The minimalization of the function that penalises a difference between actual and desired end-effector's configuration

$$\min_{\theta} \|r - f(\theta)\|$$

where $\theta$ are joint coordinates, $r$ are desired end-effector's coordinates and $f$ is the nonlinear vector equation mapping the joint coordinates to the end-effector coordinates.


## Optimization algorithm

The optimization of the objective function through the Newton-Raphson method:

$\theta \leftarrow \theta_{initial}$

$e \leftarrow r - f(\theta)$, while $\| e \| > \epsilon$:

&emsp; $\theta \leftarrow \theta + J^{+}(\theta)e$

where $J$ is the end-effector body Jacobian and $\epsilon$ - the desired accuracy.


## Demonstration

The reference trajectory tracking by the articulated robot's end-effector.

![inverse_kinematics_demo](inverse_kinematics_demo.gif)


## References

[K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and Control*, Cambridge University Press, 2017](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)