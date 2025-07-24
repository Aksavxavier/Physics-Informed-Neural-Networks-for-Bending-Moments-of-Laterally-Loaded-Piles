# Physics-Informed-Neural-Networks-for-Bending-Moments-of-Laterally-Loaded-Piles
Physics-Informed Neural Networks for Bending Moments of Laterally Loaded Piles

This repository implements Physics-Informed Neural Networks (PINNs) to predict bending moment distributions of laterally loaded piles from observed displacements. The framework includes forward, inverse, and parametric PINN models, along with tools for non-dimensionalization and analysis.



Background

The distribution of internal forces, particularly bending moments, is critical in the design of laterally loaded piles. Traditional monitoring relies on measuring displacements and back-calculating internal forces using iterative continuum or finite element models. PINNs offer a data-driven approach that enforces governing physics directly in the neural network loss function, enabling direct estimation of internal force distributions from displacement data without explicit boundary conditions. fileciteturn0file0

Objectives

Non-dimensionalize the governing equations to improve numerical stability and model generality.

Develop a forward PINN that predicts the bending moment distribution from sparse displacement measurements.

Develop an inverse PINN that learns the bending stiffness parameter  directly from displacement data (Reese case).

Build a parametric PINN with variable  as an input, allowing cross-sectional stiffness variation.

