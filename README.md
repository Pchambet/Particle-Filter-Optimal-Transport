# Optimal Transport Particle Filter with Auto Differentiation

This repository contains an implementation of an Optimal Transport Particle Filter combined with Auto Differentiation techniques to enhance state estimation accuracy. The project includes two main Python scripts:
1. `transport_optimal.py`: Implements the Optimal Transport resampling method for the particle filter.
2. `auto_differentiation.py`: Applies Auto Differentiation techniques to improve the estimation process.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)

## Project Overview
This project demonstrates a state-of-the-art approach to particle filtering, leveraging Optimal Transport for efficient resampling and Auto Differentiation for accurate state estimation. Particle filters are essential in tracking and estimating dynamic systems, and this project showcases advancements that can be applied to various fields such as robotics, finance, and environmental monitoring.

The main goal is to integrate a resampling function with optimal transport in a particular filter algorithm.

## Features
- **Optimal Transport Resampling**: Improves the efficiency and accuracy of the resampling step in particle filters.
- **Auto Differentiation**: Enhances the precision of state estimation by automatically computing derivatives.
- **Visualization**: Plots true states, classical estimates, and differentiable estimates for comparison.
- **Proof paper**: Explain how the results and the algorithms have been created and where they come from. It contains every mathematical research of this project.

## Project Structure
```plaintext
PARTICULAR_FILTERING
├── Code
│   ├── auto_differentiation.py
│   └── optimal_transport.py
├── Documents
│   ├── Annexe_Autodifférenciation
│   ├── Article_resume.pdf
│   ├── Poster.pdf
│   ├── Proof_report.pdf
│   └── Synthesis_report.pdf
├── Notebooks
│   ├── Notebook Autodifférenciation.ipynb
│   ├── Notebook Comparaison Classique et Différentiable.ipynb
│   └── PFD-Tutoriel-TP-Pierre.ipynb
├── Outputs
│   ├── output1_optimal_transport.png
│   └── output2_optimal_transport.png
└── README.md

```

## Installation
To run this project, you need to have Python 3.x installed along with the following libraries:
- `numpy`
- `matplotlib`
- `POT` (Python Optimal Transport)

## Methodology

### Optimal Transport Particle Filter
This method uses Optimal Transport to resample particles in the filter, which helps in maintaining a diverse set of particles and avoids issues like particle degeneracy.

### Auto Differentiation
Auto Differentiation is utilized to automatically compute the necessary gradients, making the state estimation process more accurate and efficient. This approach is particularly useful in non-linear dynamic systems where traditional methods struggle.

### Implementation Details
- **Transition Function**: Models the state transition with added noise.
- **Likelihood Function**: Computes the likelihood of observations given the current particle states.
- **Resampling**: Uses the Sinkhorn algorithm from the POT library to perform Optimal Transport resampling.

### Code Structure
- `transport_optimal.py`: Contains the main particle filter implementation with Optimal Transport resampling.
- `auto_differentiation.py`: Demonstrates the use of Auto Differentiation in state estimation.

## Results
The project includes visualizations that compare true states, classical estimates, and differentiable estimates over time. The results show the effectiveness of combining Optimal Transport and Auto Differentiation in improving state estimation accuracy.

Example outputs can be found in the `Outputs` directory:

- `output1_optimal_transport.png`
- `output2_optimal_transport.png`
