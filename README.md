# Differentiable Particle Filter with Entropy-Regularized Optimal Transport

This repository implements a state-of-the-art algorithm to solve a core problem in probabilistic tracking: making particle filters differentiable.

This project was completed for the Cassiopée research program at Télécom SudParis, based on the paper **"Differentiable Particle Filtering via Entropy-Regularized Optimal Transport"** (Corenflos et al.).

---

## The Core Idea (Explained Simply)

#### 1. The Scenario
Imagine trying to track a small, fast robot (the "true state") as it moves through a smoky room, using only a blurry, lagging camera (the "observation").

#### 2. What is a Particle Filter?
A Particle Filter (PF) works by making thousands of 'guesses' (particles) about the robot's real-time position. Based on the blurry camera image, it gives a "score" to each guess. Good guesses get high scores; bad guesses get low scores.

#### 3. The "Non-Differentiable" Problem
A *standard* particle filter then performs a 'survival of the fittest' step called **resampling**. It abruptly **kills** all the low-score guesses and **duplicates** the high-score ones.

* **The Problem:** This "kill/duplicate" step is a hard, discrete on/off switch. You cannot use calculus (i.e., gradient descent) on a hard switch. This means you can't "teach" the filter to get better or automatically tune its own parameters (like a neural network). It is **non-differentiable**.

#### 4. The Solution (This Project)
Our filter uses **Optimal Transport (OT)**. Instead of 'killing' and 'duplicating' guesses, it **smoothly moves** them.

* **The Solution:** It calculates the most efficient way to "shift" all the bad guesses over to the locations of the good guesses. Because this "shifting" is a smooth, continuous process, you *can* use calculus on it.

This makes the entire filter **differentiable**, allowing it to be optimized and integrated into modern deep learning pipelines.

---

## Key Results

We compared our Optimal Transport Particle Filter (OT-PF) against a classical particle filter (with standard resampling) on a state-tracking task.

#### 1. Trajectory Estimation
The OT-PF (black line) successfully tracks the true observations (blue line) and demonstrates a more stable and accurate estimation than the classical filter (red line).

![Trajectory Estimation Comparison](images/trajectory_estimation.png)

#### 2. Cumulative Mean Squared Error (CMSE)
This is the key result. The CMSE (a measure of total error) of our OT-PF (black line) is **consistently lower** than that of the classical filter (red line). This proves our method is not just differentiable, but also more accurate.

![CMSE Comparison](images/cmse_comparison.png)

---

## Technical Features & Implementation

* **Optimal Transport PF:** `optimal_transport.py` implements the full particle filter using OT resampling.
* **Auto-Differentiation:** `auto_differentiation.py` provides a proof-of-concept for estimating model parameters ($Q$ and $R$) via gradient descent.
* **Mathematical Proofs:** As the lead on the theoretical side of this project, **I authored the complete mathematical derivations in `Proof_report.pdf`**. This document provides the full derivation of the Optimal Transport dual problem, the saddle-point proof, and the derivation for entropic regularization.

* **[You can read the full `Proof_report.pdf` that I produced here](Documents/Proof_report.pdf)**.

---

## Installation

This project requires Python 3.x and the following libraries:

---

## Project Structure

```bash
pip install numpy matplotlib
pip install pot

📦 Particle-Filter-Optimal-Transport
│
├── 📂 Code/                             # Main Python scripts
│   ├── auto_differentiation.py          # Parameter estimation using autodifferentiation
│   └── optimal_transport.py             # Main implementation of the PF-OT filter
│
├── 📂 Documents/                        # Research and analysis reports
│   ├── Synthesis_report.pdf             # Summary report of the project
│   ├── Proof_report.pdf                 # Full mathematical derivations
│   ├── Poster.pdf                       # Project presentation poster
│   └── Annexe_Autodifferentiation.pdf   # Explanation of Q and R parameter estimation
│
├── 📂 Notebooks/                        # Jupyter notebooks for testing and visualization
│
├── 📂 Outputs/                          # Output images and generated results
│
├── 📂 Images/                           # Illustrations
│
└── README.md    
