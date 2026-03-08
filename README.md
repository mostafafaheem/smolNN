# smolNN

A lightweight, from-scratch implementation of a Multi-Layer Perceptron (Neural Network) built in Python, designed to classify Palmer Penguins based on physical characteristics. It features a graphical user interface (GUI) for easy tuning of hyperparameters and real-time model evaluation.

## Features

- **Custom Neural Network from Scratch (`nn.py`)**:
  - Implements an MLP without external deep learning frameworks (uses `numpy` and `pandas`).
  - Supports multiple optimization algorithms: **Stochastic Gradient Descent (SGD)**, **Batch Gradient Descent (BGD)**, and **Adam**.
  - Dynamic hidden layers mapping with customizable neuron counts.
  - Activation functions including **Sigmoid** and **Tanh**, accompanied by a **Softmax** output layer for robust multi-class probability extraction.

- **Graphical User Interface (`interface.py`)**:
  - A friendly desktop application built with `tkinter`.
  - Modify parameters such as the learning rate, number of epochs, activation function, and hidden layer structure on the fly.
  - Train the model dynamically, evaluate test/train accuracies, and compute confusion matrices.
  - Interactive prediction tab: Input new penguin features (e.g., *Culmen Length*, *Body Mass*, *Origin Location*) and predict the exact species immediately.

- **Data Preprocessing Pipeline**:
  - Handles missing values safely.
  - Performs One-Hot Encoding for categorical features.
  - Automatic Z-score standardization (normalization) to accelerate convergence.

## Project Structure

| File | Description |
| ---- | ----------- |
| `nn.py` | The core data structures and Neural Network logic. Includes Grid Search for automatic tuning. |
| `interface.py` | The GUI script linking directly to the Neural Network backend (`nn.py`). |
| `penguins.csv` | The Palmer Penguins dataset used to train predictions. |
| `requirements.txt` | File containing the necessary Python packages to run the project. |
| `observations.pdf` | Documentation with insights and visualizations about data preparation and training methodology. |

## Getting Started

### 1. Install Requirements

Make sure you have Python 3.x installed. Then, simply install dependencies through `pip`:
```bash
pip install -r requirements.txt
```

### 2. Running the GUI
Launch the comprehensive graphical interface to set up models dynamically:
```bash
python interface.py
```

### 3. Native Grid Search (CLI)
You can directly execute the neural network file to invoke the built-in grid search and test best parameters:
```bash
python nn.py
```
This prints the optimal configuration matrices and visualizes confusion plots directly.
