# Decision Tree Classifier from Scratch

This repository contains an implementation of a **Decision Tree Classifier in Python from scratch**, developed as part of the *Advanced Programming* course. The project includes both the algorithmic implementation and statistical analysis of the dataset.

## Dataset
- **Name:** `Airplane.csv`  
- **Size:** 103,904 samples × 24 features  
- **Target variable:** `satisfaction`  
- **Source:** Public dataset  
- Preprocessing and exploratory data analysis (EDA) are included in the `docs` directory.

## Features
- Custom Decision Tree implementation supporting:
  - **Splitting criteria:** Gini index & Information Gain
  - Hyperparameter tuning via **coarse-to-fine search**
  - Metrics: **Accuracy** and **F1-Score** (weighted 0.7 and 0.3 respectively)
- Post-pruning capability to reduce overfitting
- Visual tree export using **Graphviz** (saved as `.svg`)
- Statistical dataset interpretation and visualizations (via `seaborn`, `matplotlib`)

## Requirements
- Python 3.x
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `graphviz`

## Output
- Detailed performance metrics for train/validation sets
- Visual representation of the final decision tree
- Accuracy trends for different hyperparameter configurations

---
*This project was created to deepen understanding of machine learning algorithms by building them entirely from the ground up, without relying on pre-built models.*