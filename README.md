# COMP551_A1 - ML Project 1: Linear and Logistic Regression from Scratch

Team members - Diana Covaci, Nicholas Milin, and Carl-Elliot Bilodeau-Savaria

This project implements two fundamental machine learning models, Linear Regression and Logistic Regression, from scratch in Python. Using these models, we analyze two benchmark datasets to understand model behavior, performance, and the impact of data preprocessing, batch size, and learning rate. The projects aims to provide hands-on experience with model implementation, experimentation and performance analysis.

---

## Repository Structure

```
COMP551_A1/
├── A1.ipynb          # Main script with experiments (run in VS Code)
├── models.py         # Model definitions and helper functions
├── requirements.txt  # Dependencies
├── writeup.pdf       # Project final report
└── README.md         # Project documentation
```

---

## Datasets 
1. Parkinson's Telemonitoring Dataset (Regression)
   - Target: motor UPDRS, total UPDRS
   - Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
2. Breast Cancer Diagnostic Dataset (Classification)
   - Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Features
- Data Preprocessing
- Model Implementation: 
   * Analytical Linear Regression
   * Logistic Regression with Gradient Descent
   * Mini-batch Stochastic Gradient Descent for both models
- Experiments and Analaysis: 
   * Train/test performance with 80/20 split
   * Feature weights analysis
   * Effect of training set size on performance
   * Mini-batch size experiments
   * Learning rate experiments
   * Comparison of analytical vs. SGD linear regressions
- Visualization

### Prerequisites

- Python 3.8+  
- pip (Python package manager)  
- VS Code or any Python IDE/terminal  
- Required Python packages listed in `requirements.txt`

## Installation & Setup

1. Clone this repository: 

   ```bash
   git clone https://github.com/dianacovacii/COMP551_A1.git
   cd COMP551_A1
   ```

2. Create and activate a virtual environment: 

   ```bash 
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies: 

   ```bash
   pip install -r requirements.txt
   ```


## Usage

1. Run the cells in order, it will:

   * Load and preprocess the dataset(s)
   * Train models defined in `models.py`
   * Evaluate and compare results
   * Print metrics and summaries in the terminal/output panel

---

## Report

The accompanying report (writeup.pdf) provides a comprehensize analysis of our findings, including: 
   * Model derivations and implementation detials
   * Effects of hyperparameters 
   * Comparison between batch and stochastic gradient descent 
   * Evaluation metrics and performance discussion

--- 

## Acknowledgments

This project was completed as part of the COMP551 (Applied Machine Learning) course at McGill University in Fall 2025.
