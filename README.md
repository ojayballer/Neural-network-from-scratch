# Neural Network from Scratch 🧠
```bash

This project is a simple implementation of a **neural network (logistic regression)** from scratch using only **NumPy** and **Pandas**.  
It demonstrates forward propagation, backward propagation (gradient descent), and prediction without using machine learning libraries like scikit-learn or TensorFlow.
```
---

## 📂 Project Structure
```bash
Neural-network-from-scratch/
│── neural_network.py # Main code
│── insurance_data.csv # Sample dataset
│── README.md # Documentation
```

---

## 📊 Dataset

The project uses a small custom dataset: **`insurance_data.csv`** with the following columns:

| Column         | Description                                      |
|----------------|--------------------------------------------------|
| `age`          | Age of the person                               |
| `affordibility`| Whether the person can afford insurance (0 or 1) |
| `bought_insurance` | Target variable: 1 if bought insurance, else 0 |


## ⚙️ How it works

- **Forward Propagation**: Computes weighted sum of inputs + bias and applies the sigmoid activation.
- **Log Loss Function**: Measures prediction error.
- **Backward Propagation**: Updates weights and bias using gradient descent.
- **Prediction**: Outputs probability and class labels (0 or 1).

---

## 🚀 Usage

# Clone the repo
git clone https://github.com/<your-username>/Neural-network-from-scratch.git
cd Neural-network-from-scratch


# Run the code
```bash
python neural_network.py
```
📚 Learnings
```bash
How logistic regression can be seen as a simple neural network.

Implementing gradient descent manually.

Using sigmoid activation for binary classification.
```
🛠️ Technologies
```bash
Python 3

NumPy

Pandas
```
