# Horse Racing Outcome Prediction — Deep Learning vs Random Forest

This repository contains a deep learning group project analyzing and predicting **Hong Kong horse racing outcomes** using the Kaggle “HK Horse Racing” dataset. This project was completed as part of "STAT4012 — Deep Learning" at The Chinese University of Hong Kong (CUHK).
We develop and compare two approaches:

- **Random Forest classifier** (baseline model)  
- **Deep Neural Network (Keras / TensorFlow)** (main model)

The goal is to examine whether flexible nonlinear models can outperform classical tree-based methods in predicting the winner and evaluating finishing positions (Top-3 / Top-4).

## 📅 Project Information
- Completion date: May 2021
- Languages: Python

## 🚀 Project Overview

**Objective:**  
Predict the finishing performance of each horse in a race using historical race-level and horse-level features.

**Dataset:**  
- `races.csv` — race characteristics (venue, distance, going, class, course configuration, etc.)  
- `runs.csv` — horse-level information (age, weight, draw, country, odds, past performance)

Data source: Kaggle (Hong Kong Horse Racing dataset).  
Raw data are not included in this repo due to size and licensing; see `data/README.md` for download instructions.

---

## 🧠 Methods Summary

### 1. **Feature Engineering**
- Merge race and horse records  
- Encode categorical variables  
- Rank-encode country/type based on performance  
- Standardize and flatten features to form a fixed-length vector (up to 14 horses per race)  
- Construct labels:
  - Winner (one-hot)
  - Top-3 / Top-4 indicators

### 2. **Random Forest (Baseline)**
Implemented in `src/horse_random_forest.py`:
- Grid-search tuning  
- PCA variant tested  
- Evaluated on winner prediction and Top-3 accuracy

### 3. **Deep Neural Network (Main Model)**
Implemented in `src/horse_dnn.py`:
- 1 hidden layer, 512 neurons, `tanh` activation  
- Softmax output (14 classes)  
- Adam optimizer (`lr = 0.0005`)  
- Trained for 30 epochs with 5-fold cross-validation

---

## 📈 Key Results

| Model                    | Winner Accuracy | Top-3 Accuracy |
|--------------------------|-----------------|----------------|
| Random Forest            | ~0.08–0.09      | ~0.22–0.24     |
| **Deep Neural Network**  | **~0.18**       | **~0.46–0.47** |

🔎 **Insights:**
- Winner prediction is hard due to near-random inherent structure (1/14 baseline).  
- DNN captures nonlinear dependencies better than RF, showing substantial gains in Top-3 performance.  
- Feature flattening + deep models work reasonably well despite the dataset’s noisy nature.

Full analysis is in `report/final_report.pdf`.

---

## 📁 Repository Structure
.
├── src/
│ ├── horse_dnn.py # Deep learning model (Keras)
│ └── horse_random_forest.py # Random Forest baseline
│
├── report/
│ └── final_report.pdf # Full project write-up
│
├── data/
│ └── README.md # Instructions to obtain dataset
│
└── README.md # Project documentation

---

## ▶️ How to Run

### 1. Install dependencies  
(Example using conda)

```
conda create -n horse python=3.11
conda activate horse
pip install -r requirements.txt
```
2. Download Kaggle data

See data/README.md.

3. Run models
```
python src/horse_random_forest.py
python src/horse_dnn.py
```
