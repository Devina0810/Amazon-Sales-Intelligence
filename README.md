# 🛒 Amazon Sales Intelligence Dashboard

An end-to-end Machine Learning project that analyzes Amazon product data to forecast demand, uncover business insights, and compare ML models — deployed as an interactive Streamlit web app.

---

## 🔗 Links
- 📓 **Kaggle Notebook:** [View Notebook](https://www.kaggle.com/)
- 🚀 **Live Demo:** [Streamlit App](https://your-streamlit-link.streamlit.app/)

---

## 📌 Project Overview
This project simulates a real-world Amazon internal analytics tool that:
- Analyzes 1,462 Amazon products across 9 categories
- Engineers smart features to measure product demand
- Compares 3 ML models to find the best demand predictor
- Visualizes business insights through an interactive dashboard

---

## 📊 Dataset
- **Source:** Amazon Sales Dataset by karkavelrajaj (Kaggle)
- **Size:** 1,462 products, 16 features
- **Domain:** Amazon India product listings with ratings, prices, discounts and reviews

---

## 🔧 Features Engineered
| Feature | Description |
|---|---|
| `savings` | Actual rupee savings (actual - discounted price) |
| `weighted_rating` | Rating × log(review count) — demand proxy |
| `price_tier` | Budget / Mid / Premium classification |
| `high_discount` | Binary flag for discounts ≥ 50% |
| `category_encoded` | Numerical encoding of product category |

---

## 🤖 Models Compared
| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | 0.4914 | 5.1342 |
| Random Forest | 0.9969 | 0.2331 |
| **XGBoost** | **0.9977** | **0.2285** |

✅ **XGBoost** selected as best model

---

## 🔍 Key Business Insights
- 🎮 Toys & Games has the highest avg demand score despite fewer products
- 📦 Number of reviews is the strongest predictor of product demand
- 💰 High discount products drive slightly more demand than normal ones
- 🏷️ boAt and SanDisk dominate top demanded products

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Environment:** Kaggle Notebooks

---

## 🚀 Run Locally
```bash
git clone https://github.com/Devina0810/Amazon-Sales-Intelligence.git
cd Amazon-Sales-Intelligence
pip install -r requirements.txt
streamlit run app.py
```

---

## 👩‍💻 Author
**Devina** — [GitHub](https://github.com/Devina0810)
