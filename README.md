📊 ProfitVision AI — E-commerce Profit Predictor

Live App: https://profit-vision-ai.streamlit.app

An end-to-end machine learning application that predicts whether a product will generate **High** or **Low** profit based on category, region, quantity, and sales.

🚀 What it does
- Predicts profitability (High/Low) with probability
- Handles categorical (Category, Region) + numerical (Quantity, Sales) inputs
- Real-time inference via Streamlit

🧠 Model
- Algorithm: Random Forest Classifier
- Preprocessing: one-hot encoding (categoricals), standard scaling (numericals)
- Target: derived using median profit threshold

📊 Performance
- Accuracy: ~83%
- Balanced precision/recall across classes

🧰 Tech Stack
Python • Pandas • scikit-learn • Streamlit • Joblib

▶️ Run locally
```bash
pip install -r requirements.txt
python train_profit_model.py
streamlit run streamlit_app.py
