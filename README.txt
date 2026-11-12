POLAR STATION OPS - FIXED NLP CHATBOT

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run: python -m pip install -r requirements.txt
4. Run: python -m streamlit run app.py

The app creates polar_ops.db beside app.py and persists its records in SQLite.
NLP features: TF-IDF intent matching, dynamic inventory entity extraction,
numeric and written duration parsing, and item-specific SQL-grounded answers.
This is a simulated decision-support prototype, not a certified safety system.
