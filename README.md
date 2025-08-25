# 🥔 Potato Protector (Potato Leaf Disease Detection)

A Streamlit app that loads a Keras/TensorFlow model (`potatoes.h5`) to classify potato leaf images.

## Local Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud Deploy
1. Push this folder to a **public GitHub repo** (keep `app.py`, `potatoes.h5`, `requirements.txt`, `runtime.txt` in the repo root).
2. Go to https://share.streamlit.io and connect your GitHub.
3. Select your repo and set **Main file path** to `app.py`.
4. The app will build with **Python 3.10.13** and install from `requirements.txt`.

> If `potatoes.h5` is larger than 100 MB, use **Git LFS** or host the model on Hugging Face and download on startup.

## Hugging Face Spaces (Alternative)
1. Create a Space (type: **Streamlit**).
2. Upload the same files (`app.py`, `requirements.txt`, `potatoes.h5`).
3. Spaces will auto-build and serve the app.
