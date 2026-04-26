# 🍎 Fruit Quality AI

An intelligent fruit freshness detection system powered by deep learning. 
Upload a photo of an apple, banana, or orange — get instant freshness 
analysis, quality grade, shelf life estimate, and an AI-generated quality 
report in seconds.

**Live Demo** → https://fruit-quality-ai-qubmmbkacdkkeq2duaypge.streamlit.app/

---

## What does it do?

You take a photo of a fruit. The app tells you:

- 🟢 **Is it fresh or rotten?** — with up to 99.57% accuracy
- 📊 **Freshness score** — a precise 0.0 to 1.0 number
- 🏆 **Quality grade** — A (excellent) to F (spoiled)
- 📅 **Shelf life** — how many days left
- ⚠️ **Risk level** — Low / Medium / High
- 🔬 **Grad-CAM heatmap** — see exactly which part of the fruit 
  the AI focused on
- 🤖 **AI quality report** — storage advice, nutritional impact, 
  consumption window (powered by Llama 3.3 70B via Groq)

---

## Supported Fruits

| Fruit | Fresh | Rotten |
|-------|-------|--------|
| 🍎 Apple | ✅ | ✅ |
| 🍌 Banana | ✅ | ✅ |
| 🍊 Orange | ✅ | ✅ |

---

## How the AI works
Your Photo
↓
EfficientNetB0 backbone (pretrained on ImageNet)
↓
CBAM Attention Module (focuses on relevant fruit regions)
↓
Custom classification head
↓
Fresh / Rotten prediction + confidence score
↓
Grad-CAM heatmap generation
↓
LLM report (Groq Llama 3.3 70B → Gemini fallback)
↓
Full quality report
**Architecture:** EfficientNetB0 + CBAM (Convolutional Block Attention Module)  
**Training:** 18,984 images across 6 classes  
**Test Accuracy:** 99.57% | **Macro F1:** 0.9942  
**Misclassified:** only 8 out of 1,854 test images

---

## Model Comparison

We compared our model against the CNN_BiLSTM approach from 
Yuan et al. (2024) *Current Research in Food Science*:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| CNN | ~96% | ~92% | ~96% | 0.94 |
| BiLSTM | ~95% | ~92% | ~93% | 0.92 |
| CNN_LSTM | ~97% | ~95% | ~96% | 0.96 |
| CNN_BiLSTM (Paper) | 97.76% | 95.63% | 97.17% | 0.97 |
| **EfficientNetB0+CBAM (Ours)** | **99.57%** | **99.4%** | **99.4%** | **0.9942** |

Our model outperforms the paper's best model by **+1.81% accuracy**.

---

## Project Structure

```
fruit-quality-ai/
├── app/
│   ├── model/
│   │   └── project_config.json     # Class names, grade config
│   └── utils/
│       ├── predictor.py            # Model loading + inference
│       └── llm.py                  # Groq + Gemini LLM report
├── notebooks/
│   ├── FruitQualityAI_Training.ipynb     # Model training (Colab)
│   └── FruitQualityAI_Comparison.ipynb   # Model comparison
├── .streamlit/
│   └── config.toml                 # Streamlit theme config
├── streamlit_app.py                # Main web application
├── download_model.py               # Download weights from Drive
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
└── README.md
```

## Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fruit-quality-ai.git
cd fruit-quality-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
# Copy the example file
copy .env.example .env       # Windows
# cp .env.example .env       # Mac/Linux

# Open .env and add your keys:
# GROQ_API_KEY=your_key_here
# GEMINI_API_KEY=your_key_here
```
Get a free Groq key at [console.groq.com](https://console.groq.com)

### 5. Download model weights
```bash
python download_model.py
```

### 6. Run the app
```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## API Keys Required

| Key | Where to get | Free? |
|-----|-------------|-------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | ✅ Yes |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | ✅ Yes |

The LLM report is optional — the freshness detection works without API keys.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow 2.16 + Keras 3.3 |
| Model | EfficientNetB0 + CBAM Attention |
| Web App | Streamlit 1.35 |
| LLM (Primary) | Groq — Llama 3.3 70B |
| LLM (Fallback) | Google Gemini 1.5 Flash |
| Visualization | Grad-CAM (PIL-based) |
| Training | Google Colab (GPU) |

---

## Dataset

- **Source:** Kaggle — Fresh and Stale Images of Fruits and Vegetables
- **Total images:** 18,984 (after preprocessing)
- **Split:** 80% train / 10% val / 10% test
- **Classes:** apple_fresh, apple_rotten, banana_fresh, 
  banana_rotten, orange_fresh, orange_rotten

---

## Research Paper

This project is inspired by and compared against:

> Yuan, Y., Chen, J., Polat, K., & Alhudhaif, A. (2024). 
> *An innovative approach to detecting the freshness of fruits and 
> vegetables through the integration of convolutional neural networks 
> and bidirectional long short-term memory network.* 
> Current Research in Food Science, 8, 100723.
> https://doi.org/10.1016/j.crfs.2024.100723

---

## Limitations

- Works only with apple, banana, and orange
- Requires clear image with plain background for best results
- Pesticide detection is NOT possible from RGB camera images 
  (requires hyperspectral imaging — a hardware limitation, 
  not a software one)

---

## Future Work

- Add more fruit varieties
- 4-level ripeness detection (unripe → ripe → overripe → spoiled)
- Batch analysis using YOLO (detect multiple fruits in one photo)
- Mobile app (Flutter + FastAPI backend)
- Hyperspectral imaging integration for pesticide detection

---

## Author

Built as a final year project demonstrating the application of 
deep learning in agricultural food quality assessment.

---

## License

MIT License — free to use, modify, and distribute.
