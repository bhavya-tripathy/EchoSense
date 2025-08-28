# 🌻 EchoSense  
*A Python tool for detecting nuanced emotions in creative text.*  

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)  
![NLP](https://img.shields.io/badge/Field-NLP-yellow.svg)  
![Status](https://img.shields.io/badge/Status-Prototype-green.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  

---

## 📖 Overview  
**EchoSense** is an NLP-powered tool that identifies **fine-grained emotions** in text such as poetry, journals, and song lyrics.  
Traditional sentiment analysis usually stops at *positive, negative,* or *neutral*. EchoSense goes deeper — capturing a richer spectrum of emotions like **joy, sadness, anger, fear, trust, anticipation, surprise, and disgust**.  

This project represents **Phase 1** of a larger research initiative on **nuanced emotion detection**, with potential applications in:  
- 🧠 **Mental Health Support** – early indicators of emotional distress in personal writing.  
- 📱 **Digital Well-Being** – emotion-aware platforms for healthier online interactions.  
- 🎨 **Creative Tools** – insights for writers, poets, and musicians into the emotional texture of their work.  

---

## 🚀 Features  
- 🔤 **Preprocessing pipeline**: tokenization, stopword removal, lemmatization.  
- 📑 **Lexicon-based classification** using the [NRC Emotion Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm).  
- 📊 Outputs both **dominant emotion** and **detailed emotion distribution scores**.  
- 💻 Lightweight **command-line interface (CLI)**.  
- 🔮 Designed for extensibility — upcoming versions will integrate ML classifiers and transformer-based models (BERT, RoBERTa).  

---

## ⚙️ Installation  

Clone the repository and install dependencies:  
```bash
git clone https://github.com/YourUsername/EchoSense.git
cd EchoSense
pip install -r requirements.txt
