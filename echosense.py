import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# improved_poem_emotion_analyzer.py
import re
from collections import Counter, defaultdict
from typing import Dict, Tuple, List

from echosense import analyze_emotions_transformer

# Optional imports; analyzer still works without them.
try:
    import spacy
    SPACY_OK = True
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    SPACY_OK = False
    _nlp = None

try:
    from sentence_transformers import SentenceTransformer, util
    EMB_OK = True
except Exception:
    EMB_OK = False
    SentenceTransformer = None
    util = None

# 1) Emotion set & helpers
EMOTIONS = [
    "Joy", "Sadness", "Anger", "Fear", "Trust", "Anticipation",
    "Disgust", "Surprise", "Love", "Belonging", "Serenity", "Healing"
]

def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return {"Neutral": 1.0}
    return {k: round(v / total, 3) for k, v in scores.items()}

# -----------------------
# 2) Phrase-level rules
# -----------------------
PHRASE_MAP = {
    "no longer a stranger": ("Trust", 3),
    "no longer a wanderer": ("Belonging", 3),
    "quieter brilliance": ("Belonging", 2),
    "turns the hollow into a hearth": ("Trust", 3),
    "wear my truth unhidden": ("Trust", 2),
    "wear it like armor": ("Strength", 2),
    "like rivers wear their beds": ("Serenity", 2),
    "finally belonging": ("Belonging", 3),
    "stitched into": ("Belonging", 2),
    "teaching me that movement": ("Anticipation", 1.5),
}

# -----------------------
# 3) Poetic imagery lexicon
# -----------------------
POETIC_LEXICON = {
    "hearth": [("Trust", 2.5), ("Belonging", 2.0)],
    "home": [("Trust", 2.5), ("Belonging", 2.0)],
    "fire": [("Trust", 1.5), ("Joy", 1.0)],
    "warmth": [("Trust", 2.0), ("Serenity", 1.5)],
    "river": [("Serenity", 1.5), ("Anticipation", 1.0)],
    "current": [("Anticipation", 1.0)],
    "stone": [("Sadness", 0.7), ("Serenity", 0.5)],
    "mountain": [("Trust", 1.5), ("Serenity", 1.0)],
    "snow": [("Serenity", 0.8)],
    "constellation": [("Belonging", 2.0), ("Serenity", 1.0)],
    "moon": [("Anticipation", 1.0), ("Belonging", 0.8)],
    "sun": [("Joy", 1.5)],
    "hollow": [("Sadness", 2.0)],
    "silence": [("Sadness", 1.0)],
    "gleam": [("Joy", 1.2)],
    "brilliance": [("Joy", 1.3), ("Belonging", 1.5)],
    "armor": [("Trust", 1.0)],
    "roots": [("Trust", 1.4)],
    "wings": [("Anticipation", 1.6)],
    "ashes": [("Sadness", 1.8)],
    "press": [("Sadness", 0.4)],
    "her": [("Love", 0.6)],
    "him": [("Love", 0.6)],
    "presence": [("Trust", 1.0), ("Belonging", 1.0)],
    "gaze": [("Love", 1.2), ("Trust", 0.8)]
}

SYNONYM_MAP = {
    "home": ["house", "home", "hearth"],
    "warmth": ["warmth", "heat"],
    "river": ["river", "stream"],
    "stone": ["stone", "rock"],
    "mountain": ["mountain", "peak"],
    "constellation": ["constellation", "star", "stars"],
    "silence": ["silence", "quiet"],
}

INTENSIFIERS = {"very": 1.5, "deeply": 1.6, "truly": 1.4, "utter": 1.6, "final": 1.2}
DOWNTONERS = {"slightly": 0.6, "a little": 0.6, "softly": 0.8}
NEGATIONS = {"not", "no", "never", "n't", "without", "none", "neither"}

_EMB_MODEL = None
def _load_embedder(name: str = "all-MiniLM-L6-v2"):
    global _EMB_MODEL
    if _EMB_MODEL is None:
        if not EMB_OK:
            raise RuntimeError("sentence-transformers is not installed.")
        _EMB_MODEL = SentenceTransformer(name)
    return _EMB_MODEL

# -----------------------
# 6) Main analyzer
# -----------------------
def analyze_emotions_enhanced(text: str,
                              use_embeddings: bool = False,
                              embedder_name: str = "all-MiniLM-L6-v2",
                              return_details: bool = False) -> Dict[str, float]:
    scores = Counter()
    detail = defaultdict(list)
    text_lower = text.lower()

    for phrase, (emo, w) in PHRASE_MAP.items():
        if phrase in text_lower:
            scores[emo] += w
            detail["phrases"].append((phrase, emo, w))

    sentences = re.split(r'\n{1,}|\.(?:\s|$)', text)
    def sentence_tokens(s):
        return re.findall(r"\b[\w'-]+\b", s.lower())

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        toks = sentence_tokens(s)
        neg_present = any(tok in NEGATIONS for tok in toks)
        intensity = 1.0
        for tok in toks:
            if tok in INTENSIFIERS:
                intensity *= INTENSIFIERS[tok]
            if tok in DOWNTONERS:
                intensity *= DOWNTONERS[tok]

        for tok in toks:
            if tok in POETIC_LEXICON:
                for emo, w in POETIC_LEXICON[tok]:
                    adj_w = w * intensity * (0.25 if neg_present else 1.0)
                    scores[emo] += adj_w
                    detail["lexicon"].append((tok, emo, adj_w, s))
            else:
                for key, variants in SYNONYM_MAP.items():
                    if tok in variants and key in POETIC_LEXICON:
                        for emo, w in POETIC_LEXICON[key]:
                            adj_w = w * intensity * (0.25 if neg_present else 1.0)
                            scores[emo] += adj_w
                            detail["synonym"].append((tok, key, emo, adj_w, s))

        if any(p in toks for p in ("her", "him", "their")):
            for warmth_word in ("hearth", "home", "warmth", "presence", "gaze"):
                if warmth_word in s:
                    scores["Love"] += 1.2 * intensity
                    scores["Trust"] += 1.0 * intensity
                    detail["pronoun_boost"].append((s, warmth_word))

        m = re.search(r'like\s+([a-z\'-]+)', s.lower())
        if m:
            img = m.group(1)
            if img in POETIC_LEXICON:
                for emo, w in POETIC_LEXICON[img]:
                    scores[emo] += 0.9 * w * intensity
                    detail["simile"].append((img, emo, w, s))

    if use_embeddings and EMB_OK:
        try:
            model = _load_embedder(embedder_name)
            prototypes = {
                "Joy": ["gleam, smile, bright, delight"],
                "Sadness": ["hollow, loss, grief, lonely"],
                "Anger": ["burning rage, furious"],
                "Fear": ["afraid, scared, anxious"],
                "Trust": ["home, hearth, safe, dependable, steady"],
                "Anticipation": ["waiting, future, hope, longing"],
                "Disgust": ["rotten, repulsive, gross"],
                "Surprise": ["sudden, surprised, shocked"],
                "Love": ["belonging, closeness, intimate, love"],
                "Belonging": ["fit in, belong, stitched, part of"],
                "Serenity": ["calm, serene, peaceful, stillness"],
                "Healing": ["mend, heal, recover, repair"]
            }
            sent_emb = model.encode(text, convert_to_tensor=True)
            for emo, proto_list in prototypes.items():
                proto_emb = model.encode(proto_list, convert_to_tensor=True)
                sim = float(util.pytorch_cos_sim(sent_emb, proto_emb).max().item())
                contrib = sim * 2.0
                if contrib > 0.05:
                    scores[emo] += contrib
                    detail["embeddings"].append((emo, sim, contrib))
        except Exception as e:
            detail["emb_error"] = str(e)

    if scores["Belonging"] > 0:
        scores["Trust"] += 0.25 * scores["Belonging"]
        scores["Serenity"] += 0.15 * scores["Belonging"]

    normalized = _normalize_scores(scores)
    if return_details:
        return {"scores": normalized, "raw": dict(scores), "details": dict(detail)}
    return normalized

# --- Main App CSS ---
st.markdown("""
    <style>
    ...
    </style>
""", unsafe_allow_html=True)

# --- Sample Data ---
sample_texts = {
    "Select a Sample...": "",
    "Poem": "I miss the sunflower fields, they remind me of home. A warmth I can't touch, a memory to roam.",
    "Journal Entry": "Today was a mixture of happiness and quiet moments of worry. I fear the unknown, but hold onto hope.",
    "Song Lyrics": "This feeling of nostalgia is a wound that never heals. I long for yesterday, for the love that was real.",
    "Nuanced Example": "But the sea in me is tired of rescues, and I have learned that a harbor can still be the middle of the ocean."
}

# --- Main App Layout ---
st.markdown("<h1 class='custom-title'>🌻 EchoSense</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-tagline'>Where poetry meets AI — decoding the hidden feelings in words.</p>", unsafe_allow_html=True)
st.markdown("<hr class='sunflower-divider'>", unsafe_allow_html=True)

if 'df_scores' not in st.session_state:
    st.session_state.df_scores = None

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.markdown("<h3 class='custom-header'>📄 Input Text</h3>", unsafe_allow_html=True)
    st.markdown("""
        <p style='font-size: 0.9rem; color: #8D6E63;'>
        This version uses an advanced, context-aware model to better understand nuanced emotion.
        </p>
    """, unsafe_allow_html=True)

    sample_choice = st.selectbox("Or choose a sample text:", list(sample_texts.keys()), key="sample_choice_box")

    text_input = st.text_area(
        "Enter or paste your text here:",
        value=sample_texts[sample_choice],
        height=250,
        label_visibility="collapsed",
        key="text_input_area"
    )

    if st.button("Analyze Emotions 🌈", use_container_width=True, key="analyze_button"):
        if text_input:
            with st.spinner('Analyzing...'):
                emotion_scores = analyze_emotions_transformer(text_input)
                df_scores = pd.DataFrame(emotion_scores.items(), columns=["Emotion", "Confidence"])
                df_scores = df_scores.sort_values(by="Confidence", ascending=False)
            st.session_state.df_scores = df_scores
            st.session_state.text_input = text_input
        else:
            st.session_state.df_scores = None
            st.warning("Please enter some text to analyze.")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if st.session_state.df_scores is not None:
        df_scores = st.session_state.df_scores
        text_input = st.session_state.text_input

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("### ✨ Results")
        dominant_emotion = df_scores.iloc[0]['Emotion']
        st.markdown(f"<div class='dominant-emotion-box'><p class='dominant-emotion-text'>Dominant Emotion: {dominant_emotion}</p></div>", unsafe_allow_html=True)

        chart_type = st.radio("Choose a visualization type:", ('Bar Chart', 'Radar Chart'), key="chart_type_radio")

        if chart_type == 'Bar Chart':
            fig = px.bar(
                df_scores, x="Confidence", y="Emotion", orientation='h',
                title="Emotional Profile",
                labels={'Confidence': 'Confidence Score', 'Emotion': 'Emotion'},
                height=350,
                color='Confidence',
                color_continuous_scale=[(0, '#FFDAB9'), (1, '#FF6347')]
            )
            fig.update_layout(
                xaxis_tickformat=".0%", showlegend=False, font=dict(family="Inter"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title_font_color="#A0522D"
            )
            st.plotly_chart(fig, use_container_width=True, key="bar_chart")

        else:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=df_scores['Confidence'].values,
                theta=df_scores['Emotion'].values,
                fill='toself',
                name='Emotion Scores',
                marker=dict(color="#FF7F50"),
                line=dict(color="#FF7F50")
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    angularaxis=dict(showline=False, tickfont_size=12, tickfont_family="Inter", tickcolor="#A0522D")
                ),
                showlegend=False,
                height=450,
                title="Emotional Profile (Radar View)",
                font=dict(family="Inter"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title_font_color="#A0522D"
            )
            st.plotly_chart(fig, use_container_width=True, key="radar_chart")

        st.markdown("<div class='word-cloud-container'>", unsafe_allow_html=True)
        st.markdown("### 📝 Key Words")
        words = re.findall(r'\b\w+\b', text_input.lower())
        html_output = ""
        for word in words:
            class_name = ""
            if 'tired' in word or 'rescues' in word:
                class_name = "emotion-highlight-sadness"
            elif 'hope' in word:
                class_name = "emotion-highlight-anticipation"
            elif 'love' in word:
                class_name = "emotion-highlight-joy"
            if class_name:
                html_output += f"<span class='word-cloud-word {class_name}'>{word}</span> "
            else:
                st.html

                #this model is under development and is not fully functional yet
                #goal for completion is to have a more nuanced emotion detection model

