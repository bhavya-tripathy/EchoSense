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
# 
EMOTIONS = [
    "Joy", "Sadness", "Anger", "Fear", "Trust", "Anticipation",
    "Disgust", "Surprise", "Love", "Belonging", "Serenity", "Healing"
]

def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        # fallback to neutral-like distribution
        return {"Neutral": 1.0}
    return {k: round(v / total, 3) for k, v in scores.items()}

# -----------------------
# 2) Phrase-level rules (strong signals)
# -----------------------
PHRASE_MAP = {
    # strong multi-word cues (phrase -> (emotion, weight))
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
# 3) Poetic imagery lexicon (word -> list of (emotion, weight))
# -----------------------
POETIC_LEXICON = {
    # warmth / hearth / home
    "hearth": [("Trust", 2.5), ("Belonging", 2.0)],
    "home": [("Trust", 2.5), ("Belonging", 2.0)],
    "fire": [("Trust", 1.5), ("Joy", 1.0)],
    "warmth": [("Trust", 2.0), ("Serenity", 1.5)],

    # natural elements & processes
    "river": [("Serenity", 1.5), ("Anticipation", 1.0)],
    "current": [("Anticipation", 1.0)],
    "stone": [("Sadness", 0.7), ("Serenity", 0.5)],
    "mountain": [("Trust", 1.5), ("Serenity", 1.0)],
    "snow": [("Serenity", 0.8)],
    "constellation": [("Belonging", 2.0), ("Serenity", 1.0)],
    "moon": [("Anticipation", 1.0), ("Belonging", 0.8)],
    "sun": [("Joy", 1.5), ("Hope", 1.0)] if False else [("Joy", 1.5)],

    # emotional metaphors
    "hollow": [("Sadness", 2.0)],
    "silence": [("Sadness", 1.0)],
    "gleam": [("Joy", 1.2), ("Hope", 1.0)] if False else [("Joy", 1.2)],
    "brilliance": [("Joy", 1.3), ("Belonging", 1.5)],
    "armor": [("Trust", 1.0), ("Strength", 1.5)] if False else [("Trust", 1.0)],
    "roots": [("Trust", 1.4)],
    "wings": [("Anticipation", 1.6)],
    "ashes": [("Sadness", 1.8)],
    "press": [("Sadness", 0.4)],

    # love & closeness
    "her": [("Love", 0.6)],  # pronoun-based soft signal; strengthened by context
    "him": [("Love", 0.6)],
    "presence": [("Trust", 1.0), ("Belonging", 1.0)],
    "gaze": [("Love", 1.2), ("Trust", 0.8)]
}

# Map some common synonyms -> lexical keys (simple)
SYNONYM_MAP = {
    "home": ["house", "home", "hearth"],
    "warmth": ["warmth", "heat"],
    "river": ["river", "stream"],
    "stone": ["stone", "rock"],
    "mountain": ["mountain", "peak"],
    "constellation": ["constellation", "star", "stars"],
    "silence": ["silence", "quiet"],
}

# -----------------------
# 4) Modifiers & negation
# -----------------------
INTENSIFIERS = {"very": 1.5, "deeply": 1.6, "truly": 1.4, "utter": 1.6, "final": 1.2}
DOWNTONERS = {"slightly": 0.6, "a little": 0.6, "softly": 0.8}
NEGATIONS = {"not", "no", "never", "n't", "without", "none", "neither"}

# -----------------------
# 5) Optional embedding model loader
# -----------------------
_EMB_MODEL = None
def _load_embedder(name: str = "all-MiniLM-L6-v2"):
    global _EMB_MODEL
    if _EMB_MODEL is None:
        if not EMB_OK:
            raise RuntimeError("sentence-transformers is not installed. pip install sentence-transformers")
        _EMB_MODEL = SentenceTransformer(name)
    return _EMB_MODEL

# -----------------------
# 6) Main analyzer
# -----------------------
def analyze_emotions_enhanced(text: str,
                              use_embeddings: bool = False,
                              embedder_name: str = "all-MiniLM-L6-v2",
                              return_details: bool = False) -> Dict[str, float]:
    """
    Returns normalized multi-label scores for EMOTIONS based on:
      - phrase_map matches (strong)
      - poetic_lexicon weighted matches
      - sentence-level aggregation
      - optional semantic matches via sentence-transformers
      - negation & intensity handling

    `use_embeddings`: if True and sentence-transformers is installed, uses semantic similarity
                      between sentences and short emotion prototypes to add soft scores.
    """
    scores = Counter()
    detail = defaultdict(list)

    text_lower = text.lower()

    # 1) Phrase-level strong rules
    for phrase, (emo, w) in PHRASE_MAP.items():
        if phrase in text_lower:
            scores[emo] += w
            detail["phrases"].append((phrase, emo, w))

    # 2) Sentence-level processing (gives local context for modifiers/negation)
    # Simple sentence split (keeps poetry lines together)
    sentences = re.split(r'\n{1,}|\.(?:\s|$)', text)
    def sentence_tokens(s):
        return re.findall(r"\b[\w'-]+\b", s.lower())

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        toks = sentence_tokens(s)

        # detect negation in window (simple heuristic)
        neg_present = any(tok in NEGATIONS for tok in toks)
        # detect intensity words
        intensity = 1.0
        for tok in toks:
            if tok in INTENSIFIERS:
                intensity *= INTENSIFIERS[tok]
            if tok in DOWNTONERS:
                intensity *= DOWNTONERS[tok]

        # 2a) lexicon matching in the sentence
        for tok in toks:
            # direct lexical hits
            if tok in POETIC_LEXICON:
                for emo, w in POETIC_LEXICON[tok]:
                    adj_w = w * intensity * (0.25 if neg_present else 1.0)
                    scores[emo] += adj_w
                    detail["lexicon"].append((tok, emo, adj_w, s))
            else:
                # check synonyms simple map
                for key, variants in SYNONYM_MAP.items():
                    if tok in variants and key in POETIC_LEXICON:
                        for emo, w in POETIC_LEXICON[key]:
                            adj_w = w * intensity * (0.25 if neg_present else 1.0)
                            scores[emo] += adj_w
                            detail["synonym"].append((tok, key, emo, adj_w, s))

        # 2b) pronoun-based pattern boosting (if 'her' or 'him' near warmth/home words)
        if any(p in toks for p in ("her", "him", "their")):
            # check if hearth/home warmth also in same sentence
            for warmth_word in ("hearth", "home", "warmth", "presence", "gaze"):
                if warmth_word in s:
                    scores["Love"] += 1.2 * intensity
                    scores["Trust"] += 1.0 * intensity
                    detail["pronoun_boost"].append((s, warmth_word))

        # 2c) structural metaphors: "like X" patterns (simulate metaphor detection)
        # e.g., "Like rivers wear their beds" -> match "like <noun>"
        m = re.search(r'like\s+([a-z\'-]+)', s.lower())
        if m:
            img = m.group(1)
            if img in POETIC_LEXICON:
                for emo, w in POETIC_LEXICON[img]:
                    scores[emo] += 0.9 * w * intensity
                    detail["simile"].append((img, emo, w, s))

    # 3) Optional semantic similarity fallback (adds soft evidence)
    if use_embeddings and EMB_OK:
        try:
            model = _load_embedder(embedder_name)
            # build short emotion prototypes (phrases that capture each emotion)
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
                # scale sim (0..1) -> weight contribution
                # small multiplier because lexical signals should remain primary
                contrib = sim * 2.0
                if contrib > 0.05:
                    scores[emo] += contrib
                    detail["embeddings"].append((emo, sim, contrib))
        except Exception as e:
            detail["emb_error"] = str(e)

    # 4) small smoothing: if 'belonging' or 'love' high, also nudge 'trust' and 'serenity'
    if scores["Belonging"] > 0:
        scores["Trust"] += 0.25 * scores["Belonging"]
        scores["Serenity"] += 0.15 * scores["Belonging"]

    # 5) Final normalization & return
    normalized = _normalize_scores(scores)
    if return_details:
        return {"scores": normalized, "raw": dict(scores), "details": dict(detail)}
    return normalized



# --- Emotion Analysis Functions (Mock Transformer) ---









# --- Main App CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sunset background image for the main app container */
    [data-testid="stAppViewContainer"] {
        background-image: url("https://wallpapercave.com/wp/wp5256714.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    
    /* Make header and sidebar transparent */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    
    /* Make the main content blocks semi-transparent */
    div.block-container {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
    }

    /* Style for the main title */
    .custom-title {
        font-size: 3.5rem;
        font-weight: bold;
        color: EE7600; /* Coral for sunset feel */
        text-align: center;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    }

    /* Style for section headers like "Input Text" */
    .custom-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #FFA07A; /* Light Salmon */
        margin-bottom: 10px;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.2);
    }
    
    .header-tagline {
        font-size: 1.1rem;
        color: #FFC0CB; /* Pink */
        font-style: italic;
        text-align: center;
        margin-top: -10px;
    }

    .sunflower-divider {
        border: none;
        height: 3px;
        background: #FFD700; /* Gold */
        margin-top: 20px;
        margin-bottom: 20px;
        border-radius: 2px;
        box-shadow: 0px 2px 6px rgba(255, 215, 0, 0.6);
    }
    
    /* --- Button Styles --- */
    .st-emotion-cache-19t5016 {
        background-color: #FF7F50; /* Coral */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s, background-color 0.3s, box-shadow 0.3s;
    }
    .st-emotion-cache-19t5016:hover {
        background-color: #FA8072; /* Salmon */
        transform: translateY(-3px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    /* --- Results Card Styles --- */
    .result-card {
        background-color: rgba(255, 245, 238, 0.9); /* SeaShell */
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border: 2px solid #F4A460; /* SandyBrown */
        animation: fadeIn 0.8s ease-in-out;
    }
    .dominant-emotion-box {
        background-color: #FFF0E0; /* PeachPuff */
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    .dominant-emotion-text {
        font-size: 2rem;
        font-weight: bold;
        color: #CD5C5C; /* IndianRed */
    }
    .word-cloud-container {
        border-radius: 12px;
        padding: 20px;
        background-color: #FFF0E0; /* Light orange */
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .word-cloud-word {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 5px;
        display: inline-block;
        transition: transform 0.2s;
    }
    .word-cloud-word:hover {
        transform: scale(1.1);
    }
    
    /* --- Emotion Word Colors --- */
    .emotion-highlight-joy { color: #FF7F50; } /* Coral */
    .emotion-highlight-sadness { color: #8A2BE2; } /* BlueViolet */
    .emotion-highlight-anger { color: #FF6347; } /* Tomato */
    .emotion-highlight-fear { color: #B22222; } /* FireBrick */
    .emotion-highlight-trust { color: #228B22; } /* ForestGreen */
    .emotion-highlight-anticipation { color: #FFD700; } /* Gold */
    .emotion-highlight-surprise { color: #DA70D6; } /* Orchid */
    .emotion-highlight-disgust { color: #4B0082; } /* Indigo */

    .footer-text {
        text-align: center;
        font-style: italic;
        color: #A0522D;
        margin-top: 2rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
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

# Initialize session state for analysis results
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

# Dropdown for sample texts
sample_choice = st.selectbox("Or choose a sample text:", list(sample_texts.keys()), key="sample_text_selector")    
# Text area
text_input = st.text_area(
    "Enter or paste your text here:",
    value=sample_texts[sample_choice],
    height=250,
    label_visibility="collapsed",
    key="text_input_main"  # <-- Add this unique key
)

if st.button("Analyze Emotions 🌈", use_container_width=True, key="analyze_button"): # <--- Added key
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
        
        # Plotly chart toggle
        chart_type = st.radio("Choose a visualization type:", ('Bar Chart', 'Radar Chart'), key="chart_type_selector") # <--- Added key
        
        # Plotly Bar Chart
        if chart_type == 'Bar Chart':
            fig = px.bar(
                df_scores, x="Confidence", y="Emotion", orientation='h',
                title="Emotional Profile",
                labels={'Confidence': 'Confidence Score', 'Emotion': 'Emotion'},
                height=350,
                color='Confidence',
                color_continuous_scale=[(0, '#FFDAB9'), (1, '#FF6347')] # New sunset colors
            )
            fig.update_layout(
                xaxis_tickformat=".0%", showlegend=False, font=dict(family="Inter"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title_font_color="#A0522D"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Plotly Radar Chart
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=df_scores['Confidence'].values,
                theta=df_scores['Emotion'].values,
                fill='toself',
                name='Emotion Scores',
                marker=dict(color="#FF7F50"), # New sunset color
                line=dict(color="#FF7F50") # New sunset color
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
            st.plotly_chart(fig, use_container_width=True)

        # Word cloud visualization
        st.markdown("<div class='word-cloud-container'>", unsafe_allow_html=True)
        st.markdown("### 📝 Key Words")
        words = re.findall(r'\b\w+\b', text_input.lower())
        
        # Note: The word cloud for a transformer model is less about individual words.
        # This is a symbolic representation of the app's internal logic.
        html_output = ""
        for word in words:
            # Simple keyword matching for visualization purposes only
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
                html_output += f"<span class='word-cloud-word' style='color:#BDBDBD;'>{word}</span> "
        
        st.markdown(html_output, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
st.markdown("<p class='footer-text'>🌻 Built with love by EchoSense • Poetry + AI + Emotions</p>", unsafe_allow_html=True)
