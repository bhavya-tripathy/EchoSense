import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# A comprehensive mock of a real emotion lexicon with a much wider range of words.
nrc_emotion_lexicon = {
    'abandoned': ['sadness', 'fear'], 'alone': ['sadness', 'disgust'], 'amused': ['joy'],
    'anger': ['anger'], 'anxious': ['fear'], 'anticipation': ['anticipation'],
    'apprehensive': ['fear'], 'calm': ['calmness'], 'cheer': ['joy', 'anticipation'],
    'comfort': ['joy', 'trust'], 'contempt': ['disgust', 'anger'], 'courage': ['trust', 'joy'],
    'disgust': ['disgust'], 'disappointed': ['sadness'], 'excited': ['joy', 'anticipation'],
    'fear': ['fear'], 'forgive': ['trust', 'joy'], 'furious': ['anger'], 'grief': ['sadness'],
    'happy': ['joy', 'trust'], 'hate': ['anger', 'disgust'], 'hope': ['anticipation', 'joy'],
    'joy': ['joy'], 'lonely': ['sadness'], 'longing': ['sadness', 'anticipation'],
    'love': ['joy', 'trust'], 'miss': ['sadness', 'longing'], 'nostalgia': ['sadness', 'joy'],
    'pain': ['sadness', 'fear'], 'peaceful': ['calmness', 'trust'], 'sad': ['sadness'],
    'scared': ['fear'], 'serene': ['calmness'], 'shock': ['surprise'], 'sorrow': ['sadness'],
    'surprise': ['surprise'], 'terrified': ['fear'], 'trust': ['trust'], 'unhappy': ['sadness'],
    'upset': ['sadness', 'anger'], 'worry': ['fear', 'sadness'], 'wound': ['pain', 'sadness'],
    'sunflower': ['joy', 'calmness'], 'field': ['calmness'], 'home': ['joy', 'trust', 'nostalgia'],
    # Expanded lexicon for a more nuanced analysis
    'adored': ['joy', 'trust'], 'anxiety': ['fear', 'sadness'], 'bliss': ['joy'],
    'brave': ['trust'], 'calmness': ['calmness', 'joy'], 'care': ['joy', 'trust'],
    'celebration': ['joy', 'anticipation'], 'cold': ['sadness', 'disgust'], 'crushed': ['sadness'],
    'darkness': ['fear', 'sadness'], 'dawn': ['anticipation', 'joy'], 'despair': ['sadness'],
    'dream': ['joy', 'anticipation'], 'ecstasy': ['joy'], 'empty': ['sadness'],
    'enraged': ['anger'], 'envy': ['disgust', 'anger'], 'euphoria': ['joy'],
    'exuberant': ['joy'], 'fading': ['sadness'], 'faith': ['trust', 'anticipation'],
    'fantastic': ['joy'], 'gloom': ['sadness'], 'gratitude': ['joy', 'trust'],
    'haunted': ['fear'], 'heartbroken': ['sadness'], 'hopeful': ['anticipation', 'joy'],
    'horror': ['fear', 'disgust'], 'hurt': ['sadness', 'anger'], 'inspire': ['joy', 'anticipation'],
    'joyful': ['joy'], 'laugh': ['joy'], 'loneliness': ['sadness'], 'longing': ['sadness', 'anticipation'],
    'magic': ['surprise', 'joy'], 'miserable': ['sadness'], 'mourn': ['sadness'],
    'peace': ['calmness', 'trust'], 'pleasure': ['joy'], 'rage': ['anger'],
    'relief': ['joy', 'calmness'], 'scared': ['fear'], 'shocked': ['surprise'],
    'sorrowful': ['sadness'], 'suffering': ['sadness', 'pain'], 'surprise': ['surprise'],
    'sympathy': ['trust', 'sadness'], 'tenderness': ['joy', 'trust'], 'thrill': ['joy', 'surprise'],
    'torment': ['anger', 'pain'], 'tranquil': ['calmness', 'trust'], 'trapped': ['fear'],
    'vibrant': ['joy', 'anticipation'], 'wonder': ['surprise', 'joy'], 'yearning': ['sadness', 'longing']
}

# Mapping our mock lexicon to your 8 core emotions with emojis
emotion_mapping = {
    'joy': 'Joy 🎉', 'sadness': 'Sadness 😢', 'anger': 'Anger 😡', 'fear': 'Fear 😨',
    'trust': 'Trust 🤝', 'anticipation': 'Anticipation ⏳', 'surprise': 'Surprise 😲',
    'disgust': 'Disgust 🤢', 'longing': 'Sadness 😢', 'nostalgia': 'Sadness 😢',
    'calmness': 'Neutral', 'peace': 'Neutral', 'pain': 'Sadness 😢'
}
core_emotions = ['Joy 🎉', 'Sadness 😢', 'Anger 😡', 'Fear 😨', 'Trust 🤝', 'Anticipation ⏳', 'Disgust 🤢', 'Surprise 😲']

def analyze_emotions_lexicon(text):
    """
    Analyzes text and returns emotion scores based on a lexicon.
    """
    cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
    words = cleaned_text.split()
    
    emotion_scores = Counter()
    for word in words:
        if word in nrc_emotion_lexicon:
            for emotion in nrc_emotion_lexicon[word]:
                mapped_emotion = emotion_mapping.get(emotion, 'Neutral')
                if mapped_emotion != 'Neutral':
                    emotion_scores[mapped_emotion] += 1
    
    # Ensure all core emotions are present, even with a score of 0
    all_scores = {emotion: emotion_scores.get(emotion, 0) for emotion in core_emotions}

    total_score = sum(all_scores.values())
    if total_score == 0:
        return {'Neutral': 1}
        
    normalized_scores = {
        emotion: count / total_score for emotion, count in all_scores.items()
    }

    dominant_emotion = max(normalized_scores, key=normalized_scores.get)
    if dominant_emotion == 'Neutral' and len(normalized_scores) > 1:
        del normalized_scores['Neutral']
        total_score_no_neutral = sum(normalized_scores.values())
        if total_score_no_neutral > 0:
            return {
                emotion: count / total_score_no_neutral for emotion, count in normalized_scores.items()
            }
        else:
            return {'Neutral': 1}
    
    return normalized_scores

# --- Sunflower Themed CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background: #FDF9EE; /* Soft off-white */
    }
    .st-emotion-cache-1c5c5z {
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        background-color: white; /* Card background */
    }
    h1 {
        color: #FFC107; /* Sunflower Yellow */
        font-weight: 800;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-size: 3.5rem;
    }
    .header-tagline {
        text-align: center;
        color: #8D6E63; /* Soft Brown */
        font-style: italic;
        font-size: 1.1rem;
        margin-top: -15px;
    }
    hr.sunflower-divider {
        border: 0;
        height: 2px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), #FFC107, rgba(0, 0, 0, 0));
        margin-bottom: 2rem;
    }
    .st-emotion-cache-1p6c99c {
        border-radius: 12px;
    }
    .st-emotion-cache-13ko42a {
        border-radius: 12px;
        background-color: #FFFDE7;
        border: 1px solid #FFECB3;
        transition: box-shadow 0.3s ease-in-out;
    }
    .st-emotion-cache-13ko42a:focus-within {
        box-shadow: 0 0 0 3px #FFD54F;
    }
    .st-emotion-cache-19t5016 {
        background-color: #FFC107;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s;
    }
    .st-emotion-cache-19t5016:hover {
        background-color: #FFA000;
    }
    .result-card {
        background-color: #E8F5E9; /* Light Green */
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border: 2px solid #C8E6C9;
        animation: fadeIn 0.8s ease-in-out;
    }
    .dominant-emotion-box {
        background-color: #F1F8E9; /* Very light green */
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    .dominant-emotion-text {
        font-size: 2rem;
        font-weight: bold;
        color: #388E3C;
    }
    .word-cloud-container {
        border-radius: 12px;
        padding: 20px;
        background-color: #FFFDE7; /* Light yellow */
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
    .emotion-highlight-joy { color: #FFD700; }
    .emotion-highlight-sadness { color: #1E90FF; }
    .emotion-highlight-anger { color: #FF4500; }
    .emotion-highlight-fear { color: #8B4513; }
    .emotion-highlight-trust { color: #2E8B57; }
    .emotion-highlight-anticipation { color: #FFA500; }
    .emotion-highlight-surprise { color: #9370DB; }
    .emotion-highlight-disgust { color: #556B2F; }
    .footer-text {
        text-align: center;
        font-style: italic;
        color: #757575;
        margin-top: 2rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# Sample texts for the dropdown
sample_texts = {
    "Select a Sample...": "",
    "Poem": "I miss the sunflower fields, they remind me of home. A warmth I can't touch, a memory to roam.",
    "Journal Entry": "Today was a mixture of happiness and quiet moments of worry. I fear the unknown, but hold onto hope.",
    "Song Lyrics": "This feeling of nostalgia is a wound that never heals. I long for yesterday, for the love that was real."
}

# --- Main App Layout ---
st.title("🌻 EchoSense")
st.markdown("<p class='header-tagline'>Where poetry meets AI — decoding the hidden feelings in words.</p>", unsafe_allow_html=True)
st.markdown("<hr class='sunflower-divider'>", unsafe_allow_html=True)

# Initialize session state for analysis results
if 'df_scores' not in st.session_state:
    st.session_state.df_scores = None

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.markdown("### 📝 Input Text")

    # Dropdown for sample texts
    sample_choice = st.selectbox("Or choose a sample text:", list(sample_texts.keys()))
    
    # Text area
    text_input = st.text_area(
        "Enter or paste your text here:",
        value=sample_texts[sample_choice],
        height=250,
        label_visibility="collapsed"
    )
    
    if st.button("Analyze Emotions 🌈", use_container_width=True):
        if text_input:
            with st.spinner('Analyzing...'):
                emotion_scores = analyze_emotions_lexicon(text_input)
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
        chart_type = st.radio("Choose a visualization type:", ('Bar Chart', 'Radar Chart'))
        
        # Plotly Bar Chart
        if chart_type == 'Bar Chart':
            fig = px.bar(
                df_scores, x="Confidence", y="Emotion", orientation='h',
                title="Emotional Profile",
                labels={'Confidence': 'Confidence Score', 'Emotion': 'Emotion'},
                height=350,
                color='Confidence',
                color_continuous_scale=[(0, '#FFF59D'), (1, '#FFC107')]
            )
            fig.update_layout(
                xaxis_tickformat=".0%", showlegend=False, font=dict(family="Inter"),
                paper_bgcolor="white", plot_bgcolor="white", title_font_color="#8D6E63"
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
                marker=dict(color="#FFC107"),
                line=dict(color="#FFC107")
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    angularaxis=dict(showline=False, tickfont_size=12, tickfont_family="Inter", tickcolor="#8D6E63")
                ),
                showlegend=False,
                height=450,
                title="Emotional Profile (Radar View)",
                font=dict(family="Inter"),
                paper_bgcolor="white", plot_bgcolor="white", title_font_color="#8D6E63"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Word cloud visualization
        st.markdown("<div class='word-cloud-container'>", unsafe_allow_html=True)
        st.markdown("### 📝 Key Words")
        cleaned_text = re.sub(r'[^\w\s]', '', text_input.lower())
        words = cleaned_text.split()
        
        html_output = ""
        for word in words:
            found_emotions = []
            if word in nrc_emotion_lexicon:
                for emotion in nrc_emotion_lexicon[word]:
                    mapped_emotion = emotion_mapping.get(emotion, 'Neutral')
                    if mapped_emotion in core_emotions:
                        found_emotions.append(mapped_emotion.split(' ')[0].lower())
            
            if found_emotions:
                class_name = f"emotion-highlight-{found_emotions[0]}"
                html_output += f"<span class='word-cloud-word {class_name}'>{word}</span> "
            else:
                html_output += f"<span class='word-cloud-word' style='color:#BDBDBD;'>{word}</span> "
        
        st.markdown(html_output, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p class='footer-text'>🌻 Built with love by EchoSense • Poetry + AI + Emotions</p>", unsafe_allow_html=True)
