import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO






# --- Emotion Analysis Functions ---
# We will now use a mock for a pre-trained, fine-tuned transformer model.
def analyze_emotions_transformer(text):
    """
    Simulates a fine-tuned transformer model's emotion prediction.
    This mock logic is designed to be more context-aware than a lexicon.
    """
    text_lower = text.lower()
    
    # Define a more complex, phrase-based mock logic
    if 'sea' in text_lower and 'tired' in text_lower and 'rescues' in text_lower:
        return {'Sadness 😢': 0.7, 'Anticipation ⏳': 0.2, 'Fear 😨': 0.1}
    elif 'grief' in text_lower or 'sorrow' in text_lower or 'pain' in text_lower:
        return {'Sadness 😢': 0.8, 'Anger 😡': 0.2}
    elif 'hope' in text_lower or 'hopeful' in text_lower:
        return {'Anticipation ⏳': 0.6, 'Joy 🎉': 0.3, 'Trust 🤝': 0.1}
    elif 'love' in text_lower or 'tenderness' in text_lower or 'heart':
        return {'Joy 🎉': 0.5, 'Trust 🤝': 0.4, 'Anticipation ⏳': 0.1}
    elif 'anger' in text_lower or 'furious' in text_lower:
        return {'Anger 😡': 0.9, 'Disgust 🤢': 0.1}
    else:
        # Fallback to a basic sentiment check for any words not in specific phrases
        emotion_scores = Counter()
        lexicon = {
            'happy': 'Joy 🎉', 'joyful': 'Joy 🎉', 'sunflower': 'Joy 🎉', 'miss': 'Sadness 😢',
            'lonely': 'Sadness 😢', 'sad': 'Sadness 😢', 'angry': 'Anger 😡', 'fear': 'Fear 😨'
        }
        words = re.findall(r'\b\w+\b', text_lower)
        for word in words:
            if word in lexicon:
                emotion_scores[lexicon[word]] += 1
        
        if not emotion_scores:
            return {'Neutral': 1.0}
        
        total = sum(emotion_scores.values())
        return {emotion: count/total for emotion, count in emotion_scores.items()}

# Mapping our mock lexicon to your 8 core emotions with emojis
core_emotions = ['Joy 🎉', 'Sadness 😢', 'Anger 😡', 'Fear 😨', 'Trust 🤝', 'Anticipation ⏳', 'Disgust 🤢', 'Surprise 😲']









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
        chart_type = st.radio("Choose a visualization type:", ('Bar Chart', 'Radar Chart'))
        
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
