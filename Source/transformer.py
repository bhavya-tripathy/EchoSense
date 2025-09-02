# --- Emotion Analysis Functions (Expanded Mock Transformer) ---
import re
from typing import Counter


def analyze_emotions_transformer(text):
    """
    Simulates a fine-tuned transformer model's emotion prediction.
    Expanded with more keywords, weighted matches, and phrase detection.
    """
    text_lower = text.lower()
    emotion_scores = Counter()

    # --- Phrase-level rules (catch strong multi-word signals first) ---
    phrase_map = {
        "broken heart": ("Sadness 😢", 3),
        "lost in the dark": ("Fear 😨", 3),
        "new beginnings": ("Anticipation ⏳", 2),
        "all alone": ("Sadness 😢", 2),
        "burning rage": ("Anger 😡", 3),
        "pure joy": ("Joy 🎉", 3),
        "deep trust": ("Trust 🤝", 2),
        "utter disgust": ("Disgust 🤢", 3),
        "sudden shock": ("Surprise 😲", 3)
    }
    for phrase, (emotion, weight) in phrase_map.items():
        if phrase in text_lower:
            emotion_scores[emotion] += weight

    # --- Word-level lexicon with weights ---
    lexicon = {
        # Joy
        "happy": ("Joy 🎉", 1), "joy": ("Joy 🎉", 1), "joyful": ("Joy 🎉", 1),
        "excited": ("Joy 🎉", 1), "delight": ("Joy 🎉", 1), "cheerful": ("Joy 🎉", 1),
        "smile": ("Joy 🎉", 1), "sunflower": ("Joy 🎉", 2),

        # Sadness
        "miss": ("Sadness 😢", 1), "lonely": ("Sadness 😢", 1), "sad": ("Sadness 😢", 1),
        "grief": ("Sadness 😢", 2), "loss": ("Sadness 😢", 2), "tears": ("Sadness 😢", 1),
        "heartbroken": ("Sadness 😢", 3), "sorrow": ("Sadness 😢", 2),

        # Anger
        "angry": ("Anger 😡", 1), "mad": ("Anger 😡", 1), "rage": ("Anger 😡", 2),
        "furious": ("Anger 😡", 2), "resent": ("Anger 😡", 1), "irritated": ("Anger 😡", 1),
        "hate": ("Anger 😡", 2),

        # Fear
        "fear": ("Fear 😨", 1), "scared": ("Fear 😨", 1), "afraid": ("Fear 😨", 1),
        "anxious": ("Fear 😨", 2), "terror": ("Fear 😨", 3), "panic": ("Fear 😨", 2),
        "nervous": ("Fear 😨", 1),

        # Trust
        "trust": ("Trust 🤝", 1), "safe": ("Trust 🤝", 1), "reliable": ("Trust 🤝", 1),
        "faith": ("Trust 🤝", 2), "secure": ("Trust 🤝", 1), "depend": ("Trust 🤝", 1),

        # Anticipation
        "hope": ("Anticipation ⏳", 1), "hopeful": ("Anticipation ⏳", 1), "future": ("Anticipation ⏳", 1),
        "waiting": ("Anticipation ⏳", 1), "dream": ("Anticipation ⏳", 2), "longing": ("Anticipation ⏳", 2),

        # Disgust
        "disgust": ("Disgust 🤢", 2), "gross": ("Disgust 🤢", 1), "nasty": ("Disgust 🤢", 1),
        "rotten": ("Disgust 🤢", 2), "repulsive": ("Disgust 🤢", 2), "vile": ("Disgust 🤢", 2),

        # Surprise
        "surprised": ("Surprise 😲", 1), "shocked": ("Surprise 😲", 2), "astonished": ("Surprise 😲", 2),
        "unexpected": ("Surprise 😲", 1), "suddenly": ("Surprise 😲", 1)
    }

    # --- Tokenize text into words and match against lexicon ---
    words = re.findall(r'\b\w+\b', text_lower)
    for word in words:
        if word in lexicon:
            emotion, weight = lexicon[word]
            emotion_scores[emotion] += weight

    # --- Handle empty match case ---
    if not emotion_scores:
        return {"Neutral": 1.0}

    # --- Normalize scores ---
    total = sum(emotion_scores.values())
    return {emotion: round(count / total, 3) for emotion, count in emotion_scores.items()}
