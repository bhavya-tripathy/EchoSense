# improved_poem_emotion_analyzer.py
import re
from collections import Counter, defaultdict
from typing import Dict, Tuple, List

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