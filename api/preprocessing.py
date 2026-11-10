"""
Nettoyage de texte partagé entre l'entraînement (notebook 01) et l'API,
pour garantir que les messages envoyés à /predict subissent exactement
le même prétraitement que les données utilisées pour l'entraînement.
"""

import re
import unicodedata

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s']")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text
