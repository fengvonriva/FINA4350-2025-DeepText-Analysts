from afinn import Afinn

# Initialize AFINN
afinn = Afinn()

def analyze_afinn(text):
    """
    Analyze sentiment of a preprocessed string using AFINN dictionary.
    Args:
        text (str): Preprocessed string (lowercase, no stopwords).
    Returns:
        float: AFINN sentiment score.
    """
    return afinn.score(text)