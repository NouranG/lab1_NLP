import nltk
import os

NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")

def setup_nltk():
    nltk.data.path.clear()
    nltk.data.path.append(NLTK_DIR)

    nltk.download("punkt", download_dir=NLTK_DIR)
    nltk.download("stopwords", download_dir=NLTK_DIR)
    nltk.download("wordnet", download_dir=NLTK_DIR)
    nltk.download("omw-1.4", download_dir=NLTK_DIR)

if __name__ == "__main__":
    setup_nltk()