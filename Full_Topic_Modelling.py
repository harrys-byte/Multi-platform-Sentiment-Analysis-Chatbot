import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress only the ConvergenceWarning from sklearn
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_topic_modelling(file,topics):
    # Load your platform-specific file
    df = pd.read_csv(file)  # Change this for each

    # Use correct text column
    texts = df['Final_Text'].dropna().values
    
    df = df.dropna(subset=['Final_Text']).reset_index(drop=True)

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    n_topics = topics # You can adjust this per platform size
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(tfidf)
    
    topic_matrix = nmf.transform(tfidf)
    df['Topic'] = topic_matrix.argmax(axis=1)
    
    time.sleep(1.5)
    def display_topics(model, feature_names, no_top_words=10):
        print("\n")
        for idx, topic in enumerate(model.components_):
            print(f"🔹 Topic {idx}:")
            print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
            print()

    display_topics(nmf, feature_names)
    time.sleep(2)
    
    # Suppose df['Topic'] contains topic numbers (0-5)
    topic_counts = df['Topic'].value_counts().sort_index()

    # Ensure all topics from 0 to 5 exist
    all_topics = pd.Series([0]*topics, index=range(topics))
    topic_counts = all_topics.add(topic_counts, fill_value=0)
    
    plt.figure(figsize=(8,6))
    sns.barplot(x=topic_counts.index, y=topic_counts.values, palette='mako')
    plt.title('Topic Distribution')
    plt.xlabel('Topic')
    plt.ylabel('Count')
    plt.show()

    df.to_csv("topic_modelling.csv")
    print("\n Data saved to topic_modelling.csv\n")

