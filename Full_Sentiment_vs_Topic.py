import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import os
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress only the ConvergenceWarning from sklearn
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_sent_topic(file,model,topics):
    df = pd.read_csv(file)
        
    # Use correct text column
    texts = df['Final_Text'].dropna().values
    
    df = df.dropna(subset=['Final_Text']).reset_index(drop=True)
    df['IsTextNaN'] = df['Final_Text'].isna()

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    n_topics = topics # You can adjust this per platform size
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(tfidf)
    
    topic_matrix = nmf.transform(tfidf)
    df['Topic'] = topic_matrix.argmax(axis=1)
    
    time.sleep(1.2)
    def display_topics(model, feature_names, no_top_words=10):
        print("\n")
        for idx, topic in enumerate(model.components_):
            print(f"🔹 Topic {idx}:")
            print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
            print()

    display_topics(nmf, feature_names)
    # Mark rows with missing Text separately
    time.sleep(1.25)

    if (model=="vader"):
        # Initialize VADER
        analyzer = SentimentIntensityAnalyzer()

        # Then assign sentiment
        def analyze_sentiment(text):
            if pd.isna(text):
                return pd.Series([0, "Neutral"])
            score = analyzer.polarity_scores(text)['compound']

            if score > 0.5:
                return pd.Series([score, "Positive"])
            elif score <= -0.5:
                return pd.Series([score, "Negative"])
            else:
                return pd.Series([score, "Neutral"])


        df[['SentimentScore', 'Sentiment']] = df['Final_Text'].apply(analyze_sentiment)
        df['NeutralSource'] = df.apply(lambda x: "NaN Text" if x['IsTextNaN'] else "Real Text", axis=1)
        
        pivot_table = df.pivot_table(index='Topic', columns='Sentiment', aggfunc='size', fill_value=0)
        pivot_table = pivot_table.reindex(columns=['Negative', 'Neutral', 'Positive'], fill_value=0)
        pivot_table = pivot_table.reindex(range(topics), fill_value=0)
        
        time.sleep(1)
        
        print(df['Sentiment'].value_counts())
        print("\n")
        print(pivot_table)

        plt.figure(figsize=(8,6))
        sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt='d')
        plt.title('Sentiment vs Topic Heatmap (VADER)')
        plt.xlabel('Sentiment')
        plt.ylabel('Dominant Topic')
        plt.show()
        
    elif (model=="roberta"):
        #roberta analysis
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        def analyze_roberta_sentiment(text):
            if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
                return pd.Series([0, "Neutral"])

            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                output = model(**tokens)

            scores = softmax(output.logits[0].numpy())
            score = scores[2] - scores[0]  # Positive minus Negative (rough sentiment score)

            labels = ['Negative', 'Neutral', 'Positive']
            sentiment = labels[scores.argmax()]

            return pd.Series([score, sentiment])

        df[['SentimentScore_RoBERTa', 'Sentiment_RoBERTa']] = df['Final_Text'].apply(analyze_roberta_sentiment)
        df['NeutralSource_RoBERTa'] = df.apply(lambda x: "NaN Text" if x['IsTextNaN'] else "Real Text", axis=1)
        
        pivot_table = df.pivot_table(index='Topic', columns='Sentiment_RoBERTa', aggfunc='size', fill_value=0)
        pivot_table = pivot_table.reindex(columns=['Negative', 'Neutral', 'Positive'], fill_value=0)
        
        time.sleep(1)
        print(pivot_table)

        plt.figure(figsize=(8,6))
        sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt='d')
        plt.title('Sentiment vs Topic Heatmap (RoBERTa)')
        plt.xlabel('Sentiment')
        plt.ylabel('Dominant Topic')
        plt.show()
        
    else:
        print("\n--Invalid Option--\n")