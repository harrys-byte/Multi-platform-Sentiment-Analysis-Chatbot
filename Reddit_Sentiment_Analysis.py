import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import time
import os


def run_rd_sentiment_analysis(file,model,show):
    if model.lower()=="vader":
        df = pd.read_csv(file)

        # Mark rows with missing Text separately
        df['IsTextNaN'] = df['Final_Text'].isna()

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

        # Save the results
        df.to_csv("reddit_full_sentiment.csv", index=False)
        
        df = pd.read_csv("reddit_full_sentiment.csv")
        
        if (show.lower()=="yes"):
            # Show distribution
            print("\n")
            print(df['Sentiment'].value_counts())

            fig, axs = plt.subplots(1, 2, figsize=(16,6))

            sns.countplot(data=df, x='Sentiment', palette='Set2', ax=axs[0])
            axs[0].set_title('Reddit VADER Sentiment Distribution', fontsize=14)
            axs[0].set_xlabel('Sentiment')
            axs[0].set_ylabel('Count')

            sns.countplot(data=df, x='Sentiment', hue='NeutralSource', palette='Set2',ax=axs[1])
            axs[1].set_title('Sentiment Breakdown with Neutral Source', fontsize=14)
            axs[1].set_xlabel('Sentiment')
            axs[1].set_ylabel('Count')
            plt.show()
        
        df.to_csv("reddit_full_vader_sentiment.csv", index=False)
        print("✅ Sentiment data stored in 'reddit_full_vader_sentiment.csv'")
        
    elif model.lower()=="roberta":
        df = pd.read_csv(file)

        # Mark rows with missing Text separately
        df['IsTextNaN'] = df['Final_Text'].isna()

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
        
        if (show.lower()=="yes"):
            print("\n")
            time.sleep(1)
            print(df['Sentiment_RoBERTa'].value_counts())

            fig, axs = plt.subplots(1, 2, figsize=(16,6))

            # RoBERTa Plot
            sns.countplot(data=df, x='Sentiment_RoBERTa', palette='Set2', ax=axs[0])
            axs[0].set_title('Reddit RoBERTa Sentiment Distribution', fontsize=14)
            axs[0].set_xlabel('Sentiment')
            axs[0].set_ylabel('Count')

            sns.countplot(data=df, x='Sentiment_RoBERTa', hue='NeutralSource_RoBERTa', palette='Set2',ax=axs[1])
            axs[1].set_title('Sentiment Breakdown with Neutral Source', fontsize=14)
            axs[1].set_xlabel('Sentiment')
            axs[1].set_ylabel('Count')
            plt.show()

        df.to_csv("reddit_full_roberta_sentiment.csv", index=False)
        print("✅ Sentiment data stored in 'reddit_full_roberta_sentiment.csv'\n")
        
    else:
        print("Invalid Option\n")



