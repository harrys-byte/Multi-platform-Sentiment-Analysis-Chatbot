import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import time

def run_bing_sentiment_analysis(file, model,show):
    df = pd.read_csv(file)

    if model == "vader":
        sia = SentimentIntensityAnalyzer()
        
        # Mark rows with missing Text separately
        df['IsTextNaN'] = df['Final_Text'].isna()

        def analyze_sentiment(text):
            if pd.isna(text):
                return pd.Series([0, "Neutral"])
            score = sia.polarity_scores(str(text))['compound']

            if score >= 0.5:
                return pd.Series([score, "Positive"])
            elif score <= -0.5:
                return pd.Series([score, "Negative"])
            else:
                return pd.Series([score, "Neutral"])

        df[['SentimentScore', 'Sentiment']] = df['Final_Text'].apply(analyze_sentiment)
        df['NeutralSource'] = df.apply(lambda x: "NaN Text" if x['IsTextNaN'] else "Real Text", axis=1)
        
        if (show.lower()=="yes"):
            time.sleep(1)
            print("\n")
            sentiment_counts = df['Sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
            print(sentiment_counts)
            time.sleep(0.5)

            df['Sentiment'] = pd.Categorical(df['Sentiment'], categories=['Positive', 'Neutral', 'Negative'])

            fig, axs = plt.subplots(1, 2, figsize=(16,6))

            sns.countplot(data=df, x='Sentiment', palette='Set2', ax=axs[0])
            axs[0].set_title('Bing News VADER Sentiment Distribution', fontsize=14)
            axs[0].set_xlabel('Sentiment')
            axs[0].set_ylabel('Count')

            sns.countplot(data=df, x='Sentiment', hue='NeutralSource', palette='Set2',ax=axs[1])
            axs[1].set_title('Sentiment Breakdown with Neutral Source', fontsize=14)
            axs[1].set_xlabel('Sentiment')
            axs[1].set_ylabel('Count')
            plt.show()
        
        df.to_csv("bing_SA_vader.csv", index=False)
        print("✅ Sentiment analysis saved to 'bing_SA_vader.csv'")
        
        return 'bing_vader_SA.csv'
        
    elif model == "roberta":
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        df['IsTextNaN'] = df['Final_Text'].isna()
        
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
        
        if show.lower()=="yes":
            time.sleep(1)
            print("\n")
            sentiment_counts = df['Sentiment_RoBERTa'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
            print(sentiment_counts)
            time.sleep(0.5)

            df['Sentiment_RoBERTa'] = pd.Categorical(df['Sentiment_RoBERTa'], categories=['Positive', 'Neutral', 'Negative'])

            fig, axs = plt.subplots(1, 2, figsize=(16,6))

            # RoBERTa Plot
            sns.countplot(data=df, x='Sentiment_RoBERTa', palette='Set2', ax=axs[0])
            axs[0].set_title('Bing RoBERTa Sentiment Distribution', fontsize=14)
            axs[0].set_xlabel('Sentiment')
            axs[0].set_ylabel('Count')

            sns.countplot(data=df, x='Sentiment_RoBERTa', hue='NeutralSource_RoBERTa', palette='Set2',ax=axs[1])
            axs[1].set_title('Sentiment Breakdown with Neutral Source', fontsize=14)
            axs[1].set_xlabel('Sentiment')
            axs[1].set_ylabel('Count')
            plt.show()

        output_file = "bing_SA_roberta.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ Sentiment analysis completed and saved to {output_file}")

    else:
        print("Invalid Option.\n")