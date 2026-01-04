import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import time

# In[3]:


def run_yt_sentiment_analysis(file,model,show):
    print()
    # Load your cleaned YouTube data
    df = pd.read_csv("youtube_full_cleaned.csv")  # Change filename if needed
    
    if (model.lower()=="vader"):
        sia = SentimentIntensityAnalyzer()
        
        # Mark rows with missing Text separately
        df['IsTextNaN'] = df['Final_Text'].isna()

        def analyze_sentiment(text):
            if pd.isna(text):
                return pd.Series([0, "Neutral"])
            score = sia.polarity_scores(str(text))['compound']

            if score >= 0.48:
                return pd.Series([score, "Positive"])
            elif score <= -0.48:
                return pd.Series([score, "Negative"])
            else:
                return pd.Series([score, "Neutral"])

        df[['SentimentScore', 'Sentiment']] = df['Final_Text'].apply(analyze_sentiment)
        df['NeutralSource'] = df.apply(lambda x: "NaN Text" if x['IsTextNaN'] else "Real Text", axis=1)
        
        if show.lower()=="yes":
            time.sleep(1)
            print(df['Sentiment'].value_counts())
            time.sleep(1)

            fig, axs = plt.subplots(1, 2, figsize=(16,6))

            sns.countplot(data=df, x='Sentiment', palette='Set2', ax=axs[0])
            axs[0].set_title('YouTube VADER Sentiment Distribution', fontsize=14)
            axs[0].set_xlabel('Sentiment')
            axs[0].set_ylabel('Count')

            sns.countplot(data=df, x='Sentiment', hue='NeutralSource', palette='Set2',ax=axs[1])
            axs[1].set_title('Sentiment Breakdown with Neutral Source', fontsize=14)
            axs[1].set_xlabel('Sentiment')
            axs[1].set_ylabel('Count')
            plt.show()

        df.to_csv("youtube_SA_vader.csv", index=False)
        print("✅ Sentiment analysis saved to 'youtube_SA_vader.csv'")

        
    elif (model.lower()=="roberta"):
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
            print(df['Sentiment_RoBERTa'].value_counts())
            time.sleep(0.5)

            fig, axs = plt.subplots(1, 2, figsize=(16,6))

            # RoBERTa Plot
            sns.countplot(data=df, x='Sentiment_RoBERTa', palette='Set2', ax=axs[0])
            axs[0].set_title('YouTube RoBERTa Sentiment Distribution', fontsize=14)
            axs[0].set_xlabel('Sentiment')
            axs[0].set_ylabel('Count')

            sns.countplot(data=df, x='Sentiment_RoBERTa', hue='NeutralSource_RoBERTa', palette='Set2',ax=axs[1])
            axs[1].set_title('Sentiment Breakdown with Neutral Source', fontsize=14)
            axs[1].set_xlabel('Sentiment')
            axs[1].set_ylabel('Count')
            plt.show()
        
        df.to_csv("youtube_SA_roberta.csv", index=False)
        print("✅ Sentiment analysis saved to 'youtube_SA_roberta.csv'")

    else:
        print("Invalid Option.\n")


# In[ ]:




