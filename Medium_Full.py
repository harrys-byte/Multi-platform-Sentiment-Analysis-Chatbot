import warnings
warnings.filterwarnings('ignore', message="To exit: use 'exit', 'quit', or Ctrl-D.")
warnings.filterwarnings('ignore', message="An exception has occurred, use %tb to see the full traceback.")


from googlesearch import search
from newspaper import Article
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch


nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


def scrape_medium(keywords, maxarcs):
    query = "site:medium.com " + " ".join(keywords)
    
    print("\n🔍 Searching Medium articles for keywords:", ", ".join(keywords))
    urls = list(search(query, stop=maxarcs))    
    print(f"\n🔗 Found {len(urls)} URLs. Scraping content...\n")

    articles_data = []

    for i, url in enumerate(urls, 1):
        try:
            article = Article(url)
            article.download()
            article.parse()

            print(f"✅ [{i}/{len(urls)}] {article.title}")

            articles_data.append({
                'title': article.title,
                'authors': ", ".join(article.authors),
                'publish_date': article.publish_date,
                'text': article.text,
                'url': url
            })

            # Be nice to Medium servers

        except Exception as e:
            print(f"❌ Failed to process {url}: {e}")
            continue

    df = pd.DataFrame(articles_data)
    
    # Deduplicate on URL, Title, and Source
    df = df.drop_duplicates(subset=["url", "title", "text"], keep='first').reset_index(drop=True)

    if df.empty:
        print("\n❗ No valid articles scraped. Please check your keywords or try again later.")
        raise SystemExit

    output_file = 'medium_articles.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Scraped articles saved to '{output_file}'\n")

    return output_file


def preprocess_medium():
    try:
        df = pd.read_csv("medium_articles.csv")
    except FileNotFoundError:
        print("❗ File 'medium_articles.csv' not found.")
        return None

    # Function to clean text
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = " ".join([word for word in text.split() if word not in stop_words])
        return text

    time.sleep(1)
    tqdm.pandas(desc="🔧 Cleaning Titles")
    df["Final_Title"] = df["title"].fillna('').progress_apply(clean_text)
    
    time.sleep(1)
    tqdm.pandas(desc="🔧 Cleaning Texts")
    df["Final_Text"] = df["text"].fillna('').progress_apply(clean_text)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    after = len(df)
    
    time.sleep(0.75)
    print(f"🗑️ Removed {before - after} duplicate entries based on URLs.")

    # Remove very short content
    before = len(df)
    df = df[df["Final_Title"].str.len() >= 5]
    df = df[df["Final_Text"].str.len() >= 5]
    after = len(df)
    
    time.sleep(0.75)
    print(f"🗑️ Removed {before - after} very short or empty rows.")

    # Reset index
    df = df.reset_index(drop=True)

    # Save
    df.to_csv("medium_cleaned.csv", index=False, encoding='utf-8-sig')

    print(f"\n✅ Preprocessing complete. Cleaned rows: {len(df)}. Saved to 'medium_cleaned.csv'.\n")

    return "medium_cleaned.csv"


def run_medium_sentiment_analysis(file,model,show):
    print()
    # Load your cleaned data
    df = pd.read_csv(file)  # Change filename if needed
    
    if (model.lower()=="vader"):
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
        
        if show.lower()=="yes":
            time.sleep(0.5)
            # Ensure all sentiments appear in output, even if 0
            sentiment_order = ['Positive', 'Negative', 'Neutral']
            sentiment_counts = df['Sentiment'].value_counts().reindex(sentiment_order, fill_value=0)
            print(sentiment_counts)
            time.sleep(0.5)

            df['Sentiment'] = pd.Categorical(df['Sentiment'], categories=['Positive', 'Neutral', 'Negative'])

            fig, axs = plt.subplots(figsize=(9,6))

            # VADER Plot
            sns.countplot(data=df, x='Sentiment', palette='Set2', ax=axs)
            axs.set_title('Medium VADER Sentiment Distribution', fontsize=14)
            axs.set_xlabel('Sentiment')
            axs.set_ylabel('Count')
            plt.show()

        df.to_csv("medium_SA_vader.csv", index=False)
        print("✅ Sentiment analysis saved to 'medium_vader_SA.csv'")

        
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
            time.sleep(0.5)
            # Ensure all sentiments appear in output, even if 0
            sentiment_order = ['Positive', 'Negative', 'Neutral']
            sentiment_counts = df['Sentiment_RoBERTa'].value_counts().reindex(sentiment_order, fill_value=0)
            print(sentiment_counts)
            time.sleep(0.5)

            df['Sentiment_RoBERTa'] = pd.Categorical(df['Sentiment_RoBERTa'], categories=['Positive', 'Neutral', 'Negative'])
            fig, axs = plt.subplots(figsize=(9,6))

            # RoBERTa Plot
            sns.countplot(data=df, x='Sentiment_RoBERTa', palette='Set2', ax=axs)
            axs.set_title('Medium RoBERTa Sentiment Distribution', fontsize=14)
            axs.set_xlabel('Sentiment')
            axs.set_ylabel('Count')
            plt.show()
            
        df.to_csv("medium_SA_roberta.csv", index=False)
        print("✅ Sentiment analysis saved to 'medium_SA_roberta.csv'") 
        
    else:
        print("Invalid Option.\n")


