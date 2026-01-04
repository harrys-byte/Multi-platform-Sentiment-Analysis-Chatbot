import pandas as pd
import re
import nltk
import time
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def preprocess_bing_articles():
    df = pd.read_csv("bing_articles_with_text.csv")
    
    df['Final_Text'] = df[['title', 'text']].fillna('').agg(' '.join, axis=1)

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text.strip()

    df['Final_Text'] = df['Final_Text'].apply(clean_text)
    
    stop_words = set(stopwords.words("english"))

    def remove_stopwords(text):
        return " ".join([word for word in text.split() if word not in stop_words])

    df["Final_Text"] = df["Final_Text"].apply(remove_stopwords)
   
    """df = df.drop_duplicates(subset='title')
    df = df.drop_duplicates(subset='url')
    df = df.reset_index(drop=True)
    
    df = df.drop_duplicates(subset=["url",'title','text']).reset_index(drop=True)"""
    
    time.sleep(1)

    output_file = "bing_articles_cleaned.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ Cleaned articles saved to {output_file}")
    return output_file




