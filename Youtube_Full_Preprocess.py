import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import time

# Make sure stopwords are downloaded
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def preprocess_youtube_data():
    time.sleep(1.2)
    print("🧹 Loading data for preprocessing...")
    df = pd.read_csv("youtube_full.csv")
    
    time.sleep(0.75)
    print(f"✅ Total Comments: {len(df)}")
    time.sleep(0.4)
    
    print("🚀 Cleaning comments...")
    df["Cleaned_Text"] = df['comment'].apply(clean_text)
    df["Final_Text"] = df["Cleaned_Text"].apply(remove_stopwords)
    
    time.sleep(0.75)
    output_file = "youtube_full_cleaned.csv"
    df.to_csv(output_file, index=False)

    print(f"✅ Preprocessing completed. Cleaned data saved to youtube_full_cleaned.csv")
    return output_file



# In[ ]:




