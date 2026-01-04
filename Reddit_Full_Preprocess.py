#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import praw
from datetime import datetime
import os
import time
from prawcore.exceptions import NotFound
import re
from nltk.corpus import stopwords
import nltk


# In[4]:


def preprocess_data():

    df = pd.read_csv("reddit_full.csv")

    df = df[df['Author'].notnull()]

    # ✅ Define bot detection function
    def is_likely_bot(username):
        try:
            user = reddit.redditor(str(username))
            user_info = user._fetch()

            comment_karma = user.comment_karma
            link_karma = user.link_karma
            account_age_days = (time.time() - user.created_utc) / (60 * 60 * 24)

            if comment_karma < 10 or link_karma < 10:
                return True
            if account_age_days < 30:
                return True
            return False

        except NotFound:
            return True  # Suspended/deleted user = suspicious
        except:
            return False  # Any other error, assume not a bot

    # ✅ Apply detection and filter
    time.sleep(1)
    print("🔎 Checking users... (this may take a few minutes)")
    df["is_bot"] = df["Author"].apply(is_likely_bot)
    df = df[df["is_bot"] == False].drop(columns=["is_bot"])
    
    def clean_text(text):
        text = str(text)                          
        text = text.lower()                       
        text = re.sub(r"http\S+", "", text)       
        text = re.sub(r"@\w+", "", text)          
        text = re.sub(r"[^a-z\s]", "", text)      
        text = re.sub(r"\s+", " ", text).strip()  
        return text


    df["Cleaned_Title"] = df["Title"].apply(clean_text)
    df["Cleaned_Text"] = df["Text"].apply(clean_text)
    
    
    #nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    def remove_stopwords(text):
        return " ".join([word for word in text.split() if word not in stop_words])

    df["NoStop_Title"] = df["Cleaned_Title"].apply(remove_stopwords)
    df["Final_Text"] = df["Cleaned_Text"].apply(remove_stopwords)
    
    time.sleep(1.2)
    # ✅ Save filtered data
    df.to_csv("cleaned_reddit_full.csv", index=False)
    print("✅ Bot filtering complete. Saved to cleaned_reddit_full.csv")
    time.sleep(1)
    
    return "cleaned_reddit_full.csv"


