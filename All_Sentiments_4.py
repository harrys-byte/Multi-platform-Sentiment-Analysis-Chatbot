import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

import sys
import warnings
warnings.filterwarnings('ignore', message="To exit: use 'exit', 'quit', or Ctrl-D.")
warnings.filterwarnings('ignore', message="An exception has occurred, use %tb to see the full traceback.")

from Reddit_Full_Scrap import scrape_reddit
from Reddit_Full_Preprocess import preprocess_data
from Reddit_Sentiment_Analysis import run_rd_sentiment_analysis
from Youtube_Full_Scrap import scrape_youtube
from Youtube_Full_Preprocess import preprocess_youtube_data
from Youtube_Sentiment_Analysis import run_yt_sentiment_analysis
from Full_Topic_Modelling import run_topic_modelling
from Full_Sentiment_vs_Topic import run_sent_topic
from Bing_Full_Scrap import scrape_news
from Bing_Full_Preprocess import preprocess_bing_articles
from Bing_Sentiment_Analysis import run_bing_sentiment_analysis
from Medium_Full import scrape_medium,preprocess_medium,run_medium_sentiment_analysis

def run_all_sent(keys):
    
    keywords = keys
    
    print("You have opted for Sentiment Analaysis across all the plaforms available\n")
    
    print("Provide the necessary data required for analysis\n")
    print("Reddit")
    subreddit = input("Enter subreddit to search (default: india): ")
    
    print("\nYouTube")
    maxvids = int(input("Enter maximum count of videos : "))
    maxcom = int(input("Enter maximum comments per video : "))
    
    print("\nBing and Medium Articles")
    maxarcs = int(input("Enter maximum number of articles : "))
    
    time.sleep(1)
    print("\n⏳ Scraping data from Reddit...")
    scrape_reddit(keywords, subreddit)
    print("\n⏳ Scraping data from YouTube...")
    scrape_youtube(keywords, max_videos=maxvids, max_comments_per_video=maxcom)
    print("\n⏳ Scraping data from Bing...")
    scrape_news(keywords,max_articles=maxarcs)
    print("\n⏳ Scraping data from Medium...")
    scrape_medium(keywords,maxarcs)
    
    
    print("🧹 Preprocessing Reddit data...")
    cleaned_file_rd = preprocess_data()
    print("\n🧹 Preprocessing YouTube data...")
    cleaned_file_yt = preprocess_youtube_data() 
    print("\n🧹 Preprocessing Bing data...")
    cleaned_file_bing = preprocess_bing_articles()
    print("\n🧹 Preprocessing Medium data...")
    cleaned_file_med = preprocess_medium()
    
    
    model = input("Enter model type (vader/roberta): ").lower()
    
    run_rd_sentiment_analysis(file=cleaned_file_rd,model=model,show="no")
    run_yt_sentiment_analysis(file=cleaned_file_yt,model=model,show="no")
    run_bing_sentiment_analysis(file=cleaned_file_bing,model=model,show="no")
    run_medium_sentiment_analysis(file=cleaned_file_med, model=model,show="no")
    
    if (model=="vader"):
        reddit_df = pd.read_csv("reddit_full_vader_sentiment.csv")
        youtube_df = pd.read_csv("youtube_SA_vader.csv")
        bing_df = pd.read_csv("bing_SA_vader.csv")
        medium_df = pd.read_csv("medium_SA_vader.csv")
    else:
        reddit_df = pd.read_csv("reddit_full_roberta_sentiment.csv")
        youtube_df = pd.read_csv("youtube_SA_roberta.csv")
        bing_df = pd.read_csv("bing_SA_roberta.csv")
        medium_df = pd.read_csv("medium_SA_roberta.csv")
    
    reddit_df['Platform'] = 'Reddit'
    youtube_df['Platform'] = 'YouTube'
    bing_df['Platform'] = 'Bing'
    medium_df['Platform'] = 'Medium'
    
    combined_df = pd.concat([reddit_df, youtube_df, medium_df, bing_df], ignore_index=True)
    combined_df.to_csv("all_4_combined_sentiment.csv", index=False)
    
    if model=="vader":
        plt.figure(figsize=(10, 6))
        sns.countplot(data=combined_df, x='Platform', hue='Sentiment', palette='Set2')
        plt.title("VADER Sentiment Distribution Across Platforms")
        plt.ylabel("Number of Mentions")
        plt.xlabel("Platform")
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=combined_df, x='Platform', hue='Sentiment_RoBERTa', palette='Set2')
        plt.title("RoBERTa Sentiment Distribution Across Platforms")
        plt.ylabel("Number of Mentions")
        plt.xlabel("Platform")
        plt.show()
        
        