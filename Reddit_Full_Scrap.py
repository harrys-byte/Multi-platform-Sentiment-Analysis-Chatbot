import praw
import pandas as pd
from datetime import datetime
import os
import time
from prawcore.exceptions import NotFound
import warnings
warnings.filterwarnings('ignore', message="To exit: use 'exit', 'quit', or Ctrl-D.")
warnings.filterwarnings('ignore', message="An exception has occurred, use %tb to see the full traceback.")


# In[2]:


def scrape_reddit(keywords, subreddit):
    # Reddit API auth
    reddit = praw.Reddit(
        client_id="AFu4I0OlbCB7y5GUFOfSZg",
        client_secret="gy1eQJ0G_luNXwHIApY75Wmpn2-eDw",
        user_agent="policySA by /u/Confident-Wasabi-977"
    )

    # ✅ User Inputs
    subreddit_name = subreddit.lower()
    query = keywords

    # ✅ Time Filters
    time_filters = ['day', 'week', 'month', 'year', 'all']

    all_posts = []
    
    total = 3
    for i in range(total):
        percent = int((i + 1) / total * 100)
        bar = '█' * (percent // 10) + '-' * (10 - percent // 10)
        print(f'\r🔄 Parsing {i + 1}/{total} |{bar}| {percent}%', end='')

        for time_filter in time_filters:
            #print(f"⏳ Fetching posts with time filter: {time_filter}...")

            submissions = reddit.subreddit(subreddit_name).search(query, sort='relevance', time_filter=time_filter, limit=100)

            for submission in submissions:
                all_posts.append({
                    "Subreddit": subreddit_name,
                    "TimeFilter": time_filter,
                    "Title": submission.title,
                    "Text": submission.selftext,
                    "Upvotes": submission.score,
                    "Comments": submission.num_comments,
                    "Date": datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d'),
                    "Author": str(submission.author),
                    "URL": f"https://www.reddit.com{submission.permalink}"
                })
    
    if len(all_posts)==0:
        print("\nNo valid data found")
        raise SystemExit
        
    print("\n✅ Parsing complete!")
    time.sleep(1)
    print("Data scrapping successfully completed")
    time.sleep(1)

    # Save to CSV
    df = pd.DataFrame(all_posts)

    # Remove duplicates based on URL (or you can use 'Title')
    df_unique = df.drop_duplicates(subset=['URL']).reset_index(drop=True)

    print(f"\nTotal unique posts found: {len(df_unique)}")

    #print(f"\n✅ Scraped {len(df)} posts for query '{query}' in subreddit r/{subreddit_name}")
    df_unique.to_csv("reddit_full.csv", index=False)
    time.sleep(1.25)
    print("✅ Reddit data saved successfully in 'reddit_full.csv'\n")
    time.sleep(1)
    return df_unique
