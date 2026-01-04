get_ipython().system('pip install google-api-python-client')

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import time

import warnings
warnings.filterwarnings('ignore', message="To exit: use 'exit', 'quit', or Ctrl-D.")
warnings.filterwarnings('ignore', message="An exception has occurred, use %tb to see the full traceback.")

# Step 1: Set up YouTube API
api_key = "AIzaSyDnxGr_DNZVKR_5Eko58oba2VNITbdMSOk"  # 🔑 Replace with your real API key
youtube = build("youtube", "v3", developerKey=api_key)

# Step 2: Search YouTube Videos by Keyword
def search_youtube_videos(keywords, max_results, maxcomms):
    video_ids = set()
    
    print(f"🛠 Starting search with keywords: {keywords} and max_results: {max_results}")
    time.sleep(1)
    
    for keyword in keywords:
        print(f"🔑 Searching for keyword: {keyword}")
        
        if len(video_ids) >= max_results:
            break

        remaining_slots = max_results - len(video_ids)

        request = youtube.search().list(
            q=keyword.strip(),
            part="id",
            type="video",
            maxResults=min(remaining_slots,20)  # Never ask for more than needed
        )
        response = request.execute()

        for item in response.get("items", []):
            video_id = item["id"]["videoId"]
            video_ids.add(video_id)

            if len(video_ids) >= max_results:
                break
                
    time.sleep(1)
    #print(f"🔎 Collected {len(video_ids)} unique videos: {video_ids}")
    return list(video_ids)

# Step 3: Get Comments from Each Video
def get_video_comments(video_id, max_comments):
    comments = []

    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            textFormat="plainText"
        ).execute()

        while response and len(comments) < max_comments:
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            if "nextPageToken" in response:
                response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response["nextPageToken"],
                    maxResults=max_comments,
                    textFormat="plainText"
                ).execute()
            else:
                break
                
    except HttpError as e:
        print(f"⚠️ Skipping video {video_id} due to API error: {e}")
        return []

    except Exception as e:
        print(f"🚫 Skipping video {video_id} due to unexpected error: {e}")
        return []

    return comments


# Step 4: Run Full Pipeline
def scrape_youtube(keywords, max_videos, max_comments_per_video):
    collected_videos = []
    all_comments = []

    video_ids = search_youtube_videos(keywords, max_results=max_videos*2, maxcomms=max_comments_per_video)  # Fetch more than needed, since we'll filter
    
    if len(video_ids)==0:
        print(f"\n🔍 Found {len(video_ids)} potential videos for keywords: {keywords}")
        raise SystemExit
        
    print(f"\n🔍 Found {len(video_ids)} potential videos for keywords: {keywords}")
    time.sleep(1)
    
    for vid in video_ids:
        if len(collected_videos) >= max_videos:
            break

        print(f"⏳ Fetching comments for video ID: {vid}")
        comments = get_video_comments(vid, max_comments=max_comments_per_video)
        print(f"✅ Retrieved {len(comments)} comments")

        if len(comments) >= 10:
            for c in comments:
                all_comments.append({'video_id': vid, 'comment': c})
            collected_videos.append(vid)
            print(f"✅ Video ID {vid} accepted (>= 10 comments)")
        else:
            print(f"⚠️ Video ID {vid} skipped (less than 10 comments)")

        time.sleep(1)

    df = pd.DataFrame(all_comments)
    time.sleep(1)

    output_file = "youtube_full.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ All valid comments saved to {output_file}")
    print(f"✅ Total valid videos collected: {len(collected_videos)} / {max_videos}\n")
    return output_file

