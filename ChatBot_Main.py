import sys
import time
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
from All_Sentiments_4 import run_all_sent

def msg():
    print("\nHold on! We’re collecting the content and tidying it up for you. This might take a little while—thank you for your patience!")
    
def main():
    while True:
        text =  "📢 Welcome to Policy Insights Chatbot!"
        centered_text = text.center(120, ' ')
        print(centered_text)
        
        print("\nAvailable Platforms for Analysis :")
        print("1. Reddit\n2. YouTube\n3. Bing Articles\n4. Medium\n5. Sentiment Distribution across platforms\n6. Exit")
        op = int(input("\nEnter your choice : "))
        print()
        if (op==6):
            print("👋 Exiting the chatbot. Thank you!")
            return
        keys = input("Enter keywords (comma separated): ").split(",")

        if (op==1):
            def reddit(keys):
                print("\nYou have opted for REDDIT Data Analysis")

                keywords = keys
                subreddit = input("Enter subreddit to search (default: india): ")
                msg()
                
                # Step 2: Scrape data
                time.sleep(1)
                print("\n⏳ Scraping data...")
                scrape_reddit(keywords, subreddit)

                # Step 3: Preprocess data
                print("🧹 Preprocessing data...")
                cleaned_file = preprocess_data() #because I automatically read the file in Reddit_Full_Preprocess.py

                # Step 4: Main Loop
                while True:
                    print("\nWhat would you like to do?")
                    print("1️⃣ Sentiment Analysis")
                    print("2️⃣ Topic Modeling")
                    print("3️⃣ Sentiment vs Topics")
                    print("4️⃣ Return to Home")
                    print("5️⃣ Exit")
                    choice = input("Enter your choice: ")

                    if choice == "1":
                        model = input("Choose Sentiment Model (vader/roberta): ").lower()
                        run_rd_sentiment_analysis(file=cleaned_file,model=model,show="yes")

                    elif choice == "2":
                        topics = int(input("Enter number of topics : "))
                        run_topic_modelling(file=cleaned_file,topics=topics)

                    elif choice == "3":
                        # Optional: You can implement joint graphs if needed
                        model = input("Enter model type : ").lower()
                        topics = int(input("Enter number of topics : "))
                        run_sent_topic(file=cleaned_file,model=model,topics=topics)

                    elif choice == "4":
                        break

                    elif choice == "5":
                        print("\n👋 Exiting the chatbot. Thank you!")
                        raise SystemExit
                    else:
                        print("---❗ Invalid choice. Please try again ---")
            reddit(keys=keys)
        

        elif (op==2):
            def yt(keys):
                print("\nYou have opted for YOUTUBE Data Analysis")

                keywords = keys
                maxvids = int(input("Enter maximum count of videos : "))
                maxcom = int(input("Enter maximum comments per video : "))
                msg()

                # Step 2: Scrape YouTube data
                time.sleep(1)
                print("\n⏳ Scraping YouTube data...")
                scrape_youtube(keywords, max_videos=maxvids, max_comments_per_video=maxcom)  # 👉 You need to define this function separately

                # Step 3: Preprocess data
                print("🧹 Preprocessing YouTube data...")
                cleaned_file = preprocess_youtube_data() 

                # Step 4: Main Loop
                while True:
                    print("\nWhat would you like to do?")
                    print("1️⃣ Sentiment Analysis")
                    print("2️⃣ Topic Modeling")
                    print("3️⃣ Sentiment vs Topics")
                    print("4️⃣ Return to Home")
                    print("5️⃣ Exit")
                    choice = input("Enter your choice: ")

                    if choice == "1":
                        model = input("Choose Sentiment Model (vader/roberta): ").lower()
                        run_yt_sentiment_analysis(file=cleaned_file,model=model,show="yes")

                    elif choice == "2":
                        topics = int(input("Enter number of topics : "))
                        run_topic_modelling(file=cleaned_file,topics=topics)

                    elif choice == "3":
                        model = input("Enter model type : ").lower()
                        topics = int(input("Enter number of topics : "))
                        run_sent_topic(file=cleaned_file,model=model,topics=topics)

                    elif choice == "4":
                        break

                    elif choice == "5":
                        print("\n👋 Exiting the chatbot. Thank you!")
                        raise SystemExit

                    else:
                        print("---❗ Invalid choice. Please try again ---")
            yt(keys=keys)
        

        elif (op==3):
            def ba(keys):
                print("\nYou have opted for BING ARTICLES Data Analysis")
                
                time.sleep(0.5)

                # Step 1: Get policy name & keywords
                keywords = keys
                maxarcs = int(input("Enter maximum number of articles : "))
                msg()

                # Step 2: Scrape data
                time.sleep(1)
                print("\n⏳ Scraping data...")
                print()
                scrape_news(keywords,max_articles=maxarcs)

                # Step 3: Preprocess data
                print("🧹 Preprocessing data...")
                cleaned_file = preprocess_bing_articles() #because I automatically read the file in Reddit_Full_Preprocess.py

                # Step 4: Main Loop
                while True:
                    print("\nWhat would you like to do?")
                    print("1️⃣ Sentiment Analysis")
                    print("2️⃣ Topic Modeling")
                    print("3️⃣ Sentiment vs Topics")
                    print("4️⃣ Return to Home")
                    print("5️⃣ Exit")
                    choice = input("Enter your choice: ")

                    if choice == "1":
                        model = input("Choose Sentiment Model (vader/roberta): ").lower()
                        run_bing_sentiment_analysis(file=cleaned_file,model=model,show="yes")

                    elif choice == "2":
                        topics = int(input("Enter number of topics : "))
                        run_topic_modelling(file=cleaned_file,topics=topics)

                    elif choice == "3":
                        model = input("Enter model type : ").lower()
                        topics = int(input("Enter number of topics : "))
                        run_sent_topic(file=cleaned_file,model=model,topics=topics)

                    elif choice == "4":
                        break

                    elif choice == "5":
                        print("\n👋 Exiting the chatbot. Thank you!")
                        raise SystemExit
                        

                    else:
                        print("---❗ Invalid choice. Please try again ---")
            ba(keys=keys)
        

        elif (op==4):
            def medium(keys):
                print("\nYou have opted for MEDIUM Data Analysis")

                time.sleep(0.5)
                # Step 1: Get policy name & keywords
                keywords = keys
                maxarcs = int(input("Enter maximum number of articles : "))
                msg()  # existing function for user-friendly messaging

                # Step 2: Scrape data
                time.sleep(1)
                print("\n⏳ Scraping data from Medium...")
                print()
                scrape_medium(keywords,maxarcs) 
                
                # Step 3: Preprocess data
                print("🧹 Preprocessing data...")
                cleaned_file = preprocess_medium() 

                # Step 4: Main Loop for further actions
                while True:
                    print("\nWhat would you like to do?")
                    print("1️⃣ Sentiment Analysis")
                    print("2️⃣ Topic Modeling")
                    print("3️⃣ Sentiment vs Topics")
                    print("4️⃣ Return to Home")
                    print("5️⃣ Exit")
                    choice = input("Enter your choice: ")

                    if choice == "1":
                        model = input("Choose Sentiment Model (vader/roberta): ").lower()
                        run_medium_sentiment_analysis(file=cleaned_file, model=model,show="yes")

                    elif choice == "2":
                        topics = int(input("Enter number of topics : "))
                        run_topic_modelling(file=cleaned_file, topics=topics)

                    elif choice == "3":
                        model = input("Enter model type : ").lower()
                        topics = int(input("Enter number of topics : "))
                        run_sent_topic(file=cleaned_file, model=model, topics=topics)

                    elif choice == "4":
                        main()  # Back to main menu

                    elif choice == "5":
                        print("\n👋 Exiting the chatbot. Thank you!")
                        raise SystemExit

                    else:
                        print("---❗ Invalid choice. Please try again ---")

            medium(keys=keys)

        elif (op==5):
            run_all_sent(keys)
        
        else:
            print(" --Invalid Option--")

