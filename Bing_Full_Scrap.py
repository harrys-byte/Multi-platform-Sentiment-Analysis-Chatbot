import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', message="To exit: use 'exit', 'quit', or Ctrl-D.")
warnings.filterwarnings('ignore', message="An exception has occurred, use %tb to see the full traceback.")

def extract_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Try to extract main article text heuristically
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text().strip() for p in paragraphs])
        return text.strip()
    except Exception as e:
        return ""

def enrich_bing_with_text(csv_file):
    df = pd.read_csv(csv_file)

    print("\n📄 Collecting full text from each URL...")
    texts = []

    for url in tqdm(df['url'], desc="🔍 Scraping article content"):
        if not url.startswith('http'):
            texts.append("")
            continue
        article_text = extract_article_text(url)
        texts.append(article_text)

    df['text'] = texts
    df = df[df['text'].str.strip() != ''].reset_index(drop=True)
    output_file = "bing_articles_with_text.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ Saved with full texts to '{output_file}'\n")

    return output_file


def scrape_news(keywords, max_articles):
    query = "+".join([kw.strip() for kw in keywords])
    base_url = f"https://www.bing.com/news/search?q={query}&FORM=HDRSC6"
    headers = {"User-Agent": "Mozilla/5.0"}

    print(f"🔍 Searching Bing News for keywords: {', '.join(keywords)}")
    print("⏳ Fetching results...\n")
    time.sleep(1)

    articles = []
    unique_titles = set()
    page = 0
    collected = 0
    max_pages = 30  # Limit to avoid infinite scraping

    print("🔎 Deduplicating articles and validating content...")
    with tqdm(total=max_articles, desc="🔍 Collecting unique articles with valid text") as pbar:
        while collected < max_articles and page < max_pages:
            url = f"{base_url}&first={page * 10}"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            news_cards = soup.select('.news-card, .t_s, .title')
            if not news_cards:
                break  # Stop if no more pages found

            for card in news_cards:
                link = card.find('a')
                title = card.get_text(strip=True)
                href = link.get('href') if link else None

                if href and title and title not in unique_titles:
                    text = extract_article_text(href)
                    if text and len(text.strip()) > 50:  # Validity check
                        articles.append({
                            'title': title,
                            'url': href,
                            'text': text.strip()
                        })
                        unique_titles.add(title)
                        collected += 1
                        pbar.update(1)

                if collected >= max_articles:
                    break

            page += 1
            time.sleep(0.5)

    if not articles:
        print("\n⚠️ No valid articles were found for these keywords.")
        raise SystemExit
    
    if collected < max_articles:
        print(f"\n⚠️ Only {collected} valid articles were found for these keywords.")

    df = pd.DataFrame(articles)

    output_file = "bing_articles_with_text.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("\n📄 Sample of Collected Articles:\n")
    for idx, row in df.iterrows():
        short_title = row['title'] if len(row['title']) <= 60 else row['title'][:57].rstrip() + "..."
        print(f"📌 [{idx + 1}/{len(df)}] {short_title}")

    print(f"\n✅ Saved with valid text to '{output_file}'\n")

    enrich_bing_with_text(output_file)
    
    