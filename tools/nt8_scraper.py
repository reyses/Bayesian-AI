import requests
from bs4 import BeautifulSoup
import markdownify
import os
import re
import time
from urllib.parse import urlparse, urljoin

def fetch_and_parse(url, output_dir):
    print(f"Fetching: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return False

    soup = BeautifulSoup(response.text, 'html.parser')

    main_content = soup.find('article')
    if not main_content:
        main_content = soup.find('div', class_='blog-post__content') or soup.find('main')
    
    if not main_content:
        print("WARNING: Could not find strict wrapper. Dumping body.")
        main_content = soup.body
        if main_content:
            for tag in main_content(['nav', 'footer', 'header', 'aside', 'script', 'style']):
                tag.decompose()

    if not main_content:
        return False

    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else "Untitled Article"
    
    date_tag = soup.find('time')
    date_str = date_tag.get_text(strip=True) if date_tag else ""

    md_content = markdownify.markdownify(str(main_content), heading_style="ATX")
    md_content = re.sub(r'\n{3,}', '\n\n', md_content).strip()

    if len(md_content) < 50:
        print(f"Skipping {url}, content too short (likely index or error page).")
        return False

    final_doc = f"# {title}\n"
    if date_str:
        final_doc += f"**Date:** {date_str}\n"
    final_doc += f"**Source:** {url}\n\n"
    final_doc += "---\n\n"
    final_doc += md_content

    path = urlparse(url).path
    slug = [p for p in path.split('/') if p][-1]
    if not slug:
        slug = "index"
    
    filename = f"{slug}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_doc)
    
    print(f"Saved: {filepath}")
    return True

def crawl_index(base_index_url, output_dir, max_articles=50):
    print(f"\n--- Crawling Index: {base_index_url} ---")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    articles_processed = 0
    page = 1
    
    while articles_processed < max_articles:
        if page == 1:
            index_url = base_index_url
        else:
            index_url = f"{base_index_url.rstrip('/')}/?page={page}"
            
        print(f"\nChecking index page: {index_url}")
        
        try:
            response = requests.get(index_url, headers=headers)
            if response.status_code == 404:
                print("Hit 404 on index pagination. Stopping.")
                break
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch index page {page}: {e}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        article_links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Target blog posts but exclude sub-indices
            if href.startswith('/futures/blogs/') and href != '/futures/blogs/':
                if '/page/' not in href and '/author/' not in href and '/category/' not in href:
                    full_url = urljoin("https://ninjatrader.com", href)
                    article_links.add(full_url)
                
        if not article_links:
            print("No articles found on this index page. Stopping.")
            break
            
        print(f"Found {len(article_links)} potential articles on page {page}.")
        
        new_articles_on_page = 0
        for url in article_links:
            if articles_processed >= max_articles:
                break
                
            path = urlparse(url).path
            slug = [p for p in path.split('/') if p][-1]
            if not slug:
                slug = "index"
            filepath = os.path.join(output_dir, f"{slug}.md")
            
            if os.path.exists(filepath):
                print(f"Already have {slug}, skipping.")
                continue
                
            success = fetch_and_parse(url, output_dir)
            if success:
                articles_processed += 1
                new_articles_on_page += 1
            time.sleep(0.1) # Faster rate limit
            
        if new_articles_on_page == 0:
            print(f"All articles on page {page} were already cataloged.")
            
        page += 1
        
    print(f"\nCrawling complete. Fetched {articles_processed} new articles.")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "research", "nt8_catalog")
    os.makedirs(out_dir, exist_ok=True)
    
    # Crawl the entire index, traversing pagination to find EVERYTHING
    crawl_index("https://ninjatrader.com/futures/blogs/", out_dir, max_articles=10000)
