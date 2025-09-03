# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
from urllib.parse import urlparse, urljoin
import xmltodict
import re
import time
import os
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import wikipediaapi
import praw
from newspaper import Article

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Content Gap Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4;}
    .subheader {font-size: 1.5rem; color: #ff7f0e;}
    .highlight {background-color: #f0f2f6; padding: 15px; border-radius: 5px;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 15px;}
    .stProgress > div > div > div > div {background-image: linear-gradient(to right, #1f77b4, #ff7f0e);}
</style>
""", unsafe_allow_html=True)

class ContentGapAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize Wikipedia API
        self.wiki_wiki = wikipediaapi.Wikipedia(
            user_agent='ContentGapAnalyzer/1.0 (https://example.com; contact@example.com)',
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        
        # Initialize Reddit API (optional)
        self.reddit = None
        try:
            # You need to set these environment variables
            reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
            reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "ContentGapAnalyzer/1.0 by YourUsername")
            
            if reddit_client_id and reddit_client_secret:
                self.reddit = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent
                )
        except:
            pass
    
    def find_sitemap(self, domain):
        """Try to find the sitemap for a domain"""
        sitemap_urls = [
            f"https://{domain}/sitemap.xml",
            f"https://{domain}/sitemap_index.xml",
            f"https://{domain}/sitemap",
            f"https://{domain}/wp-sitemap.xml",
        ]
        
        for url in sitemap_urls:
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    return url
            except:
                continue
        
        # Try to find sitemap in robots.txt
        try:
            robots_url = f"https://{domain}/robots.txt"
            response = requests.get(robots_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                for line in response.text.split('\n'):
                    if line.lower().startswith('sitemap:'):
                        return line.split(':', 1)[1].strip()
        except:
            pass
        
        return None
    
    def parse_sitemap(self, sitemap_url):
        """Parse a sitemap and return all URLs"""
        try:
            response = requests.get(sitemap_url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return []
            
            # Parse XML sitemap
            data = xmltodict.parse(response.content)
            urls = []
            
            # Handle different sitemap formats
            if 'urlset' in data and 'url' in data['urlset']:
                for url_data in data['urlset']['url']:
                    if 'loc' in url_data:
                        urls.append(url_data['loc'])
            elif 'sitemapindex' in data and 'sitemap' in data['sitemapindex']:
                for sitemap_data in data['sitemapindex']['sitemap']:
                    if 'loc' in sitemap_data:
                        # Recursively parse nested sitemaps
                        nested_urls = self.parse_sitemap(sitemap_data['loc'])
                        urls.extend(nested_urls)
            
            return urls
        except:
            return []
    
    def extract_content_from_url(self, url):
        """Extract text content from a URL"""
        try:
            # Use newspaper3k for article extraction
            article = Article(url)
            article.download()
            article.parse()
            
            # If newspaper3k fails, fall back to BeautifulSoup
            if not article.text.strip():
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code != 200:
                    return ""
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    element.decompose()
                
                # Extract text from main content areas
                text = ""
                content_selectors = [
                    'main', 'article', '.content', '#content', '.post', 
                    '.article', '.main-content', '[role="main"]'
                ]
                
                for selector in content_selectors:
                    elements = soup.select(selector)
                    for element in elements:
                        text += element.get_text() + "\n"
                
                # If no specific content found, use body
                if not text.strip():
                    text = soup.body.get_text() if soup.body else ""
                
                # Clean up text
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            return article.text
        except:
            return ""
    
    def extract_topics_from_text(self, text, num_topics=5):
        """Extract topics from text using TF-IDF and LDA"""
        if not text.strip():
            return []
        
        # Tokenize and clean text
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        if len(words) < 10:  # Not enough content
            return []
        
        # Create document-term matrix
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        dtm = vectorizer.fit_transform([' '.join(words)])
        
        # Apply LDA
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(dtm)
        
        # Extract top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
            topics.append(' '.join(top_words))
        
        return topics
    
    def get_wikipedia_topics(self, domain, max_topics=10):
        """Get relevant Wikipedia topics based on domain content"""
        # Extract some content from the domain to find relevant Wikipedia topics
        sample_url = f"https://{domain}"
        content = self.extract_content_from_url(sample_url)
        
        if not content:
            return []
        
        # Extract keywords from content
        topics = self.extract_topics_from_text(content, num_topics=3)
        
        # Search Wikipedia for related topics
        wiki_topics = set()
        
        for topic in topics:
            try:
                page = self.wiki_wiki.page(topic)
                if page.exists():
                    wiki_topics.add(page.title)
                
                # Get links from the page
                links = page.links
                for link_title in list(links.keys())[:5]:  # Get first 5 links
                    wiki_topics.add(link_title)
                    
                    if len(wiki_topics) >= max_topics:
                        break
            except:
                continue
            
            if len(wiki_topics) >= max_topics:
                break
        
        return list(wiki_topics)
    
    def get_reddit_topics(self, domain, max_topics=10):
        """Get relevant Reddit topics based on domain"""
        if not self.reddit:
            return []
        
        try:
            # Extract some content from the domain to find relevant subreddits
            sample_url = f"https://{domain}"
            content = self.extract_content_from_url(sample_url)
            
            if not content:
                return []
            
            # Extract keywords from content
            topics = self.extract_topics_from_text(content, num_topics=3)
            
            reddit_topics = set()
            
            for topic in topics:
                try:
                    # Search for subreddits related to the topic
                    subreddits = list(self.reddit.subreddits.search_by_name(topic, include_nsfw=False))
                    
                    for subreddit in subreddits[:3]:  # Get top 3 subreddits
                        reddit_topics.add(subreddit.display_name)
                        
                        # Get hot posts from the subreddit
                        for post in subreddit.hot(limit=5):
                            reddit_topics.add(post.title)
                            
                            if len(reddit_topics) >= max_topics:
                                break
                        
                        if len(reddit_topics) >= max_topics:
                            break
                except:
                    continue
                
                if len(reddit_topics) >= max_topics:
                    break
            
            return list(reddit_topics)
        except:
            return []
    
    def get_news_topics(self, domain, max_topics=10):
        """Get relevant news topics based on domain"""
        try:
            # Extract some content from the domain to find relevant news topics
            sample_url = f"https://{domain}"
            content = self.extract_content_from_url(sample_url)
            
            if not content:
                return []
            
            # Extract keywords from content
            topics = self.extract_topics_from_text(content, num_topics=3)
            
            news_topics = set()
            
            for topic in topics:
                try:
                    # Use News API (you would need an API key for the real News API)
                    # This is a mock implementation
                    mock_news = [
                        f"Breaking: New developments in {topic}",
                        f"How {topic} is changing the industry",
                        f"Experts weigh in on the future of {topic}",
                        f"5 trends in {topic} you need to know",
                        f"{topic.capitalize()} market analysis and predictions"
                    ]
                    
                    for news_title in mock_news:
                        news_topics.add(news_title)
                        
                        if len(news_topics) >= max_topics:
                            break
                except:
                    continue
                
                if len(news_topics) >= max_topics:
                    break
            
            return list(news_topics)
        except:
            return []
    
    def analyze_urls(self, urls, max_urls=20):
        """Analyze a list of URLs and extract topics"""
        all_topics = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, url in enumerate(urls[:max_urls]):
            status_text.text(f"Analyzing {i+1}/{min(len(urls), max_urls)}: {url}")
            progress_bar.progress((i + 1) / min(len(urls), max_urls))
            
            content = self.extract_content_from_url(url)
            if content:
                topics = self.extract_topics_from_text(content)
                all_topics.extend(topics)
            
            time.sleep(0.5)  # Be polite to servers
        
        progress_bar.empty()
        status_text.empty()
        
        return list(set(all_topics))  # Return unique topics
    
    def analyze_content_gap(self, my_sitemap_url, competitor_sitemap_url):
        """Analyze content gaps between my sitemap and competitor sitemap"""
        # Parse sitemaps
        my_urls = self.parse_sitemap(my_sitemap_url)
        competitor_urls = self.parse_sitemap(competitor_sitemap_url)
        
        if not my_urls:
            st.error(f"Could not parse your sitemap: {my_sitemap_url}")
            return None
        if not competitor_urls:
            st.error(f"Could not parse competitor sitemap: {competitor_sitemap_url}")
            return None
        
        st.info(f"Found {len(my_urls)} URLs in your sitemap and {len(competitor_urls)} URLs in competitor sitemap")
        
        # Extract domains for API-based topic discovery
        my_domain = urlparse(my_sitemap_url).netloc
        competitor_domain = urlparse(competitor_sitemap_url).netloc
        
        # Get topics from various sources
        with st.spinner("Getting Wikipedia topics..."):
            my_wiki_topics = self.get_wikipedia_topics(my_domain)
            competitor_wiki_topics = self.get_wikipedia_topics(competitor_domain)
        
        with st.spinner("Getting Reddit topics..."):
            my_reddit_topics = self.get_reddit_topics(my_domain)
            competitor_reddit_topics = self.get_reddit_topics(competitor_domain)
        
        with st.spinner("Getting news topics..."):
            my_news_topics = self.get_news_topics(my_domain)
            competitor_news_topics = self.get_news_topics(competitor_domain)
        
        # Combine all topics
        my_topics = list(set(my_wiki_topics + my_reddit_topics + my_news_topics))
        competitor_topics = list(set(competitor_wiki_topics + competitor_reddit_topics + competitor_news_topics))
        
        # Analyze URLs if we have few topics from APIs
        if len(my_topics) < 5 or len(competitor_topics) < 5:
            st.info("Not enough topics from APIs, analyzing website content...")
            my_topics.extend(self.analyze_urls(my_urls, max_urls=10))
            competitor_topics.extend(self.analyze_urls(competitor_urls, max_urls=10))
        
        # Find content gaps
        content_gaps = list(set(competitor_topics) - set(my_topics))
        
        return {
            "my_urls": my_urls,
            "competitor_urls": competitor_urls,
            "my_topics": my_topics,
            "competitor_topics": competitor_topics,
            "content_gaps": content_gaps,
            "source_breakdown": {
                "my_wiki": my_wiki_topics,
                "my_reddit": my_reddit_topics,
                "my_news": my_news_topics,
                "competitor_wiki": competitor_wiki_topics,
                "competitor_reddit": competitor_reddit_topics,
                "competitor_news": competitor_news_topics
            }
        }

def main():
    st.title("🔍 Content Gap Analyzer")
    st.markdown("Identify content opportunities by analyzing your competitor's content using free APIs")
    
    # Initialize analyzer
    analyzer = ContentGapAnalyzer()
    
    # Input section
    st.sidebar.header("Input Parameters")
    
    input_method = st.sidebar.radio(
        "Input method",
        ["Direct sitemap URLs", "Domain names (auto-discover sitemap)"]
    )
    
    if input_method == "Direct sitemap URLs":
        my_sitemap = st.sidebar.text_input("Your Sitemap URL", "https://example.com/sitemap.xml")
        competitor_sitemap = st.sidebar.text_input("Competitor Sitemap URL", "https://competitor.com/sitemap.xml")
    else:
        my_domain = st.sidebar.text_input("Your Domain", "example.com")
        competitor_domain = st.sidebar.text_input("Competitor Domain", "competitor.com")
        
        if st.sidebar.button("Discover Sitemaps"):
            with st.spinner("Looking for sitemaps..."):
                my_sitemap = analyzer.find_sitemap(my_domain)
                competitor_sitemap = analyzer.find_sitemap(competitor_domain)
                
                if my_sitemap:
                    st.sidebar.success(f"Your sitemap found: {my_sitemap}")
                else:
                    st.sidebar.error("Could not find your sitemap")
                
                if competitor_sitemap:
                    st.sidebar.success(f"Competitor sitemap found: {competitor_sitemap}")
                else:
                    st.sidebar.error("Could not find competitor sitemap")
        else:
            my_sitemap = ""
            competitor_sitemap = ""
    
    # Optional API keys
    st.sidebar.subheader("Optional API Keys")
    st.sidebar.info("For enhanced results, provide these API keys (not required for basic analysis)")
    
    reddit_client_id = st.sidebar.text_input("Reddit Client ID", type="password")
    reddit_client_secret = st.sidebar.text_input("Reddit Client Secret", type="password")
    reddit_user_agent = st.sidebar.text_input("Reddit User Agent", "ContentGapAnalyzer/1.0 by YourUsername")
    
    if reddit_client_id and reddit_client_secret:
        os.environ["REDDIT_CLIENT_ID"] = reddit_client_id
        os.environ["REDDIT_CLIENT_SECRET"] = reddit_client_secret
        os.environ["REDDIT_USER_AGENT"] = reddit_user_agent
        
        # Reinitialize Reddit client
        try:
            analyzer.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
        except:
            st.sidebar.error("Failed to initialize Reddit client")
    
    analyze_btn = st.sidebar.button("Analyze Content Gap")
    
    if analyze_btn:
        if not my_sitemap or not competitor_sitemap:
            st.error("Please provide valid sitemap URLs")
            return
        
        with st.spinner("Analyzing content gaps..."):
            results = analyzer.analyze_content_gap(my_sitemap, competitor_sitemap)
        
        if not results:
            return
        
        # Display results
        st.header("Content Gap Analysis Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Your URLs", len(results["my_urls"]))
        with col2:
            st.metric("Competitor URLs", len(results["competitor_urls"]))
        with col3:
            st.metric("Your Topics", len(results["my_topics"]))
        with col4:
            st.metric("Content Gaps", len(results["content_gaps"]))
        
        # Content gaps
        st.subheader("Content Opportunities")
        if results["content_gaps"]:
            st.info("These are topics your competitor covers that you might want to create content about:")
            for gap in results["content_gaps"]:
                st.write(f"- {gap}")
        else:
            st.success("No significant content gaps found! You cover all the topics your competitor does.")
        
        # Topic sources breakdown
        st.subheader("Topic Sources Breakdown")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Your Topics by Source")
            source_data = {
                "Source": ["Wikipedia", "Reddit", "News"],
                "Count": [
                    len(results["source_breakdown"]["my_wiki"]),
                    len(results["source_breakdown"]["my_reddit"]),
                    len(results["source_breakdown"]["my_news"])
                ]
            }
            source_df = pd.DataFrame(source_data)
            fig = px.pie(source_df, values='Count', names='Source', title='Your Topics by Source')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("Competitor Topics by Source")
            source_data = {
                "Source": ["Wikipedia", "Reddit", "News"],
                "Count": [
                    len(results["source_breakdown"]["competitor_wiki"]),
                    len(results["source_breakdown"]["competitor_reddit"]),
                    len(results["source_breakdown"]["competitor_news"])
                ]
            }
            source_df = pd.DataFrame(source_data)
            fig = px.pie(source_df, values='Count', names='Source', title='Competitor Topics by Source')
            st.plotly_chart(fig, use_container_width=True)
        
        # Topic comparison
        st.subheader("Topic Coverage Comparison")
        
        # Prepare data for visualization
        topics_data = {
            "Topic": results["my_topics"] + results["competitor_topics"],
            "Domain": ["Your Domain"] * len(results["my_topics"]) + ["Competitor"] * len(results["competitor_topics"])
        }
        
        topics_df = pd.DataFrame(topics_data)
        topic_counts = topics_df.groupby(["Domain", "Topic"]).size().reset_index(name="Count")
        
        # Create visualization
        if not topic_counts.empty:
            fig = px.sunburst(
                topic_counts, 
                path=['Domain', 'Topic'], 
                values='Count',
                color='Domain',
                color_discrete_map={'Your Domain':'lightblue', 'Competitor':'lightcoral'}
            )
            fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data to create visualization")
        
        # Recommendations
        st.subheader("Recommendations")
        if results["content_gaps"]:
            st.markdown(f"""
            1. Create content on these gap topics: **{", ".join(results['content_gaps'][:5])}**
            2. Analyze competitor's top-performing content for ideas on content format and angle
            3. Consider creating pillar content on topics where you have partial coverage
            4. Improve your existing content on similar topics to better compete
            """)
        else:
            st.markdown("""
            1. Focus on creating deeper content on topics you already cover
            2. Look for new emerging topics in your industry
            3. Consider different content formats (video, infographics, etc.)
            4. Improve promotion of your existing content
            """)
        
        # Export option
        st.subheader("Export Results")
        
        # Create a DataFrame for export
        max_len = max(len(results["my_urls"]), len(results["competitor_urls"]), 
                     len(results["my_topics"]), len(results["competitor_topics"]),
                     len(results["content_gaps"]))
        
        export_df = pd.DataFrame({
            "Your_URLs": results["my_urls"] + [None] * (max_len - len(results["my_urls"])),
            "Competitor_URLs": results["competitor_urls"] + [None] * (max_len - len(results["competitor_urls"])),
            "Your_Topics": results["my_topics"] + [None] * (max_len - len(results["my_topics"])),
            "Competitor_Topics": results["competitor_topics"] + [None] * (max_len - len(results["competitor_topics"])),
            "Content_Gaps": results["content_gaps"] + [None] * (max_len - len(results["content_gaps"]))
        })
        
        # Convert to CSV
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="content_gap_analysis.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
