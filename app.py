# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
from urllib.parse import urlparse
import re
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
</style>
""", unsafe_allow_html=True)

class ContentGapAnalyzer:
    def __init__(self):
        self.serp_api_key = os.getenv("SERPAPI_KEY")
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_domain(self, url):
        """Extract domain from URL"""
        parsed_uri = urlparse(url)
        return '{uri.netloc}'.format(uri=parsed_uri)
    
    def get_serp_data(self, query, domain=None):
        """
        Get SERP data for a query (mock implementation - in real scenario, use SERP API)
        """
        # This is a mock implementation. In a real scenario, you would use:
        # params = {
        #   "q": query,
        #   "location": "United States",
        #   "hl": "en",
        #   "gl": "us",
        #   "api_key": self.serp_api_key
        # }
        # response = requests.get("https://serpapi.com/search", params=params)
        # return response.json()
        
        # Mock data for demonstration
        time.sleep(0.5)  # Simulate API delay
        
        mock_data = {
            "organic_results": [
                {
                    "position": 1,
                    "title": f"Top 10 {query} Strategies for 2023",
                    "link": "https://example.com/article1",
                    "snippet": f"Learn about the best {query} strategies that will help you grow your business in 2023.",
                    "displayed_link": "example.com/blog/..."
                },
                {
                    "position": 2,
                    "title": f"How to Master {query} in 5 Easy Steps",
                    "link": "https://competitor.com/article1",
                    "snippet": f"Discover the five steps to mastering {query} and outperforming your competition.",
                    "displayed_link": "competitor.com/blog/..."
                },
                {
                    "position": 3,
                    "title": f"The Ultimate Guide to {query}",
                    "link": "https://example.com/article2",
                    "snippet": f"Comprehensive guide covering everything you need to know about {query}.",
                    "displayed_link": "example.com/guides/..."
                },
                {
                    "position": 4,
                    "title": f"{query} Trends and Statistics for 2023",
                    "link": "https://competitor.com/article2",
                    "snippet": f"Latest trends and statistics about {query} that you need to know.",
                    "displayed_link": "competitor.com/research/..."
                },
                {
                    "position": 5,
                    "title": f"Beginner's Guide to {query}",
                    "link": "https://example.com/article3",
                    "snippet": f"Perfect for beginners looking to understand the basics of {query}.",
                    "displayed_link": "example.com/beginner/..."
                }
            ]
        }
        
        # Filter by domain if provided
        if domain:
            filtered_results = []
            for result in mock_data["organic_results"]:
                if domain in result["link"]:
                    filtered_results.append(result)
            return {"organic_results": filtered_results}
        
        return mock_data
    
    def get_content_topics(self, url):
        """
        Extract main topics from a URL (mock implementation)
        """
        # In a real scenario, you would fetch the page and analyze its content
        # response = requests.get(url, headers=self.headers)
        # soup = BeautifulSoup(response.content, 'html.parser')
        # text = soup.get_text()
        
        # Mock content based on URL
        time.sleep(0.3)  # Simulate processing time
        
        topic_map = {
            "example.com": ["SEO", "Content Marketing", "Social Media", "Email Marketing", "Digital Strategy"],
            "competitor.com": ["SEO", "PPC Advertising", "Video Marketing", "Influencer Marketing", "Analytics"]
        }
        
        domain = self.extract_domain(url)
        for site, topics in topic_map.items():
            if site in domain:
                return topics
        
        # Default return if no match
        return ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"]
    
    def analyze_content_gap(self, my_domain, competitor_domain):
        """
        Analyze content gaps between my domain and competitor domain
        """
        # Get common keywords both sites rank for
        common_keywords = ["digital marketing", "SEO tips", "content strategy", "social media marketing"]
        
        my_content = {}
        competitor_content = {}
        
        # Analyze content for each keyword
        for keyword in common_keywords:
            my_results = self.get_serp_data(keyword, my_domain)
            competitor_results = self.get_serp_data(keyword, competitor_domain)
            
            my_content[keyword] = my_results
            competitor_content[keyword] = competitor_results
        
        # Get unique topics for each domain
        my_topics = self.get_content_topics(f"https://{my_domain}")
        competitor_topics = self.get_content_topics(f"https://{competitor_domain}")
        
        # Find content gaps (topics competitor covers but I don't)
        content_gaps = list(set(competitor_topics) - set(my_topics))
        
        return {
            "my_content": my_content,
            "competitor_content": competitor_content,
            "my_topics": my_topics,
            "competitor_topics": competitor_topics,
            "content_gaps": content_gaps
        }

def main():
    st.title("🔍 Content Gap Analyzer")
    st.markdown("Identify content opportunities by analyzing your competitor's website")
    
    # Initialize analyzer
    analyzer = ContentGapAnalyzer()
    
    # Input section
    st.sidebar.header("Input Parameters")
    my_domain = st.sidebar.text_input("Your Domain", "example.com")
    competitor_domain = st.sidebar.text_input("Competitor Domain", "competitor.com")
    
    # Add API key input (optional)
    serp_api_key = st.sidebar.text_input("SERP API Key (optional)", type="password")
    if serp_api_key:
        os.environ["SERPAPI_KEY"] = serp_api_key
        analyzer.serp_api_key = serp_api_key
    
    analyze_btn = st.sidebar.button("Analyze Content Gap")
    
    if analyze_btn:
        with st.spinner("Analyzing content gaps..."):
            results = analyzer.analyze_content_gap(my_domain, competitor_domain)
        
        # Display results
        st.header("Content Gap Analysis Results")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Content Topics", len(results["my_topics"]))
        with col2:
            st.metric("Competitor Content Topics", len(results["competitor_topics"]))
        with col3:
            st.metric("Content Gaps Identified", len(results["content_gaps"]))
        
        # Content gaps
        st.subheader("Content Opportunities")
        if results["content_gaps"]:
            st.info("These are topics your competitor covers that you might want to create content about:")
            for gap in results["content_gaps"]:
                st.write(f"- {gap}")
        else:
            st.success("No significant content gaps found! You cover all the topics your competitor does.")
        
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
        fig = px.sunburst(
            topic_counts, 
            path=['Domain', 'Topic'], 
            values='Count',
            color='Domain',
            color_discrete_map={'Your Domain':'lightblue', 'Competitor':'lightcoral'}
        )
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Keyword analysis
        st.subheader("Keyword Performance Comparison")
        
        # Mock keyword data
        keyword_data = {
            "Keyword": ["digital marketing", "SEO tips", "content strategy", "social media marketing"],
            "Your Ranking": [3, 5, 2, 8],
            "Competitor Ranking": [1, 2, 4, 3]
        }
        keyword_df = pd.DataFrame(keyword_data)
        
        # Display keyword table
        st.dataframe(keyword_df)
        
        # Recommendations
        st.subheader("Recommendations")
        st.markdown("""
        1. Create content on these gap topics: **{", ".join(results['content_gaps'])}**
        2. Improve your content on keywords where competitor ranks higher
        3. Consider creating pillar content on topics where you have partial coverage
        4. Analyze competitor's top-performing content for ideas on content format and angle
        """)
        
        # Export option
        if st.button("Export Results"):
            # Create a DataFrame for export
            export_data = {
                "Content Gaps": results["content_gaps"],
                "Your Topics": results["my_topics"],
                "Competitor Topics": results["competitor_topics"]
            }
            
            # Convert to DataFrame (needs adjustment for uneven list lengths)
            max_len = max(len(results["content_gaps"]), len(results["my_topics"]), len(results["competitor_topics"]))
            export_df = pd.DataFrame({
                "Content_Gaps": results["content_gaps"] + [None] * (max_len - len(results["content_gaps"])),
                "Your_Topics": results["my_topics"] + [None] * (max_len - len(results["my_topics"])),
                "Competitor_Topics": results["competitor_topics"] + [None] * (max_len - len(results["competitor_topics"]))
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
