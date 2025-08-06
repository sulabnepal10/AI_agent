import asyncio
import json
import logging
import os
import re
import time
import subprocess
from typing import List, Dict, Set, Optional
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from pydantic import BaseModel
import google.generativeai as genai
from openai import AsyncOpenAI
from urllib.parse import urljoin, urlparse
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

# Install Playwright browsers
def install_playwright_browsers():
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
    except Exception as e:
        logging.error(f"Failed to install Playwright browsers: {str(e)}")

install_playwright_browsers()

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StartupData(BaseModel):
    url: str
    name: str
    founders: List[str]
    emails: List[str]
    pages_crawled: int
    confidence_score: float
    extraction_method: str

def is_valid_name(name: str) -> bool:
    """Enhanced name validation with better filtering."""
    name = name.strip()
    if not name or len(name) < 3:
        return False
    
    words = name.split()
    if len(words) < 2 or len(words) > 4:  
        return False
    
    invalid_patterns = [
        r'\d+',  
        r'info|team|about|contact|support|admin|sales|marketing|press',
        r'www\.|\.com|\.org|\.net',  
        r'CEO|CTO|CFO|VP|Director|Manager|Lead|Head|Senior|Junior',  
        r'Company|Corp|Inc|LLC|Ltd|Group|Solutions|Technologies',
        r'@|#|\$|%|\&|\*',  
        r'[A-Z]{3,}', 
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, name, re.IGNORECASE):
            return False
    
    for word in words:
        if not word[0].isupper() or not word[1:].islower():
            return False
    
    try:
        tokens = word_tokenize(name)
        pos_tags = pos_tag(tokens)
        proper_nouns = [word for word, pos in pos_tags if pos in ['NNP', 'NNPS']]
        if len(proper_nouns) < len(words) * 0.5: 
            return False
    except:
        pass  
    
    return True

def is_valid_email(email: str) -> bool:
    """Enhanced email validation."""
    email = email.lower().strip()
    
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False
    
    invalid_patterns = [
        r'info@|support@|contact@|sales@|team@|admin@|help@|service@|customer@',
        r'marketing@|press@|media@|news@|careers@|jobs@|hr@|legal@',
        r'noreply@|no-reply@|donotreply@|bounce@|mailer@|daemon@',
        r'test@|example@|sample@|demo@|fake@|dummy@',
        r'webmaster@|postmaster@|hostmaster@|root@|www@',
        r'@example\.|@test\.|@localhost|@domain\.'
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, email):
            return False
    
    return True

def is_startup_url(url: str, title: str = "", snippet: str = "") -> bool:
    """Enhanced startup URL filtering."""
    excluded_domains = [
        'linkedin.com', 'twitter.com', 'facebook.com', 'instagram.com', 'youtube.com',
        'wikipedia.org', 'crunchbase.com', 'substack.com', 'medium.com', 'forbes.com',
        'techcrunch.com', 'venturebeat.com', 'bloomberg.com', 'reuters.com', 'nytimes.com',
        'wsj.com', 'ft.com', 'cnbc.com', 'businessinsider.com', 'fortune.com',
        'github.com', 'stackoverflow.com', 'reddit.com', 'quora.com'
    ]

    try:
        domain = urlparse(url).netloc.lower()
        domain = domain.replace('www.', '')
    except:
        return False

    if any(excluded in domain for excluded in excluded_domains):
        return False
    
    startup_indicators = [
        r'\.ai$', r'\.io$', r'\.co$', r'\.tech$', r'\.app$',
        r'startup', r'founder', r'co-founder', r'ceo', r'team',
        r'about', r'company', r'leadership', r'innovation'
    ]
    
    text = f"{url} {title} {snippet}".lower()
    return any(re.search(pattern, text) for pattern in startup_indicators)

def extract_company_name(url: str, html_content: str = "") -> str:
    """Extract company name from URL or HTML content."""
    try:
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.find('title')
            if title:
                title_text = title.get_text().strip()
                title_text = re.sub(r'\s*[-|–]\s*(About|Home|Company|Startup).*$', '', title_text, flags=re.IGNORECASE)
                if title_text and len(title_text) < 100:
                    return title_text
 
        domain = urlparse(url).netloc.replace('www.', '')
        company_name = domain.split('.')[0]
        return company_name.title()
    except:
        return "Unknown Company"

async def enhanced_search_serpapi(keyword: str, serpapi_key: str, num_results: int = 20) -> List[Dict]:
    """Enhanced SerpAPI search with better query strategies."""
    cache_file = f"serpapi_cache_{keyword.replace(' ', '_')}.json"

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_results = json.load(f)
                if len(cached_results) >= num_results:
                    return cached_results[:num_results]
        except:
            pass
    
    logger.info(f"Searching SerpAPI for: {keyword}")

    query_variations = [
        f'"{keyword}" founder CEO "about us" -linkedin -crunchbase -news',
        f'{keyword} startup "team" "leadership" -job -career -article',
        f'"{keyword}" "founded by" OR "co-founder" site:*.com OR site:*.ai OR site:*.io',
        f'{keyword} company "executive team" "management" -wikipedia -blog',
        f'"{keyword}" "our team" "meet the team" startup'
    ]
    
    all_results = []
    seen_urls = set()
    
    for query in query_variations:
        try:
            params = {
                "q": query,
                "num": 20,
                "api_key": serpapi_key,
                "gl": "us",
                "hl": "en"
            }
            
            response = requests.get("https://serpapi.com/search", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for result in data.get("organic_results", []):
                url = result.get("link", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                if url and url not in seen_urls and is_startup_url(url, title, snippet):
                    all_results.append({
                        'url': url,
                        'title': title,
                        'snippet': snippet,
                        'query': query
                    })
                    seen_urls.add(url)
            
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Search error for query '{query}': {str(e)}")
            continue
    
    def relevance_score(result):
        text = f"{result['title']} {result['snippet']}".lower()
        score = 0
        score += text.count('founder') * 3
        score += text.count('ceo') * 2
        score += text.count('team') * 1
        score += text.count('about') * 1
        return score
    
    all_results.sort(key=relevance_score, reverse=True)
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(all_results, f)
    except Exception as e:
        logger.error(f"Error saving cache: {str(e)}")
    
    return all_results[:num_results]

def advanced_regex_extraction(text: str, html: str = "") -> Dict:
    """Advanced regex extraction with context analysis."""
    founders = set()
    emails = set()
    
    founder_patterns = [
        r'(?:founded by|co-founded by|started by|created by)[:\s,-]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'(?:founder|co-founder)[:\s,-]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[,\s]+(?:is|was)?\s*(?:the\s+)?(?:founder|co-founder)',
        r'(?:CEO|Chief Executive Officer)[:\s,-]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[,\s]+(?:is|was)?\s*(?:the\s+)?(?:CEO|Chief Executive Officer)',
        r'<h[1-6][^>]*>([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)</h[1-6]>[^<]*(?:founder|ceo|chief)',
    ]
    
    content_sources = [text]
    if html:
        content_sources.append(html)
        soup = BeautifulSoup(html, 'html.parser')
        clean_text = soup.get_text()
        content_sources.append(clean_text)
    
    for content in content_sources:
        for pattern in founder_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else ""
                
                name = match.strip()
                if name and is_valid_name(name):
                    founders.add(name)
    
    email_patterns = [
        r'\b([a-zA-Z0-9][a-zA-Z0-9._%+-]{1,50}@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
        r'mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        r'(?:email|contact)[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
    ]
    
    for content in content_sources:
        for pattern in email_patterns:
            email_matches = re.findall(pattern, content, re.IGNORECASE)
            for email in email_matches:
                if is_valid_email(email):
                    emails.add(email.lower())
    
    return {"founders": list(founders), "emails": list(emails)}

async def find_relevant_pages(url: str, max_pages: int = 5) -> List[str]:
    """Find relevant pages with improved link detection."""
    relevant_pages = [url]  
    
    try:
        async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
            result = await crawler.arun(url)
            
            if not result.html:
                return relevant_pages
            
            soup = BeautifulSoup(result.html, 'html.parser')
            links = soup.find_all('a', href=True)
            
            relevant_keywords = {
                'about': 5, 'team': 5, 'leadership': 4, 'founder': 5,
                'executive': 3, 'company': 3, 'management': 3,
                'our-team': 4, 'meet-team': 4, 'who-we-are': 3,
                'staff': 2, 'people': 2, 'board': 2
            }
            
            link_scores = []
            
            for link in links:
                href = link.get('href', '')
                text = link.get_text().strip().lower()
                
                if not href:
                    continue
                
                try:
                    full_url = urljoin(url, href)
                    if not full_url.startswith('http'):
                        continue   
                
                    if urlparse(full_url).netloc != urlparse(url).netloc:
                        continue
                        
                except:
                    continue
              
                score = 0
                content_to_check = f"{href} {text}".lower()
                
                for keyword, weight in relevant_keywords.items():
                    if keyword in content_to_check:
                        score += weight
                
                if score > 0:
                    link_scores.append((full_url, score))
          
            link_scores.sort(key=lambda x: x[1], reverse=True)
            
            for page_url, score in link_scores[:max_pages-1]:
                if page_url not in relevant_pages:
                    relevant_pages.append(page_url)
            
    except Exception as e:
        logger.error(f"Error finding relevant pages for {url}: {str(e)}")
    
    return relevant_pages[:max_pages]

async def call_openai_enhanced(text: str, api_key: str, model_name: str, url: str = "") -> Dict:
    """Enhanced OpenAI call with better prompting."""
    client = AsyncOpenAI(api_key=api_key)
    results = {"founders": set(), "emails": set()}
  
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)] if len(text) > 4000 else [text]
    
    for chunk in chunks:
        try:
            prompt = f"""
You are an expert at extracting founder information from company websites. 

STRICT RULES:
1. Extract ONLY full names (First Last) of actual founders, co-founders, or CEOs
2. Names must be real people, not companies or generic terms
3. Extract ONLY personal email addresses (name@company.com format)
4. EXCLUDE generic emails like info@, support@, contact@, etc.
5. Return ONLY valid JSON format

EXAMPLES:
 Good: "Mohan kumar" (founder), "mohan.kumar@company.com"
 Bad: "Company Team", "CEO", "info@company.com", "Mohan" (first name only)

Website: {url}
Content to analyze:
{chunk}

Extract and return ONLY a JSON object:
{{"founders": ["Full Name 1", "Full Name 2"], "emails": ["sulab1@company.com", "sulab2@company.com"]}}
"""

            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400
            )
            
            response_text = response.choices[0].message.content.strip()
                
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'\s*```', '', response_text)
            
            try:
                data = json.loads(response_text)
                founders = data.get("founders", [])
                emails = data.get("emails", [])
          
                for name in founders:
                    if isinstance(name, str) and is_valid_name(name):
                        results["founders"].add(name.strip())
                
                for email in emails:
                    if isinstance(email, str) and is_valid_email(email):
                        results["emails"].add(email.lower().strip())
                        
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from OpenAI: {response_text[:100]}")
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            break
    
    return {"founders": list(results["founders"]), "emails": list(results["emails"])}

async def extract_startup_data(url: str, api_key: str, model: str, max_pages: int = 3) -> StartupData:
    """Enhanced startup data extraction."""
    founders = set()
    emails = set()
    pages_crawled = 0
    extraction_methods = []
    
    try:
       
        relevant_pages = await find_relevant_pages(url, max_pages)
        
        async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
            for page_url in relevant_pages:
                try:
                    result = await crawler.arun(
                        url=page_url,
                        config=CrawlerRunConfig(cache_mode="BYPASS")
                    )
                    
                    if not result.markdown and not result.html:
                        continue
                    
                    pages_crawled += 1
                    
            
                    if model.startswith("OpenAI") and api_key:
                        openai_model = "gpt-4o" if "gpt-4o" in model else "gpt-4o-mini"
                        ai_result = await call_openai_enhanced(result.markdown, api_key, openai_model, page_url)
                        
                        if ai_result["founders"] or ai_result["emails"]:
                            founders.update(ai_result["founders"])
                            emails.update(ai_result["emails"])
                            extraction_methods.append("AI")
                    
                    regex_result = advanced_regex_extraction(result.markdown, result.html)
                    if regex_result["founders"] or regex_result["emails"]:
                        founders.update(regex_result["founders"])
                        emails.update(regex_result["emails"])
                        if "Regex" not in extraction_methods:
                            extraction_methods.append("Regex")
                    
                except Exception as e:
                    logger.error(f"Error processing {page_url}: {str(e)}")
                    continue
        
    
        company_name = extract_company_name(url)
        
    
        confidence_score = min(1.0, (len(founders) * 0.4 + len(emails) * 0.6) / max_pages)
        
        return StartupData(
            url=url,
            name=company_name,
            founders=list(founders),
            emails=list(emails),
            pages_crawled=pages_crawled,
            confidence_score=confidence_score,
            extraction_method=", ".join(extraction_methods) or "None"
        )
        
    except Exception as e:
        logger.error(f"Error extracting data from {url}: {str(e)}")
        return StartupData(
            url=url,
            name="Error",
            founders=[],
            emails=[],
            pages_crawled=pages_crawled,
            confidence_score=0.0,
            extraction_method="Error"
        )

async def run_enhanced_crawler(keyword: str, serpapi_key: str, api_key: str, model: str, max_results: int = 10) -> pd.DataFrame:
    """Main crawler function with enhanced processing."""
    logger.info("Starting enhanced crawler workflow")
   
    search_results = await enhanced_search_serpapi(keyword, serpapi_key, max_results * 2)
    
    if not search_results:
        st.warning("No suitable startup URLs found.")
        return pd.DataFrame()
    
    st.write(f" Found {len(search_results)} potential startup URLs")
    
    with st.expander("🔍 View Search Results"):
        for i, result in enumerate(search_results[:max_results], 1):
            st.write(f"**{i}. {result['title']}**")
            st.write(f"URL: {result['url']}")
            st.write(f"Snippet: {result['snippet'][:150]}...")
            st.write("---")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    urls_to_process = [result['url'] for result in search_results[:max_results]]
    extraction_results = []
    successful_extractions = 0
    
    for i, url in enumerate(urls_to_process):
        status_text.text(f"Processing {i+1}/{len(urls_to_process)}: {url}")
        
        try:
            result = await extract_startup_data(url, api_key, model, max_pages=3)
            extraction_results.append(result.model_dump())
            
            if result.founders or result.emails:
                successful_extractions += 1
                with results_container:
                    st.success(f"✅ **{result.name}** - Found {len(result.founders)} founders, {len(result.emails)} emails")
                    if result.founders:
                        st.write(f"Founders: {', '.join(result.founders)}")
                    if result.emails:
                        st.write(f"Emails: {', '.join(result.emails)}")
            else:
                with results_container:
                    st.info(f"ℹ️ **{result.name}** - No founder data found")
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            extraction_results.append({
                "url": url,
                "name": "Error",
                "founders": [],
                "emails": [],
                "pages_crawled": 0,
                "confidence_score": 0.0,
                "extraction_method": "Error"
            })
        
        progress_bar.progress((i + 1) / len(urls_to_process))
        
        if i < len(urls_to_process) - 1:
            await asyncio.sleep(2)
    
    status_text.text(f"✅ Completed! Found data for {successful_extractions}/{len(extraction_results)} companies")
    
    df = pd.DataFrame({
        "Company Name": [r["name"] for r in extraction_results],
        "Website URL": [r["url"] for r in extraction_results],
        "Founders": [", ".join(r["founders"]) for r in extraction_results],
        "Founder Emails": [", ".join(r["emails"]) for r in extraction_results],
        "Pages Crawled": [r["pages_crawled"] for r in extraction_results],
        "Confidence Score": [f"{r['confidence_score']:.2f}" for r in extraction_results],
        "Extraction Method": [r["extraction_method"] for r in extraction_results],
        "Data Found": ["✅" if (r["founders"] or r["emails"]) else "❌" for r in extraction_results]
    })
    
    try:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"startup_founders_enhanced_{timestamp}.xlsx"
        csv_filename = f"startup_founders_enhanced_{timestamp}.csv"
        
        df.to_excel(excel_filename, index=False, engine="openpyxl")
        df.to_csv(csv_filename, index=False)
        
        st.session_state['excel_file'] = excel_filename
        st.session_state['csv_file'] = csv_filename
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        st.error(f"Failed to save results: {str(e)}")
    
    return df

def main():
    """Streamlit app main function."""
    st.set_page_config(page_title="Founder Extractor Agent", page_icon="", layout="wide")
    
    st.title("Startup Founder Extractor")
    st.markdown("Extract founder names and emails from startup websites")
    
    with st.form("crawler_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            keyword = st.text_input(
                "Search Keyword", 
                value="AI fintech startups 2025",
                help="Try specific terms like 'AI fintech startups 2025' or 'B2B SaaS startups founders'"
            )
            serpapi_key = st.text_input("SerpAPI Key", type="password")
            
        with col2:
            api_key = st.text_input("OpenAI API Key", type="password")
            model = st.selectbox("Model", [
                "OpenAI GPT-4o",
                "OpenAI GPT-4o-mini"
            ])
        
        max_results = st.slider("Maximum Results to Process", 5, 30, 10)
        
        submit_button = st.form_submit_button("Start Extraction", use_container_width=True)
    
    if submit_button:
        if not serpapi_key:
            st.error("Please provide a SerpAPI key.")
            return
        
        if not api_key and model.startswith("OpenAI"):
            st.warning("OpenAI API key not provided. Will use regex extraction only.")
        
        st.markdown("---")
        st.write("### Processing Results")
    
        try:
            df = asyncio.run(run_enhanced_crawler(
                keyword=keyword,
                serpapi_key=serpapi_key,
                api_key=api_key,
                model=model,
                max_results=max_results
            ))
            
            if not df.empty:
                st.markdown("---")
                st.write("### Final Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Companies", len(df))
                with col2:
                    successful = len(df[df["Data Found"] == "✅"])
                    st.metric("Successful Extractions", successful)
                with col3:
                    total_founders = sum(len(founders.split(", ")) if founders else 0 for founders in df["Founders"])
                    st.metric("Total Founders Found", total_founders)
                with col4:
                    total_emails = sum(len(emails.split(", ")) if emails else 0 for emails in df["Founder Emails"])
                    st.metric("Total Emails Found", total_emails)
                
                st.dataframe(df, use_container_width=True)
                
                if 'excel_file' in st.session_state and 'csv_file' in st.session_state:
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(st.session_state['excel_file'], 'rb') as f:
                            st.download_button(
                                "Download Excel",
                                f,
                                file_name=st.session_state['excel_file'],
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    with col2:
                        with open(st.session_state['csv_file'], 'rb') as f:
                            st.download_button(
                                "Download CSV",
                                f,
                                file_name=st.session_state['csv_file'],
                                mime="text/csv"
                            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main()