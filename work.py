import streamlit as st
import requests
import asyncio
import pandas as pd
import re
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
import logging
import os
from typing import List, Dict, Set, Optional
import time
from playwright.async_api import async_playwright
import google.generativeai as genai
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Data Model ---
from startup_model import StartupData

# --- URL FILTERING ---
EXCLUDED_DOMAINS = {
    'medium.com', 'forbes.com', 'techcrunch.com', 'venturebeat.com', 'wired.com',
    'bloomberg.com', 'reuters.com', 'cnn.com', 'bbc.com', 'theguardian.com',
    'wsj.com', 'nytimes.com', 'businessinsider.com', 'fastcompany.com', 'inc.com',
    'entrepreneur.com', 'fortune.com', 'harvard.edu', 'mit.edu', 'youtube.com',
    'reddit.com', 'quora.com', 'stackoverflow.com', 'wikipedia.org', 'linkedin.com',
    'twitter.com', 'facebook.com', 'instagram.com', 'pinterest.com', 'github.com',
    'substack.com', 'hackernoon.com', 'dev.to', 'ycombinator.com', 'crunchbase.com',
    'pitchbook.com', 'wellfound.com', 'dealroom.co', 'tracxn.com', 'cbinsights.com',
    'startupranking.com', 'eu-startups.com', 'builtwith.com', 'g2.com', 'capterra.com',
    'f6s.com', 'apollo.io', 'zoominfo.com', 'amazon.com', 'google.com', 'apple.com',
    'microsoft.com', 'meta.com', 'adobe.com', 'oracle.com', 'ibm.com', 'salesforce.com',
    'sap.com', 'shopify.com'
}

STARTUP_POSITIVE_INDICATORS = {
    '.ai', '.io', '.co', '.app', '.tech', '.dev', '.ml', '.xyz', '.ly',
    'startup', 'labs', 'technologies', 'solutions', 'systems', 'platform',
    'software', 'app', 'tool', 'service', 'api', 'saas'
}

def is_likely_startup(url: str, title: str = "", snippet: str = "") -> bool:
    """Enhanced startup detection."""
    if not url:
        return False
    
    try:
        parsed = urlparse(url.lower())
        domain = parsed.netloc.replace('www.', '')
        
        if any(excluded in domain for excluded in EXCLUDED_DOMAINS):
            return False
        
        startup_signals = sum(1 for indicator in STARTUP_POSITIVE_INDICATORS if indicator in domain)
        content = f"{title} {snippet}".lower()
        content_signals = sum(1 for keyword in ['startup', 'founder', 'ceo', 'venture', 'funding', 'seed', 'series'] if keyword in content)
        negative_signals = sum(1 for keyword in ['blog', 'news', 'article', 'research', 'university', 'government', 'wiki', 'forum', 'job', 'career'] if keyword in content or keyword in domain)
        
        return (startup_signals + content_signals >= 3) and (negative_signals == 0)
    except:
        return False

def search_multiple_engines(keyword: str, serpapi_key: str, num_results: int = 100) -> List[Dict]:
    """Search with multiple query variations."""
    logger.info(f"Searching for: {keyword}")
    
    query_variations = [
        f'"{keyword}" founder CEO -blog -news -wikipedia',
        f'{keyword} "about us" "team" "founder" -linkedin -crunchbase',
        f'{keyword} startup "co-founder" site:*.com OR site:*.ai OR site:*.io',
        f'{keyword} company "leadership" "executive team" -job -career',
        f'"{keyword}" "founded by" OR "started by" -article -post'
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
                
                if url and url not in seen_urls and is_likely_startup(url, title, snippet):
                    all_results.append({
                        'url': url,
                        'title': title,
                        'snippet': snippet,
                        'query': query
                    })
                    seen_urls.add(url)
            
            time.sleep(1)
        except Exception as e:
            logger.error(f"Search error for query '{query}': {str(e)}")
            continue
    
    all_results.sort(key=lambda x: (x['title'].lower().count('founder') + x['snippet'].lower().count('founder')), reverse=True)
    return all_results[:num_results]

def find_all_relevant_pages(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Find relevant pages for founder info."""
    relevant_pages = set()
    
    page_patterns = [
        r'\babout\b', r'\babout.us\b', r'\bteam\b', r'\bour.team\b', r'\bleadership\b',
        r'\bfounders?\b', r'\bcontact\b', r'\bcontact.us\b', r'\bmission\b', r'\bour.story\b',
        r'\bexecutives?\b', r'\bwho.we.are\b', r'\bmanagement\b'
    ]
    
    all_links = soup.find_all('a', href=True)
    
    for link in all_links:
        href = link.get('href', '').lower()
        text = link.get_text(strip=True).lower()
        
        for pattern in page_patterns:
            if re.search(pattern, href) or re.search(pattern, text):
                full_url = urljoin(base_url, link['href'])
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    relevant_pages.add(full_url)
                break
    
    base_domain = urlparse(base_url).netloc
    base_scheme = urlparse(base_url).scheme
    common_paths = [
        '/about', '/about-us', '/team', '/our-team', '/company', '/leadership',
        '/founders', '/contact', '/contact-us', '/mission', '/our-story', '/executive-team'
    ]
    
    for path in common_paths:
        relevant_pages.add(f"{base_scheme}://{base_domain}{path}")
    
    return list(relevant_pages)[:10]

def extract_structured_data(soup: BeautifulSoup) -> Dict:
    """Extract structured data from JSON-LD, microdata."""
    structured_data = {"founders": [], "emails": []}
    
    json_scripts = soup.find_all('script', type='application/ld+json')
    for script in json_scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, dict):
                for field in ['founder', 'founders', 'person', 'employee']:
                    if field in data:
                        founder_data = data[field]
                        if isinstance(founder_data, list):
                            for founder in founder_data:
                                if isinstance(founder, dict) and 'name' in founder:
                                    structured_data["founders"].append(founder['name'])
                                elif isinstance(founder, str):
                                    structured_data["founders"].append(founder)
                        elif isinstance(founder_data, dict) and 'name' in founder_data:
                            structured_data["founders"].append(founder_data['name'])
        except:
            continue
    
    founder_elements = soup.find_all(attrs={"itemtype": re.compile(r"person|founder", re.I)})
    for element in founder_elements:
        name_elem = element.find(attrs={"itemprop": "name"})
        if name_elem:
            structured_data["founders"].append(name_elem.get_text(strip=True))
    
    return structured_data

def advanced_regex_extraction(text: str, html: str) -> Dict:
    """Regex patterns for founder and email extraction."""
    founders = set()
    emails = set()
    
    founder_patterns = [
        r'(?:founder|co-founder|ceo|chief executive officer)[:\s,-]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[,\s]+(?:is|was)?\s*(?:the\s+)?(?:founder|co-founder|ceo)',
        r'(?:founded by|started by|created by)[:\s,-]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'meet\s+(?:our\s+)?(?:founder|ceo)[:\s,-]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:is|was)\s+(?:the\s+)?(?:founder|co-founder|ceo)\s*(?:since|in)?\s*\d{4}?'
    ]
    
    content_sources = [text, html] if html else [text]
    
    for content in content_sources:
        for pattern in founder_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else ""
                if match and is_valid_name(match.strip()):
                    founders.add(match.strip())
    
    email_patterns = [
        r'\b([a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
        r'mailto:([a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        r'email[:\s]*([a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    ]
    
    for content in content_sources:
        for pattern in email_patterns:
            email_matches = re.findall(pattern, content, re.IGNORECASE)
            for email in email_matches:
                if is_valid_email(email):
                    emails.add(email.lower())
    
    return {"founders": list(founders), "emails": list(emails)}

def is_valid_name(name: str) -> bool:
    """Name validation."""
    if not isinstance(name, str):
        return False
    
    name = re.sub(r'\s+', ' ', name.strip())
    if not (3 < len(name) < 60) or len(name.split()) < 2:
        return False
    
    if not re.match(r"^[A-Za-z\s.'-]+$", name):
        return False
    
    exclude_words = {
        'ceo', 'cto', 'founder', 'co-founder', 'chief', 'officer', 'director',
        'president', 'vice', 'company', 'inc', 'llc', 'corporation', 'team'
    }
    
    name_words = set(word.lower() for word in name.split())
    if name_words.intersection(exclude_words):
        return False
    
    words = name.split()
    return all(word[0].isupper() for word in words)

def is_valid_email(email: str) -> bool:
    """Email validation."""
    if not email or '@' not in email or '.' not in email:
        return False
    
    email = email.lower().strip()
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False
    
    generic_prefixes = {
        'info', 'contact', 'support', 'sales', 'admin', 'hello', 'help', 'service', 'team'
    }
    
    email_prefix = email.split('@')[0]
    return email_prefix not in generic_prefixes and ('.' in email_prefix or '_' in email_prefix or email_prefix.isalpha())

def chunk_content(text: str, max_length: int = 7000) -> List[str]:
    """Chunk content for API."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def call_model_with_retry(text: str, api_key: str, model: str, url: str = "", max_retries: int = 3) -> Dict:
    """Call selected model API with chunking and retry logic."""
    results = {"founders": set(), "emails": set()}
    
    for chunk in chunk_content(text):
        for attempt in range(max_retries):
            try:
                if model == "Gemini":
                    genai.configure(api_key=api_key)
                    model_instance = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"""
You are an expert at extracting founder information from startup websites. Extract full names of founders/co-founders/CEOs and their personal email addresses.

RULES:
1. FOUNDER NAMES: Extract full names (First Last) for roles like Founder, Co-Founder, CEO.
2. EMAILS: Extract personal emails (e.g., john.smith@company.com), not generic ones (e.g., info@company.com).
3. OUTPUT: Return a JSON object with "founders" and "emails" arrays.
4. IGNORE: News articles, job listings, blog posts, or directory listings.

EXAMPLES:
Input: "Our CEO John Smith founded Acme in 2020. Contact: john.smith@acme.com"
Output: {{"founders": ["John Smith"], "emails": ["john.smith@acme.com"]}}

Input: "Founded by Sarah Lee and Tom Wilson. Contact: support@company.com"
Output: {{"founders": ["Sarah Lee", "Tom Wilson"], "emails": []}}

WEBSITE: {url}
CONTENT:
{chunk}

Return ONLY a valid JSON object:
"""
                    response = model_instance.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=500,
                            top_p=0.8
                        )
                    )
                    response_text = response.text.strip()
                    response_text = re.sub(r'```json\s*', '', response_text)
                    response_text = re.sub(r'\s*```', '', response_text)
                
                elif model == "OpenAI":
                    client = OpenAI(api_key=api_key)
                    prompt = f"""
You are an expert at extracting founder information from startup websites. Extract full names of founders/co-founders/CEOs and their personal email addresses.

RULES:
1. FOUNDER NAMES: Extract full names (First Last) for roles like Founder, Co-Founder, CEO.
2. EMAILS: Extract personal emails (e.g., john.smith@company.com), not generic ones (e.g., info@company.com).
3. OUTPUT: Return a JSON object with "founders" and "emails" arrays.
4. IGNORE: News articles, job listings, blog posts, or directory listings.

EXAMPLES:
Input: "Our CEO John Smith founded Acme in 2020. Contact: john.smith@acme.com"
Output: {{"founders": ["John Smith"], "emails": ["john.smith@acme.com"]}}

Input: "Founded by Sarah Lee and Tom Wilson. Contact: support@company.com"
Output: {{"founders": ["Sarah Lee", "Tom Wilson"], "emails": []}}

WEBSITE: {url}
CONTENT:
{chunk}

Return ONLY a valid JSON object:
"""
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=500
                    )
                    response_text = response.choices[0].message.content.strip()
                    response_text = re.sub(r'```json\s*', '', response_text)
                    response_text = re.sub(r'\s*```', '', response_text)
                
                elif model == "DeepSeek":
                    response = requests.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "deepseek-reasoner",
                            "messages": [{
                                "role": "user",
                                "content": f"""
You are an expert at extracting founder information from startup websites. Extract full names of founders/co-founders/CEOs and their personal email addresses.

RULES:
1. FOUNDER NAMES: Extract full names (First Last) for roles like Founder, Co-Founder, CEO.
2. EMAILS: Extract personal emails (e.g., john.smith@company.com), not generic ones (e.g., info@company.com).
3. OUTPUT: Return a JSON object with "founders" and "emails" arrays.
4. IGNORE: News articles, job listings, blog posts, or directory listings.

EXAMPLES:
Input: "Our CEO John Smith founded Acme in 2020. Contact: john.smith@acme.com"
Output: {{"founders": ["John Smith"], "emails": ["john.smith@acme.com"]}}

Input: "Founded by Sarah Lee and Tom Wilson. Contact: support@company.com"
Output: {{"founders": ["Sarah Lee", "Tom Wilson"], "emails": []}}

WEBSITE: {url}
CONTENT:
{chunk}

Return ONLY a valid JSON object:
"""
                            }],
                            "temperature": 0.1,
                            "max_tokens": 500
                        }
                    )
                    response.raise_for_status()
                    response_text = response.json()['choices'][0]['message']['content'].strip()
                    response_text = re.sub(r'```json\s*', '', response_text)
                    response_text = re.sub(r'\s*```', '', response_text)
                
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    founders = data.get("founders", [])
                    emails = data.get("emails", [])
                    results["founders"].update([name for name in founders if is_valid_name(str(name))])
                    results["emails"].update([email for email in emails if is_valid_email(str(email))])
                    break
                else:
                    logger.warning(f"No JSON found in response for {model} (attempt {attempt + 1})")
                
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"{model} API error (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    break
                time.sleep(2 ** attempt)
    
    return {"founders": list(results["founders"]), "emails": list(results["emails"])}

async def crawl_with_playwright(url: str) -> tuple[str, bool]:
    """Crawl page using Playwright."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30000)
            content = await page.content()
            await browser.close()
            return content, True
    except Exception as e:
        logger.error(f"Playwright error for {url}: {str(e)}")
        return "", False

async def extract_from_multiple_pages(url: str, api_key: str, model: str, max_pages: int = 6) -> StartupData:
    """Extract founder info with selected model or regex/structured data."""
    logger.info(f"Starting extraction for: {url}")
    
    all_founders = set()
    all_emails = set()
    pages_crawled = 0
    total_content = []
    
    # Crawl main page
    html, success = await crawl_with_playwright(url)
    if not success:
        logger.error(f"Failed to crawl main page: {url}")
        return StartupData(url=url, name="Failed to load", founders=[], emails=[], pages_crawled=0, confidence_score=0.0)
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract company name
    company_name = "Unknown"
    if soup.title:
        title_text = soup.title.text.strip()
        company_name = re.split(r'\s*[-|–]\s*', title_text)[0].strip()
        company_name = re.sub(r'\s*\|\s*.*$', '', company_name).strip()
    
    logger.info(f"Company: {company_name}")
    
    # Process main page
    main_text = BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)
    total_content.append(main_text)
    pages_crawled += 1
    
    structured_data = extract_structured_data(soup)
    regex_data = advanced_regex_extraction(main_text, html)
    
    all_founders.update(structured_data.get("founders", []))
    all_founders.update(regex_data.get("founders", []))
    all_emails.update(structured_data.get("emails", []))
    all_emails.update(regex_data.get("emails", []))
    
    # Crawl related pages
    relevant_pages = find_all_relevant_pages(soup, url)
    logger.info(f"Found {len(relevant_pages)} potentially relevant pages")
    
    for page_url in relevant_pages[:max_pages-1]:
        try:
            if page_url != url:
                await asyncio.sleep(1)
                page_html, page_success = await crawl_with_playwright(page_url)
                if page_success and page_html:
                    page_soup = BeautifulSoup(page_html, 'html.parser')
                    page_text = page_soup.get_text(separator=' ', strip=True)
                    
                    total_content.append(page_text)
                    pages_crawled += 1
                    
                    page_structured = extract_structured_data(page_soup)
                    page_regex = advanced_regex_extraction(page_text, page_html)
                    
                    all_founders.update(page_structured.get("founders", []))
                    all_founders.update(page_regex.get("founders", []))
                    all_emails.update(page_structured.get("emails", []))
                    all_emails.update(page_regex.get("emails", []))
                    
                    logger.info(f"Successfully crawled: {page_url}")
                else:
                    logger.warning(f"Failed to crawl: {page_url}")
        except Exception as e:
            logger.error(f"Error crawling {page_url}: {str(e)}")
            continue
    
    # Combine content for AI analysis
    combined_content = "\n\n".join(total_content)
    
    # Model extraction (if API key is provided and valid)
    if api_key and len(combined_content) > 100:
        try:
            ai_result = call_model_with_retry(combined_content, api_key, model, url)
            all_founders.update(ai_result.get("founders", []))
            all_emails.update(ai_result.get("emails", []))
            logger.info(f"{model} extraction successful for {url}")
        except Exception as e:
            logger.warning(f"{model} extraction failed for {url}: {str(e)}. Falling back to regex/structured data.")
    
    # Final validation
    final_founders = [re.sub(r'\s+', ' ', str(founder).strip()) for founder in all_founders if is_valid_name(str(founder))]
    final_emails = [str(email).lower().strip() for email in all_emails if is_valid_email(str(email))]
    
    # Confidence score
    confidence = min(1.0, (len(final_founders) * 0.5 + len(final_emails) * 0.4 + pages_crawled * 0.05))
    
    logger.info(f"Extraction complete for {company_name}: {len(final_founders)} founders, {len(final_emails)} emails, {pages_crawled} pages")
    
    return StartupData(
        url=url,
        name=company_name,
        founders=final_founders,
        emails=final_emails,
        pages_crawled=pages_crawled,
        confidence_score=confidence
    )

async def run_enhanced_crawler(keyword: str, serpapi_key: str, api_key: str, model: str, max_results: int = 20, max_pages_per_site: int = 6) -> pd.DataFrame:
    """Crawler workflow with selected model and fallback to regex."""
    logger.info("Starting crawler workflow")
    
    search_results = search_multiple_engines(keyword, serpapi_key, max_results * 3)
    
    if not search_results:
        st.warning("No suitable startup URLs found.")
        return pd.DataFrame()
    
    st.write(f"🎯 Found {len(search_results)} high-quality startup URLs")
    with st.expander("View Search Results"):
        for i, result in enumerate(search_results[:max_results], 1):
            st.write(f"{i}. **{result['title']}**")
            st.write(f"   URL: {result['url']}")
            st.write(f"   Snippet: {result['snippet'][:100]}...")
            st.write("---")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    extraction_results = []
    successful_extractions = 0
    batch_size = 3
    urls_to_process = [result['url'] for result in search_results[:max_results]]
    
    for i in range(0, len(urls_to_process), batch_size):
        batch_urls = urls_to_process[i:i + batch_size]
        status_text.text(f"Processing batch {i//batch_size + 1}/{len(urls_to_process)//batch_size + 1}")
        
        tasks = [extract_from_multiple_pages(url, api_key, model, max_pages_per_site) for url in batch_urls]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for j, result in enumerate(batch_results):
            current_idx = i + j
            if isinstance(result, Exception):
                logger.error(f"Exception processing URL {batch_urls[j]}: {str(result)}")
                result = StartupData(url=batch_urls[j], name="Error", founders=[], emails=[], pages_crawled=0, confidence_score=0.0)
            
            extraction_results.append(result.model_dump())
            
            if result.founders or result.emails:
                successful_extractions += 1
                with results_container:
                    st.success(f"✅ **{result.name}** - Found {len(result.founders)} founders, {len(result.emails)} emails ({result.pages_crawled} pages)")
            else:
                with results_container:
                    st.info(f"ℹ️ **{result.name}** - No founder data found ({result.pages_crawled} pages)")
            
            progress_bar.progress((current_idx + 1) / len(urls_to_process))
        
        if i + batch_size < len(urls_to_process):
            await asyncio.sleep(2)
    
    status_text.text(f"✅ Completed! Found data for {successful_extractions}/{len(extraction_results)} companies")
    
    df = pd.DataFrame({
        "Company Name": [r["name"] for r in extraction_results],
        "Website URL": [r["url"] for r in extraction_results],
        "Founders": [", ".join(r["founders"]) for r in extraction_results],
        "Founder Emails": [", ".join(r["emails"]) for r in extraction_results],
        "Pages Crawled": [r["pages_crawled"] for r in extraction_results],
        "Confidence Score": [f"{r['confidence_score']:.2f}" for r in extraction_results],
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
    st.set_page_config(page_title="Startup Founder Extractor", page_icon="🚀", layout="wide")
    st.title("🚀 Startup Founder Extractor")
    st.markdown("**Crawler for startup founder data extraction with multiple model support (falls back to regex if API unavailable)**")
    
    if 'extraction_stats' not in st.session_state:
        st.session_state.extraction_stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'total_pages_crawled': 0
        }
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        serpapi_key = st.text_input("🔍 SerpAPI Key", type="password")
        
        model = st.selectbox("🤖 Select Model", ["Gemini", "OpenAI", "DeepSeek"])
        api_key = st.text_input(f"🤖 {model} API Key (optional)", type="password")
        
        keyword = st.text_input("🎯 Search Keywords", value="AI fintech startups 2024")
        
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider("📊 Companies", 5, 20, 10)
        with col2:
            max_pages = st.slider("📄 Pages/Site", 3, 10, 5)
        
        st.subheader("📊 Session Stats")
        stats = st.session_state.extraction_stats
        st.metric("Companies Processed", stats['total_processed'])
        st.metric("Successful Extractions", stats['successful_extractions'])
        st.metric("Pages Crawled", stats['total_pages_crawled'])
        
        if stats['total_processed'] > 0:
            success_rate = (stats['successful_extractions'] / stats['total_processed']) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Start Extraction", type="primary", use_container_width=True):
            if not serpapi_key or not keyword:
                st.error("❌ Please provide SerpAPI key and keywords")
                return
            
            with st.spinner("🔍 Searching and extracting..."):
                start_time = time.time()
                df = asyncio.run(run_enhanced_crawler(keyword, serpapi_key, api_key, model, max_results, max_pages))
                end_time = time.time()
            
            if not df.empty:
                st.success(f"🎉 Extraction completed in {end_time - start_time:.1f} seconds!")
                
                st.session_state.extraction_stats['total_processed'] += len(df)
                st.session_state.extraction_stats['successful_extractions'] += len(df[df['Data Found'] == '✅'])
                st.session_state.extraction_stats['total_pages_crawled'] += df['Pages Crawled'].sum()
                
                st.subheader("📊 Extraction Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Companies", len(df))
                with col2:
                    successful = len(df[df['Data Found'] == '✅'])
                    st.metric("With Data", successful)
                with col3:
                    total_founders = sum(len(founders.split(', ')) if founders else 0 for founders in df['Founders'])
                    st.metric("Total Founders", total_founders)
                with col4:
                    total_emails = sum(len(emails.split(', ')) if emails else 0 for emails in df['Founder Emails'])
                    st.metric("Total Emails", total_emails)
                
                st.subheader("📋 Detailed Results")
                show_filter = st.selectbox("Show results:", ["All companies", "Only successful extractions", "Only failed extractions"])
                
                if show_filter == "Only successful extractions":
                    display_df = df[df['Data Found'] == '✅']
                elif show_filter == "Only failed extractions":
                    display_df = df[df['Data Found'] == '❌']
                else:
                    display_df = df
                
                display_df = display_df.sort_values('Confidence Score', ascending=False)
                st.dataframe(display_df, use_container_width=True, height=400)
                
                st.subheader("📥 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'excel_file' in st.session_state and os.path.exists(st.session_state['excel_file']):
                        with open(st.session_state['excel_file'], "rb") as file:
                            st.download_button(
                                label="📊 Download Excel",
                                data=file,
                                file_name=st.session_state['excel_file'],
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                
                with col2:
                    if 'csv_file' in st.session_state and os.path.exists(st.session_state['csv_file']):
                        with open(st.session_state['csv_file'], "rb") as file:
                            st.download_button(
                                label="📄 Download CSV",
                                data=file,
                                file_name=st.session_state['csv_file'],
                                mime="text/csv",
                                use_container_width=True
                            )
                
                st.subheader("🔍 Company Details")
                successful_companies = df[df['Data Found'] == '✅'].sort_values('Confidence Score', ascending=False)
                
                for _, row in successful_companies.iterrows():
                    with st.expander(f"✅ **{row['Company Name']}** (Confidence: {row['Confidence Score']})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Website:** {row['Website URL']}")
                            st.write(f"**Pages Crawled:** {row['Pages Crawled']}")
                        with col2:
                            if row['Founders']:
                                st.write(f"**Founders:** {row['Founders']}")
                            if row['Founder Emails']:
                                st.write(f"**Emails:** {row['Founder Emails']}")
    
    with col2:
        st.subheader("💡 Tips")
        st.info("Use specific keywords like 'AI fintech startups 2024' for best results. Select a model and provide its API key for enhanced extraction; regex fallback ensures functionality if API is unavailable.")

if __name__ == "__main__":
    main()