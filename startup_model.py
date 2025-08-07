import streamlit as st  # Imports Streamlit for web-based UI
import pandas as pd  # Imports Pandas for DataFrame operations
import os  # Imports os for file system operations
import json  # Imports json for reading/writing JSON data
import asyncio  # Imports asyncio for asynchronous programming
import re  # Imports re for regular expression operations
import logging  # Imports logging for logging messages
from typing import List, Dict, Set  # Imports type hints for clarity
from urllib.parse import urljoin, urlparse  # Imports urljoin for URL construction, urlparse for URL parsing
from bs4 import BeautifulSoup  # Imports BeautifulSoup for HTML parsing
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig  # Imports crawl4ai components for web crawling
from crawl4ai.extraction_strategy import NoExtractionStrategy  # Imports strategy to disable default extraction,,, Tells the web crawler to grab all page content (like HTML or text) without automatically filtering or picking specific parts.
from crawl4ai.chunking_strategy import RegexChunking  # Imports regex-based chunking for content
from openai import AsyncOpenAI  # Imports AsyncOpenAI for async OpenAI API calls
from tenacity import retry, stop_after_attempt, wait_exponential  # Imports retry logic for API calls
from pydantic import BaseModel, Field  # Imports Pydantic for structured data models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Configures logging format
logger = logging.getLogger(__name__)  # Creates logger for this module

# Pydantic model for structured data
class StartupData(BaseModel):  # Defines StartupData model with Pydantic
    url: str = Field(..., description="The website URL of the startup.")  # Required URL field
    name: str = Field(..., description="The name of the startup company.")  # Required name field
    founders: list[str] = Field(default=[], description="List of founder names.")  # Optional founders list
    emails: list[str] = Field(default=[], description="List of email addresses.")  # Optional emails list

# Validation functions
def is_valid_name(name: str) -> bool:  # Validates a person's name
    if not name or not isinstance(name, str):  # Checks if name is empty or not a string
        return False
    name = name.strip()  # Removes whitespace
    if not (3 < len(name) < 50):  # Ensures name length is 4-49 characters
        return False
    words = name.split()  # Splits name into words
    if not (1 < len(words) < 5):  # Ensures 2-4 words
        return False
    if not all(re.match(r'^[A-Z][a-zA-Z\'-.]+$', word) for word in words):  # Ensures each word starts with capital
        return False
    exclude_keywords = ['team', 'about', 'contact', 'support', 'admin', 'sales', 'marketing', 'press', 'company', 'corp', 'inc', 'llc', 'group', 'solutions', 'technologies', 'ventures']  # Keywords to exclude
    if any(keyword in name.lower() for keyword in exclude_keywords):  # Checks for excluded keywords
        return False
    if re.search(r'[0-9@#$%&*+=/\\<>{}[\]|]', name):  # Checks for invalid characters
        return False
    return True  # Returns True if all checks pass

def is_valid_email(email: str) -> bool:  # Validates an email address
    if not email or not isinstance(email, str):  # Checks if email is empty or not a string
        return False
    email = email.strip().lower()  # Removes whitespace, converts to lowercase
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):  # Checks email format
        return False
    exclude_local_parts = ['info', 'contact', 'support', 'sales', 'admin', 'marketing', 'press', 'hello', 'team', 'careers', 'jobs', 'noreply', 'no-reply']  # Excludes generic local parts
    if email.split('@')[0] in exclude_local_parts:  # Checks local part
        return False
    exclude_domains = ['example.com', 'test.com', 'domain.com']  # Excludes test domains
    if email.split('@')[1] in exclude_domains:  # Checks domain
        return False
    return True  # Returns True if all checks pass

def is_startup_website(url: str, title: str = "", snippet: str = "") -> bool:  # Checks if URL is a startup website
    exclude_domains = ['wikipedia.org', 'linkedin.com', 'crunchbase.com', 'forbes.com', 'techcrunch.com', 'medium.com', 'twitter.com', 'facebook.com', 'instagram.com', 'youtube.com', 'github.com', 'reddit.com', 'bloomberg.com', 'reuters.com', 'wsj.com', 'ft.com', 'cnbc.com', 'businessinsider.com', 'venturebeat.com', 'google.com', 'apple.com', 'microsoft.com', 'amazon.com', 'ycombinator.com', 'pitchbook.com', 'apollo.io', 'seniorexecutive.com', 'forumvc.com', 'topstartups.io', 'review.firstround.com', 'mobileappdaily.com', 'startupriders.com', 'ellfound.com', 'eweek.com', 'scribd.com', 'failory.com', 'foundersnetwork.com', 'eranyc.com', 'designrush.com', 'tpalmeragency.com', 'thefinancialtechnologyreport.com', 'startupsavant.com', 'techstars.com', 'ascendixtech.com']  # Excludes non-startup domains
    domain = urlparse(url.lower()).netloc.lstrip('www.')  # Extracts domain without 'www.'
    if any(excluded in domain for excluded in exclude_domains):  # Checks for excluded domains
        return False
    if any(p in urlparse(url.lower()).path for p in ['/blog/', '/news/', '/article/', '/profile/', '/list/', '/directory']):  # Checks for non-homepage paths
        return False
    content = f"{title} {snippet}".lower()  # Combines title and snippet
    startup_indicators = ['startup', 'founded', 'founder', 'co-founder', 'ceo', 'fintech', 'ai', 'biotech', 'venture', 'seed', 'series a', 'series b', 'our team', 'about us', 'our story']  # Startup keywords
    return any(indicator in content for indicator in startup_indicators) and ('.com' in domain or '.ai' in domain or '.io' in domain or '.co' in domain)  # Checks for keywords and domain endings

# SerpAPI search with pagination and caching
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))  # Retries API calls 3 times
async def search_serpapi(query: str, serpapi_key: str, start: int = 0) -> List[Dict]:  # Async search with SerpAPI
    import requests  # Imports requests for HTTP calls
    cache_file = f"serpapi_cache_{hash(query + str(start)) % 10000}.json"  # Generates cache filename
    if os.path.exists(cache_file):  # Checks if cache exists
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:  # Reads cache file
                return json.load(f)  # Returns cached results
        except Exception as e:
            logger.warning(f"Cache read error: {e}")  # Logs cache read error
    try:
        params = {"q": query, "num": 20, "start": start, "api_key": serpapi_key, "gl": "us", "hl": "en"}  # Sets SerpAPI parameters
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)  # Makes HTTP request
        response.raise_for_status()  # Raises exception for failed requests
        results = response.json().get("organic_results", [])  # Extracts organic results
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:  # Writes results to cache
                json.dump(results, f, ensure_ascii=False, indent=2)  # Saves as JSON
        except Exception as e:
            logger.warning(f"Cache write error: {e}")  # Logs cache write error
        return results  # Returns search results
    except Exception as e:
        logger.error(f"SerpAPI error for query '{query}' at start={start}: {e}")  # Logs API error
        return []  # Returns empty list on failure

async def discover_startups(keyword: str, serpapi_key: str, num_results: int = 20) -> Dict[str, str]:  # Discovers startup websites
    cache_file = f"startups_cache_{hash(keyword) % 10000}.json"  # Generates cache filename
    if os.path.exists(cache_file):  # Checks if cache exists
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:  # Reads cache
                cached = json.load(f)  # Loads cached startups
                if len(cached) >= num_results:  # Checks if enough results
                    return dict(list(cached.items())[:num_results])  # Returns limited results
        except Exception as e:
            logger.warning(f"Startup cache read error: {e}")  # Logs cache read error
    queries = [  # Defines search queries for startups
        f'"{keyword}" startup founders (site:*.com | site:*.ai | site:*.io | site:*.co) -inurl:(crunchbase.com linkedin.com ...)',
        f'"{keyword}" startup "about us" founders team',
        f'top "{keyword}" startups 2024 2025 site:*.com | site:*.ai | site:*.io | site:*.co -inurl:(list directory)',
        f'"{keyword}" startup company homepage -inurl:(list directory)',
        f'"{keyword}" fintech AI startup founders contact'
    ]
    startups = {}  # Initializes startups dictionary
    seen_domains = set()  # Tracks seen domains
    async def process_query(query: str, start: int):  # Processes a single query
        results = await search_serpapi(query, serpapi_key, start)  # Fetches search results
        for res in results:  # Iterates over results
            url = res.get("link", "")  # Gets URL
            title = res.get("title", "").split('|')[0].strip()  # Gets title before '|'
            snippet = res.get("snippet", "")  # Gets snippet
            if url and is_startup_website(url, title, snippet):  # Checks if startup website
                domain = urlparse(url).netloc.lstrip('www.')  # Extracts domain
                if domain not in seen_domains:  # Ensures domain is unique
                    startups[domain] = title or "Unknown"  # Adds to startups
                    seen_domains.add(domain)  # Marks domain as seen
    tasks = []  # Initializes task list
    for query in queries:  # Iterates over queries
        for start in [0, 20, 40]:  # Paginates results
            tasks.append(process_query(query, start))  # Adds query task
            if len(startups) >= num_results:  # Stops if enough startups
                break
        if len(startups) >= num_results:  # Stops outer loop
            break
    await asyncio.gather(*tasks, return_exceptions=True)  # Runs tasks concurrently
    if len(startups) < num_results:  # Checks if more startups needed
        list_query = f'top "{keyword}" startups list 2024 2025 -inurl:(crunchbase.com linkedin.com ...)'  # Query for startup lists
        list_results = await search_serpapi(list_query, serpapi_key)  # Fetches list results
        if list_results:  # If results exist
            for list_result in list_results[:2]:  # Processes top 2 results
                list_url = list_result.get("link", "")  # Gets list URL
                async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:  # Initializes crawler
                    result = await crawler.arun(list_url)  # Crawls list page
                    if result and result.markdown:  # If crawling successful
                        prompt = f"Extract a list of startup company names from the following text. Return ONLY a JSON list of strings, e.g., [\"Company A\", \"Company B\"].\n\n{result.markdown[:8000]}"  # Prompt for company names
                        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", st.session_state.get("openai_key", "")))  # Initializes OpenAI client
                        response = await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})  # Calls OpenAI
                        company_names = json.loads(response.choices[0].message.content).get("names", [])  # Parses company names
                        for name in company_names[:num_results - len(startups)]:  # Processes needed names
                            homepage_query = f'"{name}" official website homepage'  # Query for homepage
                            homepage_results = await search_serpapi(homepage_query, serpapi_key)  # Fetches homepage results
                            if homepage_results:  # If results exist
                                url = homepage_results[0].get("link", "")  # Gets first URL
                                if url and is_startup_website(url):  # Checks if startup website
                                    domain = urlparse(url).netloc.lstrip('www.')  # Extracts domain
                                    if domain not in seen_domains:  # Ensures unique domain
                                        startups[domain] = name  # Adds to startups
                                        seen_domains.add(domain)  # Marks domain as seen
                            await asyncio.sleep(0.5)  # Pauses to avoid API overload
                if len(startups) >= num_results:  # Stops if enough startups
                    break
    startups = dict(list(startups.items())[:num_results])  # Limits to num_results
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:  # Writes to cache
            json.dump(startups, f, ensure_ascii=False, indent=2)  # Saves as JSON
    except Exception as e:
        logger.warning(f"Startup cache write error: {e}")  # Logs cache write error
    return startups  # Returns startups dictionary

async def find_relevant_pages(url: str) -> List[str]:  # Finds relevant pages on website
    relevant_pages = []  # Initializes list for relevant pages
    config = BrowserConfig(headless=True, browser_type="chromium", viewport_width=1920, viewport_height=1080, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", wait_for_load=True, wait_for_idle=True, timeout=30)  # Configures crawler
    try:
        async with AsyncWebCrawler(config=config) as crawler:  # Initializes crawler
            run_config = CrawlerRunConfig(word_count_threshold=50, extraction_strategy=NoExtractionStrategy(), chunking_strategy=RegexChunking(), css_selector="body", max_retries=2)  # Configures crawler run
            result = await crawler.arun(url, config=run_config)  # Crawls URL
            if not result.success or not result.html:  # Checks for crawl failure
                logger.warning(f"Failed to crawl {url}")  # Logs warning
                return relevant_pages  # Returns empty list
            soup = BeautifulSoup(result.html, 'html.parser')  # Parses HTML
            about_keywords = ['about', 'team', 'leadership', 'founders', 'our-story', 'who-we-are', 'management', 'executive', 'company', 'people', 'our-team', 'meet-us', 'story']  # Keywords for relevant pages
            scored_links = []  # Initializes list for scored links
            for link in soup.find_all('a', href=True):  # Iterates over links
                href = link.get('href', '').strip().lower()  # Gets href
                text = link.get_text(strip=True).lower()  # Gets link text
                if not href:  # Skips empty hrefs
                    continue
                full_url = urljoin(url, href)  # Constructs full URL
                if not full_url.startswith(('http://', 'https://')) or urlparse(full_url).netloc != urlparse(url).netloc:  # Skips invalid or external URLs
                    continue
                skip_patterns = ['blog', 'news', 'careers', 'jobs', 'press', 'media', 'login', 'signup', 'privacy', 'terms']  # Patterns to skip
                if any(pattern in href for pattern in skip_patterns):  # Skips irrelevant links
                    continue
                score = 0  # Initializes score
                link_content = f"{href} {text}"  # Combines href and text
                for keyword in about_keywords:  # Scores based on keywords
                    if keyword in link_content:
                        score += 12 if keyword == text else 8 if keyword in text else 5  # Assigns scores
                if score > 0:  # Adds scored links
                    scored_links.append((full_url, score))
            scored_links.sort(key=lambda x: x[1], reverse=True)  # Sorts by score
            relevant_pages.extend([page_url for page_url, _ in scored_links[:5]])  # Adds top 5 links
            main_content = soup.get_text().lower()  # Extracts page text
            if any(keyword in main_content for keyword in ['founder', 'co-founder', 'ceo', 'team']):  # Checks for founder keywords
                relevant_pages.insert(0, url)  # Adds homepage if relevant
    except Exception as e:
        logger.error(f"Error finding pages for {url}: {e}")  # Logs error
    return list(dict.fromkeys(relevant_pages))[:6]  # Returns up to 6 unique pages

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))  # Retries OpenAI calls
async def extract_with_openai(text: str, url: str, company_name: str, api_key: str, model: str) -> Dict:  # Extracts data with OpenAI
    client = AsyncOpenAI(api_key=api_key)  # Initializes OpenAI client
    max_chars = 16000  # Sets max input length
    if len(text) > max_chars:  # Checks if text exceeds limit
        text = text[:max_chars] + "..."  # Truncates text
    prompt = f"""
You are an expert at extracting founder information from startup websites for '{company_name}' at {url}.
TASK: Extract founder names and their email addresses with high precision.
INSTRUCTIONS:
- Extract ONLY human names (first + last, or first + middle + last) of founders or CEOs.
- Look for explicit titles: Founder, Co-Founder, CEO, Chief Executive Officer.
- GOOD names: "John Smith", "Sarah J. Wilson", "Dr. Mike Chen"
- BAD names: "CEO", "The Team", "Company Inc.", "Leadership", generic terms, or company names.
- Names must have proper capitalization (e.g., "John Smith", not "john smith").
- Extract ONLY personal emails likely belonging to founders, matching the company domain or related domains.
- GOOD emails: "john@company.com", "sarah.w@startup.ai"
- BAD emails: "info@", "support@", "contact@", "sales@", or emails from unrelated domains (e.g., gmail.com, yahoo.com).
- Ignore domains: example.com, test.com, domain.com.
- Return JSON: {{"founders": [], "emails": []}}
- Ensure names and emails are contextually linked (e.g., email near a founder's name).
CONTENT:
{text}
JSON:"""  # Defines OpenAI prompt
    try:
        response = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=600)  # Calls OpenAI
        content = response.choices[0].message.content.strip()  # Extracts response
        content = re.sub(r'```json\s*|\s*```', '', content).strip()  # Cleans JSON markers
        data = json.loads(content)  # Parses JSON
        return {
            "founders": [name.strip() for name in data.get("founders", []) if is_valid_name(name)],  # Validates founders
            "emails": [email.lower().strip() for email in data.get("emails", []) if is_valid_email(email)]  # Validates emails
        }
    except Exception as e:
        logger.error(f"OpenAI error for {url}: {e}")  # Logs error
        return {"founders": [], "emails": []}  # Returns empty results

async def extract_emails_from_contact_page(url: str, api_key: str, model: str) -> List[str]:  # Extracts emails from contact page
    config = BrowserConfig(headless=True, browser_type="chromium", viewport_width=1920, viewport_height=1080, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", wait_for_load=True, wait_for_idle=True, timeout=30)  # Configures crawler
    try:
        async with AsyncWebCrawler(config=config) as crawler:  # Initializes crawler
            run_config = CrawlerRunConfig(word_count_threshold=50, extraction_strategy=NoExtractionStrategy(), chunking_strategy=RegexChunking(), css_selector="body", max_retries=2)  # Configures crawler run
            result = await crawler.arun(url, config=run_config)  # Crawls contact page
            if not result.success or not result.markdown:  # Checks for failure
                return []  # Returns empty list
            prompt = f"""
Extract personal email addresses from the contact page content for a startup. Ignore generic emails (e.g., info@, contact@, support@).
Emails must likely belong to founders or executives and match the company domain or related domains.
Exclude emails from unrelated domains (e.g., gmail.com, yahoo.com).
Return JSON: {{"emails": []}}
CONTENT:
{result.markdown[:8000]}
JSON:"""  # Defines prompt for emails
            client = AsyncOpenAI(api_key=api_key)  # Initializes OpenAI client
            response = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=250)  # Calls OpenAI
            content = response.choices[0].message.content.strip()  # Extracts response
            content = re.sub(r'```json\s*|\s*```', '', content).strip()  # Cleans JSON
            data = json.loads(content)  # Parses JSON
            return [email.lower().strip() for email in data.get("emails", []) if is_valid_email(email)]  # Returns validated emails
    except Exception as e:
        logger.warning(f"Error extracting emails from contact page {url}: {e}")  # Logs error
        return []  # Returns empty list

def extract_with_regex(text: str, html: str = "") -> Dict:  # Extracts data with regex
    founders = set()  # Initializes set for founders
    emails = set()  # Initializes set for emails
    all_content = [text, html] if html else [text]  # Combines text and HTML
    founder_patterns = [  # Regex patterns for founders
        r'(?:founded by|co-founded by|started by|ceo)[:\s,]+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)',
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[,\s]*(?:founder|co-founder|ceo|chief executive)',
        r'<h[1-6][^>]*>([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)</h[1-6]>[^<]*(?:founder|co-founder|ceo)',
        r'meet\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[,\s]*(?:founder|co-founder|ceo)',
    ]
    email_patterns = [  # Regex patterns for emails
        r'\b([a-zA-Z0-9][a-zA-Z0-9._%+-]{0,50}@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
        r'mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        r'(?:email|contact)[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
    ]
    for content in all_content:  # Iterates over content
        if not content:  # Skips empty content
            continue
        for pattern in founder_patterns:  # Applies founder patterns
            for match in re.findall(pattern, content, re.IGNORECASE | re.MULTILINE):  # Finds matches
                name = match.strip() if isinstance(match, str) else match[0].strip()  # Extracts name
                if is_valid_name(name):  # Validates name
                    founders.add(name)  # Adds to set
        for pattern in email_patterns:  # Applies email patterns
            for email in re.findall(pattern, content, re.IGNORECASE):  # Finds matches
                email = email.strip().lower()  # Cleans email
                if is_valid_email(email):  # Validates email
                    emails.add(email)  # Adds to set
    return {"founders": list(founders), "emails": list(emails)}  # Returns extracted data

async def process_startup(url: str, name: str, openai_key: str, model: str) -> StartupData:  # Processes a single startup
    logger.info(f"Processing: {url} ({name})")  # Logs processing start
    startup_data = StartupData(url=f"https://{url}", name=name)  # Initializes StartupData
    try:
        relevant_pages = await find_relevant_pages(f"https://{url}")  # Finds relevant pages
        all_founders = set()  # Initializes set for founders
        all_emails = set()  # Initializes set for emails
        config = BrowserConfig(headless=True, browser_type="chromium", viewport_width=1920, viewport_height=1080, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", wait_for_load=True, wait_for_idle=True, timeout=30)  # Configures crawler
        async with AsyncWebCrawler(config=config) as crawler:  # Initializes crawler
            async def crawl_page(page_url: str):  # Crawls a single page
                try:
                    run_config = CrawlerRunConfig(word_count_threshold=50, extraction_strategy=NoExtractionStrategy(), chunking_strategy=RegexChunking(), css_selector="body", max_retries=2)  # Configures crawler run
                    result = await crawler.arun(page_url, config=run_config)  # Crawls page
                    if not result.success:  # Checks for failure
                        return {"founders": [], "emails": []}, None  # Returns empty results
                    page_data = {"founders": [], "emails": []}  # Initializes page data
                    if page_url == f"https://{url}" and result.html:  # Checks if homepage
                        soup = BeautifulSoup(result.html, 'html.parser')  # Parses HTML
                        title = soup.find('title')  # Finds title tag
                        if title and title.text.strip():  # Checks for valid title
                            startup_data.name = title.text.strip()[:100]  # Updates name
                    if result.markdown or result.html:  # Checks for content
                        ai_result = await extract_with_openai(result.markdown or "", page_url, startup_data.name, openai_key, model)  # Extracts with OpenAI
                        page_data["founders"].extend(ai_result["founders"])  # Adds founders
                        page_data["emails"].extend(ai_result["emails"])  # Adds emails
                        regex_result = extract_with_regex(result.markdown or "", result.html or "")  # Extracts with regex
                        page_data["founders"].extend(regex_result["founders"])  # Adds founders
                        page_data["emails"].extend(regex_result["emails"])  # Adds emails
                    return page_data, result.html  # Returns page data and HTML
                except Exception as e:
                    logger.warning(f"Error processing page {page_url}: {e}")  # Logs error
                    return {"founders": [], "emails": []}, None  # Returns empty results
            async def crawl_contact_page(contact_path: str):  # Crawls contact page
                contact_url = f"https://{url}{contact_path}"  # Constructs contact URL
                contact_emails = await extract_emails_from_contact_page(contact_url, openai_key, model)  # Extracts emails
                return contact_emails  # Returns emails
            page_tasks = [crawl_page(page_url) for page_url in relevant_pages]  # Creates page tasks
            contact_tasks = [crawl_contact_page(path) for path in ['/contact', '/contact-us', '/get-in-touch', '/connect']]  # Creates contact page tasks
            page_results = await asyncio.gather(*page_tasks, return_exceptions=True)  # Runs page tasks
            contact_results = await asyncio.gather(*contact_tasks, return_exceptions=True)  # Runs contact tasks
            for result, html in page_results:  # Processes page results
                if isinstance(result, dict):  # Checks if valid result
                    all_founders.update(result["founders"])  # Adds founders
                    all_emails.update(result["emails"])  # Adds emails
            for result in contact_results:  # Processes contact results
                if isinstance(result, list):  # Checks if valid result
                    all_emails.update(result)  # Adds emails
        startup_data.founders = sorted(list(all_founders))  # Sets sorted founders
        startup_data.emails = sorted(list(all_emails))  # Sets sorted emails
        logger.info(f"Extracted {len(startup_data.founders)} founders and {len(startup_data.emails)} emails from {url}")  # Logs results
    except Exception as e:
        logger.error(f"Error processing startup {url}: {e}")  # Logs error
    return startup_data  # Returns StartupData

async def run_extraction_workflow(keyword: str, serpapi_key: str, openai_key: str, model: str, max_results: int) -> pd.DataFrame:  # Runs extraction workflow
    st.info("🚀 Starting startup discovery...")  # Displays start message
    os.environ['OPENAI_API_KEY'] = openai_key  # Sets OpenAI API key
    startups = await discover_startups(keyword, serpapi_key, max_results)  # Discovers startups
    if not startups:  # Checks if no startups found
        st.warning("No startup websites found. Try a different keyword.")  # Displays warning
        return pd.DataFrame()  # Returns empty DataFrame
    st.success(f"Found {len(startups)} potential startup websites.")  # Displays success message
    progress_bar = st.progress(0)  # Initializes progress bar
    status_text = st.empty()  # Initializes status text
    results = []  # Initializes results list
    async def process_batch(batch: List[tuple]):  # Processes a batch of startups
        batch_tasks = [process_startup(domain, name, openai_key, model) for domain, name in batch]  # Creates batch tasks
        return await asyncio.gather(*batch_tasks, return_exceptions=True)  # Runs tasks
    batch_size = 5  # Sets batch size
    for i in range(0, len(startups), batch_size):  # Processes in batches
        batch = list(startups.items())[i:i + batch_size]  # Selects batch
        status_text.text(f"Processing batch {i//batch_size + 1}/{(len(startups) + batch_size - 1)//batch_size}")  # Updates status
        batch_results = await process_batch(batch)  # Processes batch
        for j, startup_data in enumerate(batch_results):  # Processes results
            if isinstance(startup_data, StartupData):  # Checks if valid result
                results.append(startup_data)  # Adds to results
                if startup_data.founders or startup_data.emails:  # Checks for data
                    st.success(f"✅ {startup_data.name}: {len(startup_data.founders)} founders, {len(startup_data.emails)} emails")  # Displays success
                else:
                    st.info(f"ℹ️ {startup_data.name}: No data found")  # Displays info
        progress_bar.progress(min((i + batch_size) / len(startups), 1.0))  # Updates progress
        await asyncio.sleep(1)  # Pauses between batches
    status_text.text("✅ Extraction completed!")  # Displays completion
    df_data = []  # Initializes DataFrame data
    for data in results:  # Formats results for DataFrame
        df_data.append({
            "Company Name": data.name,
            "Website URL": data.url,
            "Founders": ", ".join(data.founders) if data.founders else "",
            "Founder Emails": ", ".join(data.emails) if data.emails else "",
            "Data Found": "✅" if (data.founders or data.emails) else "❌"
        })
    df = pd.DataFrame(df_data)  # Creates DataFrame
    if not df.empty:  # Checks if DataFrame has data
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")  # Generates timestamp
        csv_filename = f"startup_founders_{timestamp}.csv"  # Sets CSV filename
        excel_filename = f"startup_founders_{timestamp}.xlsx"  # Sets Excel filename
        try:
            df.to_csv(csv_filename, index=False)  # Saves to CSV
            df.to_excel(excel_filename, index=False, engine='openpyxl')  # Saves to Excel
            st.session_state['csv_file'] = csv_filename  # Stores CSV filename
            st.session_state['excel_file'] = excel_filename  # Stores Excel filename
        except Exception as e:
            logger.error(f"Error saving files: {e}")  # Logs error
            st.error(f"Failed to save results: {e}")  # Displays error
    return df  # Returns DataFrame

def main():  # Defines main Streamlit app
    st.set_page_config(page_title="Startup Founder Crawler", layout="wide")  # Configures page
    st.title("🚀 Startup Founder & Email Crawler")  # Sets title
    st.write("Extract founder names and email addresses from startup websites using SerpAPI, Crawl4AI, and OpenAI")  # Displays description
    with st.sidebar:  # Creates sidebar
        st.header("Configuration")  # Adds header
        serpapi_key = st.text_input("SerpAPI Key", type="password", help="Get from serpapi.com")  # Input for SerpAPI key
        openai_key = st.text_input("OpenAI API Key", type="password", help="Get from platform.openai.com")  # Input for OpenAI key
        model = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], index=0)  # Dropdown for model
        keyword = st.text_input("Search Keyword", value="AI startups 2024 founders site:*.com | site:*.ai | site:*.io | site:*.co", help="Keywords to search for startups (e.g., 'AI startups 2024 founders')")  # Input for keyword
        max_results = st.slider("Max Results", 5, 50, 20)  # Slider for max results
        st.markdown("---")  # Adds separator
        st.markdown("**Instructions:**")  # Adds instructions header
        st.markdown("1. Enter your API keys")  # Instruction 1
        st.markdown("2. Specify search keywords")  # Instruction 2
        st.markdown("3. Click 'Start Extraction'")  # Instruction 3
        st.markdown("4. Download results as CSV/Excel")  # Instruction 4
    col1, col2 = st.columns([3, 1])  # Creates two columns
    with col2:  # Smaller column
        if st.button(" Test APIs", type="secondary"):  # Test APIs button
            with st.spinner("Testing APIs..."):  # Shows spinner
                if serpapi_key:  # Checks for SerpAPI key
                    try:
                        import requests  # Imports requests
                        response = requests.get("https://serpapi.com/search", params={"q": "test", "api_key": serpapi_key, "num": 1}, timeout=10)  # Tests SerpAPI
                        if response.status_code == 200:  # Checks for success
                            st.success("✅ SerpAPI working")  # Displays success
                        else:
                            st.error(f"❌ SerpAPI failed: {response.status_code}")  # Displays error
                    except Exception as e:
                        st.error(f"❌ SerpAPI error: {e}")  # Displays error
                else:
                    st.warning("SerpAPI key missing")  # Displays warning
                if openai_key:  # Checks for OpenAI key
                    try:
                        async def test_openai():  # Tests OpenAI
                            client = AsyncOpenAI(api_key=openai_key)  # Initializes client
                            response = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Say 'test'"}], max_tokens=5)  # Makes test call
                            return "success"  # Returns success
                        result = asyncio.run(test_openai())  # Runs test
                        st.success("✅ OpenAI working")  # Displays success
                    except Exception as e:
                        st.error(f"❌ OpenAI error: {e}")  # Displays error
                else:
                    st.warning("OpenAI key missing")  # Displays warning
    with col1:  # Larger column
        if st.button("🚀 Start Extraction", type="primary", use_container_width=True):  # Start Extraction button
            if not serpapi_key:  # Checks for SerpAPI key
                st.error("❌ SerpAPI key is required")  # Displays error
            elif not openai_key:  # Checks for OpenAI key
                st.error("❌ OpenAI API key is required")  # Displays error
            else:
                with st.spinner("🔍 Extracting founder data..."):  # Shows spinner
                    df = asyncio.run(run_extraction_workflow(keyword, serpapi_key, openai_key, model, max_results))  # Runs workflow
                    if not df.empty:  # Checks for results
                        st.success(f"✅ Extraction completed! Found data for {len(df[df['Data Found'] == '✅'])} companies.")  # Displays success
                        st.subheader("Results")  # Adds subheader
                        st.dataframe(df, use_container_width=True)  # Displays DataFrame
                        col1, col2 = st.columns(2)  # Creates two columns
                        if 'csv_file' in st.session_state:  # Checks for CSV file
                            with col1:
                                with open(st.session_state['csv_file'], "rb") as f:  # Opens CSV
                                    st.download_button(label="Download CSV", data=f.read(), file_name=st.session_state['csv_file'], mime="text/csv", use_container_width=True)  # Adds CSV download button
                        if 'excel_file' in st.session_state:  # Checks for Excel file
                            with col2:
                                with open(st.session_state['excel_file'], "rb") as f:  # Opens Excel
                                    st.download_button(label=" Download Excel", data=f.read(), file_name=st.session_state['excel_file'], mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)  # Adds Excel download button
                    else:
                        st.warning(" No results found. Try different keywords or check your API keys.")  # Displays warning

if __name__ == "__main__":  # Checks if script is run directly
    main()  # Runs main function