import streamlit as st
import pandas as pd
import os
import json
import asyncio
import re
import logging
from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import NoExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic model for structured data


class StartupData(BaseModel):
    url: str = Field(..., description="The website URL of the startup.")
    name: str = Field(..., description="The name of the startup company.")
    founders: list[str] = Field(
        default=[], description="List of founder names.")
    emails: list[str] = Field(
        default=[], description="List of email addresses.")

# Validation functions


def is_valid_name(name: str) -> bool:
    if not name or not isinstance(name, str):
        return False
    name = name.strip()
    if not (3 < len(name) < 50):
        return False
    words = name.split()
    if not (1 < len(words) < 5):
        return False
    if not all(re.match(r'^[A-Z][a-zA-Z\'-.]+$', word) for word in words):
        return False
    exclude_keywords = [
        'team', 'about', 'contact', 'support', 'admin', 'sales', 'marketing', 'press',
        'company', 'corp', 'inc', 'llc', 'group', 'solutions', 'technologies', 'ventures'
    ]
    if any(keyword in name.lower() for keyword in exclude_keywords):
        return False
    if re.search(r'[0-9@#$%&*+=/\\<>{}[\]|]', name):
        return False
    return True


def is_valid_email(email: str) -> bool:
    if not email or not isinstance(email, str):
        return False
    email = email.strip().lower()
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False
    exclude_local_parts = [
        'info', 'contact', 'support', 'sales', 'admin', 'marketing', 'press', 'hello',
        'team', 'careers', 'jobs', 'noreply', 'no-reply'
    ]
    if email.split('@')[0] in exclude_local_parts:
        return False
    exclude_domains = ['example.com', 'test.com', 'domain.com']
    if email.split('@')[1] in exclude_domains:
        return False
    return True


def is_startup_website(url: str, title: str = "", snippet: str = "") -> bool:
    exclude_domains = [
        'wikipedia.org', 'linkedin.com', 'crunchbase.com', 'forbes.com', 'techcrunch.com',
        'medium.com', 'twitter.com', 'facebook.com', 'instagram.com', 'youtube.com',
        'github.com', 'reddit.com', 'bloomberg.com', 'reuters.com', 'wsj.com', 'ft.com',
        'cnbc.com', 'businessinsider.com', 'venturebeat.com', 'google.com', 'apple.com',
        'microsoft.com', 'amazon.com', 'ycombinator.com', 'pitchbook.com', 'apollo.io',
        'seniorexecutive.com', 'forumvc.com', 'topstartups.io', 'review.firstround.com',
        'mobileappdaily.com', 'startupriders.com', 'ellfound.com', 'eweek.com', 'scribd.com',
        'failory.com', 'foundersnetwork.com', 'eranyc.com', 'designrush.com', 'tpalmeragency.com',
        'thefinancialtechnologyreport.com', 'startupsavant.com', 'techstars.com', 'ascendixtech.com'
    ]
    domain = urlparse(url.lower()).netloc.lstrip('www.')
    if any(excluded in domain for excluded in exclude_domains):
        return False
    if any(p in urlparse(url.lower()).path for p in ['/blog/', '/news/', '/article/', '/profile/', '/list/', '/directory']):
        return False
    content = f"{title} {snippet}".lower()
    startup_indicators = [
        'startup', 'founded', 'founder', 'co-founder', 'ceo', 'fintech', 'ai', 'biotech',
        'venture', 'seed', 'series a', 'series b', 'our team', 'about us', 'our story'
    ]
    return any(indicator in content for indicator in startup_indicators) and ('.com' in domain or '.ai' in domain or '.io' in domain or '.co' in domain)

# SerpAPI search with pagination and caching


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
async def search_serpapi(query: str, serpapi_key: str, start: int = 0) -> List[Dict]:
    import requests
    cache_file = f"serpapi_cache_{hash(query + str(start)) % 10000}.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

    try:
        params = {
            "q": query,
            "num": 20,
            "start": start,
            "api_key": serpapi_key,
            "gl": "us",
            "hl": "en"
        }
        response = requests.get(
            "https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        results = response.json().get("organic_results", [])
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
        return results
    except Exception as e:
        logger.error(
            f"SerpAPI error for query '{query}' at start={start}: {e}")
        return []


async def discover_startups(keyword: str, serpapi_key: str, num_results: int = 20) -> Dict[str, str]:
    cache_file = f"startups_cache_{hash(keyword) % 10000}.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                if len(cached) >= num_results:
                    return dict(list(cached.items())[:num_results])
        except Exception as e:
            logger.warning(f"Startup cache read error: {e}")

    queries = [
        f'"{keyword}" startup founders (site:*.com | site:*.ai | site:*.io | site:*.co) -inurl:(crunchbase.com linkedin.com wikipedia.org review.firstround.com topstartups.io seniorexecutive.com forumvc.com mobileappdaily.com startupriders.com ellfound.com eweek.com scribd.com failory.com foundersnetwork.com eranyc.com designrush.com tpalmeragency.com thefinancialtechnologyreport.com startupsavant.com techstars.com ascendixtech.com)',
        f'"{keyword}" startup "about us" founders team',
        f'top "{keyword}" startups 2024 2025 site:*.com | site:*.ai | site:*.io | site:*.co -inurl:(list directory)',
        f'"{keyword}" startup company homepage -inurl:(list directory)',
        f'"{keyword}" fintech AI startup founders contact'
    ]
    startups = {}
    seen_domains = set()

    async def process_query(query: str, start: int):
        results = await search_serpapi(query, serpapi_key, start)
        for res in results:
            url = res.get("link", "")
            title = res.get("title", "").split('|')[0].strip()
            snippet = res.get("snippet", "")
            if url and is_startup_website(url, title, snippet):
                domain = urlparse(url).netloc.lstrip('www.')
                if domain not in seen_domains:
                    startups[domain] = title or "Unknown"
                    seen_domains.add(domain)

    tasks = []
    for query in queries:
        for start in [0, 20, 40]:
            tasks.append(process_query(query, start))
            if len(startups) >= num_results:
                break
        if len(startups) >= num_results:
            break
    await asyncio.gather(*tasks, return_exceptions=True)

    if len(startups) < num_results:
        list_query = f'top "{keyword}" startups list 2024 2025 -inurl:(crunchbase.com linkedin.com wikipedia.org review.firstround.com topstartups.io seniorexecutive.com forumvc.com mobileappdaily.com startupriders.com ellfound.com eweek.com scribd.com failory.com foundersnetwork.com eranyc.com designrush.com tpalmeragency.com thefinancialtechnologyreport.com startupsavant.com techstars.com ascendixtech.com)'
        list_results = await search_serpapi(list_query, serpapi_key)
        if list_results:
            for list_result in list_results[:2]:
                list_url = list_result.get("link", "")
                async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
                    result = await crawler.arun(list_url)
                    if result and result.markdown:
                        prompt = f"Extract a list of startup company names from the following text. Return ONLY a JSON list of strings, e.g., [\"Company A\", \"Company B\"].\n\n{result.markdown[:8000]}"
                        client = AsyncOpenAI(api_key=os.environ.get(
                            "OPENAI_API_KEY", st.session_state.get("openai_key", "")))
                        response = await client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"}
                        )
                        company_names = json.loads(
                            response.choices[0].message.content).get("names", [])
                        for name in company_names[:num_results - len(startups)]:
                            homepage_query = f'"{name}" official website homepage'
                            homepage_results = await search_serpapi(homepage_query, serpapi_key)
                            if homepage_results:
                                url = homepage_results[0].get("link", "")
                                if url and is_startup_website(url):
                                    domain = urlparse(
                                        url).netloc.lstrip('www.')
                                    if domain not in seen_domains:
                                        startups[domain] = name
                                        seen_domains.add(domain)
                            await asyncio.sleep(0.5)
                if len(startups) >= num_results:
                    break

    startups = dict(list(startups.items())[:num_results])
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(startups, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Startup cache write error: {e}")
    return startups


async def find_relevant_pages(url: str) -> List[str]:
    relevant_pages = []
    config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        viewport_width=1920,
        viewport_height=1080,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        wait_for_load=True,
        wait_for_idle=True,
        timeout=30
    )
    try:
        async with AsyncWebCrawler(config=config) as crawler:
            run_config = CrawlerRunConfig(
                word_count_threshold=50,
                extraction_strategy=NoExtractionStrategy(),
                chunking_strategy=RegexChunking(),
                css_selector="body",
                max_retries=2
            )
            result = await crawler.arun(url, config=run_config)
            if not result.success or not result.html:
                logger.warning(f"Failed to crawl {url}")
                return relevant_pages
            soup = BeautifulSoup(result.html, 'html.parser')
            about_keywords = [
                'about', 'team', 'leadership', 'founders', 'our-story', 'who-we-are',
                'management', 'executive', 'company', 'people', 'our-team', 'meet-us', 'story'
            ]
            scored_links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').strip().lower()
                text = link.get_text(strip=True).lower()
                if not href:
                    continue
                full_url = urljoin(url, href)
                if not full_url.startswith(('http://', 'https://')) or urlparse(full_url).netloc != urlparse(url).netloc:
                    continue
                skip_patterns = ['blog', 'news', 'careers', 'jobs',
                                 'press', 'media', 'login', 'signup', 'privacy', 'terms']
                if any(pattern in href for pattern in skip_patterns):
                    continue
                score = 0
                link_content = f"{href} {text}"
                for keyword in about_keywords:
                    if keyword in link_content:
                        score += 12 if keyword == text else 8 if keyword in text else 5
                if score > 0:
                    scored_links.append((full_url, score))
            scored_links.sort(key=lambda x: x[1], reverse=True)
            relevant_pages.extend(
                [page_url for page_url, _ in scored_links[:5]])
            main_content = soup.get_text().lower()
            if any(keyword in main_content for keyword in ['founder', 'co-founder', 'ceo', 'team']):
                relevant_pages.insert(0, url)
    except Exception as e:
        logger.error(f"Error finding pages for {url}: {e}")
    return list(dict.fromkeys(relevant_pages))[:6]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def extract_with_openai(text: str, url: str, company_name: str, api_key: str, model: str) -> Dict:
    client = AsyncOpenAI(api_key=api_key)
    max_chars = 16000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
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
JSON:"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r'```json\s*|\s*```', '', content).strip()
        data = json.loads(content)
        return {
            "founders": [name.strip() for name in data.get("founders", []) if is_valid_name(name)],
            "emails": [email.lower().strip() for email in data.get("emails", []) if is_valid_email(email)]
        }
    except Exception as e:
        logger.error(f"OpenAI error for {url}: {e}")
        return {"founders": [], "emails": []}


async def extract_emails_from_contact_page(url: str, api_key: str, model: str) -> List[str]:
    config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        viewport_width=1920,
        viewport_height=1080,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        wait_for_load=True,
        wait_for_idle=True,
        timeout=30
    )
    try:
        async with AsyncWebCrawler(config=config) as crawler:
            run_config = CrawlerRunConfig(
                word_count_threshold=50,
                extraction_strategy=NoExtractionStrategy(),
                chunking_strategy=RegexChunking(),
                css_selector="body",
                max_retries=2
            )
            result = await crawler.arun(url, config=run_config)
            if not result.success or not result.markdown:
                return []
            prompt = f"""
Extract personal email addresses from the contact page content for a startup. Ignore generic emails (e.g., info@, contact@, support@).
Emails must likely belong to founders or executives and match the company domain or related domains.
Exclude emails from unrelated domains (e.g., gmail.com, yahoo.com).
Return JSON: {{"emails": []}}
CONTENT:
{result.markdown[:8000]}
JSON:"""
            client = AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250
            )
            content = response.choices[0].message.content.strip()
            content = re.sub(r'```json\s*|\s*```', '', content).strip()
            data = json.loads(content)
            return [email.lower().strip() for email in data.get("emails", []) if is_valid_email(email)]
    except Exception as e:
        logger.warning(f"Error extracting emails from contact page {url}: {e}")
        return []


def extract_with_regex(text: str, html: str = "") -> Dict:
    founders = set()
    emails = set()
    all_content = [text, html] if html else [text]
    founder_patterns = [
        r'(?:founded by|co-founded by|started by|ceo)[:\s,]+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)',
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[,\s]*(?:founder|co-founder|ceo|chief executive)',
        r'<h[1-6][^>]*>([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)</h[1-6]>[^<]*(?:founder|co-founder|ceo)',
        r'meet\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[,\s]*(?:founder|co-founder|ceo)',
    ]
    email_patterns = [
        r'\b([a-zA-Z0-9][a-zA-Z0-9._%+-]{0,50}@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
        r'mailto:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        r'(?:email|contact)[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
    ]
    for content in all_content:
        if not content:
            continue
        for pattern in founder_patterns:
            for match in re.findall(pattern, content, re.IGNORECASE | re.MULTILINE):
                name = match.strip() if isinstance(
                    match, str) else match[0].strip()
                if is_valid_name(name):
                    founders.add(name)
        for pattern in email_patterns:
            for email in re.findall(pattern, content, re.IGNORECASE):
                email = email.strip().lower()
                if is_valid_email(email):
                    emails.add(email)
    return {"founders": list(founders), "emails": list(emails)}


async def process_startup(url: str, name: str, openai_key: str, model: str) -> StartupData:
    logger.info(f"Processing: {url} ({name})")
    startup_data = StartupData(url=f"https://{url}", name=name)
    try:
        relevant_pages = await find_relevant_pages(f"https://{url}")
        all_founders = set()
        all_emails = set()
        config = BrowserConfig(
            headless=True,
            browser_type="chromium",
            viewport_width=1920,
            viewport_height=1080,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            wait_for_load=True,
            wait_for_idle=True,
            timeout=30
        )
        async with AsyncWebCrawler(config=config) as crawler:
            async def crawl_page(page_url: str):
                try:
                    run_config = CrawlerRunConfig(
                        word_count_threshold=50,
                        extraction_strategy=NoExtractionStrategy(),
                        chunking_strategy=RegexChunking(),
                        css_selector="body",
                        max_retries=2
                    )
                    result = await crawler.arun(page_url, config=run_config)
                    if not result.success:
                        return {"founders": [], "emails": []}, None
                    page_data = {"founders": [], "emails": []}
                    if page_url == f"https://{url}" and result.html:
                        soup = BeautifulSoup(result.html, 'html.parser')
                        title = soup.find('title')
                        if title and title.text.strip():
                            startup_data.name = title.text.strip()[:100]
                    if result.markdown or result.html:
                        ai_result = await extract_with_openai(result.markdown or "", page_url, startup_data.name, openai_key, model)
                        page_data["founders"].extend(ai_result["founders"])
                        page_data["emails"].extend(ai_result["emails"])
                        regex_result = extract_with_regex(
                            result.markdown or "", result.html or "")
                        page_data["founders"].extend(regex_result["founders"])
                        page_data["emails"].extend(regex_result["emails"])
                    return page_data, result.html
                except Exception as e:
                    logger.warning(f"Error processing page {page_url}: {e}")
                    return {"founders": [], "emails": []}, None

            async def crawl_contact_page(contact_path: str):
                contact_url = f"https://{url}{contact_path}"
                contact_emails = await extract_emails_from_contact_page(contact_url, openai_key, model)
                return contact_emails

            page_tasks = [crawl_page(page_url) for page_url in relevant_pages]
            contact_tasks = [crawl_contact_page(path) for path in [
                '/contact', '/contact-us', '/get-in-touch', '/connect']]
            page_results = await asyncio.gather(*page_tasks, return_exceptions=True)
            contact_results = await asyncio.gather(*contact_tasks, return_exceptions=True)

            for result, html in page_results:
                if isinstance(result, dict):
                    all_founders.update(result["founders"])
                    all_emails.update(result["emails"])

            for result in contact_results:
                if isinstance(result, list):
                    all_emails.update(result)

        startup_data.founders = sorted(list(all_founders))
        startup_data.emails = sorted(list(all_emails))
        logger.info(
            f"Extracted {len(startup_data.founders)} founders and {len(startup_data.emails)} emails from {url}")
    except Exception as e:
        logger.error(f"Error processing startup {url}: {e}")
    return startup_data


async def run_extraction_workflow(keyword: str, serpapi_key: str, openai_key: str, model: str, max_results: int) -> pd.DataFrame:
    st.info("🚀 Starting startup discovery...")
    os.environ['OPENAI_API_KEY'] = openai_key
    startups = await discover_startups(keyword, serpapi_key, max_results)
    if not startups:
        st.warning("No startup websites found. Try a different keyword.")
        return pd.DataFrame()

    st.success(f"Found {len(startups)} potential startup websites.")
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []

    async def process_batch(batch: List[tuple]):
        batch_tasks = [process_startup(
            domain, name, openai_key, model) for domain, name in batch]
        return await asyncio.gather(*batch_tasks, return_exceptions=True)

    batch_size = 5
    for i in range(0, len(startups), batch_size):
        batch = list(startups.items())[i:i + batch_size]
        status_text.text(
            f"Processing batch {i//batch_size + 1}/{(len(startups) + batch_size - 1)//batch_size}")
        batch_results = await process_batch(batch)
        for j, startup_data in enumerate(batch_results):
            if isinstance(startup_data, StartupData):
                results.append(startup_data)
                if startup_data.founders or startup_data.emails:
                    st.success(
                        f"✅ {startup_data.name}: {len(startup_data.founders)} founders, {len(startup_data.emails)} emails")
                else:
                    st.info(f"ℹ️ {startup_data.name}: No data found")
        progress_bar.progress(min((i + batch_size) / len(startups), 1.0))
        await asyncio.sleep(1)
    status_text.text("✅ Extraction completed!")
    df_data = []
    for data in results:
        df_data.append({
            "Company Name": data.name,
            "Website URL": data.url,
            "Founders": ", ".join(data.founders) if data.founders else "",
            "Founder Emails": ", ".join(data.emails) if data.emails else "",
            "Data Found": "✅" if (data.founders or data.emails) else "❌"
        })

    df = pd.DataFrame(df_data)
    if not df.empty:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"startup_founders_{timestamp}.csv"
        excel_filename = f"startup_founders_{timestamp}.xlsx"
        try:
            df.to_csv(csv_filename, index=False)
            df.to_excel(excel_filename, index=False, engine='openpyxl')
            st.session_state['csv_file'] = csv_filename
            st.session_state['excel_file'] = excel_filename
        except Exception as e:
            logger.error(f"Error saving files: {e}")
            st.error(f"Failed to save results: {e}")
    return df


def main():
    st.set_page_config(page_title="Startup Founder Crawler", layout="wide")
    st.title("🚀 Startup Founder & Email Crawler")
    st.write("Extract founder names and email addresses from startup websites using SerpAPI, Crawl4AI, and OpenAI")

    with st.sidebar:
        st.header("Configuration")
        serpapi_key = st.text_input(
            "SerpAPI Key", type="password", help="Get from serpapi.com")
        openai_key = st.text_input(
            "OpenAI API Key", type="password", help="Get from platform.openai.com")
        model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        keyword = st.text_input(
            "Search Keyword",
            value="AI startups 2024 founders site:*.com | site:*.ai | site:*.io | site:*.co",
            help="Keywords to search for startups (e.g., 'AI startups 2024 founders')"
        )
        max_results = st.slider("Max Results", 5, 50, 20)

        st.markdown("---")
        st.markdown("** Instructions:**")
        st.markdown("1. Enter your API keys")
        st.markdown("2. Specify search keywords")
        st.markdown("3. Click 'Start Extraction'")
        st.markdown("4. Download results as CSV/Excel")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button(" Test APIs", type="secondary"):
            with st.spinner("Testing APIs..."):
                if serpapi_key:
                    try:
                        import requests
                        response = requests.get(
                            "https://serpapi.com/search",
                            params={"q": "test",
                                    "api_key": serpapi_key, "num": 1},
                            timeout=10
                        )
                        if response.status_code == 200:
                            st.success("✅ SerpAPI working")
                        else:
                            st.error(
                                f"❌ SerpAPI failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ SerpAPI error: {e}")
                else:
                    st.warning("SerpAPI key missing")

                if openai_key:
                    try:
                        async def test_openai():
                            client = AsyncOpenAI(api_key=openai_key)
                            response = await client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "user", "content": "Say 'test'"}],
                                max_tokens=5
                            )
                            return "success"
                        result = asyncio.run(test_openai())
                        st.success("✅ OpenAI working")
                    except Exception as e:
                        st.error(f"❌ OpenAI error: {e}")
                else:
                    st.warning("OpenAI key missing")

    with col1:
        if st.button("🚀 Start Extraction", type="primary", use_container_width=True):
            if not serpapi_key:
                st.error("❌ SerpAPI key is required")
            elif not openai_key:
                st.error("❌ OpenAI API key is required")
            else:
                with st.spinner("🔍 Extracting founder data..."):
                    df = asyncio.run(run_extraction_workflow(
                        keyword, serpapi_key, openai_key, model, max_results))
                    if not df.empty:
                        st.success(
                            f"✅ Extraction completed! Found data for {len(df[df['Data Found'] == '✅'])} companies.")
                        st.subheader("Results")
                        st.dataframe(df, use_container_width=True)
                        col1, col2 = st.columns(2)
                        if 'csv_file' in st.session_state:
                            with col1:
                                with open(st.session_state['csv_file'], "rb") as f:
                                    st.download_button(
                                        label="Download CSV",
                                        data=f.read(),
                                        file_name=st.session_state['csv_file'],
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                        if 'excel_file' in st.session_state:
                            with col2:
                                with open(st.session_state['excel_file'], "rb") as f:
                                    st.download_button(
                                        label=" Download Excel",
                                        data=f.read(),
                                        file_name=st.session_state['excel_file'],
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )
                    else:
                        st.warning(
                            " No results found. Try different keywords or check your API keys.")


if __name__ == "__main__":
    main()
