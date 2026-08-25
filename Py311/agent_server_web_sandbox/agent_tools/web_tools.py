# web_tools.py
# Smart DuckDuckGo HTML search + robust scraping + summarization pipeline
import os
import logging

# Configure logging to write to app.log in the project root directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_BASE_DIR, "app.log")

logging.basicConfig(
    filename=_LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import json

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# =========================================================
#  SMART SEARCH (DuckDuckGo HTML)
# =========================================================

def web_search(query: str) -> str:
    """
    Smart DuckDuckGo HTML search.
    Returns titles, snippets, and URLs.
    Works for general queries, cities, events, sports, etc.
    """

    if not query or not query.strip():
        return "[ERROR] Blank query strings cannot process search queries."

    try:
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"[ERROR] DuckDuckGo HTML request failed: {str(e)}"

    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for result in soup.select(".result"):
        title_tag = result.select_one(".result__title")
        snippet_tag = result.select_one(".result__snippet")
        link_tag = result.select_one("a.result__a")

        title = title_tag.get_text(strip=True) if title_tag else None
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else None
        link = link_tag["href"] if link_tag and link_tag.has_attr("href") else None

        if title or snippet:
            results.append({
                "title": title,
                "snippet": snippet,
                "url": link
            })

    if not results:
        return "[System Notice] DuckDuckGo returned no visible search results."

    return json.dumps(results, indent=2)


# =========================================================
#  URL EXTRACTION
# =========================================================

def extract_urls(query: str) -> str:
    """
    Returns only URLs from smart search results.
    Useful for agentic workflows where the LLM chooses which pages to scrape.
    """

    raw = web_search(query)
    try:
        data = json.loads(raw)
    except:
        return "[ERROR] Could not parse search results."

    urls = [item["url"] for item in data if item.get("url")]

    if not urls:
        return "[System Notice] No URLs found in search results."

    return json.dumps(urls, indent=2)


# =========================================================
#  SCRAPER (Visible text only)
# =========================================================

def web_scrape(url: str) -> str:
    """
    Scrapes visible text from a webpage.
    Removes scripts, styles, nav, footer, etc.
    Returns full text (no truncation).
    """

    if not url or not url.startswith(("http://", "https://")):
        return "[ERROR] Invalid URL. Must start with http:// or https://"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return f"[ERROR] Scraping platform network operations failure: {str(e)}"

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "meta"]):
        tag.decompose()

    raw_lines = soup.get_text("\n").splitlines()
    cleaned = [line.strip() for line in raw_lines if line.strip() and len(line.strip()) > 3]

    final_text = "\n".join(cleaned)

    if not final_text.strip():
        return "[System Notice] Page fetched successfully but contains zero readable text."

    return final_text


# =========================================================
#  SUMMARIZATION (LLM-agnostic)
# =========================================================

def summarize_text(text: str, llm_client, model_name: str) -> str:
    """
    Summarizes long text using a fast LLM (e.g., phi-4-mini).
    llm_client must be an OpenAI-compatible client.
    """

    if not text or len(text.strip()) < 20:
        return "[ERROR] Not enough content to summarize."

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Summarize the following text clearly and concisely."},
                {"role": "user", "content": text}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] Summarization failed: {str(e)}"


# =========================================================
#  SCRAPE + SUMMARIZE
# =========================================================

def scrape_and_summarize(url: str, llm_client, model_name: str) -> str:
    """
    Scrapes a webpage and summarizes it using a fast LLM.
    """

    scraped = web_scrape(url)
    if scraped.startswith("[ERROR]"):
        return scraped

    return summarize_text(scraped, llm_client, model_name)


# =========================================================
#  SEARCH + SUMMARIZE (Perplexity-style)
# =========================================================

def search_and_summarize(query: str, llm_client, model_name: str) -> str:
    """
    Performs smart search, scrapes top pages, summarizes each,
    and returns a combined summary.
    """

    raw = web_search(query)
    try:
        results = json.loads(raw)
    except:
        return "[ERROR] Could not parse search results."

    urls = [item["url"] for item in results if item.get("url")]

    if not urls:
        return "[System Notice] No URLs found to summarize."

    summaries = []
    for url in urls[:3]:  # scrape top 3 pages
        scraped = web_scrape(url)
        if scraped.startswith("[ERROR]"):
            continue

        summary = summarize_text(scraped, llm_client, model_name)
        summaries.append(f"URL: {url}\n\n{summary}\n\n---\n")

    if not summaries:
        return "[System Notice] All pages failed to scrape or summarize."

    return "\n".join(summaries)


# # web_tools.py
# # Smart DuckDuckGo HTML search + robust visible-text scraper

# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import quote_plus

# HEADERS = {
#     "User-Agent": (
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#         "AppleWebKit/537.36 (KHTML, like Gecko) "
#         "Chrome/120.0.0.0 Safari/537.36"
#     )
# }

# # ---------------------------------------------------------
# #  SMART WEB SEARCH (DuckDuckGo HTML)
# # ---------------------------------------------------------

# def web_search(query: str) -> str:
#     """
#     Performs a smart web search using DuckDuckGo HTML results.
#     Extracts titles, snippets, and URLs.
#     Works for general queries, cities, events, sports, etc.
#     """

#     if not query or not query.strip():
#         return "[ERROR] Blank query strings cannot process search queries."

#     try:
#         url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
#         resp = requests.get(url, headers=HEADERS, timeout=10)
#         resp.raise_for_status()
#     except Exception as e:
#         return f"[ERROR] DuckDuckGo HTML request failed: {str(e)}"

#     soup = BeautifulSoup(resp.text, "html.parser")

#     results = []
#     for result in soup.select(".result"):
#         title_tag = result.select_one(".result__title")
#         snippet_tag = result.select_one(".result__snippet")
#         link_tag = result.select_one("a.result__a")

#         title = title_tag.get_text(strip=True) if title_tag else None
#         snippet = snippet_tag.get_text(strip=True) if snippet_tag else None
#         link = link_tag["href"] if link_tag and link_tag.has_attr("href") else None

#         if title or snippet:
#             results.append({
#                 "title": title,
#                 "snippet": snippet,
#                 "url": link
#             })

#     if not results:
#         return "[System Notice] DuckDuckGo returned no visible search results."

#     # Format results into readable text
#     output_lines = []
#     for r in results[:8]:  # limit to top 8 results
#         if r["title"]:
#             output_lines.append(f"• {r['title']}")
#         if r["snippet"]:
#             output_lines.append(f"  {r['snippet']}")
#         if r["url"]:
#             output_lines.append(f"  {r['url']}\n")

#     final_text = "\n".join(output_lines)

#     if len(final_text) > 4000:
#         return final_text[:4000] + "\n\n...[Content truncated for safety]..."

#     return final_text


# # ---------------------------------------------------------
# #  WEB SCRAPE (Visible text only)
# # ---------------------------------------------------------

# def web_scrape(url: str) -> str:
#     """
#     Scrapes visible text from a webpage.
#     Removes scripts, styles, nav, footer, etc.
#     Returns clean text or an error.
#     """

#     if not url or not url.startswith(("http://", "https://")):
#         return "[ERROR] Invalid URL. Must start with http:// or https://"

#     try:
#         resp = requests.get(url, headers=HEADERS, timeout=15)
#         resp.raise_for_status()
#     except Exception as e:
#         return f"[ERROR] Scraping platform network operations failure: {str(e)}"

#     soup = BeautifulSoup(resp.text, "html.parser")

#     # Remove non-visible elements
#     for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "meta"]):
#         tag.decompose()

#     # Extract visible text
#     raw_lines = soup.get_text("\n").splitlines()
#     cleaned = []

#     for line in raw_lines:
#         s = line.strip()
#         if s and len(s) > 3:
#             cleaned.append(s)

#     final_text = "\n".join(cleaned)

#     if not final_text.strip():
#         return "[System Notice] Page fetched successfully but contains zero readable text."

#     if len(final_text) > 4000:
#         return final_text[:4000] + "\n\n...[Content truncated for safety]..."

#     return final_text


# # # web_tools.py
# # # Fast, reliable web search + scraping using DuckDuckGo Instant Answer API

# # import requests
# # from bs4 import BeautifulSoup
# # from urllib.parse import quote_plus

# # HEADERS = {
# #     "User-Agent": (
# #         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
# #         "AppleWebKit/537.36 (KHTML, like Gecko) "
# #         "Chrome/120.0.0.0 Safari/537.36"
# #     )
# # }

# # # ---------------------------------------------------------
# # #  WEB SEARCH (DuckDuckGo Instant Answer API)
# # # ---------------------------------------------------------

# # def web_search(query: str) -> str:
# #     """
# #     Performs a web search using DuckDuckGo Instant Answer API.
# #     Returns clean text results or a structured fallback.
# #     """

# #     if not query or not query.strip():
# #         return "[ERROR] Blank query strings cannot process search queries."

# #     try:
# #         url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_redirect=1&no_html=1"
# #         resp = requests.get(url, headers=HEADERS, timeout=10)
# #         resp.raise_for_status()
# #         data = resp.json()
# #     except Exception as e:
# #         return f"[ERROR] DuckDuckGo API request failed: {str(e)}"

# #     # 1. Direct Answer
# #     if data.get("Answer"):
# #         return data["Answer"]

# #     # 2. Abstract (Wikipedia-style summary)
# #     if data.get("AbstractText"):
# #         return data["AbstractText"]

# #     # 3. Related Topics (fallback)
# #     related = data.get("RelatedTopics", [])
# #     if related:
# #         lines = []
# #         for item in related[:5]:
# #             if isinstance(item, dict) and item.get("Text"):
# #                 lines.append(item["Text"])
# #         if lines:
# #             return "\n".join(lines)

# #     return "[System Notice] DuckDuckGo returned no useful results."


# # # ---------------------------------------------------------
# # #  WEB SCRAPE (Visible text only)
# # # ---------------------------------------------------------

# # def web_scrape(url: str) -> str:
# #     """
# #     Scrapes visible text from a webpage.
# #     Removes scripts, styles, nav, footer, etc.
# #     Returns clean text or an error.
# #     """

# #     if not url or not url.startswith(("http://", "https://")):
# #         return "[ERROR] Invalid URL. Must start with http:// or https://"

# #     try:
# #         resp = requests.get(url, headers=HEADERS, timeout=15)
# #         resp.raise_for_status()
# #     except Exception as e:
# #         return f"[ERROR] Scraping platform network operations failure: {str(e)}"

# #     soup = BeautifulSoup(resp.text, "html.parser")

# #     # Remove non-visible elements
# #     for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "meta"]):
# #         tag.decompose()

# #     # Extract visible text
# #     raw_lines = soup.get_text("\n").splitlines()
# #     cleaned = []

# #     for line in raw_lines:
# #         s = line.strip()
# #         if s and len(s) > 3:
# #             cleaned.append(s)

# #     final_text = "\n".join(cleaned)

# #     if not final_text.strip():
# #         return "[System Notice] Page fetched successfully but contains zero readable text."

# #     # Safety: limit output size
# #     if len(final_text) > 4000:
# #         return final_text[:4000] + "\n\n...[Content truncated for safety]..."

# #     return final_text


# # # # agent_tools/web_tools.py - Comprehensive Web Interaction Tools
# # # # Merged version combining best features from multiple implementations

# # # import logging
# # # import time
# # # import requests
# # # from bs4 import BeautifulSoup
# # # from urllib.parse import quote_plus, urlparse
# # # from .common import sanitize_edge_metadata

# # # # Configure logging
# # # logger = logging.getLogger(__name__)

# # # HEADERS = {
# # #     "User-Agent": (
# # #         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
# # #         "AppleWebKit/537.36 (KHTML, like Gecko) "
# # #         "Chrome/120.0.0.0 Safari/537.36"
# # #     )
# # # }

# # # def _retry_request(url, params=None, headers=None, timeout=15, max_retries=3):
# # #     """Generic retry mechanism for HTTP requests."""
# # #     for attempt in range(max_retries):
# # #         try:
# # #             logger.info(f"Attempting request to {url} (attempt {attempt + 1}/{max_retries})")
# # #             resp = requests.get(url, params=params, headers=headers, timeout=timeout)
# # #             if resp.status_code == 200:
# # #                 return resp
# # #         except requests.exceptions.Timeout:
# # #             logger.warning(f"Request timed out (attempt {attempt + 1}/{max_retries})")
# # #             if attempt == max_retries - 1:
# # #                 return None
# # #         except Exception as e:
# # #             logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
# # #             if attempt == max_retries - 1:
# # #                 return None
# # #         time.sleep(1.0)
# # #     return None

# # # def web_search(query: str) -> str:
# # #     """
# # #     Perform a comprehensive web search using multiple fallback engines.
# # #     Returns formatted result snippets or error/notice strings.
# # #     """
# # #     if not query or not query.strip():
# # #         logger.warning("Empty or non-string query received for web_search")
# # #         return "[ERROR] Blank query strings cannot process search queries."

# # #     clean_query = sanitize_edge_metadata(query).strip()
# # #     if not clean_query:
# # #         logger.warning("Query became empty after sanitization")
# # #         return "[ERROR] Query became empty after sanitization."

# # #     # Try multiple search engines with fallback
# # #     search_engines = [
# # #         ("DuckDuckGo", "https://duckduckgo.com/html/", "q"),
# # #         ("Brave", "https://search.brave.com/search", "q"),
# # #         ("Google", "https://www.google.com/search", "q")
# # #     ]

# # #     for engine_name, base_url, param_name in search_engines:
# # #         params = {param_name: clean_query}
# # #         resp_text = ""

# # #         try:
# # #             logger.info(f"Attempting {engine_name} search")
# # #             resp = requests.get(base_url, params=params, headers=HEADERS, timeout=15)
# # #             if resp.status_code == 200:
# # #                 resp_text = resp.text
# # #                 break
# # #         except Exception as e:
# # #             logger.warning(f"{engine_name} search failed: {e}")
# # #             continue

# # #     if not resp_text:
# # #         logger.error("Failed to retrieve search results from any engine.")
# # #         return "[ERROR] Unable to extract page data from search engine."

# # #     soup = BeautifulSoup(resp_text, "html.parser")
# # #     results = []

# # #     # Try multiple selectors for robustness
# # #     selectors = [
# # #         "div.result", "div.web-result",  # DuckDuckGo
# # #         ".snippet", "div.web-results",     # Brave
# # #         "a.result__a", "div.r"            # Google
# # #     ]

# # #     candidates = []
# # #     for selector in selectors:
# # #         candidates = soup.select(selector)
# # #         if candidates:
# # #             logger.info(f"Found {len(candidates)} results using selector '{selector}'")
# # #             break

# # #     for res in candidates[:5]:
# # #         # Try multiple title/snippet selectors
# # #         title_el = res.select_one("a.result__a") or res.find("a") or res.select_one(".snippet-title, .title, h2, a")
# # #         snippet_el = (
# # #             res.select_one("a.result__snippet")
# # #             or res.select_one(".result__snippet")
# # #             or res.find("p")
# # #             or res.select_one(".snippet-content, .desc, .body, p")
# # #         )

# # #         title = title_el.get_text(" ", strip=True) if title_el else ""
# # #         snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

# # #         if title or snippet:
# # #             results.append(f"TITLE: {title}\nSUMMARY: {snippet}")

# # #     if not results:
# # #         logger.info("Search completed successfully but returned zero readable result snippets.")
# # #         return "[System Notice] Search completed successfully but returned zero readable result snippets."

# # #     return "\n\n".join(results)


# # # def web_scrape(url: str) -> str:
# # #     """
# # #     Fetches a remote target page layout block, cleans boilerplate layouts,
# # #     and drops raw text structures.
# # #     """
# # #     logger.info(f"Scraping URL: {url}")
# # #     if not isinstance(url, str):
# # #         logger.warning("Non-string input received for web_scrape")
# # #         return "[ERROR] Invalid URL type provided."

# # #     clean_url = sanitize_edge_metadata(url).strip()
# # #     if not clean_url or not clean_url.startswith(("http://", "https://")):
# # #         logger.warning(f"Invalid URL scheme: {clean_url}")
# # #         return "[ERROR] Invalid target scheme configuration. URL strings must explicitly begin with http:// or https//"

# # #     try:
# # #         logger.info(f"Fetching page content from {clean_url}")
# # #         resp = requests.get(clean_url, headers=HEADERS, timeout=15)
# # #         resp.raise_for_status()
# # #         logger.info("Page fetched successfully")
# # #     except requests.exceptions.Timeout:
# # #         logger.error("Network timeout during page fetch.")
# # #         return "[ERROR] Network timeout during page fetch."
# # #     except Exception as e:
# # #         logger.error(f"Scraping platform network operations failure: {e}")
# # #         return f"[ERROR] Scraping platform network operations failure: {str(e)}"

# # #     try:
# # #         soup = BeautifulSoup(resp.text, "html.parser")

# # #         # Prune functional interface nodes, headers, or execution script payloads
# # #         for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
# # #             tag.decompose()

# # #         # Capture textual layout tokens cleanly
# # #         raw_lines = soup.get_text("\n").splitlines()
# # #         cleaned_lines = []

# # #         for line in raw_lines:
# # #             s_line = line.strip()
# # #             # Omit massive string tables or repetitive whitespace spacer characters
# # #             if s_line and len(s_line) > 4:
# # #                 cleaned_lines.append(s_line)

# # #         final_text = "\n".join(cleaned_lines)

# # #         # Token Safety Barrier: limits context footprint lengths to 4000 characters
# # #         if len(final_text) > 4000:
# # #             logger.info(f"Scraped content truncated to 4000 chars (original length: {len(final_text)})")
# # #             return final_text[:4000] + "\n\n...[Content truncated to save local CPU memory capacity]..."

# # #         logger.info("Page scraped successfully")
# # #         return final_text if final_text.strip() else "[System Notice] Page layout fetched successfully but contains zero readable text values."
# # #     except Exception as e:
# # #         logger.error(f"Failed to parse scraped page content: {e}")
# # #         return f"[ERROR] Failed to parse scraped page content: {str(e)}"


# # # def brave_web_search(query: str) -> str:
# # #     """Performs a web search using Brave Search API."""
# # #     logger.info(f"Performing web search for query: {query[:50]}...")
# # #     if not isinstance(query, str) or not query.strip():
# # #         logger.warning("Empty or non-string query received for web_search")
# # #         return "[ERROR] Blank query strings cannot process search queries."

# # #     url = "https://search.brave.com/search"
# # #     encoded_query = quote_plus(sanitize_edge_metadata(query).strip())
# # #     params = {"q": encoded_query}
# # #     resp_text = ""

# # #     # Network retry execution mapping
# # #     for attempt in range(3):
# # #         try:
# # #             logger.info(f"Attempting Brave search (attempt {attempt + 1}/3)")
# # #             resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
# # #             if resp.status_code == 200:
# # #                 resp_text = resp.text
# # #                 logger.info("Brave search request successful")
# # #                 break
# # #         except requests.exceptions.Timeout:
# # #             logger.warning(f"Search request timed out (attempt {attempt + 1}/3)")
# # #             if attempt == 2:
# # #                 return "[ERROR] Search gateway transaction timed out or server refused connection profiles."
# # #             time.sleep(1.0)
# # #         except Exception as e:
# # #             logger.error(f"Search request failed (attempt {attempt + 1}/3): {e}")
# # #             if attempt == 2:
# # #                 return "[ERROR] Search gateway transaction timed out or server refused connection profiles."
# # #             time.sleep(1.0)

# # #     if not resp_text:
# # #         logger.warning("Unable to extract page data from search engine.")
# # #         return "[ERROR] Unable to extract page data from search engine."

# # #     try:
# # #         soup = BeautifulSoup(resp_text, "html.parser")
# # #         results = []

# # #         # Cascade Selector Check: attempts standard snippets, layout targets, or structural fallback elements
# # #         selectors = [".snippet", "div.web-results", "div[data-loc='main'] .snippet-title", "div.result"]

# # #         # Try searching across broad generic tags if target structural items are missing
# # #         found_elements = []
# # #         for sel in selectors:
# # #             found_elements = soup.select(sel)
# # #             if found_elements:
# # #                 logger.info(f"Found {len(found_elements)} elements using selector '{sel}'")
# # #                 break

# # #         for res in found_elements[:5]:
# # #             title_el = res.select_one(".snippet-title, .title, h2, a")
# # #             snippet_el = res.select_one(".snippet-content, .desc, .body, p")

# # #             t = title_el.get_text(" ", strip=True) if title_el else ""
# # #             s = snippet_el.get_text(" ", strip=True) if snippet_el else ""

# # #             if t or s:
# # #                 results.append(f"TITLE: {t}\nSUMMARY: {s}")

# # #         logger.info(f"Web search returned {len(results)} results")
# # #         return "\n\n".join(results) if results else "[System Notice] Brave Search completed successfully but returned zero index contents."
# # #     except Exception as e:
# # #         logger.error(f"Failed to parse search results: {e}")
# # #         return f"[ERROR] Failed to parse search results: {str(e)}"


# # # def smart_search(query: str) -> str:
# # #     """
# # #     Smart search that automatically chooses the best engine based on query type.
# # #     Combines features from all previous implementations.
# # #     """
# # #     # Determine optimal search engine based on query characteristics
# # #     if "weather" in query.lower() or "temperature" in query.lower():
# # #         # Weather queries work better with DuckDuckGo
# # #         return web_search(query)
# # #     elif "news" in query.lower() or "latest" in query.lower():
# # #         # News queries work better with Brave
# # #         return brave_web_search(query)
# # #     else:
# # #         # Default to DuckDuckGo for general queries
# # #         return web_search(query)


# # # def smart_scrape(url: str, max_content_length: int = 4000) -> str:
# # #     """
# # #     Smart scraping with configurable content length limits.
# # #     Automatically handles different page types and structures.
# # #     """
# # #     logger.info(f"Smart scraping URL: {url} (max length: {max_content_length})")

# # #     # Validate URL
# # #     if not isinstance(url, str) or not url.strip():
# # #         return "[ERROR] Invalid URL provided."

# # #     clean_url = sanitize_edge_metadata(url).strip()
# # #     if not clean_url.startswith(("http://", "https://")):
# # #         return "[ERROR] Invalid URL scheme."

# # #     try:
# # #         resp = requests.get(clean_url, headers=HEADERS, timeout=15)
# # #         resp.raise_for_status()
# # #     except Exception as e:
# # #         logger.error(f"Failed to fetch {url}: {e}")
# # #         return f"[ERROR] Failed to fetch content: {str(e)}"

# # #     try:
# # #         soup = BeautifulSoup(resp.text, "html.parser")

# # #         # Smart content extraction based on page type
# # #         if soup.find("script"):
# # #             # Page has scripts - extract text without script tags
# # #             for tag in soup(["script", "style"]):
# # #                 tag.decompose()

# # #         # Extract meaningful content
# # #         paragraphs = soup.find_all("p")
# # #         headings = soup.find_all(["h1", "h2", "h3"])

# # #         content_parts = []

# # #         # Add headings if present
# # #         for heading in headings:
# # #             content_parts.append(heading.get_text(strip=True))

# # #         # Add paragraphs if present
# # #         for paragraph in paragraphs:
# # #             text = paragraph.get_text(strip=True)
# # #             if text and len(text) > 10:  # Skip very short paragraphs
# # #                 content_parts.append(text)

# # #         final_text = "\n\n".join(content_parts)

# # #         # Apply length limit
# # #         if len(final_text) > max_content_length:
# # #             logger.info(f"Content truncated to {max_content_length} chars")
# # #             return final_text[:max_content_length] + "\n\n...[Content truncated]"

# # #         return final_text if final_text.strip() else "[System Notice] No readable content found."
# # #     except Exception as e:
# # #         logger.error(f"Smart scrape failed: {e}")
# # #         return f"[ERROR] Smart scrape failed: {str(e)}"


# # # # Export functions for external use
# # # __all__ = ['web_search', 'web_scrape', 'brave_web_search', 'smart_search', 'smart_scrape']