"""
Entrez API Utilities with Rate Limiting
======================================

This module provides utilities for making NCBI Entrez API calls with proper
rate limiting, configuration, and error handling. Designed to work safely
with multiple concurrent jobs.
"""

import os
import time
import threading
import logging
from typing import Optional, Any, Callable
from functools import wraps
from contextlib import contextmanager
from dotenv import load_dotenv
from Bio import Entrez

logger = logging.getLogger(__name__)

# Global rate limiter instance
_rate_limiter = None
_rate_limiter_lock = threading.Lock()


class EntrezRateLimiter:
    """Thread-safe rate limiter for Entrez API calls."""
    
    def __init__(self, requests_per_second: float = 3.0):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second to allow
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.lock = threading.Lock()
        
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                logger.debug("Rate limiting: sleeping for %.2f seconds", sleep_time)
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()


def get_rate_limiter() -> EntrezRateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    
    with _rate_limiter_lock:
        if _rate_limiter is None:
            # Determine rate based on API key availability
            api_key = os.getenv('ENTREZ_API_KEY')
            if api_key:
                # With API key: 10 requests/second
                rate = 10.0
                logger.info("🔑 Entrez API key detected - using rate limit: %.1f req/sec", rate)
            else:
                # Without API key: 3 requests/second (conservative)
                rate = 3.0
                logger.warning("⚠️ No Entrez API key - using conservative rate limit: %.1f req/sec", rate)
            
            _rate_limiter = EntrezRateLimiter(rate)
    
    return _rate_limiter


def configure_entrez():
    """Configure Entrez with email and API key from environment variables."""
    load_dotenv()
    
    # Set email (required by NCBI)
    email = os.getenv('ENTREZ_EMAIL')
    if not email:
        raise ValueError(
            "ENTREZ_EMAIL environment variable is required. "
            "Please add your email address to the .env file."
        )
    
    Entrez.email = email
    logger.info("📧 Configured Entrez email: %s", email)
    
    # Set API key if available
    ncbi_key = os.getenv('ENTREZ_API_KEY')
    if ncbi_key:
        Entrez.api_key = ncbi_key
        logger.info("🔑 Configured Entrez API key (first 8 chars): %s...", ncbi_key[:8])
    else:
        logger.warning("⚠️ No ENTREZ_API_KEY found - API calls will be rate-limited to 3/second")
    
    # Set tool name for better tracking
    Entrez.tool = "UORCA"


@contextmanager
def rate_limited_entrez():
    """Context manager for rate-limited Entrez calls."""
    rate_limiter = get_rate_limiter()
    rate_limiter.wait_if_needed()
    yield


def rate_limited(func: Callable) -> Callable:
    """Decorator to add rate limiting to Entrez functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with rate_limited_entrez():
            return func(*args, **kwargs)
    return wrapper


def safe_entrez_call(entrez_func: Callable, *args, max_retries: int = 3, **kwargs) -> Any:
    """
    Make a safe Entrez API call with rate limiting and retry logic.
    
    Args:
        entrez_func: The Entrez function to call (e.g., Entrez.efetch)
        *args: Arguments to pass to the function
        max_retries: Maximum number of retry attempts
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        The result of the Entrez function call
    
    Raises:
        Exception: If all retry attempts fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            with rate_limited_entrez():
                logger.debug("Entrez API call attempt %d/%d: %s", 
                           attempt + 1, max_retries + 1, entrez_func.__name__)
                return entrez_func(*args, **kwargs)
                
        except Exception as e:
            last_exception = e
            logger.warning("Entrez API call failed (attempt %d/%d): %s", 
                         attempt + 1, max_retries + 1, str(e))
            
            if attempt < max_retries:
                # Exponential backoff: 1s, 2s, 4s, etc.
                sleep_time = 2 ** attempt
                logger.info("Retrying in %d seconds...", sleep_time)
                time.sleep(sleep_time)
            else:
                logger.error("All Entrez API retry attempts failed")
                break
    
    raise last_exception


def fetch_taxonomy_info(taxid: str) -> Optional[dict]:
    """
    Fetch taxonomy information for a given taxonomic ID.
    
    Args:
        taxid: NCBI taxonomic ID
    
    Returns:
        Dictionary with taxonomy information or None if failed
    """
    try:
        logger.info("🔍 Fetching taxonomy info for taxid: %s", taxid)
        
        handle = safe_entrez_call(Entrez.efetch, db="taxonomy", id=str(taxid), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        if records and len(records) > 0:
            record = records[0]
            result = {
                'taxid': taxid,
                'scientific_name': record.get("ScientificName", "Unknown"),
                'common_name': record.get("OtherNames", {}).get("CommonName", [""])[0] if record.get("OtherNames", {}).get("CommonName") else "",
                'lineage': record.get("Lineage", ""),
                'rank': record.get("Rank", "")
            }
            logger.info("✅ Found taxonomy: %s (%s)", result['scientific_name'], result.get('common_name', ''))
            return result
        else:
            logger.warning("⚠️ No taxonomy records found for taxid: %s", taxid)
            return None
            
    except Exception as e:
        logger.error("❌ Error fetching taxonomy info for taxid %s: %s", taxid, str(e))
        return None


def search_sra_runs(srx_id: str) -> list:
    """
    Search for SRA run IDs (SRR) associated with an SRA experiment ID (SRX).
    
    Args:
        srx_id: SRA experiment ID (e.g., "SRX123456")
    
    Returns:
        List of SRA run IDs (SRR IDs)
    """
    try:
        logger.debug("🔍 Searching SRA runs for experiment: %s", srx_id)
        
        # Search for the SRX ID
        search_handle = safe_entrez_call(
            Entrez.esearch, 
            db="sra", 
            term=srx_id, 
            retmax=100  # Should be plenty for most experiments
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        id_list = search_results.get('IdList', [])
        if not id_list:
            logger.warning("⚠️ No SRA records found for experiment: %s", srx_id)
            return []
        
        # Fetch detailed information to get SRR IDs
        fetch_handle = safe_entrez_call(
            Entrez.efetch,
            db="sra",
            id=",".join(id_list),
            rettype="runinfo",
            retmode="text"
        )
        
        runinfo_data = fetch_handle.read()
        fetch_handle.close()
        
        # Parse the runinfo to extract SRR IDs
        srr_ids = []
        for line in runinfo_data.split('\n'):
            if line.startswith('SRR'):
                srr_id = line.split(',')[0]  # First column is the Run ID
                srr_ids.append(srr_id)
        
        logger.info("✅ Found %d SRA runs for experiment %s", len(srr_ids), srx_id)
        return srr_ids
        
    except Exception as e:
        logger.error("❌ Error searching SRA runs for %s: %s", srx_id, str(e))
        return []


# Auto-configure Entrez when module is imported
try:
    configure_entrez()
except Exception as e:
    logger.error("❌ Failed to configure Entrez: %s", str(e))
    logger.error("Please ensure ENTREZ_EMAIL is set in your .env file")