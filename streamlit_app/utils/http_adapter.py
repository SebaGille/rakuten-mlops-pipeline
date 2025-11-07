"""HTTP adapter for host-based routing support"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class HostHeaderAdapter(HTTPAdapter):
    """HTTP adapter that adds custom Host header for host-based routing"""
    
    def __init__(self, host_header: Optional[str] = None, *args, **kwargs):
        """
        Initialize adapter with host header
        
        Args:
            host_header: Host header value to add to requests
        """
        self.host_header = host_header
        super().__init__(*args, **kwargs)
    
    def add_headers(self, request, **kwargs):
        """Add Host header to request if configured"""
        if self.host_header:
            request.headers['Host'] = self.host_header
            logger.debug(f"Added Host header: {self.host_header} for URL: {request.url}")


def create_session_with_host_header(
    host_header: Optional[str] = None,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_factor: float = 1.0
) -> requests.Session:
    """
    Create a requests Session with host header support and retry logic
    
    Args:
        host_header: Host header value for host-based routing
        timeout: Default timeout for requests
        max_retries: Maximum number of retries
        backoff_factor: Backoff factor for retries
    
    Returns:
        Configured requests.Session
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    
    # Mount adapter for both HTTP and HTTPS
    adapter = HostHeaderAdapter(host_header=host_header, max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set default timeout
    session.timeout = timeout
    
    if host_header:
        logger.info(f"Created session with Host header: {host_header}")
    
    return session


def patch_mlflow_requests(tracking_uri: str, host_header: str) -> None:
    """
    Patch MLflow's requests to use host header adapter
    This is a temporary solution until MLflow supports custom sessions
    
    Note: This still uses monkey patching but in a more controlled way
    """
    try:
        import mlflow
        from functools import wraps
        
        # Store original request method if not already stored
        if not hasattr(requests.Session, '_original_request'):
            requests.Session._original_request = requests.Session.request
        
        original_request = requests.Session._original_request
        tracking_uri_base = tracking_uri.rstrip("/")
        
        @wraps(original_request)
        def request_with_host_header(session_self, method, url, *args, **kwargs):
            """Add Host header for MLflow requests"""
            url_normalized = url.rstrip("/")
            if tracking_uri_base and (url_normalized.startswith(tracking_uri_base) or url.startswith(tracking_uri_base)):
                if 'headers' not in kwargs:
                    kwargs['headers'] = {}
                kwargs['headers']['Host'] = host_header
                logger.debug(f"MLflow request: {method} {url} with Host: {host_header}")
            return original_request(session_self, method, url, *args, **kwargs)
        
        # Patch the Session class
        requests.Session.request = request_with_host_header
        logger.info(f"Patched MLflow requests for {tracking_uri} -> Host: {host_header}")
    except Exception as e:
        logger.warning(f"Failed to patch MLflow requests: {e}", exc_info=True)

