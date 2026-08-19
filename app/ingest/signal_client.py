import asyncio
import json
import os
from datetime import datetime
from typing import List, Optional, Tuple

import httpx
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.schemas import SignalPage, SignalSummary

DEFAULT_TIMEOUT = int(os.getenv("SIGNAL_TIMEOUT_S", "10"))
DEFAULT_PAGE_SIZE = int(os.getenv("SIGNAL_PAGE_SIZE", "500"))

class SignalClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        timeout_s: int = DEFAULT_TIMEOUT,
        page_size: int = DEFAULT_PAGE_SIZE,
        mock_path: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.page_size = page_size
        self.mock_path = mock_path

    def _headers(self) -> dict:
        hdrs = {"Accept": "application/json"}
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        return hdrs

    async def fetch_summary(
        self,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
    ) -> Tuple[List[SignalSummary], dict]:
        """
        Returns: (records, debug_meta)
        debug_meta: dict(latency_ms, pages_fetched, retries)
        """
        if self.mock_path:
            return await self._fetch_mock()

        params = {}
        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts
        params["page_size"] = str(self.page_size)

        t0 = datetime.now()
        pages = 0
        retries = 0
        out: List[SignalSummary] = []

        async with httpx.AsyncClient(
            base_url=self.base_url, headers=self._headers(), timeout=self.timeout_s, http2=True
        ) as client:
            next_page = None
            while True:
                req_params = dict(params)
                if next_page:
                    req_params["page"] = next_page

                try:
                    page = await self._get_page(client, "/signals/summary", req_params)
                except httpx.RequestError as e:
                    retries += 1
                    raise e

                pages += 1
                out.extend(page.data)
                next_page = page.next_page
                if not next_page:
                    break

        latency_ms = int((datetime.now() - t0).total_seconds() * 1000)
        meta = {
            "latency_ms": latency_ms,
            "pages_fetched": pages,
            "retries": retries,
        }
        return out, meta

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    )
    async def _get_page(self, client: httpx.AsyncClient, path: str, params: dict) -> SignalPage:
        resp = await client.get(path, params=params)
        resp.raise_for_status()
        try:
            return SignalPage.model_validate(resp.json())
        except ValidationError as ve:
            raise ValueError(f"Invalid response schema: {ve}") from ve

    async def _fetch_mock(self) -> Tuple[List[SignalSummary], dict]:
        t0 = datetime.now()
        with open(self.mock_path, "r") as f:
            raw = json.load(f)

        page = SignalPage.model_validate(raw)
        latency_ms = int((datetime.now() - t0).total_seconds() * 1000)
        meta = {"latency_ms": latency_ms, "pages_fetched": 1, "retries": 0}
        return page.data, meta
