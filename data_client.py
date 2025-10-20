import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

RICK_MORTY_GRAPHQL = "https://rickandmortyapi.com/graphql"
_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".cache_locations.json")


def _load_cache() -> Optional[Dict[str, Any]]:
    if not os.path.exists(_CACHE_FILE):
        return None
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(data: Dict[str, Any]) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(5))
def _graphql(query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp = requests.post(
        RICK_MORTY_GRAPHQL,
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload and payload["errors"]:
        raise RuntimeError(str(payload["errors"]))
    return payload["data"]


def _locations_query() -> str:
    return (
        """
        query ($page: Int) {
          locations(page: $page) {
            info { count pages next prev }
            results {
              id
              name
              type
              dimension
              residents {
                id
                name
                status
                species
                image
                gender
                origin { name }
              }
            }
          }
        }
        """
    )


def fetch_all_locations() -> List[Dict[str, Any]]:
    """Fetch all locations with their residents via GraphQL, paging across all pages.

    Uses a tiny JSON file cache to avoid repeated network calls during local runs.
    """
    cached = _load_cache()
    if cached and "locations" in cached:
        return cached["locations"]

    all_locations: List[Dict[str, Any]] = []
    page = 1
    while True:
        data = _graphql(_locations_query(), {"page": page})
        locs = data["locations"]["results"]
        all_locations.extend(locs)
        next_page = data["locations"]["info"]["next"]
        if not next_page:
            break
        page = next_page

    _save_cache({"locations": all_locations})
    return all_locations


@lru_cache(maxsize=1)
def get_locations() -> List[Dict[str, Any]]:
    return fetch_all_locations()


def get_characters_index() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for loc in get_locations():
        for resident in loc.get("residents", []):
            index[resident["id"]] = resident
    return index
