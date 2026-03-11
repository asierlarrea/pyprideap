from __future__ import annotations

import requests

_BASE_URL = "https://www.ebi.ac.uk/pride/ws/archive/v3"


class PrideClient:
    def __init__(self, base_url: str = _BASE_URL):
        self.base_url = base_url

    def get_project(self, accession: str) -> dict:
        resp = requests.get(f"{self.base_url}/projects/{accession}")
        resp.raise_for_status()
        return resp.json()

    def list_files(self, accession: str) -> list[dict]:
        resp = requests.get(
            f"{self.base_url}/files/byProject",
            params={"accession": accession, "pageSize": 100},
        )
        resp.raise_for_status()
        return resp.json()

    def get_download_urls(self, accession: str) -> dict[str, str]:
        files = self.list_files(accession)
        urls = {}
        for f in files:
            name = f.get("fileName", "")
            locations = f.get("publicFileLocations", [])
            for loc in locations:
                if loc.get("name") == "FTP Protocol":
                    urls[name] = loc["value"]
                    break
        return urls
