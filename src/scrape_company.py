#!/usr/bin/env python3
"""
Scrape GitHub repositories for a company/organization.

Usage:
    # Scrape all repos for a GitHub account
    python scrape_company.py --account disney

    # Scrape with explicit token
    python scrape_company.py --account google --token ghp_xxxx

    # Scrape only public repos
    python scrape_company.py --account netflix --repo-type public

    Output: ../data/{account}_projects_raw.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dotenv import load_dotenv
import requests

GITHUB_API = "https://api.github.com"
API_VERSION = "2022-11-28"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv("./.env")

def parse_next_link(link_header: Optional[str]) -> Optional[str]:
    """
    GitHub REST pagination: Link header provides rel="next", rel="last", etc.
    """
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' in part:
            l = part.find("<")
            r = part.find(">")
            if l != -1 and r != -1 and r > l:
                return part[l + 1 : r]
    return None


class GitHubClient:
    def __init__(self, token: Optional[str], timeout_s: int = 30):
        self.session = requests.Session()
        self.timeout_s = timeout_s
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "github-account-repo-scraper/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)

    def _sleep_for_rate_limit(self, resp: requests.Response) -> None:
        """
        GitHub best practice:
          - If Retry-After: wait that many seconds
          - Else if X-RateLimit-Remaining == 0: wait until X-RateLimit-Reset (epoch seconds)
          - Else: secondary limit fallback (short pause)
        """
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                wait_s = max(int(retry_after), 1)
            except ValueError:
                wait_s = 60
            logger.warning(f"Rate limited (Retry-After). Waiting {wait_s}s...")
            time.sleep(wait_s)
            return

        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if remaining == "0" and reset:
            try:
                wait_s = max(int(reset) - int(time.time()), 1)
            except ValueError:
                wait_s = 60
            logger.warning(f"Rate limit exhausted. Waiting {wait_s}s until reset...")
            time.sleep(wait_s)
            return

        logger.warning("Secondary rate limit hit. Waiting 10s...")
        time.sleep(10)

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Any, requests.Response]:
        backoff = 2
        for _ in range(10):
            resp = self.session.get(url, params=params, timeout=self.timeout_s)
            if resp.status_code == 200:
                return resp.json(), resp

            if resp.status_code in (403, 429):
                logger.warning(f"Received {resp.status_code} from {url}")
                self._sleep_for_rate_limit(resp)
                continue

            if resp.status_code in (500, 502, 503, 504):
                logger.warning(f"Server error {resp.status_code}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            raise RuntimeError(f"GitHub API error {resp.status_code} for {url}: {resp.text[:500]}")

        raise RuntimeError(f"Exceeded retries for {url}")

    def get_account_type(self, account: str) -> str:
        """
        Detect account type using GET /users/{username}.
        Returns: "User" or "Organization"
        """
        logger.info(f"Detecting account type for '{account}'...")
        data, _ = self.get_json(f"{GITHUB_API}/users/{account}")
        if not isinstance(data, dict) or "type" not in data:
            raise RuntimeError(f"Unexpected response for /users/{account}")
        t = str(data["type"])
        if t not in ("User", "Organization"):
            raise RuntimeError(f"Unknown account type {t!r} for {account}")
        logger.info(f"Account '{account}' is a {t}")
        return t

    def iter_repos_for_account(self, account: str, repo_type: str = "all") -> Iterable[dict]:
        """
        If Organization: GET /orgs/{org}/repos
        If User:         GET /users/{username}/repos

        repo_type is applied as:
          - org endpoint: type=all|public|private|forks|sources|member
          - user endpoint: type=all|owner|member
        """
        acc_type = self.get_account_type(account)

        if acc_type == "Organization":
            url = f"{GITHUB_API}/orgs/{account}/repos"
            params = {
                "per_page": 100,
                "type": repo_type,          # org semantics
                "sort": "full_name",
                "direction": "asc",
            }
        else:
            url = f"{GITHUB_API}/users/{account}/repos"
            params = {
                "per_page": 100,
                "type": "all" if repo_type == "all" else "owner",  # user semantics (best-effort mapping)
                "sort": "full_name",
                "direction": "asc",
            }

        page = 1
        total_repos = 0
        while True:
            logger.info(f"Fetching repos for '{account}' (page {page})...")
            data, resp = self.get_json(url, params=params)
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected repos response shape for {account}")
            total_repos += len(data)
            logger.info(f"Fetched {len(data)} repos (total so far: {total_repos})")
            for repo in data:
                yield repo

            nxt = parse_next_link(resp.headers.get("Link"))
            if not nxt:
                logger.info(f"Finished fetching repos for '{account}' ({total_repos} total)")
                break
            url = nxt
            params = None  # already encoded in nxt
            page += 1

    def get_latest_commit(self, owner: str, repo: str, default_branch: Optional[str] = None) -> Optional[dict]:
        """
        Fetch the latest commit for a repository.
        Returns a dict with sha, date, message, and author, or None if unavailable.
        """
        branch = default_branch or "main"
        url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
        params = {"sha": branch, "per_page": 1}
        try:
            data, _ = self.get_json(url, params=params)
            if isinstance(data, list) and len(data) > 0:
                commit = data[0]
                commit_info = commit.get("commit", {})
                author_info = commit_info.get("author", {})
                return {
                    "sha": commit.get("sha"),
                    "date": author_info.get("date"),
                    "message": (commit_info.get("message") or "").split("\n")[0],  # First line only
                    "author": author_info.get("name"),
                }
        except RuntimeError:
            logger.warning(f"Could not fetch latest commit for {owner}/{repo}")
        return None


def repo_to_row(repo: dict, account: str, latest_commit: Optional[dict] = None) -> dict:
    owner = repo.get("owner") or {}
    license_info = repo.get("license") or {}
    commit = latest_commit or {}

    return {
        "account": account,
        "id": repo.get("id"),
        "name": repo.get("name"),
        "full_name": repo.get("full_name"),
        "owner_login": owner.get("login"),
        "html_url": repo.get("html_url"),
        "clone_url": repo.get("clone_url"),
        "ssh_url": repo.get("ssh_url"),
        "description": repo.get("description"),
        "homepage": repo.get("homepage"),
        "topics": ";".join(repo.get("topics") or []),
        "private": repo.get("private"),
        "visibility": repo.get("visibility"),
        "fork": repo.get("fork"),
        "archived": repo.get("archived"),
        "disabled": repo.get("disabled"),
        "is_template": repo.get("is_template"),
        "created_at": repo.get("created_at"),
        "updated_at": repo.get("updated_at"),
        "pushed_at": repo.get("pushed_at"),
        "size_kb": repo.get("size"),
        "language": repo.get("language"),
        "default_branch": repo.get("default_branch"),
        "stargazers_count": repo.get("stargazers_count"),
        "watchers_count": repo.get("watchers_count"),
        "forks_count": repo.get("forks_count"),
        "open_issues_count": repo.get("open_issues_count"),
        "license_spdx_id": license_info.get("spdx_id"),
        "license_name": license_info.get("name"),
        "latest_commit_sha": commit.get("sha"),
        "latest_commit_date": commit.get("date"),
        "latest_commit_message": commit.get("message"),
        "latest_commit_author": commit.get("author"),
    }


def write_csv(path: str, rows: List[dict]) -> None:
    header = [
        "account", "id", "name", "full_name", "owner_login",
        "html_url", "clone_url", "ssh_url",
        "description", "homepage", "topics",
        "private", "visibility", "fork", "archived", "disabled", "is_template",
        "created_at", "updated_at", "pushed_at",
        "size_kb", "language", "default_branch",
        "stargazers_count", "watchers_count", "forks_count", "open_issues_count",
        "license_spdx_id", "license_name",
        "latest_commit_sha", "latest_commit_date", "latest_commit_message", "latest_commit_author",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True, help="GitHub account name (user or org), e.g. google, netflix, microsoft.")
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token (env GITHUB_TOKEN also works).")
    ap.add_argument(
        "--repo-type",
        default="all",
        help="Repo type filter. For orgs: all|public|private|forks|sources|member. For users: best-effort (all/owner).",
    )
    args = ap.parse_args()

    account = args.account.strip()
    logger.info(f"Starting scrape for account: {account}")

    gh = GitHubClient(token=args.token)
    repos = list(gh.iter_repos_for_account(account, repo_type=args.repo_type))

    logger.info(f"Fetching latest commits for {len(repos)} repos...")
    rows = []
    for i, repo in enumerate(repos, 1):
        owner = (repo.get("owner") or {}).get("login", account)
        repo_name = repo.get("name")
        default_branch = repo.get("default_branch")
        logger.info(f"[{i}/{len(repos)}] Fetching latest commit for {owner}/{repo_name}")
        latest_commit = gh.get_latest_commit(owner, repo_name, default_branch)
        rows.append(repo_to_row(repo, account=account, latest_commit=latest_commit))

    out_path = f"../data/{account}_projects_raw.csv"
    logger.info(f"Writing {len(rows)} repos to {out_path}")
    write_csv(out_path, rows)
    logger.info(f"Done! {account}: {len(rows)} repos -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
