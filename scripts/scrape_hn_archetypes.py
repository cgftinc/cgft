"""
Hacker News Archetype Dataset Scraper
======================================
Builds (Context, Archetype, Ground-Truth Comment) triplets from HN comments
using the public Algolia HN API.  No API key required for scraping.

Strategy
--------
The /items/{id} tree-traversal endpoint does NOT expose comment scores —
they are always null.  Instead we use the /search endpoint and query it
per-archetype with keyword terms.  For each matching comment we then
fetch its parent (story or comment) to form the context field.

To maximise yield, each (search-term, archetype) pair is queried across
multiple two-to-three-year time windows (Algolia caps results per query,
but different windows surface different comments).  Results are deduped
by objectID before assembly.

Optional LLM filter pass (--llm-filter) sends each candidate to an LLM
to remove job postings, ads, and off-topic content before saving.
Requires OPENAI_API_KEY in the environment.

Output: hn_archetypes_dataset.jsonl  (one JSON line per HNTrainingSample)

Usage:
    uv run python scripts/scrape_hn_archetypes.py
    uv run python scripts/scrape_hn_archetypes.py --per-archetype 500
    uv run python scripts/scrape_hn_archetypes.py --llm-filter
    uv run python scripts/scrape_hn_archetypes.py --per-archetype 500 --llm-filter
"""

import argparse
import asyncio
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import aiohttp
import openai
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

# ── API endpoints ──────────────────────────────────────────────────────────────

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
HN_ITEM_URL = "https://hn.algolia.com/api/v1/items/{id}"

# ── Tuning knobs ───────────────────────────────────────────────────────────────

CONCURRENCY = 10         # max simultaneous outbound HTTP requests
MAX_RETRIES = 3          # per-request retry attempts
RETRY_DELAY = 1.0        # base back-off in seconds (multiplied by attempt#)
RESULTS_PER_ARCHETYPE = 1250  # target comment results to pull per archetype → 5 000 total raw
OUTPUT_FILE = Path("hn_archetypes_dataset.jsonl")

# Time windows for paginated querying — Algolia caps results per query, so
# querying separate date ranges surfaces different comments and raises total yield.
# Unix timestamps: 2015-01-01, 2018-01-01, 2021-01-01, 2024-01-01, open-ended.
TIME_WINDOWS = [
    (1420070400, 1514764800),   # 2015 – 2018
    (1514764800, 1609459200),   # 2018 – 2021
    (1609459200, 1704067200),   # 2021 – 2024
    (1704067200, None),         # 2024 – present
]

# ── LLM filter settings ────────────────────────────────────────────────────────

AZURE_ENDPOINT = "https://giris-m8gatqe4-eastus2.cognitiveservices.azure.com/"
AZURE_API_VERSION = "2024-12-01-preview"
LLM_FILTER_MODEL = "gpt-4.1-mini"
LLM_FILTER_CONCURRENCY = 3        # keep well below Azure RPM limits
LLM_FILTER_MAX_RETRIES = 6        # retry budget for 429 rate-limit responses
LLM_FILTER_RETRY_BASE  = 2.0      # exponential back-off base (seconds)


def load_env_local() -> None:
    """Load key=value pairs from .env.local into os.environ (if not already set)."""
    env_file = Path(__file__).parent.parent / ".env.local"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ── Archetype definitions ──────────────────────────────────────────────────────
# search_terms    → sent to the Algolia query string
# verify_keywords → at least one must appear in cleaned comment text

ARCHETYPES: dict[str, dict] = {
    "The Rust/Rewrite Evangelist": {
        "search_terms": [
            "rust language", "rewrite in rust", "memory safety", "borrow checker",
            "unsafe rust", "zero cost abstractions", "fearless concurrency", "rust ownership",
        ],
        "verify_keywords": [
            "rust", "memory safety", "borrow checker", "c++", "segfault",
            "rewrite", "cargo", "unsafe", "lifetime", "ownership",
        ],
    },
    "The Cynical Systems Veteran": {
        "search_terms": [
            "unix philosophy", "software bloat", "overengineered", "reinventing the wheel",
            "systemd bad", "microservices complexity", "kubernetes overhead", "everything is terrible",
        ],
        "verify_keywords": [
            "unix", "posix", "bloat", "reinventing", "overengineered",
            "back in my day", "systemd", "web scale", "k8s",
            "microservices", "kubernetes", "complexity",
        ],
    },
    "The Solo Bootstrapper": {
        "search_terms": [
            "monthly recurring revenue", "saas profitable", "bootstrap startup", "stripe revenue",
            "indie hacker mrr", "solopreneur revenue", "churn rate saas", "customer acquisition",
        ],
        "verify_keywords": [
            "mrr", "bootstrap", "indie", "profitable", "stripe",
            "saas", "churn", "customers", "solopreneur", "revenue", "arr",
        ],
    },
    "The Academic Pedant": {
        "search_terms": [
            "empirical evidence", "formally defined", "monad category theory", "technically correct",
            "peer reviewed", "strictly speaking", "by definition", "academic literature",
        ],
        "verify_keywords": [
            "technically", "defined as", "literature", "empirical",
            "actually", "monad", "formal", "peer-reviewed", "strictly speaking",
            "by definition", "academic",
        ],
    },
}

# ── Pydantic output model ──────────────────────────────────────────────────────


class HNTrainingSample(BaseModel):
    story_title: str
    story_url: str
    parent_text: str           # text of the comment/story this comment replies to
    archetype_profile: str     # matched archetype name
    ground_truth_comment: str
    score: int
    created_at_i: int          # Unix timestamp from Algolia — used for temporal split


# ── Low-level HTTP helper ──────────────────────────────────────────────────────


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: dict | None = None,
    *,
    sem: asyncio.Semaphore,
) -> dict | None:
    """
    GET *url* and return parsed JSON, or None on permanent failure.
    Respects the shared semaphore and retries with exponential back-off.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with sem:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json(content_type=None)

                    if resp.status == 429:
                        wait = RETRY_DELAY * attempt * 3
                        print(f"[rate-limit] 429 — backing off {wait:.1f}s", file=sys.stderr)
                        await asyncio.sleep(wait)
                        continue

                    print(f"[warn] HTTP {resp.status} for {url}", file=sys.stderr)
                    return None

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt == MAX_RETRIES:
                print(f"[error] {exc!r} after {MAX_RETRIES} attempts — {url}", file=sys.stderr)
                return None
            await asyncio.sleep(RETRY_DELAY * attempt)

    return None


# ── Text utilities ─────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_HTML_ENTITIES = {
    "&gt;": ">", "&lt;": "<", "&amp;": "&",
    "&#x27;": "'", "&quot;": '"', "&#x2F;": "/", "&nbsp;": " ",
}


def strip_html(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    return _WHITESPACE_RE.sub(" ", text).strip()


# ── Archetype classification ───────────────────────────────────────────────────


def verify_archetype(text: str, archetype_name: str) -> bool:
    """Confirm that at least one verify_keyword appears in lowercased *text*."""
    lower = text.lower()
    return any(kw in lower for kw in ARCHETYPES[archetype_name]["verify_keywords"])


# ── Step 1: search for candidate comments ─────────────────────────────────────


async def search_comments_for_archetype(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    archetype_name: str,
    target: int,
) -> list[dict]:
    """
    Query Algolia for comments matching *archetype_name* keywords.

    Iterates over (search_term, time_window) pairs to maximise yield — Algolia
    caps results per query, so querying separate date ranges surfaces different
    comments.  Results are deduped by objectID.

    Returns raw Algolia hit dicts (not yet HNTrainingSample).
    """
    hits: list[dict] = []
    seen_ids: set[str] = set()
    search_terms = ARCHETYPES[archetype_name]["search_terms"]

    for term in search_terms:
        if len(hits) >= target:
            break

        for (window_start, window_end) in TIME_WINDOWS:
            if len(hits) >= target:
                break

            # Build Algolia numeric filter for this time window
            if window_end is not None:
                numeric_filter = f"created_at_i>{window_start},created_at_i<{window_end}"
            else:
                numeric_filter = f"created_at_i>{window_start}"

            page = 0
            while len(hits) < target:
                data = await fetch_json(
                    session,
                    HN_SEARCH_URL,
                    params={
                        "tags": "comment",
                        "query": term,
                        "hitsPerPage": 50,
                        "page": page,
                        "numericFilters": numeric_filter,
                    },
                    sem=sem,
                )
                if not data or not data.get("hits"):
                    break

                for hit in data["hits"]:
                    oid = hit.get("objectID", "")
                    if oid in seen_ids:
                        continue
                    seen_ids.add(oid)

                    raw_text = hit.get("comment_text") or ""
                    if not raw_text:
                        continue

                    text = strip_html(raw_text)
                    if not verify_archetype(text, archetype_name):
                        continue

                    hits.append(hit)

                page += 1
                # Partial page → no more results for this term+window
                if len(data["hits"]) < 50:
                    break

    return hits[:target]


# ── Step 2: resolve parent text ────────────────────────────────────────────────


async def resolve_parent_text(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    hit: dict,
) -> str:
    """
    Return the text of the comment's direct parent.

    - If the comment is top-level (parent_id == story_id), return the story
      title (+ body for Ask/Show HN posts).
    - Otherwise fetch the parent comment from /items/{id} and return its text.
    """
    story_id = str(hit.get("story_id") or "")
    parent_id = str(hit.get("parent_id") or "")

    if parent_id == story_id:
        title = hit.get("story_title") or ""
        body = strip_html(hit.get("story_text") or "")
        return body if body else title

    parent = await fetch_json(session, HN_ITEM_URL.format(id=parent_id), sem=sem)
    if parent:
        raw = parent.get("text") or ""
        text = strip_html(raw)
        if text:
            return text

    return hit.get("story_title") or ""


# ── Step 3: assemble samples ───────────────────────────────────────────────────


async def build_sample(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    hit: dict,
    archetype_name: str,
) -> HNTrainingSample | None:
    """Resolve context and assemble a single HNTrainingSample from an Algolia hit."""
    text = strip_html(hit.get("comment_text") or "")
    if not text:
        return None

    story_title = hit.get("story_title") or ""
    story_id = hit.get("story_id") or ""
    story_url = hit.get("story_url") or f"https://news.ycombinator.com/item?id={story_id}"
    score = hit.get("points") or 0
    created_at_i = int(hit.get("created_at_i") or 0)

    parent_text = await resolve_parent_text(session, sem, hit)

    return HNTrainingSample(
        story_title=story_title,
        story_url=story_url,
        parent_text=parent_text,
        archetype_profile=archetype_name,
        ground_truth_comment=text,
        score=score,
        created_at_i=created_at_i,
    )


# ── LLM filter ────────────────────────────────────────────────────────────────

_FILTER_SYSTEM = (
    "You are a strict data-quality filter for a Hacker News comment training dataset. "
    "Respond with a single word: YES to keep a sample, NO to discard it."
)

_FILTER_TEMPLATE = """\
Archetype : {archetype}
Story     : {story_title}
Parent    : {parent_text}
Comment   : {comment}

Should this comment be KEPT as a genuine example of the "{archetype}" archetype \
engaged in authentic HN technical/startup/academic discussion?

Discard (NO) if the comment is:
- A job posting, hiring ad, or recruiting message
- A product listing, advertisement, or pure self-promotion
- A generic "who is hiring" / "seeking freelancer" post
- Clearly off-topic or spam
- Too short/generic to meaningfully represent the archetype

Keep (YES) if it is a real discussion comment that authentically represents \
the archetype's voice and concerns.

Answer YES or NO."""


async def llm_keep_sample(
    client: AsyncAzureOpenAI,
    sem: asyncio.Semaphore,
    sample: HNTrainingSample,
    model: str,
) -> bool:
    """Ask the LLM whether *sample* is worth keeping.

    Retries up to LLM_FILTER_MAX_RETRIES times on 429 rate-limit responses
    using exponential back-off with jitter.  Non-rate-limit errors fail open
    (sample is kept) to avoid silently dropping data.
    """
    prompt = _FILTER_TEMPLATE.format(
        archetype=sample.archetype_profile,
        story_title=sample.story_title[:120],
        parent_text=sample.parent_text[:300],
        comment=sample.ground_truth_comment[:600],
    )
    for attempt in range(1, LLM_FILTER_MAX_RETRIES + 1):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _FILTER_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=5,
                    temperature=0,
                )
            answer = resp.choices[0].message.content.strip().upper()
            return answer.startswith("YES")
        except openai.RateLimitError:
            wait = LLM_FILTER_RETRY_BASE ** attempt + random.uniform(0, 1)
            print(
                f"[llm-filter] rate limited — backing off {wait:.1f}s "
                f"(attempt {attempt}/{LLM_FILTER_MAX_RETRIES})",
                file=sys.stderr,
            )
            await asyncio.sleep(wait)
        except Exception as exc:
            print(f"[llm-filter] error — keeping sample: {exc!r}", file=sys.stderr)
            return True  # fail-open on unexpected errors

    print("[llm-filter] max retries exceeded — keeping sample", file=sys.stderr)
    return True


async def llm_filter_pass(
    samples: list[HNTrainingSample],
    model: str,
) -> list[HNTrainingSample]:
    """
    Run all *samples* through the LLM filter concurrently.
    Prints a per-archetype breakdown of how many were kept vs discarded.
    """
    client = AsyncAzureOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )
    sem = asyncio.Semaphore(LLM_FILTER_CONCURRENCY)

    print(f"[llm-filter] Filtering {len(samples)} samples with {model} …")
    tasks = [llm_keep_sample(client, sem, s, model) for s in samples]
    keep_flags = await asyncio.gather(*tasks)

    kept = [s for s, keep in zip(samples, keep_flags) if keep]
    dropped = len(samples) - len(kept)

    kept_counts = Counter(s.archetype_profile for s in kept)
    drop_counts = Counter(
        s.archetype_profile for s, keep in zip(samples, keep_flags) if not keep
    )
    width = max((len(k) for k in ARCHETYPES), default=0) + 2
    print(f"[llm-filter] Kept {len(kept)}, discarded {dropped}.")
    print(f"\n  {'Archetype':<{width}}  {'kept':>5}  {'dropped':>7}")
    print(f"  {'-'*width}  {'-----':>5}  {'-------':>7}")
    for name in ARCHETYPES:
        print(f"  {name:<{width}}  {kept_counts.get(name, 0):>5}  {drop_counts.get(name, 0):>7}")
    print()

    return kept


# ── Main pipeline ──────────────────────────────────────────────────────────────


async def run_pipeline(
    per_archetype: int = RESULTS_PER_ARCHETYPE,
    llm_filter: bool = False,
    filter_model: str = LLM_FILTER_MODEL,
) -> None:
    """
    Full pipeline:
      1. For each archetype, search Algolia across all time windows
      2. Concurrently resolve each comment's parent text
      3. (Optional) LLM filter pass to remove noise
      4. Write results to OUTPUT_FILE as newline-delimited JSON
         sorted by created_at_i ascending (oldest first) so downstream
         temporal train/eval splits are straightforward.
    """
    load_env_local()

    if llm_filter and not os.environ.get("OPENAI_API_KEY"):
        print(
            "error: --llm-filter requires OPENAI_API_KEY to be set.\n"
            "  export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    sem = asyncio.Semaphore(CONCURRENCY)

    all_samples: list[HNTrainingSample] = []

    async with aiohttp.ClientSession(connector=connector) as session:
        for archetype_name in ARCHETYPES:
            print(f"[search]  {archetype_name} …")
            hits = await search_comments_for_archetype(
                session, sem, archetype_name, target=per_archetype
            )
            print(f"          {len(hits)} candidates found across time windows. Resolving parents …")

            tasks = [build_sample(session, sem, hit, archetype_name) for hit in hits]
            results = await asyncio.gather(*tasks)

            samples = [s for s in results if s is not None]
            all_samples.extend(samples)
            print(f"          {len(samples)} samples assembled.\n")

    # ── Optional LLM filter ────────────────────────────────────────────────────
    if llm_filter:
        all_samples = await llm_filter_pass(all_samples, model=filter_model)

    # Sort by timestamp ascending so the JSONL file is chronologically ordered.
    # This makes it easy to do a clean temporal train/eval split downstream
    # (train = older records, eval = newer records) with no data leakage.
    all_samples.sort(key=lambda s: s.created_at_i)

    # ── Write output ───────────────────────────────────────────────────────────
    print(f"Writing {len(all_samples)} total samples to {OUTPUT_FILE} …")
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for sample in all_samples:
            fh.write(sample.model_dump_json() + "\n")

    print(f"Saved to {OUTPUT_FILE.resolve()}\n")

    # ── Summary ────────────────────────────────────────────────────────────────
    counts = Counter(s.archetype_profile for s in all_samples)
    width = max((len(k) for k in counts), default=0) + 2
    if all_samples:
        from datetime import datetime, timezone
        oldest = datetime.fromtimestamp(all_samples[0].created_at_i, tz=timezone.utc).strftime("%Y-%m-%d")
        newest = datetime.fromtimestamp(all_samples[-1].created_at_i, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"── Date range: {oldest} → {newest}")
    print("── Archetype breakdown " + "─" * 28)
    for name, count in counts.most_common():
        print(f"  {name:<{width}} {count:>4} samples")
    print("─" * 50)
    print(f"\n  Total samples : {len(all_samples)}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape HN comments into archetype training data."
    )
    parser.add_argument(
        "--per-archetype",
        type=int,
        default=RESULTS_PER_ARCHETYPE,
        metavar="N",
        help=f"Target comments to collect per archetype (default: {RESULTS_PER_ARCHETYPE})",
    )
    parser.add_argument(
        "--llm-filter",
        action="store_true",
        default=False,
        help="Run an LLM pass to remove job posts, ads, and off-topic comments",
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        default=LLM_FILTER_MODEL,
        metavar="MODEL",
        help=f"Azure deployment name for filtering (default: {LLM_FILTER_MODEL})",
    )
    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            per_archetype=args.per_archetype,
            llm_filter=args.llm_filter,
            filter_model=args.filter_model,
        )
    )
