# Heuristic Filters Spec

Cheap, deterministic pre-filters for agentic trace training data. Run before LLM-based pivot filtering to reduce cost and improve signal quality.

---

## Problem

The current pipeline has one heuristic filter (`min_completion_chars`) and then jumps straight to LLM-based pivot filtering ($0.27/100 traces). Many examples that the pivot judge would trivially classify as "downstream" or "trivial" can be caught with free heuristic checks first. Additionally, dataset-level issues (outcome imbalance, duplicate completions) are invisible until training fails.

---

## Research Basis

| Filter | Research | Key Finding |
|--------|----------|-------------|
| **Deduplication** | SemDeDup (arXiv:2303.09540), DRIVE (arXiv:2511.06307) | Dedup removes 50% of pretraining data with minimal quality loss, improves OOD perf. Duplicate completions → identical gradients → wasted compute. |
| **Tool result relay** | PivotRL (arXiv:2603.21383) | 71% of turns produce zero gradient. Relay turns are canonical zero-variance turns — any policy would produce the same output. |
| **Outcome balance** | RC-GRPO (arXiv:2602.03025), Online Difficulty Filtering (arXiv:2504.03380) | Training on all-success data → variance collapse → no learning. Optimal pass rate range is (0.3, 0.7). |

---

## Design Principles

1. **Every filter returns `FilterResult` with structured `DropReason`.** Kept, dropped (with machine-readable reasons), inspectable. No silent data loss.
2. **Filters are composable.** Each is standalone for notebook users. `apply_filters()` composes them for the wizard/Modal with accumulated results.
3. **Deterministic, no LLM calls.** These are free to run. The expensive pivot judge runs on the reduced set.
4. **Conservative defaults.** False positives (dropping a good example) are worse than false negatives (keeping a bad one). The pivot judge catches what heuristics miss.
5. **Per-example filters are commutative; dataset-level filters are order-dependent.** `apply_filters()` enforces correct ordering.

---

## Core Types

### `DropReason` — structured drop metadata

Replaces unstructured string reasons. Machine-readable for the wizard UI.

```python
@dataclass(frozen=True)
class DropReason:
    filter: str        # stage name: "heuristic", "tool_relay", "dedup", "tool_calls"
    reason: str        # specific reason: "too_short", "relay", "duplicate", "excluded"
    detail: str | None = None  # optional: cluster ID, tool name, etc.

    def __str__(self) -> str:
        if self.detail:
            return f"{self.filter}:{self.reason}:{self.detail}"
        return f"{self.filter}:{self.reason}"
```

### Updated `FilterResult`

```python
@dataclass
class FilterResult:
    kept: list[TrainingExample]
    dropped: list[tuple[TrainingExample, DropReason]]

    @property
    def summary(self) -> dict[str, int]:
        """Counts per filter stage."""
        counts: dict[str, int] = {}
        for _, reason in self.dropped:
            key = reason.filter
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def summary_detail(self) -> dict[str, int]:
        """Counts per filter:reason pair."""
        counts: dict[str, int] = {}
        for _, reason in self.dropped:
            key = f"{reason.filter}:{reason.reason}"
            counts[key] = counts.get(key, 0) + 1
        return counts
```

**Migration:** `FilterResult.dropped` changes from `list[tuple[TrainingExample, str]]` to `list[tuple[TrainingExample, DropReason]]`. Existing code that reads `reason` as a string will break — this is intentional. The old `apply_heuristic_filters` is the only consumer and will be updated in the same PR.

---

## Composition Layer

### `apply_filters()` — pipeline runner

Runs filters in sequence, accumulates all drops into one `FilterResult`. Config is data (serializable for Modal).

```python
# Filter registry: name → function
FILTER_REGISTRY: dict[str, Callable] = {
    "heuristic": apply_heuristic_filters,
    "tool_relay": filter_tool_result_relay,
    "tool_calls": filter_by_tool_calls,
    "dedup": deduplicate_completions,
}

def apply_filters(
    examples: list[TrainingExample],
    steps: list[tuple[str, dict[str, Any]]],
) -> FilterResult:
    """Run named filters in sequence, accumulate all drops.

    Each step is ``(filter_name, kwargs_dict)``. Filter names are looked
    up in ``FILTER_REGISTRY``.

    Per-example filters (heuristic, tool_relay, tool_calls) are
    commutative. Dataset-level filters (dedup) must run after per-example
    filters — the function enforces this by running all per-example
    steps first, then dataset-level steps, regardless of input order.

    Example::

        result = apply_filters(examples, [
            ("heuristic", {"min_completion_chars": 50}),
            ("tool_relay", {"overlap_threshold": 0.5}),
            ("tool_calls", {"exclude_tools": ["get_user", "find_order"]}),
            ("dedup", {"similarity_threshold": 0.85}),
        ])
    """
```

**Ordering enforcement:** `apply_filters()` partitions steps into per-example and dataset-level, runs per-example first regardless of input order. This prevents the order-dependence bug where dedup sees different inputs based on step ordering.

**Wizard integration:** The Modal worker receives `steps` as JSON (list of `[name, kwargs]` pairs). No callables over the wire.

### Diagnostics are separate

`check_outcome_balance()` is NOT a filter and is NOT part of `apply_filters()`. It's called separately on the final dataset. The wizard calls it after all filtering and displays a warning banner if imbalanced.

---

## Filters

### 1. `filter_tool_result_relay` (per-example, commutative)

Detect text-response turns where the assistant mostly restates what a tool returned. The decision point was the tool call (which tool, what args), not the formatting of its result.

```python
def filter_tool_result_relay(
    examples: list[TrainingExample],
    *,
    overlap_threshold: float = 0.5,
) -> FilterResult:
```

**Algorithm:**
1. Collect ALL `role="tool"` messages from `prompt_messages`
2. If no tool results precede this turn, keep the example
3. If the completion has tool calls of its own, keep (it's making a new decision)
4. Extract text content from tool results. For JSON payloads, extract leaf string values only (not keys or structural tokens)
5. Tokenize both the tool result text and the completion text (lowercased words, stop words removed)
6. Compute max token overlap ratio against any single tool result: `max(|completion_tokens ∩ tool_i_tokens| / |completion_tokens|)`
7. If max overlap > threshold, drop

**JSON handling:** Tool results are often structured (`{"order_id": "ORD-123", "status": "shipped"}`). Tokenizing raw JSON produces garbage. Strategy: attempt `json.loads()`. If successful, recursively extract leaf string/number values and tokenize those. If not JSON, tokenize as plain text.

**Stop words** (English function words only, 25 words):
```
the, a, an, is, are, was, were, has, have, had, been,
i, you, we, it, they, that, this,
for, to, of, and, or, but, in, on, with
```

This list is deliberately minimal. Domain terms (order, refund, shipping) are NOT stop words — they carry signal for overlap detection. Only grammatical function words that inflate overlap without indicating relay are removed.

**Drop reason:** `DropReason(filter="tool_relay", reason="relay")`.

**Guard:** Only applies to text-only completions (no tool calls). If the assistant calls another tool after relaying, that's a decision — keep it.

**Expected reduction:** 15-25% of text-response turns.

### 2. `deduplicate_completions` (dataset-level, order-dependent)

Remove near-duplicate assistant completions. Training on many copies of the same formulaic response biases the policy and wastes gradient updates.

```python
def deduplicate_completions(
    examples: list[TrainingExample],
    *,
    similarity_threshold: float = 0.85,
    max_per_cluster: int = 3,
) -> FilterResult:
```

**Algorithm:**
1. **Canonical sort** by `(trace_id, turn_index)` before clustering — ensures deterministic results regardless of input order
2. Normalize each completion: lowercase, replace entities with placeholders
3. Compute trigram Jaccard similarity between normalized completions
4. Greedy clustering: assign each example to first cluster with similarity > threshold, or start new cluster
5. Keep first `max_per_cluster` examples per cluster (by canonical sort order), drop the rest

**Entity normalization patterns:**
- `[A-Z]{2,}\d+` → `<ID>` (order IDs like ORD-123, AB1234)
- `\d{3,}` → `<NUM>` (numbers with 3+ digits)
- `\d{1,2}/\d{1,2}/\d{2,4}` → `<DATE>` (dates like 3/15/2024)
- `\$[\d,.]+` → `<PRICE>` (prices like $29.99)
- `\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b` → `<EMAIL>`

These are conservative — they target common entity formats in customer service traces without destroying meaningful content.

**Drop reason:** `DropReason(filter="dedup", reason="duplicate", detail=f"cluster_{cluster_id}")`.

**Complexity:** O(n^2) worst case (all completions unique, every example compared to every cluster). O(n*k) typical for production traces where k << n (many similar completions collapse into few clusters). For datasets < 5k examples this is under 1 second. If it becomes a bottleneck, switch to MinHash locality-sensitive hashing.

**Expected reduction:** 10-20% of total examples. Higher for production traces.

### 3. `filter_by_tool_calls` (per-example, user-configured)

Already implemented. User specifies which tool calls to exclude after inspecting `detect_tools()` output. Different from other filters: requires user input, not automatic.

```python
def filter_by_tool_calls(
    examples: list[TrainingExample],
    exclude_tools: list[str],
) -> FilterResult:
```

**Drop reason:** `DropReason(filter="tool_calls", reason="excluded", detail=tool_name)`.

### 4. `apply_heuristic_filters` (per-example, commutative)

Existing filter. Drops short completions.

```python
def apply_heuristic_filters(
    examples: list[TrainingExample],
    *,
    min_completion_chars: int = 50,
) -> FilterResult:
```

**Drop reason:** `DropReason(filter="heuristic", reason="too_short")`.

---

## Diagnostic

### `check_outcome_balance` (dataset-level, advisory)

Not a filter. Returns a diagnostic. Called separately from `apply_filters()`.

```python
def check_outcome_balance(
    examples: list[TrainingExample],
    *,
    score_name: str = "task_success",
    success_threshold: float = 0.5,
    min_failure_fraction: float = 0.15,
) -> OutcomeBalance:
```

**Score semantics:** Binary classification using `success_threshold`. An example is a success if `score >= success_threshold`, failure if `score < success_threshold`. Examples missing the score are `unknown`. This handles both binary scores (0/1) and continuous scores (0.0-1.0) — threshold 0.5 works for both.

```python
@dataclass
class OutcomeBalance:
    total: int
    success_count: int
    failure_count: int
    unknown_count: int
    failure_fraction: float   # failure / (success + failure), excludes unknown
    is_balanced: bool
    message: str | None
```

---

## What We're NOT Adding

- **Completion entropy filter** — not research-backed as a training data filter
- **Echo turn detection** — overlaps with tool result relay
- **Context length filter** — the trainer handles truncation
- **Turn position filters** — too blunt
- **Response length filters** — weak signal
- **Regex content patterns** — brittle, domain-specific (removed `_AUTH_PATTERNS` for this reason)

---

## User Interface

### Notebook flow (standalone filters)

```python
from cgft.traces.processing import (
    build_training_examples,
    apply_heuristic_filters,
    filter_tool_result_relay,
    filter_by_tool_calls,
    deduplicate_completions,
    check_outcome_balance,
    detect_tools,
    split_dataset,
)
from cgft.traces.pivot import apply_pivot_filter

# 1. Build examples
examples = build_training_examples(traces)

# 2. Cheap heuristic filters
result = apply_heuristic_filters(examples, min_completion_chars=50)
examples = result.kept

result = filter_tool_result_relay(examples)
examples = result.kept

# 3. User-specified tool exclusions (before pivot to save LLM cost)
tools = detect_tools(traces)
for t in tools.tools:
    print(f"  {t.name} ({t.call_count} calls)")
result = filter_by_tool_calls(examples, ["get_user_details", "find_order"])
examples = result.kept

# 4. Dedup (dataset-level, after per-example filters)
result = deduplicate_completions(examples)
examples = result.kept

# 5. Dataset health check
balance = check_outcome_balance(examples)
if not balance.is_balanced:
    print(f"Warning: {balance.message}")

# 6. LLM pivot filtering (expensive, on reduced set)
result = apply_pivot_filter(examples, traces=traces, llm_client=client)
examples = result.kept

# 7. Split
train, eval_data = split_dataset(examples, train_count=400, eval_count=100)
```

### Wizard flow (composed pipeline via Modal)

```python
# Modal worker receives serializable config:
result = apply_filters(examples, [
    ("heuristic", {"min_completion_chars": 50}),
    ("tool_relay", {"overlap_threshold": 0.5}),
    ("tool_calls", {"exclude_tools": ["get_user_details"]}),
    ("dedup", {"similarity_threshold": 0.85}),
])

# result.summary → {"heuristic": 12, "tool_relay": 180, "tool_calls": 45, "dedup": 95}
# result.summary_detail → {"heuristic:too_short": 12, "tool_relay:relay": 180, ...}
```

---

## Pipeline Ordering

```
fetch_traces()
  → build_training_examples()
    → apply_heuristic_filters()          # per-example: min_completion_chars
    → filter_tool_result_relay()         # per-example: drop relay turns
    → filter_by_tool_calls()             # per-example: user-specified (FREE, before pivot)
    → deduplicate_completions()          # dataset-level: dedup (after per-example)
    → check_outcome_balance()            # diagnostic: advisory warning
    → apply_pivot_filter()               # LLM judge (expensive, on reduced set)
    → split_dataset()
```

`filter_by_tool_calls` runs before `apply_pivot_filter` — it's free, so don't pay for LLM analysis on examples that will be dropped.

---

## Known Inconsistency

`apply_score_filter()` (existing) returns `list[TrainingExample]`, not `FilterResult`. This predates the `FilterResult` contract. It should be migrated in a follow-up PR but is not blocking for this work.

---

## Implementation Plan

### Phase 1: Core types
- Add `DropReason` dataclass to `processing.py`
- Migrate `FilterResult.dropped` from `tuple[TrainingExample, str]` to `tuple[TrainingExample, DropReason]`
- Update `apply_heuristic_filters` and `filter_by_tool_calls` to use `DropReason`
- Update all existing tests

### Phase 2: Tool result relay filter
- Add `filter_tool_result_relay()` to `processing.py`
- JSON value extraction for structured tool results
- Stop word list (25 English function words)
- Tests: relay detected, non-relay kept, no tool result kept, tool-call completion kept, JSON tool result, threshold boundary, max overlap across multiple tool results

### Phase 3: Deduplication
- Add `deduplicate_completions()` to `processing.py`
- Canonical sort by `(trace_id, turn_index)`
- Entity normalization (5 regex patterns)
- Trigram Jaccard + greedy clustering
- Tests: exact duplicates, near-duplicates, unique kept, max_per_cluster, entity normalization, determinism (same result regardless of input order)

### Phase 4: Outcome balance
- Add `OutcomeBalance` dataclass and `check_outcome_balance()` to `processing.py`
- Tests: balanced, imbalanced, missing scores, custom threshold, continuous scores

### Phase 5: Composition
- Add `FILTER_REGISTRY` and `apply_filters()` to `processing.py`
- Ordering enforcement (per-example before dataset-level)
- Tests: accumulated drops, ordering enforcement, unknown filter name error, serializable config roundtrip

### Phase 6: Integration
- Update `docs/traces.md`
- Update E2E notebook

---

## Complexity Budget

All per-example filters: O(n). Dedup: O(n^2) worst case, O(n*k) typical. For datasets < 5k examples, all heuristic filters combined complete in < 1 second.
