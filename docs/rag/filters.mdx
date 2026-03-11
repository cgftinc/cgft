# Corpus Search Query Filtering

We support structured metadata filtering using query DSL for retrieval across chunk sources.

### Default Corpora backend query
If using our default corpus backend, you can add a `filters` field to your searches

```python
result = client.search(
    corpus_id=CORPUS_ID,
    query="meeting follow-up",
    limit=10,
    filters={
        "and": [
            {"field": "date_start", "op": "gte", "value": "2024-01-01"},
            {"field": "participants", "op": "contains_any", "value": ["alice"]},
        ]
    },
)
```

### 2) BE-agnostic: ChunkSource.search
This works independently of the corpus BE (corpora, turbopuffer, future extensions). Can be used in your custom environment or in extending QA generation.

```python
from synthetic_data_prep.corpus.search_schema.builders import all_of, field

filters = all_of(
    field("date_start").gte("2024-01-01"),
    field("participants").contains_any(["alice"]),
)

chunks = source.search_text("meeting follow-up", filter=filters)
```

#### Notes:
- source.search_text(...) works across supported backends.
- Filters are validated against backend capabilities.
- Unsupported operators fail fast with explicit errors.


<br>

### Example of providing a search tool in environment
```python
search_tool_definition = ToolDefinition(
            name="search",
            description="Search using BM25 with optional metadata and filename filtering.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional structured filter object using domain-specific query language (DSL). "
                            "Field condition shape: "
                            "{\"field\":\"<name>\",\"op\":\"eq|in|gte|lte|contains_any|contains_all\",\"value\":...}. "
                            "Logical shape: {\"and\":[...]} or {\"or\":[...]} or {\"not\":{...}}. "
                            "Example: "
                            "{\"and\":[{\"field\":\"metadata_name\",\"op\":\"eq\",\"value\":\"example_value\"},"
                            "{\"field\":\"metadata_name_2\",\"op\":\"gte\",\"value\":123}]}"
                        ),
                    },
                    "filename": {
                        "type": "string",
                        "description": (
                            "Optional filename filter. Simple string for substring match "
                            "(e.g., 'config') or regex pattern (e.g., '.*\\.json$')."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results to return (default 10).",
                    },
                },
                "required": ["query"]
            },
        )
```
