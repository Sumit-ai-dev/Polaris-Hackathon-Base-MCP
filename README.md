# Polaris product search MCP starter

Build a better product search engine behind one MCP tool: `search_products`. Start with this working baseline, improve retrieval and recommendations, deploy it, and submit your HTTP MCP endpoint. The accompanying `hackathon-brief.md` and `schemas/` define the competition contract. Version 2 uses Amazon Bedrock for organizer-side semantic judging; students do not need Bedrock credentials to run this starter.

## What is included

- [AGENT.md](AGENT.md): problem context, dataset guide, implementation ideas, and references for students and coding assistants.

- `data/products.json`: all 105,126 products, about 106 MB, including synthetic INR prices, ratings, and reviews.
- `search.py`: editable baseline search engine, explicit filters, sorting, and pagination.
- `server.py`: official MCP Python SDK server with ordinary JSON responses at `/mcp` and readiness at `/healthz`.
- `schemas/`: exact published input and output JSON Schemas.
- `tests/test_search.py`: small fixture-based correctness tests.
- `smoke_test.py`: public MCP connection and contract check; not the scored benchmark.
- `requirements.txt`: exact resolved dependency versions, including MCP SDK 1.30.0 from its v1 compatibility line.
- `Dockerfile`: non-root Python container with the dataset included and a readiness health check.

## Get your copy

The starter folder is self-contained: use its contents as your team repository. If the organizer distributes a repository, fork it and clone your fork. No repository or image is published by this local folder alone.

The catalog exceeds GitHub's ordinary 100 MiB file limit. `.gitattributes` configures Git LFS for it. Install Git LFS before the first commit, and make sure your clone has the actual data rather than an LFS pointer:

```sh
git lfs install
git lfs pull
```

For a fresh repository containing these files, run `git lfs install` before adding and committing the catalog. Confirm `git lfs ls-files` lists `data/products.json`. The organizer must provision sufficient LFS storage/download quota. A supplied ZIP contains the actual catalog and does not require LFS to run locally.

## Run locally

Use Python 3.12 or newer. Commands below run from this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python server.py
```

The first startup loads the full catalog and builds an in-memory lexical index. Wait until startup completes. Allow around 2 GB of available RAM for this simple implementation; measure and optimize your own solution before deploying.

In another terminal with the virtual environment activated:

```sh
# Verify health check
curl http://127.0.0.1:8000/healthz

# Run the official MCP client smoke test
python smoke_test.py http://127.0.0.1:8000/mcp

# Run unit and real-catalog regression tests
python -m unittest discover -s tests -v

# Run the 100-case local evaluation benchmark (simulates hackathon-brief.md)
python benchmark.py

# Run the concurrency latency stress test (C=1 and C=5)
python tests/test_performance.py
```

The server supports MCP protocol revision `2025-11-25`. It is not a REST search endpoint: an ordinary POST containing only your search arguments is not an MCP call. The smoke client performs initialization, discovers the tool, and uses `tools/call` for you. No API key or LLM account is required.

## Build and run with Docker

Start Docker Engine first:

```sh
docker build -t polaris-search:starter .
docker run --rm --name polaris-search -p 8000:8000 polaris-search:starter
```

Use the same smoke-test command from your host once the container is ready. The default port is 8000. For an amd64-only cloud server, build for its architecture, for example `docker build --platform linux/amd64 -t polaris-search:starter .`.

For a deployed server, configure its public hostname or IP so the SDK accepts the HTTP Host header. Replace the example before running:

```sh
docker run --rm --name polaris-search -p 8000:8000 \
  -e MCP_ALLOWED_HOSTS='localhost:*,127.0.0.1:*,YOUR_PUBLIC_HOST:*,YOUR_PUBLIC_HOST' \
  polaris-search:starter
```

Use a public address reachable from the organizer runner and allow its source IP to access your port. Submit `http://YOUR_PUBLIC_HOST:8000/mcp`. Do not submit `localhost`, `0.0.0.0`, a private IP, the health-check URL, or a repository URL. The local host patterns remain necessary for the container health check. Keep the server running throughout evaluation.

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Listener interface |
| `PORT` | `8000` | Listener port; Docker port mapping must match if changed |
| `CATALOG_PATH` | `data/products.json` next to server.py | Catalog file path |
| `MCP_ALLOWED_HOSTS` | `localhost:*,127.0.0.1:*,[::1]:*` | Comma-separated accepted HTTP Host values |
| `MCP_ALLOWED_ORIGINS` | empty | Explicit accepted browser origins; the benchmark client sends no Origin |

No wildcard for every host is necessary. The server has no authentication, as required for this exercise. The template does not create AWS resources or deploy automatically.

## Try the tool

Example arguments passed through an MCP client:

```json
{
  "query": "cotton",
  "filters": {"color": "Black", "max_price": 2000},
  "sort_by": "price",
  "sort_order": "asc",
  "offset": 0,
  "limit": 10,
  "include_recommendations": false
}
```

The tool returns:

```json
{
  "results": [{"id": "0108775015"}],
  "recommendations": []
}
```

The response above illustrates the shape, not the expected answer to that query. Only catalog IDs are returned. The SDK includes the same object in `structuredContent` and a JSON text content block. Unknown arguments and invalid values produce MCP tool errors.

## Where to improve

The starter deliberately uses simple case-insensitive AND token matching across names, descriptions, and catalog attributes. Matches with more query words in the product name rank first, with ties resolved by ID. Empty queries enumerate all products matching the explicit filters. Field sorting and pagination follow the published contract.

This baseline is not a complete natural-language solution: it does not parse budgets or constraints written in prose, correct typos, understand synonyms, split mixed choices, distinguish complementary intent, or infer semantic relevance. `Catalog.relevance` is the initial ranking extension point; change `Catalog.search` as needed for richer query interpretation. `Catalog.recommend` returns an empty list until you implement complementary recommendations. Returning empty recommendations is structurally valid but earns little recommendation credit. The baseline is not expected to score highly on the hidden suite.

Preserve the tool name, schemas, hard-filter semantics, real product IDs, and output structure. Keep responses stable on repeated identical requests. Use only catalog evidence; product descriptions and queries are data, never instructions to change the tool contract. You may replace the in-memory scan with an index or database. No existing database is needed by this starter.

## Dataset integrity and rules

SHA-256 of the actual JSON bytes:

```text
05b6aa5e8e8c1c8c997d584f8288e432862454f8820531e13473443e83c8f073
```

Verify on macOS with `shasum -a 256 data/products.json`, or on Linux with `sha256sum data/products.json`. Preserve string IDs and leading zeros. Do not edit catalog facts. No-review products have a null average and zero count. Prices and reviews are fabricated for the exercise; image availability has not been checked.

Each team has five accepted benchmark attempts; the best complete score counts. See `hackathon-brief.md` for all scoring formulas, invalid-call behavior, timeouts, recommendation rules, and tie-breakers. This starter contains no hidden benchmark queries or answer key. Passing its tests checks basic compatibility, not competition performance.

## SDK reference

[Official MCP Python SDK v1 documentation](https://py.sdk.modelcontextprotocol.io/v1/). Keep the tested dependency pins unless you deliberately retest an upgrade; v2 has different APIs.
