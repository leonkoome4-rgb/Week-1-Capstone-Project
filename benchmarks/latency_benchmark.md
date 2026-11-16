Prompt v3, 7 test cases x 3 run(s) per engine. Engines called directly (no fallback, no safety net).

| Engine | Model | Calls | Valid JSON+schema | Mean latency | Median | p95 | Cold start | Critical-flag accuracy |
|---|---|---|---|---|---|---|---|---|
| cloud | gpt-4o-mini | 21 | 0% | n/a | n/a | n/a | n/a | n/a |
| local | llama3.2 | 21 | 86% | 8.05s | 7.60s | 9.60s | 19.60s | 100% |

- cloud failure example: MissingCredentials: OPENAI_API_KEY is not set (see .env.example)
- local failure example: SchemaError: clinical_reasoning_summary must be a non-empty string
