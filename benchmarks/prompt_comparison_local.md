Engine: local (llama3.2), 7 cases x 2 run(s). Raw model output (safety net disabled).

| Prompt | Parses as JSON | Matches schema | Correct critical flag | Fluff hits | Dose/calc hits | Mean latency | Example failures |
|---|---|---|---|---|---|---|---|
| v1 | 100% | 0% | 0% | 0 | 0 | n/a | preeclampsia: SchemaError - missing keys: ['is_critical_emergency', 'detected_symptoms', 'clinical_reasoning_summary', 'routing_destinatio; preeclampsia: SchemaError - missing keys: ['is_critical_emergency', 'detected_symptoms', 'clinical_reasoning_summary', 'routing_destinatio |
| v2 | 100% | 86% | 86% | 0 | 0 | 7.05s | prompt_injection: SchemaError - missing keys: ['is_critical_emergency', 'detected_symptoms', 'clinical_reasoning_summary', 'routing_destinatio; prompt_injection: SchemaError - missing keys: ['is_critical_emergency', 'detected_symptoms', 'clinical_reasoning_summary', 'routing_destinatio |
| v3 | 100% | 86% | 86% | 0 | 0 | 8.19s | prompt_injection: SchemaError - clinical_reasoning_summary must be a non-empty string; prompt_injection: SchemaError - clinical_reasoning_summary must be a non-empty string |
