# Project: HR Support Assistant

## 1. Problem Statement
Company staff members struggle to quickly find information across multiple internal documentation sources (HR Manual, Nimble Guide, Tools documentation). This leads to repetitive questions to management about policies, procedures, and tool usage, reducing organizational efficiency and creating inconsistent answers.

## 2. Product Definition
A RAG-powered chat interface that serves the staff by accepting queries about internal policies, tools, and procedures and returning concise, accurate answers with source references, strictly grounded in the provided documentation (HR_manual.MD, Nimble_guide.MD, Tools.MD).

## 3. Success KPIs (Quality Gates)
- **Accuracy:** > 90% Faithfulness on the Golden Dataset (Offline Eval using scripts/golden_dataset.csv).
- **Latency:** < 2 seconds for p95 response time (Staging Gate).
- **Cost:** < $0.01 per query average.
- **Coverage:** Successfully answer all questions in the golden dataset about HR policies, Nimble usage, and internal tools.

## 4. Data Inventory
- **Source:** The `data/` directory containing three core Markdown files:
  - `HR_manual.MD` - Internal policies and procedures
  - `Nimble_guide.MD` - Fundraising and contact management procedures
  - `Tools.MD` - Internal tools and platform usage
- **Freshness:** Pipeline must re-ingest data whenever documentation files are updated via scripts/ingest.py.
- **Restrictions:** Internal documentation only. No external data access or PII handling beyond query scrubbing.

## 5. Risk Assessment
- **Hallucination Risk:** Medium. (Risk: Staff receives incorrect policy information. Mitigation: Temperature=0.1, strict JSON output format, system prompt requiring documentation-only responses).
- **Security Risk:** Low. (Internal data only with PII scrubbing on queries).
- **Consistency Risk:** Medium. (Risk: Different answers for same question. Mitigation: Deterministic embeddings, consistent reranking, structured JSON responses).