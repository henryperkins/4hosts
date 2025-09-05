
# Enhancing Source-to-Answer Quality

This note distills several ideas for tightening the evidence pipeline and improving the relevance, depth and factual quality of the final LLM answer.

## 1. Collection

- **Domain-specific APIs** – tap Scholar, PubMed, SSRN, patent search, etc. when the classifier flags academic / technical intent.
- **Vector-store recall** – embed crawled documents once, retrieve semantically similar passages before resorting to external search.
- **Cross-encoder re-ranking** – apply a model such as *ms-marco-MiniLM* or *ColBERT* on the merged result set to surface the 15–20 most relevant hits.
- **Reference mining** – crawl citations inside top papers/articles to discover authoritative “second-hop” sources that basic web search misses.

## 2. Filtering & Scoring

- Replace the binary credibility gate with a **weighted score** combining:
  1. Domain reputation.
  2. Author authority (e.g., h-index, organisation).
  3. Evidence density (numbers, quotes, refs).
  4. Recency bonus/penalty.
- **Semantic snippet filter** – drop paragraphs whose sentence-level embedding similarity to the query’s key themes is below a threshold.

## 3. Compression

- Allocate the token budget **proportionally to `(credibility × relevance)`** rather than a flat per-source cut.
- Run retained articles through an **extractive summariser** (TextRank or BART fine-tune) so snippets already capture key facts.

## 4. Isolation / Evidence Structuring

- Emit fragments with a rich schema:

  ```jsonc
  {
    "claim": "Quantum error rates fell 55 % year-on-year",
    "metric": 0.55,
    "units": "fraction",
    "polarity": "positive",
    "confidence": 0.82
  }
  ```

- Group fragments into **agree / disagree / neutral** buckets to force the generator to address conflicts explicitly.

## 5. LLM Prompt & Generation

- Add a short **coverage table** to the prompt (`Theme | Covered? | Best URL`) so the model fills remaining gaps.
- Send fragments as a **JSON array** and require the model to cite by index (`[3]`) – prevents hallucinated URLs.
- Use **role-based prompting** – analytical paradigm receives stats & methods, strategic gets SWOT & market metrics, etc.

## 6. Agentic Loop Tuning

- Lower first-pass threshold to 0.55 but continue only if **quality gain ≥ 0.05** per iteration – stops endless churn.
- Generate follow-up queries from **unsatisfied bigrams** instead of full phrases to reduce query drift.

## 7. Post-Generation Validation

- Pass the draft through a lightweight **fact-checker / claim verifier**. If flagged, feed the claim back into the agentic loop for targeted sourcing.

---

Implementing even a subset of these ideas should shorten research time, raise factual coverage above the 0.75 threshold in fewer iterations, and produce richer, better-grounded answers for end users.

