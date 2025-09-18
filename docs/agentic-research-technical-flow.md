# Comprehensive Breakdown of the Four Hosts Agentic Research Application Technical Flow

Date: 2025-08-31
Status: Adopted as canonical flow doc for the repo. See the Implementation Map at the end for file-level pointers.

---

## 1. User Interface Layer
Purpose: Capture user input, display real-time paradigm distribution, and allow customization of research parameters.

### Components

#### A. Query Input
- Visual: A text box containing the example query:
  > "How can small businesses compete with Amazon?"
- Function:
  - Accepts natural language queries.
  - Triggers the classification engine upon submission.

#### B. Advanced Options
- Visual: A panel with configurable settings:
  - Depth: Standard (default) or Deep (more exhaustive research).
  - Paradigm: Auto (system selects) or Manual (user specifies).
- Function:
  - Adjusts the scope and intensity of the research pipeline.
  - Standard Depth: Faster, focused on high-confidence insights.
  - Deep Depth: Broader search, more sources, higher computational cost.

#### C. Paradigm Distribution Display
- Visual: Real-time bar showing paradigm allocation:
  - Maeve: 40% (Strategic)
  - Dolores: 25% (David vs. Goliath narrative)
  - Bernard: 20% (Analytical)
  - Teddy: 15% (Protective/ethical)
  - Primary Paradigm: Maeve (highlighted).
- Function:
  - Provides transparency into the system’s cognitive focus.
  - Updates dynamically as the query is analyzed.

---

## 2. Paradigm Classification Engine
Purpose: Analyze the query to determine the optimal paradigm(s) for processing.

### Components

#### A. Query Analyzer
- Visual: A box listing extracted metadata:
  - Entities: [small businesses, Amazon]
  - Intent: competitive_strategy
  - Domain: business/commerce
  - Urgency: moderate
  - Complexity: high
- Function:
  - Uses NLP to decompose the query into structured attributes.
  - Entities: Identifies key actors/objects.
  - Intent: Classifies the goal (e.g., strategy, analysis, prediction).
  - Domain: Determines the field (e.g., business, science, politics).
  - Urgency/Complexity: Adjusts research depth and paradigm weighting.

#### B. Paradigm Classifier
- Visual: Keyword-to-paradigm mappings:
  - "compete" → Maeve (strategy)
  - "small vs Amazon" → Dolores (underdog narrative)
  - "businesses" → Bernard (analytical)
  - "help small" → Teddy (protective)
- Function:
  - Uses a rule-based + ML hybrid system to match query terms to paradigms.
  - Maeve: Dominant for competitive strategy queries.
  - Dolores: Triggered by power imbalance themes (e.g., small vs. large).
  - Bernard: Activated for analytical or data-driven questions.
  - Teddy: Engaged for ethical or protective contexts.

#### C. Paradigm Decision
- Visual:
  - Primary Paradigm: Maeve (Strategic) (bolded)
  - Secondary: Dolores
  - Confidence: 78%
- Function:
  - Finalizes the primary paradigm (Maeve) based on:
    - Keyword matches.
    - Domain intent.
    - Historical performance data (e.g., Maeve excels in business strategy).
  - Assigns a confidence score (78% here) to validate the choice.
  - Secondary paradigm (Dolores) provides supplementary context (e.g., antitrust issues).

---

## 3. Context Engineering Pipeline (Maeve Configuration)
Purpose: Tailor the research process to the selected paradigm (here, Maeve for strategic focus).

### Components

#### A. Write Layer
- Visual:
  - Maeve Mode instructions:
    - Map competitive landscape:
      - Amazon strengths/weaknesses.
      - Small business advantages.
      - Market opportunities.
      - Success stories.
    - + Dolores: Note monopoly issues.
- Function:
  - Defines the research scope based on Maeve’s strategic lens.
  - Ensures the system prioritizes competitive analysis.
  - Dolores’ influence: Adds systemic context (e.g., antitrust).

#### B. Select Layer
- Visual:
  - Search Strategy:
    - Primary searches:
      - "compete with Amazon strategy"
      - "small business advantages"
      - "niche market strategies"
      - "Industry reports + case studies"
    - + Dolores: "Amazon antitrust"
- Function:
  - Generates paradigm-specific search queries.
  - Maeve: Focuses on actionable strategies.
  - Dolores: Adds queries about systemic issues (e.g., monopolies).

#### C. Compress Layer
- Visual:
  - Maeve: 40% compression
  - Focus on:
    - Actionable strategies.
    - Implementation steps.
    - Success metrics.
    - Quick wins.
  - Remove:
    - Theory, history.
- Function:
  - Filters out non-actionable or redundant information.
  - 40% compression: Retains only high-value, strategic content.
  - Aligns with Maeve’s preference for pragmatism.

#### D. Isolate Layer
- Visual:
  - Extract Key Intel:
    1. Local advantage strategies.
    2. Niche domination tactics.
    3. Community integration.
    4. Tech leverage points.
    5. Partnership opportunities.
    - + Policy changes coming (Dolores).
- Function:
  - Distills core insights for the final output.
  - Maeve: Prioritizes tactical advantages.
  - Dolores: Ensures systemic factors (e.g., policy) are included.

---

## 4. Research Execution Layer
Purpose: Execute searches, evaluate sources, and extract key findings.

### Components

#### A. Paradigm-Aware Search
- Visual:
  - Searches executed (Maeve priority):
    1. "small business beat Amazon 2024" → 847 results.
    2. "local retailer advantages Amazon" → 523 results.
    3. "niche market strategy examples" → 1,204 results.
    4. "Amazon weakness exploit business" → 392 results.
    5. [Dolores] "Amazon monopoly local" → 621 results.
    6. [Industry] "Retail strategy reports" → 178 results.
  - Total sources analyzed: 3,765.
- Function:
  - Executes paradigm-prioritized searches.
  - Maeve: Dominates with strategy-focused queries.
  - Dolores: Contributes systemic searches (e.g., antitrust).
  - Aggregates results for credibility analysis.

#### B. Source Credibility
- Visual:
  - MAEVE source weights applied:
    - HBR strategy article: 0.95
    - McKinsey report: 0.92
    - Successful case study: 0.88
    - Reddit discussion: 0.45
    - Opinion blog: 0.25
  - High-quality sources: 47.
- Function:
  - Assigns credibility scores based on:
    - Source authority (e.g., HBR > Reddit).
    - Relevance to the paradigm (Maeve favors strategic content).
  - Filters out low-quality sources (e.g., opinion blogs).

#### C. Key Findings
- Visual:
  - Strategic insights extracted:
    - Local same-day > Prime delivery.
    - Personal service competitive edge.
    - Community integration strategies.
    - Successful niche examples (12).
    - Policy changes favoring local.
  - Actionable strategies: 23.
- Function:
  - Extracts paradigm-aligned insights.
  - Maeve: Focuses on tactical advantages (e.g., same-day delivery).
  - Dolores: Highlights systemic opportunities (e.g., policy changes).

---

## 5. Synthesis & Presentation
Purpose: Integrate findings, structure the response, and validate quality.

### Components

#### A. Paradigm Integration
- Visual:
  - MAEVE (Primary): Strategic framework.
  - DOLORES (Support): Systemic context.
  - Output structure:
    - 23 actionable strategies organized.
    - 5 systemic advantages highlighted.
    - 3 case studies for inspiration.
    - Policy momentum documented.
- Function:
  - Merges Maeve’s strategic insights with Dolores’ systemic context.
  - Ensures a cohesive narrative that addresses both tactics and broader trends.

#### B. Answer Generation
- Visual:
  - MAEVE-style response:
    1. Executive Summary (3 key strategies).
    2. Immediate Actions (quick wins).
    3. Strategic Framework (4 phases).
    4. Success Metrics (KPIs).
    - + Context: Amazon vulnerabilities.
- Function:
  - Structures the answer to mirror Maeve’s strategic thinking:
    - Executive Summary: High-level takeaways.
    - Immediate Actions: Quick wins (e.g., same-day delivery).
    - Strategic Framework: Long-term plan.
    - Success Metrics: KPIs to track progress.

#### C. Self-Healing
- Visual:
  - Quality checks passed:
    - Actionable content: 87%
    - Source diversity: Good
    - Bias check: Balanced
    - Paradigm fit: Optimal
  - No paradigm switch needed.
- Function:
  - Validates the output against quality benchmarks:
    - Actionable content: >85% required.
    - Source diversity: Ensures no over-reliance on one source type.
    - Bias check: Confirms balanced perspectives.
    - Paradigm fit: Verifies Maeve was the right choice.
  - Self-corrects if thresholds aren’t met (e.g., switches paradigms if confidence is low).

---

## 6. Final Research Output
Purpose: Deliver a polished, paradigm-optimized response to the user.

### Components

#### A. Strategic Framework
- Visual:
  - Title: "Strategic Framework: Small Businesses Competing with Amazon"
  - Immediate Opportunities (Maeve):
    1. Local Advantage Strategy: Same-day delivery without Prime fees.
    2. Relationship Commerce: Personal service Amazon can’t replicate.
    3. Niche Expertise: Deep category knowledge (e.g., specialty foods).
  - Systemic Context (Dolores):
    - Antitrust scrutiny creating policy opportunities.
    - "Shop local" movement gaining momentum.
  - Implementation Roadmap: [Full plan with citations].
- Function:
  - Presents a clear, actionable strategic plan.
  - Maeve’s influence: Dominates with tactical advice.
  - Dolores’ influence: Provides macro-level context.

#### B. Research Metrics
- Visual:
  - Sources analyzed: 3,765
  - High-quality sources: 47
  - Paradigms used: 2 (Maeve + Dolores)
  - Research time: 4.7 sec
  - Confidence: 92%
- Function:
  - Provides transparency into the research process.
  - Confidence score reflects certainty in the output.
  - Speed demonstrates efficiency.

---

## Key System Behaviors & Design Principles
1. Paradigm-Driven Processing:
   - The system dynamically selects paradigms based on query analysis.
   - Each paradigm (Maeve, Dolores, Bernard, Teddy) has unique strengths:
     - Maeve: Strategy, competition, actionable insights.
     - Dolores: Power dynamics, systemic issues, narratives.
     - Bernard: Analysis, data, structured thinking.
     - Teddy: Ethics, protection, risk mitigation.
2. Layered Context Engineering:
   - Write/Select/Compress/Isolate layers ensure precision and relevance.
   - Each layer is paradigm-configured (e.g., Maeve compresses theory, focuses on action).
3. Self-Optimizing Pipeline:
   - Self-Healing checks validate output quality.
   - Paradigm switching occurs if confidence is low.
4. Speed vs. Depth Trade-offs:
   - Standard Depth: Fast, focused.
   - Deep Depth: Slower, more exhaustive.
5. Transparency & Explainability:
   - Real-time paradigm distribution display.
   - Research metrics (sources, confidence, time) build user trust.

---

## Visual Flow Summary
```
User Query
│
▼
1. User Interface Layer (Input + Paradigm Display)
│
▼
2. Paradigm Classification Engine (Analyze Query → Select Paradigm)
│
▼
3. Context Engineering (Write/Select/Compress/Isolate for Maeve)
│
▼
4. Research Execution (Search → Evaluate Sources → Extract Insights)
│
▼
5. Synthesis (Integrate Paradigms → Generate Answer → Self-Healing)
│
▼
6. Final Output (Strategic Framework + Metrics)
```

---

## Example Walkthrough: "How can small businesses compete with Amazon?"
1. User Input:
   - Query entered; Maeve selected as primary paradigm (40%).
2. Classification:
   - Intent: competitive_strategy → Maeve.
   - Keywords: "small vs Amazon" → Dolores secondary.
3. Context Engineering (Maeve):
   - Write: Focus on competitive landscape.
   - Select: Search for "niche market strategies".
   - Compress: Remove theory, keep actionable tactics.
   - Isolate: Extract local advantage strategies.
4. Research Execution:
   - Sources analyzed; high-quality sources retained.
   - Key finding: "Local same-day > Prime delivery".
5. Synthesis:
   - Maeve + Dolores integrated:
     - Tactics (Maeve) + policy context (Dolores).
6. Final Output:
   - 3 immediate opportunities + systemic trends.
   - High confidence, fast execution.

---

## Potential Edge Cases & Adaptations
| Scenario | System Response |
|---|---|
| Low confidence (<70%) | Switch primary paradigm (e.g., Maeve → Bernard if analytical depth needed). |
| High complexity query | Auto-select Deep Depth; engage multiple paradigms (e.g., Maeve + Bernard). |
| Biased source distribution | Adjust credibility weights; flag in Self-Healing. |
| Urgent query | Prioritize quick wins in Compress Layer; reduce compression to 20%. |
| Ambiguous intent | Request clarification or default to Bernard (analytical) for decomposition. |

---

## Technical Underpinnings (Inferred)
1. NLP & Intent Classification:
   - Transformer-based models for query analysis.
2. Paradigm Selection:
   - Rule-based + ML hybrid:
     - Rules for keyword-paradigm mappings.
     - ML for dynamic weighting (confidence calibration).
3. Search & Credibility Scoring:
   - Custom search orchestration prioritizing paradigm-aligned sources.
   - Credibility model trained on source authority and relevance.
4. Self-Healing:
   - Threshold-based validation (e.g., actionable content >85%).
   - Feedback loop to adjust paradigms or depth if checks fail.

---

## Implementation Map (This Repo)
- UI Layer
  - Frontend entry and form: `frontend/src/components/ResearchFormEnhanced.tsx`
  - Live classification preview: `frontend/src/components/ResearchPage.tsx` (debounced calls to `/paradigms/classify`), `frontend/src/components/ParadigmDisplay.tsx`
- Classification
  - Engine and features: `backend/services/classification_engine.py`
  - API route: `backend/routes/paradigms.py` → POST `/paradigms/classify`
- Context Engineering (W‑S‑C‑I)
  - Pipeline and layers: `backend/services/context_engineering.py`
  - Models: `backend/models/context_models.py`
- Research Execution
  - Orchestrator: `backend/services/research_orchestrator.py`
  - Search APIs: `backend/services/search_apis.py`
  - Credibility: `backend/services/credibility.py`
- Synthesis
  - Answer generation: `backend/services/answer_generator.py`
  - HTTP orchestration: `backend/routes/research.py`
- Tests & Examples
  - Context pipeline test: `backend/test_context_engineering.py`
  - End‑to‑end features: `backend/test_enhanced_features.py`
  - Brave search example: `backend/examples/brave_search_example.py`

---

## Quick Run
- Backend: `cd four-hosts-app/backend && uvicorn core.app:create_app --factory --reload`
- Frontend: `cd four-hosts-app/frontend && npm install && npm run dev`
- Try the example query above, and watch the live paradigm preview update while typing.

