# Genie User SOP: Maximizing Document Intelligence

This guide serves as a Standard Operating Procedure (SOP) for interacting with Genie. It explains how the system "thinks," interprets your prompts, and determines whether an answer is safe to display.

---

## 🛡️ The Genie Decision Model
Genie is designed to be **uncompromisingly safe**. Instead of guessing or "hallucinating" facts, it employs a dual-lock gatekeeping system.

1. **Gate 1: Citation Lock**: Every factual answer MUST contain a reference in parentheses, e.g., `(EU_AI_Act.pdf, p. 12)`. If the AI fails to cite its source, the response is **BLOCKED**.
2. **Gate 2: Semantic Lock**:
   - **Factual Mode**: Needs a high grounding score (**≥ 0.55**) to **PASS**.
   - **Comprehension Mode**: Needs a moderate grounding score (**≥ 0.40**) to trigger **SYNTHESIS**.

---

## 📂 Representative Interaction Table

| Query Type | Example Prompt | Expected Genie Response | Logic / Value Extract |
| :--- | :--- | :--- | :--- |
| **Factual Lookup** | "What are the transparency requirements for chatbots?" | ✅ **PASS** | Genie finds the specific Article, formats the answer with a citation, and displays it. |
| **Comprehension** | "**Summarize** the main risks for Tier-1 AI models." | ⚠️ **SYNTHESIS** | Keywords like "Summarize" lower the threshold to 0.40, allowing for a broader, synthesized overview. |
| **Specialized Mode** | "**Quiz me** on the compliance section." | 📝 **QUIZ MODE** | Triggers the `quiz_generator` to create a verified MCQ test based on extracted facts. |
| **Structured Task** | "**Extract** all definitions into a **Markdown Table**." | ✅ **PASS** | Triggers the `structured_extractor` to format retrieved facts into a usable table. |
| **Out-of-Scope** | "What is the best recipe for spaghetti?" | 🚫 **BLOCK** | The information isn't in your document. Genie blocks the response to prevent hallucination. |

---

## 💡 How to Prompt for Maximum Value

To get the most out of Genie's v3.2.0 engine, follow these "Prompt Engineering" best practices:

### 1. Trigger the Right Mode
Genie listens for specific keywords to adjust its strictness:
- **For precise facts**: Use direct questions (Who, What, When, Where).
- **For broad overviews**: Start your prompt with **"Explain," "Summarize," "Overview of,"** or **"Describe."** This tells Genie you want a comprehensive synthesis rather than a single-point fact.

### 2. Leverage Hybrid Search
Genie uses both **Semantic (Context)** and **Keyword (Exact)** search. 
- *Pro Tip*: Use exact terms from your documents (e.g., "Article 15," "Section 2.4," or "Annex III"). This helps the BM25 keyword engine find the precise "needle" in the haystack.

### 3. Request Specific Structures
Genie is excellent at formatting data. Don't just ask for an answer; ask for it in a format you can use:
- *"Extract X into a JSON object."*
- *"Compare Y and Z in a Markdown Table."*
- *"Provide a bulleted checklist for onboarding."*

### 4. Continuous Evaluation
If Genie **BLOCKS** an answer, it means the system couldn't find a high-confidence, cited source in your uploaded documents. 
- *Fix*: Try rephrasing with simpler terms or confirm that the information exists in the specific PDF you uploaded. 

---

## 🚦 Understanding UI Indicators
- **✅ PASS (Factual)**: High confidence, direct citation found.
- **⚠️ SYNTHESIS (Comprehension)**: Moderate confidence, gathered from multiple spots.
- **🚫 BLOCK (Refusal)**: Security mechanism triggered. No information was found or the AI couldn't prove where it came from.

---
*Manual Version: 3.2.0 | Policy: Zero-Hallucination Enforcement*
