## Overview

MMLONGBENCH-DOC addresses a critical gap in evaluating Large Vision-Language Models (LVLMs) on long-context document understanding. While current LVLMs excel at single-page document tasks, their performance on lengthy, multi-modal documents spanning tens or hundreds of pages remains largely unexplored. This research introduces the first comprehensive benchmark specifically designed to evaluate multi-modal long-context document understanding capabilities.

Real-world documents such as financial reports, academic papers, and user manuals often contain complex layouts with diverse information modalities including text, tables, charts, and images distributed across multiple pages. Understanding these documents requires two key capabilities: localization (finding relevant information within massive heterogeneous content) and cross-page comprehension (reasoning over information spread across different pages).

## Dataset Construction and Methodology

The researchers constructed MMLONGBENCH-DOC through a rigorous multi-stage process involving expert annotation and quality control. The dataset comprises 135 PDF documents averaging 47.5 pages and 21,214 textual tokens, sourced from both existing datasets (DUDE, SlideVQA, ChartQA, FinanceBench) and newly collected materials from ArXiv, ManualsLib, and web searches.

The documents span seven diverse domains: Research Reports, Financial Reports, Academic Papers, Brochures, Guidelines, Administration & Industry Files, and Tutorials/Workshops. This diversity ensures comprehensive evaluation across different document types and information structures.

Ten expert annotators (doctoral-level researchers) created 1,082 questions following strict guidelines. The annotation process involved two approaches: systematically reviewing and revising 425 existing questions from previous datasets, and creating 898 entirely new questions. Annotators classified existing questions as "Retain," "Revise" (32.2%), or "Remove" (46.1%) based on quality criteria including difficulty appropriateness, document relevance, and potential shortcuts that might undermine multi-hop reasoning.

The final question set includes three critical categories:
- **Single-page questions (494)**: Testing localization abilities within individual pages
- **Cross-page questions (365, 33.7%)**: Requiring evidence integration across multiple pages
- **Unanswerable questions (223, 20.6%)**: Evaluating hallucination detection when information is absent

Each question includes comprehensive meta-information such as reference answers, evidence page locations, evidence sources (text, layout, chart, table, image), and answer formats.

## Quality Control and Evaluation Protocol

A three-round quality control process ensured benchmark reliability. First, questions were tested with GPT-4o without documents to identify and remove 94 questions answerable through parametric knowledge alone. Second, GPT-4o predictions with documents were compared to human annotations, with inconsistencies reviewed by annotators. Third, cross-checking between annotators and meta-annotation by primary authors resolved disagreements.

The evaluation protocol adapts a three-step process: response generation (freestyle answers from LVLMs), answer extraction (using GPT-4o to convert responses into standardized formats), and rule-based scoring calculating both accuracy and F1 scores across different answer types (Integer, Float, String, List, Not Answerable, Fail to Answer).

## Experimental Results and Key Findings

The researchers evaluated 14 LVLMs (4 proprietary, 10 open-source) and 10 LLMs as baselines. For LVLMs, PDF documents were converted to PNG screenshots, with proprietary models receiving all original images while open-source models used concatenation strategies due to multi-image input limitations. LLMs processed OCR-extracted text using Tesseract.

The results reveal several striking findings:

**Overall Performance Challenges**: The best-performing model, GPT-4o, achieved only 44.9% F1 score, while GPT-4V reached 30.5%. All other LVLMs performed considerably worse, often around 20% or lower. Human annotators achieved 66.0% F1, indicating a substantial 20+ percentage point gap even with the most advanced models.

**Counterintuitive LVLM vs. OCR+LLM Comparison**: Most surprisingly, 12 out of 14 LVLMs performed worse than LLMs processing OCR-extracted text. For example, Gemini-1.5-Pro and Claude-3 Opus showed 4.2% and 6.4% F1 score degradations respectively when processing visual documents versus OCR text. Only GPT-4o (+14.4% F1) and GPT-4V (+5.3% F1) demonstrated superior performance on visual documents, highlighting their unique ability to effectively leverage multimodal information.

**Performance Analysis by Document Characteristics**: 

The evaluation across different evidence sources revealed that only GPT-4o maintained balanced performance across text, layout, chart, table, and image evidence. Other LVLMs showed particular weakness with chart and image-based questions. Performance generally declined as evidence appeared deeper in documents, confirming localization challenges. All models achieved higher scores on single-page versus cross-page questions, demonstrating the difficulty of cross-page comprehension.

For unanswerable questions, models showed different behaviors: GPT-4o and Claude-3 Opus were "aggressive" (often providing incorrect answers), while Gemini-1.5-Pro and DeepSeek-VL-Chat were "cautious" (frequently refusing to answer when uncertain).

## Error Analysis and Oracle Experiments

Oracle experiments, where models received only relevant evidence pages instead of full documents, revealed that long context length significantly impacts performance. For instance, Gemini-1.5-Pro showed over 20% absolute performance degradation when processing complete documents versus oracle pages. However, even with oracle pages, performance remained limited (40% for Gemini-1.5-Pro), indicating challenges beyond mere context length.

Error analysis of GPT-4o revealed that Hallucinated Evidence (33%) and Perceptual Error (28%) were the most common failure modes, followed by Irrelevant Answer (11%) and Incomplete Evidence (10%). This analysis pinpoints critical areas for model improvement: enhanced visual understanding and reduced tendency to generate unsupported information.

## Significance and Impact

MMLONGBENCH-DOC makes several important contributions to the field. It provides the first comprehensive benchmark for evaluating long-context multimodal document understanding, filling a critical gap in current evaluation frameworks. The benchmark reveals that this task remains largely unsolved for current LVLMs, with even state-of-the-art models significantly underperforming human capabilities.

The counterintuitive finding that most LVLMs perform worse than OCR-based LLMs challenges assumptions about the universal superiority of end-to-end multimodal models. This suggests that current LVLMs may lack sufficient training on extremely long visual contexts and highlights the need for architectural and training improvements.

The research establishes clear directions for future development, emphasizing the need for LVLMs capable of processing extensive visual contexts without performance degradation, improved cross-page reasoning abilities, and reduced hallucination tendencies. The benchmark serves as both a diagnostic tool for current model limitations and a target for next-generation multimodal AI systems.

For practical applications, advances driven by this benchmark could revolutionize document processing in legal, financial, healthcare, academic, and administrative domains where understanding lengthy, complex documents is crucial. By explicitly evaluating hallucination detection, the benchmark also contributes to developing more trustworthy and reliable AI systems suitable for sensitive applications.

MMLONGBENCH-DOC represents a foundational step toward achieving human-level document understanding capabilities in AI systems, providing both the evaluation framework and empirical insights needed to guide future research in this critical area.