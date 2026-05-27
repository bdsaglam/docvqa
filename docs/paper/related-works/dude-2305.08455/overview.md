## Introduction

The field of Document Understanding (DocAI) has made significant strides with deep learning approaches, yet a critical gap remains between laboratory performance and real-world applications. The Document Understanding Dataset and Evaluation (DUDE) addresses this challenge by introducing a comprehensive benchmark that reflects the true complexity of document processing in practical settings.

![Document examples from DUDE showing various question types and multi-page documents with annotations for different complexity categories](https://paper-assets.alphaxiv.org/figures/2305.08455v3/img-0.jpeg)

*Figure 1: Examples from DUDE showcasing diverse question types across multi-page documents, including extractive questions requiring specific information extraction, abstractive questions demanding comprehension, counting tasks, layout-navigating questions about document structure, and multi-hop reasoning across pages.*

Current Document Visual Question Answering (DocVQA) benchmarks predominantly focus on simplified scenarios: single-page documents, extractive questions with answers directly found in text, and narrow domain coverage. DUDE represents a paradigm shift by introducing multi-page, multi-domain documents with diverse question types including layout navigation, arithmetic operations, and crucially, non-answerable questions that reflect real-world ambiguities.

## Dataset Construction and Methodology

The DUDE dataset construction follows a rigorous multi-phase approach designed to ensure both quality and real-world relevance. The researchers collected 5,019 documents from public domain sources including archive.org, commons.wikimedia.org, and documentcloud.org, spanning from 1860 to 2022 across diverse industries and document types.

The annotation process involved a hybrid approach combining qualified linguists and Amazon Mechanical Turk workers. The workflow consisted of four phases:

**Phase 1: Question & Answer Generation** - Annotators created diverse question-answer pairs, with particular emphasis on generating abstractive, extractive, and list-type questions. Non-answerable questions were primarily handled by qualified linguists to ensure high quality.

**Phase 2: Answer Verification** - Independent annotators answered questions from Phase 1, with question-answer triples achieving high inter-annotator agreement (>0.8 ANLS) being promoted to the next phase.

**Phase 3: Correctness Adjudication** - Top-performing annotators resolved discrepancies from Phase 2, with outliers reviewed by qualified linguists.

**Phase 4: Test Set Diagnostics** - Qualified linguists performed final review of the test set, adding detailed diagnostic metadata for fine-grained analysis.

The resulting dataset comprises over 41,000 questions with 90% uniqueness, featuring documents averaging 5.72 pages and 1,832 tokens. The evaluation framework uses three key metrics:

$$
\text{ANLS} = \frac{1}{N} \sum_{i=1}^N \text{NL}(a_i, o_i)
$$

where NL represents the Normalized Levenshtein similarity, adapted for list-type and non-answerable questions. Additionally, Expected Calibration Error (ECE) and Area-Under-Risk-Coverage-Curve (AURC) assess model confidence calibration and selective classification capabilities.

## Unique Contributions and Innovations

DUDE introduces several distinctive features that differentiate it from existing document understanding benchmarks:

**Multi-Page Document Focus**: Unlike previous datasets that primarily handle single-page documents, DUDE's documents average 5.72 pages, requiring models to reason across extended content and maintain coherence over longer sequences.

**Diverse Question Taxonomy**: The dataset includes multiple question categories:
- **Layout-navigating questions**: Requiring understanding of visual document structure (e.g., "How many columns are in the table?")
- **Multi-hop reasoning**: Demanding sequential logic across document sections
- **Arithmetic operations**: Involving numerical computations on extracted data
- **List-type answers**: Supporting complex information aggregation
- **Non-answerable questions**: Reflecting real-world scenarios where information is unavailable

**Domain and Visual Diversity**: Statistical analysis using Simpson diversity coefficients confirms DUDE's significantly higher diversity compared to existing datasets. The documents span multiple industries, layouts, and visual presentations, from born-digital to scanned historical documents.

![Comparison of dataset diversity through t-SNE visualization of document embeddings](https://paper-assets.alphaxiv.org/figures/2305.08455v3/img-1.jpeg)

*Figure 2: t-SNE visualization comparing DUDE's document diversity against existing datasets, demonstrating significantly broader coverage of document types and visual characteristics.*

**Comprehensive Diagnostic Framework**: The test set includes rich diagnostic metadata categorizing questions by:
- Complexity levels (simple, hard multi-hop, hard meta/layout-navigating)
- Answer evidence types (free text, handwriting, tables, layouts, graphics)
- Required operations (arithmetic, comparison, counting, normalization)
- Answer formats (dates, numeric values, proper names)

## Experimental Results and Key Findings

The evaluation of state-of-the-art models on DUDE reveals significant limitations in current document understanding approaches. The best-performing model, T5-2D large with 8192 token context, achieved only 46.06 ANLS compared to the human baseline of 74.76 ANLS, highlighting a substantial performance gap.

**Model Performance Analysis**:

Text-only models struggled significantly, particularly with abstractive questions requiring visual understanding, scoring below 10% ANLS. Even sophisticated multimodal models incorporating layout embeddings (T5-2D) or vision features (Hi-VT5) showed unsatisfactory performance compared to human capabilities.

**Impact of Context Length**: Models with longer input sequences (8192 tokens vs. 512 tokens) demonstrated notable improvements of 4.4-5.0 ANLS points, underscoring the importance of processing extensive multi-page documents effectively.

**Large Language Model Insights**: 
- GPT-3 and ChatGPT showed strength in list-type questions (36-40% ANLS) and multi-hop reasoning (52.51% ANLS)
- ChatGPT excelled at identifying non-answerable questions (77.45% ANLS)
- However, LLMs performed poorly on abstractive questions (22% vs. 47% for T5-base) and arithmetic operations (\x3C25% ANLS)
- Their text-only nature severely limited performance on visually-dependent questions (average 21% ANLS)

**Confidence Calibration**: The study revealed that calibration often worsens with increased sequence length, and multi-page prediction strategies assuming conditional independence across pages generally lead to poorer calibration.

## Statistical Characteristics and Comparison

DUDE's statistical properties demonstrate its advancement over existing benchmarks. The dataset exhibits significantly longer documents (average 1,832 tokens) compared to single-page focused datasets, with more complex answer distributions including substantial representation of numerical, date, and list-type responses.

![Distribution comparison of answer types across different VQA datasets](https://paper-assets.alphaxiv.org/figures/2305.08455v3/img-14.jpeg)

*Figure 3: Comparison of answer type distributions across VQA datasets, showing DUDE's unique inclusion of diverse answer formats including lists and non-answerable questions.*

The question complexity distribution shows a balanced representation across different difficulty levels, with substantial portions requiring visual evidence interpretation, layout navigation, and multi-step reasoning processes that reflect real-world document understanding challenges.

## Significance and Future Impact

DUDE represents a crucial advancement in document understanding evaluation by establishing a more realistic and challenging benchmark that addresses the gap between academic research and practical applications. The substantial performance gap between current models and human performance (46.06 vs. 74.76 ANLS) clearly identifies areas requiring focused research attention.

The dataset's emphasis on multi-page documents, diverse question types, and comprehensive evaluation metrics (including confidence calibration) provides a roadmap for developing more robust document AI systems. The inclusion of non-answerable questions and complex reasoning tasks pushes the field toward models that can handle real-world ambiguities and uncertainties.

By making DUDE publicly available on Hugging Face, the researchers have lowered barriers to adoption while establishing a foundation for continued research. The diagnostic framework enables fine-grained analysis of model capabilities, supporting targeted improvements in specific areas like visual understanding, numerical reasoning, and multi-page coherence.

The benchmark's design as an extensible platform positions it to evolve with the field, accommodating new tasks and domains while maintaining its core focus on practical document understanding challenges. This work fundamentally shifts the evaluation paradigm in DocAI from simplified academic scenarios toward the complex, multi-faceted requirements of real-world document processing systems.