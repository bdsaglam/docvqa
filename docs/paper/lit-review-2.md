# **Methodological Frontiers and Benchmarking Standards in Document Visual Question Answering: A Comprehensive Literature Review**

The evolution of Document Visual Question Answering (DocVQA) represents a fundamental convergence of computer vision, natural language processing, and document intelligence. Historically, the field was bifurcated between Optical Character Recognition (OCR) systems designed for literal text extraction and Natural Language Processing (NLP) models designed for semantic reasoning over clean text. The contemporary landscape, however, is defined by the integration of layout-aware pre-training, frontier multimodal large language models, and the recent emergence of agentic scaffolds that treat document understanding as an iterative, tool-augmented reasoning process.1 This report provides an exhaustive mapping of the DocVQA landscape, detailing the transition from static, single-page encoders to dynamic, multi-agent frameworks capable of navigating complex, long-form document structures.

## **The Methodological Landscape of Document Intelligence**

The technical strategies employed to solve DocVQA tasks have matured through several distinct architectural generations. These generations are characterized by how they fuse visual signals, textual content, and spatial layout information.

### **Trained Document Encoders and Layout-Aware Models**

The first major wave of innovation in DocVQA was driven by models that explicitly integrated spatial coordinates into the transformer architecture. The LayoutLM family pioneered this approach by incorporating 2D position embeddings alongside standard text and image features, allowing the model to learn the "visual language" of document structures such as tables, forms, and headers.1 These models typically follow a pre-train-then-fine-tune paradigm, utilizing massive unlabeled datasets like the IIT-CDIP collection.

Subsequent iterations, such as LayoutLMv2 and v3, moved toward more holistic multimodal fusion. LayoutLMv3, for instance, unified text and image masking objectives, treating document patches and text tokens as a single sequence.1 Other architectures in this cluster focused on specific structural nuances:

* **UDOP (Unified Document Optimization):** Proposed a unified generative framework for various document tasks, treating layout as a language that can be predicted alongside text.  
* **DocFormer:** Introduced a multi-pronged attention mechanism to fuse text, vision, and spatial features at every layer, rather than just at the input stage.  
* **ERNIE-Layout:** Integrated layout knowledge into the pre-training phase using a layout-aware masking strategy.  
* **LiLT (Language-Independent Layout Transformer):** Decoupled the layout and language encoders to improve cross-lingual transferability.

The current state-of-the-art (SOTA) among these trained encoders is often held by large-scale unified models like UDOP or late-stage LayoutLM variants, although their dominance has been challenged by the zero-shot capabilities of much larger general-purpose models. The primary limitation of this cluster remains their fixed context window, which typically caps at 512 or 1024 tokens, making them fundamentally unsuited for multi-page document understanding without an external retrieval mechanism.1

### **General Multimodal Large Language Models**

The emergence of Multimodal Large Language Models (MLLMs) such as GPT-4o, Gemini 1.5 Pro, and Claude 3.5 Sonnet has shifted the field toward zero-shot and few-shot reasoning. These models do not require domain-specific fine-tuning on DocVQA datasets; instead, they leverage their massive parameter counts and diverse training data to interpret documents through direct prompting.6

Open-source alternatives have rapidly matured to compete with proprietary models. The Qwen-VL family, specifically Qwen2.5-VL and the recently previewed Qwen3-VL, currently lead many self-reported leaderboards on benchmarks like the original DocVQA.6 These models utilize high-resolution vision encoders and sophisticated alignment techniques, such as the "image-to-text" and "text-to-image" cross-modal alignment objectives. Qwen3-VL 235B, for example, reports an ANLS score of 0.971 on the DocVQA test set, which exceeds estimated human baselines.7

Other notable models in this cluster include:

* **InternVL:** Utilizes a massive vision backbone (e.g., 6B parameters) to provide high-fidelity visual representations to an LLM decoder.  
* **LLaVA-family:** Pioneered the use of instruction tuning for multimodal models, with recent versions like LLaVA-NeXT significantly improving document understanding.  
* **MiniCPM-V:** Focuses on efficiency, delivering high performance with smaller parameter footprints (e.g., 8B) suitable for edge deployment.

The strength of these models lies in their ability to handle "high thinking" tasks where questions require complex reasoning over visual and textual cues.8 However, as documents grow in length and resolution, even these frontier models encounter "visual token explosion," where the number of tokens required to represent a high-resolution page overwhelms the model's context limit.1

### **Tool-Using and Agentic DocVQA Frameworks**

The most significant contemporary shift in the field is the move toward agentic architectures. This paradigm, which aligns closely with the Recursive Language Model (RLM) framework, treats DocVQA not as a single inference step but as a multi-turn navigation and information-seeking problem.3 In this setting, a planning agent orchestrates various tools—such as OCR engines, retrievers, code interpreters, and visual "look" or "zoom" sub-calls—to gather evidence before synthesizing an answer.

Notable frameworks in this category include:

* **ARIAL (Agentic Reasoning for Interpretable Answer Localization):** A modular framework that orchestrates specialized tools for OCR (TrOCR), semantic retrieval, and answer generation. ARIAL achieves SOTA results on DocVQA (88.7 ANLS) while providing a traceable reasoning path that links every answer to specific pixel coordinates.13  
* **VISOR (Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning):** Features a structured "Evidence Space" to accumulate query-relevant observations across retrieval iterations. It uses mechanisms like "Intent Injection" to prevent search drift during long-horizon interactions.11  
* **AgenticOCR:** Transforms OCR from a static pre-processing step into a query-driven, on-demand extraction system. It identifies and selectively recognizes regions of interest, effectively decoupling retrieval granularity from page-level chunking.10  
* **ORCA (Orchestrated Reasoning with Collaborative Agents):** Employs a "thinker" agent to generate a reasoning path, which then routes sub-tasks to specialized agents for tables, figures, forms, or handwritten text.16  
* **MDocAgent:** A multi-modal multi-agent framework that integrates both text and image retrieval to enable collaborative reasoning across modalities.17  
* **DocDancer:** Formulates DocVQA as an information-seeking problem, utilizing search and read tools for iterative document exploration and localized comprehension.3

These agentic approaches are particularly effective for long-form documents where the "needle-in-a-haystack" problem is compounded by the need to integrate evidence from disparate pages.11 By using RL-based alignment techniques like Group Relative Policy Optimization (GRPO), these agents can be trained to optimize for both retrieval precision and final answer accuracy.10

### **Specialized and Document-Specific Methods**

Certain document types require specialized inductive biases that general-purpose models may lack.

* **Chart-Specific:** Models like UniChart, ChartLlama, ChartGemma, and MatCha are optimized for the visual and numerical reasoning required to parse complex charts and graphs. These models often utilize specialized pre-training on synthetic chart data.19  
* **Infographic-Specific:** These methods handle the high visual density and non-linear layouts of infographics, where text is often embedded in icons or arrows.22  
* **Table-Specific:** Models such as NII-TablQA focus on table structure recognition and cell-level reasoning, which is critical for financial and scientific documents.25  
* **Handwritten-Specific:** Competition-winning methods like PA\_VCG address the unique style variability and noise associated with handwritten notes.27

### **The OCR-Free vs. OCR-Augmented Debate**

The field is currently split on the necessity of an explicit OCR step.

* **OCR-Free (End-to-End):** Models like Donut, Pix2Struct, and Nougat map document images directly to text or structured outputs. Proponents argue this avoids the "lossy" nature of OCR and better captures layout and non-textual cues.1  
* **OCR-Augmented:** These methods extract text via an external engine (e.g., Azure, Amazon Textract, docling-serve) and feed the resulting text—often with spatial coordinates—into an LLM. This remains the dominant paradigm for enterprise applications due to its higher accuracy on complex, high-resolution scans and its inherent interpretability.10

Recent evidence suggests that for mid-sized models, OCR-augmented pipelines are significantly more reliable, whereas frontier models are becoming increasingly capable of handling visual retrieval in an OCR-free manner.31

## **SOTA Benchmarks and Performance Analysis**

The following table summarizes the current state-of-the-art across the canonical and emerging benchmarks in the DocVQA domain.

| Benchmark | Method (ArXiv/Ref) | SOTA Score | Year | Notes |
| :---- | :---- | :---- | :---- | :---- |
| **DocVQA (Original)** | Qwen3-VL 235B 7 | 0.971 ANLS | 2026 | Leader on self-reported leaderboard.6 |
| **DocVQA (Original)** | Qwen2.5-VL 72B 6 | 0.964 ANLS | 2025 | Standard benchmark; nearing human parity.6 |
| **MP-DocVQA** | AVIR Framework 5 | 0.8458 ANLS | 2025 | Surpasses baseline by 22.5 ANLS via RAG.5 |
| **InfographicVQA** | Applica.ai 32 | 0.6120 ANLS | 2021 | 2021 ICDAR Task 3 winner.24 |
| **ChartQA** | Qwen2-VL-7B 18 | (High Ref) | 2025 | Specialized visual-numerical reasoning. |
| **SlideVQA** | VISOR 11 | (Reported) | 2026 | Multi-page sequential reasoning.11 |
| **VisualMRC** | (Various) | (Variable) | 2024 | Focused on visual reading comprehension.34 |
| **DUDE** | AVIR 5 | (Verified) | 2025 | Document understanding for diverse docs.35 |
| **ARIAL (DocVQA)** | ARIAL 13 | 0.887 ANLS | 2025 | SOTA for spatially grounded extraction.15 |
| **VQAonBD** | Team Upstage 25 | 95.9% Acc | 2023 | ICDAR'23 Business Doc VQA winner.25 |
| **Handwritten VQA** | PA\_VCG 27 | 0.643 ANLS | 2024 | ICDAR'24 Handwritten VQA winner.27 |
| **MMLongBench-Doc** | GPT-4o 36 | 44.9% F1 | 2024 | 1,082 expert-annotated long-context questions.19 |
| **MMLongBench-Doc** | GPT-4o 23 | 42.7% F1 | 2024 | Demonstrates high difficulty of long DU.37 |
| **MMLongBench** | MDocAgent 18 | 0.315 (T4) | 2025 | Agentic framework using Qwen2-VL backbone. |
| **DocVQA 2026 (Val)** | Gemini 3 Pro 9 | 37.50% Acc | 2026 | Official baseline; reflects high difficulty.38 |

The performance numbers across these benchmarks highlight a critical insight: while single-page factual extraction is largely solved (scores \> 0.95), tasks requiring cross-page reasoning, long-context navigation, or complex visual reasoning (e.g., infographics, long documents) remain significantly below human performance.23

## **ICDAR DocVQA Challenge History**

The International Conference on Document Analysis and Recognition (ICDAR) has been the primary venue for tracking and driving progress in DocVQA. The challenge series has evolved from simple single-page extraction to multifaceted multimodal reasoning.

### **ICDAR 2020: The Genesis**

The first edition introduced the canonical DocVQA dataset, focusing on 12,000 document images and 50,000 questions. The task was purely extractive, and early winners utilized BERT-based architectures combined with simple OCR engines.6 This edition established ANLS as the standard metric.

### **ICDAR 2021: Collections and Infographics**

The 2021 competition expanded the scope to include Document Collection VQA (Task 2\) and Infographics VQA (Task 3).

* **Winner (Task 2):** Team Infrrd with the Infrrd-RADAR (Retrieval of Answers by Document Analysis and Re-ranking) method. This was a pioneering retrieval-augmented approach that beat entries from Amazon and iFLYTEK.41  
* **Winner (Task 3):** Team Applica.ai, which utilized a multimodal transformer to interpret the layout-heavy infographics, scoring 0.6120 ANLS.32

### **ICDAR 2023: Business Documents and Privacy**

The 2023 edition emphasized industrial applications through the VQAonBD (VQA on Business Document Images) task and a Privacy-Preserving Federated Learning track.

* **VQAonBD Winner:** Team Upstage KR, scoring 95.9% accuracy. They utilized a specialized pre-processing pipeline for table-related questions, splitting complex arithmetic or ratio queries into simpler sub-questions processed by a fine-tuned Donut model.25  
* **PFL-DocVQA:** Focused oninvoice processing using federated learning and differential privacy, bringing document analysis into the realm of secure AI.42

### **ICDAR 2024: Handwritten and Real-World Scenarios**

This edition introduced challenges for handwritten documents (HWD) and documents captured through Aria Glasses (mixed reality).

* **Handwritten Winner:** Team PA\_VCG, achieving 0.643 ANLS. This highlighted the ongoing difficulty of handling non-standard, variable handwriting across multiple languages.27  
* **Aria Glasses:** Focused on low-resolution word recognition and reading order prediction in "in-the-wild" scenarios.44

### **ICDAR 2026: The Reasoning Frontier**

The current competition (DocVQA 2026\) marks a shift toward high-reasoning tasks across diverse domains such as maps, engineering drawings, and scientific papers.9

* **Key Trends:** The introduction of categories like "maps" and "posters" with zero OCR characters necessitates models that can perform visual reasoning without text anchors. The official baselines (Gemini 3 Pro at 37.5%) suggest that the benchmark is significantly harder and less saturated than prior years.9

## **Evaluation Conventions: The ANLS Metric**

Average Normalized Levenshtein Similarity (ANLS) is the de facto standard for evaluating DocVQA, designed to provide a smooth reward for responses that are "mostly correct" but contain minor OCR or recognition errors.46

### **Mathematical Definition**

The similarity ![][image1] between a ground truth answer ![][image2] and a model prediction ![][image3] is calculated as:

![][image4], where ![][image5] is the Normalized Levenshtein Distance, defined as ![][image6].

Crucially, ANLS employs a threshold ![][image7]:

If ![][image8], the score for that question is ![][image9].

If ![][image10], the score is the raw similarity ![][image1].

### **Thresholding Philosophies**

The choice of ![][image7] is critical. Historically, a threshold of 0.5 was used to allow for substantial recognition errors while filtering out cases where the model likely retrieved the wrong text segment entirely.46 However, the ICDAR 2026 challenge has moved toward a more stringent threshold of 0.80. This reflects the increased maturity of OCR and the requirement for higher precision in complex reasoning tasks where a single digit error in a numeric answer is catastrophic.9

### **Alternative Metrics**

While ANLS is standard, researchers are increasingly reporting:

* **mAP@IoU:** For localization tasks, measuring how well the model identifies the bounding box of the answer.13  
* *ANLS:*\* A recent extension designed to handle structured outputs like dictionaries or lists, often encountered in information extraction from invoices or forms.51  
* **SMuDGE:** A composite score that accounts for grounding and semantic category (e.g., did the model return a number when a number was expected?).54

## **Handling Long and Multi-Page Documents**

Long-document understanding is the primary bottleneck for current MLLMs. The strategies to address this can be categorized by their approach to information selection and representation.

### **Retrieval-Augmented Generation (RAG)**

The most common strategy is to convert pages into a searchable index.

* **Text-Based Retrieval:** Using BM25 or dense embeddings over OCR-extracted text. This is effective for factual "needle" questions but fails on visual-only content.2  
* **Visual Retrieval:** Using vision encoders (e.g., ColPali, ColQwen) to embed page images directly, allowing retrieval based on visual features.5  
* **Hierarchical Retrieval:** Systems like AVIR or DocLens first score pages for relevance using a lightweight model, then feed the Top-K pages to a large model. AVIR reports a 70% reduction in page count while improving ANLS by \+22.5 on MP-DocVQA.5

### **Agentic Context Engineering**

Frameworks like VISOR and AgenticOCR move beyond static retrieval.

* **Dynamic Trajectories:** Maintaining a "sliding window" of recent interactions while keeping the "Evidence Space" (accumulated findings) pinned in the context window.11  
* **Query-Driven Cropping:** Instead of feeding entire pages, the agent identifies regions of interest and crops/zooms into them, maximizing the signal-to-token ratio.10

### **Architectural Native Scaling**

Models like Llama 4 Scout and Gemini 1.5 Pro are expanding context windows to handle millions of tokens.6 However, empirical results from MMLongBench-Doc indicate that simply increasing context length is insufficient, as models still suffer from "lost in the middle" effects and struggle with multi-hop reasoning across 50+ pages.19

## **OCR Pipelines in the Field**

The choice of OCR engine is often the silent determinant of success in DocVQA pipelines.

| Engine | Characteristics | Impact on DocVQA |
| :---- | :---- | :---- |
| **Azure Document Intelligence** | High accuracy on semi-structured documents; excellent table parsing. | Generally considered the "gold standard" for commercial-grade extraction.30 |
| **docling-serve** | Optimized for document-to-markdown; handles complex layouts. | Used in the ICDAR 2026 challenge baseline; preferred for its structured output.9 |
| **PaddleOCR** | Leading open-source toolkit with support for many languages. | Frequently cited in academic benchmarks as a strong open-source baseline.30 |
| **TrOCR** | Transformer-based OCR; high precision for isolated text segments. | Used in the ARIAL framework for grounding and precise localization.13 |
| **Tesseract / EasyOCR** | Legacy and lightweight open-source engines. | Often used as baselines, but significantly outperformed by modern DL models.30 |

Recent work has demonstrated that reordering prompts (Image-First vs. Text-First) can result in a 13-18% relative improvement in ANLS, suggesting that the interaction between the OCR output and the vision tokens is highly sensitive to positional attention bias.48

## **Strategic Positioning and Recommended Baselines**

For a research paper utilizing an agentic scaffold on the ICDAR 2026 challenge, the following baseline comparisons are essential to establish rigor and novelty.

### **Recommended Baseline Shortlist**

* **Qwen2.5-VL (72B & 7B):** This represents the SOTA for direct multimodal prompting. Comparing against it demonstrates how much an agentic scaffold "lifts" a natively strong model.6  
* **ARIAL:** As the leading agentic framework for spatially-grounded DocVQA, ARIAL is the most direct competitor. Comparisons should focus on localization precision and reasoning transparency.13  
* **VISOR:** Represents the SOTA in iterative visual RAG for multi-page documents. It is a crucial baseline for any method claiming novelty in long-context navigation.11  
* **MDocAgent:** Provides a benchmark for multi-agent collaboration and multimodal retrieval fusion.17  
* **Gemini 3 Pro / GPT-5.2 (Official Baselines):** These are the numbers reported by the competition organizers. Beating these numbers is the baseline requirement for a competitive submission.9  
* **AVIR:** A strong baseline for efficient retrieval-based DocVQA, particularly on multi-page datasets like MP-DocVQA.5

### **Claims to Avoid**

The literature contradicts several potential claims of novelty:

* **"No prior work uses an LLM with OCR and VLM tools":** This is exactly what ARIAL, AgenticOCR, and VISOR do.10 Instead, focus on the specific orchestration (e.g., the use of a REPL or recursive sub-calls).  
* **"Our method is the first to handle multi-page reasoning":** Infrrd (2021) and MP-DocVQA (2022) have addressed this for years.5  
* **"We are the first to use code for DocVQA reasoning":** ARIAL and MDocAgent already incorporate code/tool execution as part of their agentic loops.12  
* **"OCR-free is always worse/better":** The literature suggests a capacity-dependent trade-off where the best approach depends on the model scale and document complexity.30

## **Must-Cite Literature and Bibliography**

The following papers form the essential foundation for any contemporary DocVQA research and should be cited according to their methodological clusters.

### **Cluster 1: Trained Document Encoders**

* Xu et al., "LayoutLM: Pre-training of Text and Layout for Document Image Understanding," KDD 2020\. 1  
* Kim et al., "Donut: OCR-free Document Understanding Transformer," ECCV 2022\. 25  
* Lee et al., "Pix2Struct: Pre-training by Generating Visual-Language Representations," 2023\. 1  
* Tito et al., "Hierarchical Multimodal Transformers for Multi-page DocVQA," 2023\. 5

### **Cluster 2: General Multimodal LLMs**

* Alibaba Cloud / Qwen Team, "Qwen2.5-VL: Thinking with Images," 2025\. 6  
* Reid et al., "Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens," 2024\. 1  
* Anthropic, "Claude 3.5 Sonnet Model Card," 2024\. 6

### **Cluster 3: Agentic and Tool-Augmented Frameworks**

* Mohammadshirazi et al., "ARIAL: An Agentic Framework for Document VQA with Precise Answer Localization," arXiv:2511.18192, 2025\. 13  
* Wu et al., "VISOR: Visual Retrieval Augmented Generation via Iterative Search," arXiv:2604.09508, 2026\. 11  
* Jin et al., "AgenticOCR: Dynamic Parsing for Long-Document Understanding," arXiv:2602.24134, 2026\. 10  
* Wang et al., "MDocAgent: A Multi-Modal Multi-Agent Framework for Document Understanding," 2025\. 17  
* Zhang et al., "ORCA: Orchestrated Reasoning with Collaborative Agents for DocVQA," 2026\. 16

### **Cluster 4: Benchmarks and Evolution**

* Mathew et al., "DocVQA: A Dataset for Visual Question Answering on Document Images," WACV 2021\. 6  
* Ma et al., "MMLongBench-Doc: Benchmarking Long-context Document Understanding," NeurIPS 2024\. 19  
* Tanaka et al., "SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images," 2023\. 11  
* Biten et al., "Scene Text Visual Question Answering," ICCV 2019 (ANLS Metric Definition). 46  
* Nourbakhsh et al., "Where is this coming from? Making groundedness count in DocVQA," 2024\. 49

In conclusion, the DocVQA field has moved beyond simple extraction into a domain of complex, multi-modal spatial reasoning. The emergence of the ICDAR 2026 challenge, with its emphasis on diverse and non-standard documents, provides a critical testbed for the next generation of agentic architectures. By framing the problem as one of recursive, tool-augmented information seeking, modern frameworks are successfully lifting the performance of both small and frontier model classes, pushing the boundaries of what automated document intelligence can achieve.

#### **Works cited**

1. Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2510.15253v2](https://arxiv.org/html/2510.15253v2)  
2. Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2510.15253v3](https://arxiv.org/html/2510.15253v3)  
3. Paper page \- DocDancer: Towards Agentic Document-Grounded Information Seeking, accessed May 1, 2026, [https://huggingface.co/papers/2601.05163](https://huggingface.co/papers/2601.05163)  
4. MHier-RAG: Multi-Modal RAG for Visual-Rich Document Question-Answering via Hierarchical and Multi-Granularity Reasoning \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2508.00579v3](https://arxiv.org/html/2508.00579v3)  
5. Hierarchical multimodal transformers for Multipage DocVQA | Request PDF \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/372639409\_Hierarchical\_multimodal\_transformers\_for\_Multipage\_DocVQA](https://www.researchgate.net/publication/372639409_Hierarchical_multimodal_transformers_for_Multipage_DocVQA)  
6. DocVQA Leaderboard \- LLM Stats, accessed May 1, 2026, [https://llm-stats.com/benchmarks/docvqa](https://llm-stats.com/benchmarks/docvqa)  
7. DocVQAtest Leaderboard \- LLM Stats, accessed May 1, 2026, [https://llm-stats.com/benchmarks/docvqatest](https://llm-stats.com/benchmarks/docvqatest)  
8. MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2505.10610v3](https://arxiv.org/html/2505.10610v3)  
9. GitHub \- VLR-CVC/DocVQA2026: Official evaluation scripts and baseline prompts for the DocVQA 2026 (ICDAR 2026\) Competition on Multimodal Reasoning over Documents., accessed May 1, 2026, [https://github.com/VLR-CVC/DocVQA2026](https://github.com/VLR-CVC/DocVQA2026)  
10. AgenticOCR: Parsing Only What You Need for Efficient Retrieval-Augmented Generation \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2602.24134v1](https://arxiv.org/html/2602.24134v1)  
11. VISOR: Agentic Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2604.09508v1](https://arxiv.org/html/2604.09508v1)  
12. Data-Centric Perspectives on Agentic Retrieval-Augmented Generation: A Survey \- TechRxiv, accessed May 1, 2026, [https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176316052.24300253](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176316052.24300253)  
13. ARIAL: An Agentic Framework for Document VQA with Precise Answer Localization \- arXiv, accessed May 1, 2026, [https://arxiv.org/abs/2511.18192](https://arxiv.org/abs/2511.18192)  
14. ARIAL: An Agentic Framework for Document VQA with Precise Answer Localization \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2511.18192v1](https://arxiv.org/html/2511.18192v1)  
15. ARIAL: An Agentic Framework for Document VQA with Precise Answer Localization, accessed May 1, 2026, [https://neurips.cc/virtual/2025/128625](https://neurips.cc/virtual/2025/128625)  
16. ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2603.02438v1](https://arxiv.org/html/2603.02438v1)  
17. LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating | Request PDF \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/394270660\_LongDocURL\_a\_Comprehensive\_Multimodal\_Long\_Document\_Benchmark\_Integrating\_Understanding\_Reasoning\_and\_Locating](https://www.researchgate.net/publication/394270660_LongDocURL_a_Comprehensive_Multimodal_Long_Document_Benchmark_Integrating_Understanding_Reasoning_and_Locating)  
18. aiming-lab/MDocAgent: MDocAgent: A Multi-Modal Multi ... \- GitHub, accessed May 1, 2026, [https://github.com/aiming-lab/MDocAgent](https://github.com/aiming-lab/MDocAgent)  
19. MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2407.01523v1](https://arxiv.org/html/2407.01523v1)  
20. irpapers: A Visual Document Benchmark for Scientific Retrieval and Question Answering, accessed May 1, 2026, [https://arxiv.org/html/2602.17687v1](https://arxiv.org/html/2602.17687v1)  
21. Doc-V^∗: Coarse-to-Fine Interactive Visual Reasoning for Multi-Page Document VQA, accessed May 1, 2026, [https://arxiv.org/html/2604.13731v1](https://arxiv.org/html/2604.13731v1)  
22. Challenge 2021 \- DocVQA, accessed May 1, 2026, [https://www.docvqa.org/challenges/2021](https://www.docvqa.org/challenges/2021)  
23. (PDF) MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/381910893\_MMLongBench-Doc\_Benchmarking\_Long-context\_Document\_Understanding\_with\_Visualizations](https://www.researchgate.net/publication/381910893_MMLongBench-Doc_Benchmarking_Long-context_Document_Understanding_with_Visualizations)  
24. ICDAR 2021 Competition on Document Visual Question Answering \- Springer Professional, accessed May 1, 2026, [https://www.springerprofessional.de/en/icdar-2021-competition-on-document-visual-question-answering/19631704](https://www.springerprofessional.de/en/icdar-2021-competition-on-document-visual-question-answering/19631704)  
25. ICDAR 2023 Competition on Visual Question Answering on Business Document Images, accessed May 1, 2026, [https://www.researchgate.net/publication/373222106\_ICDAR\_2023\_Competition\_on\_Visual\_Question\_Answering\_on\_Business\_Document\_Images](https://www.researchgate.net/publication/373222106_ICDAR_2023_Competition_on_Visual_Question_Answering_on_Business_Document_Images)  
26. ICDAR 2023 Competition on Visual Question Answering on Business Document Images \- CVIT, IIIT, accessed May 1, 2026, [https://cvit.iiit.ac.in/images/ConferencePapers/2023/icdar2023\_bdi.pdf](https://cvit.iiit.ac.in/images/ConferencePapers/2023/icdar2023_bdi.pdf)  
27. ICDAR 2024 Competition on Recognition and VQA on Handwritten Documents \- CVIT, IIIT, accessed May 1, 2026, [https://cvit.iiit.ac.in/images/ConferencePapers/2024/Competition\_on\_Recognition\_and\_VQA\_on\_Handwritten\_Documents.pdf](https://cvit.iiit.ac.in/images/ConferencePapers/2024/Competition_on_Recognition_and_VQA_on_Handwritten_Documents.pdf)  
28. ICDAR 2024 HWD, accessed May 1, 2026, [https://ilocr.iiit.ac.in/icdar\_2024\_hwd/](https://ilocr.iiit.ac.in/icdar_2024_hwd/)  
29. ICDAR 2025 Competition on End-to-End Document Image Machine Translation Towards Complex Layouts | Request PDF \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/401773321\_ICDAR\_2025\_Competition\_on\_End-to-End\_Document\_Image\_Machine\_Translation\_Towards\_Complex\_Layouts](https://www.researchgate.net/publication/401773321_ICDAR_2025_Competition_on_End-to-End_Document_Image_Machine_Translation_Towards_Complex_Layouts)  
30. \[Feature Request\] Alternative OCR engines (Azure AI Vision, Google Cloud Vision etc.) · paperless-ngx paperless-ngx · Discussion \#5128 \- GitHub, accessed May 1, 2026, [https://github.com/paperless-ngx/paperless-ngx/discussions/5128](https://github.com/paperless-ngx/paperless-ngx/discussions/5128)  
31. Exploration of Augmentation Strategies in Multi-modal Retrieval-Augmented Generation for the Biomedical Domain: A Case Study Evaluating Question Answering in Glycobiology \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/398851017\_Exploration\_of\_Augmentation\_Strategies\_in\_Multi-modal\_Retrieval-Augmented\_Generation\_for\_the\_Biomedical\_Domain\_A\_Case\_Study\_Evaluating\_Question\_Answering\_in\_Glycobiology](https://www.researchgate.net/publication/398851017_Exploration_of_Augmentation_Strategies_in_Multi-modal_Retrieval-Augmented_Generation_for_the_Biomedical_Domain_A_Case_Study_Evaluating_Question_Answering_in_Glycobiology)  
32. Competitions | ICDAR 2021, accessed May 1, 2026, [https://iapr.org/archives/icdar2021/index.html%3Fp=29412.html](https://iapr.org/archives/icdar2021/index.html%3Fp=29412.html)  
33. ICDAR 2021 Competition on Document Visual Question Answering \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/354357194\_ICDAR\_2021\_Competition\_on\_Document\_Visual\_Question\_Answering](https://www.researchgate.net/publication/354357194_ICDAR_2021_Competition_on_Document_Visual_Question_Answering)  
34. Finding Needles in Images: Can Multimodal LLMs Locate Fine Details? \- ACL Anthology, accessed May 1, 2026, [https://aclanthology.org/2025.acl-long.1152.pdf](https://aclanthology.org/2025.acl-long.1152.pdf)  
35. Document Understanding Dataset and Evaluation (DUDE) \- Lirias, accessed May 1, 2026, [https://lirias.kuleuven.be/retrieve/afa1c1d8-9204-41e3-b096-f410bfe486a8](https://lirias.kuleuven.be/retrieve/afa1c1d8-9204-41e3-b096-f410bfe486a8)  
36. MMLONGBENCH-DOC: Benchmarking Long-context Document Understanding with Visualizations, accessed May 1, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/ae0e43289bffea0c1fa34633fc608e92-Paper-Datasets\_and\_Benchmarks\_Track.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/ae0e43289bffea0c1fa34633fc608e92-Paper-Datasets_and_Benchmarks_Track.pdf)  
37. MMLONGBENCH-DOC: Benchmarking Long-context Document Understanding with Visualizations | Request PDF \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/397200990\_MMLONGBENCH-DOC\_Benchmarking\_Long-context\_Document\_Understanding\_with\_Visualizations](https://www.researchgate.net/publication/397200990_MMLONGBENCH-DOC_Benchmarking_Long-context_Document_Understanding_with_Visualizations)  
38. VLR-CVC/DocVQA-2026 · Datasets at Hugging Face, accessed May 1, 2026, [https://huggingface.co/datasets/VLR-CVC/DocVQA-2026](https://huggingface.co/datasets/VLR-CVC/DocVQA-2026)  
39. DocVQA Benchmark: 99.16% Accuracy Using Agentic Document Extraction \- LandingAI, accessed May 1, 2026, [https://landing.ai/blog/superhuman-on-docvqa-without-images-in-qa-agentic-document-extraction](https://landing.ai/blog/superhuman-on-docvqa-without-images-in-qa-agentic-document-extraction)  
40. DocVQA | ICDAR 2021, accessed May 1, 2026, [https://iapr.org/archives/icdar2021/index.html%3Fp=29526.html](https://iapr.org/archives/icdar2021/index.html%3Fp=29526.html)  
41. Infrrd Engineers Triumph in ICDAR's DocVQA Challenge with AI-Powered Document Recognition, accessed May 1, 2026, [https://www.infrrd.ai/newsroom/infrrd-engineers-win-docvqa-challenge-by-icdar](https://www.infrrd.ai/newsroom/infrrd-engineers-win-docvqa-challenge-by-icdar)  
42. NeurIPS 2023 Competition: Privacy Preserving Federated Learning Document VQA \- OpenReview, accessed May 1, 2026, [https://openreview.net/pdf?id=3HKNwejEEq](https://openreview.net/pdf?id=3HKNwejEEq)  
43. ICDAR 2024 Competition on Recognition and VQA on Handwritten Documents, accessed May 1, 2026, [https://www.researchgate.net/publication/377534357\_ICDAR\_2024\_Competition\_on\_Recognition\_and\_VQA\_on\_Handwritten\_Documents](https://www.researchgate.net/publication/377534357_ICDAR_2024_Competition_on_Recognition_and_VQA_on_Handwritten_Documents)  
44. Competitions – icdar2024 Site, accessed May 1, 2026, [https://icdar2024.net/competitions/](https://icdar2024.net/competitions/)  
45. DocVQA, accessed May 1, 2026, [https://www.docvqa.org/](https://www.docvqa.org/)  
46. shunk031/ANLS: ANLS: Average Normalized Levenshtein Similarity \- GitHub, accessed May 1, 2026, [https://github.com/shunk031/ANLS](https://github.com/shunk031/ANLS)  
47. Document Visual Question Answering System — A Serviceable Case Study | by Yash Dixit, accessed May 1, 2026, [https://yashcdixit1998.medium.com/document-visual-question-answering-system-a-serviceable-case-study-a9ea662fab68](https://yashcdixit1998.medium.com/document-visual-question-answering-system-a-serviceable-case-study-a9ea662fab68)  
48. Why Your VLM Prompts Are Backwards (And How to Fix It) | by Suresh R \- GoPenAI, accessed May 1, 2026, [https://blog.gopenai.com/why-your-vlm-prompts-are-backwards-and-how-to-fix-it-4ad0c8fad429](https://blog.gopenai.com/why-your-vlm-prompts-are-backwards-and-how-to-fix-it-4ad0c8fad429)  
49. Where is this coming from? Making groundedness count in the evaluation of Document VQA models \- ACL Anthology, accessed May 1, 2026, [https://aclanthology.org/2025.findings-naacl.295.pdf](https://aclanthology.org/2025.findings-naacl.295.pdf)  
50. Insights on Model Evaluation and Fine-Tuning \- Ultralytics YOLO Docs, accessed May 1, 2026, [https://docs.ultralytics.com/guides/model-evaluation-insights/](https://docs.ultralytics.com/guides/model-evaluation-insights/)  
51. ANLS\* \- A Universal Document Processing Metric for Generative Large Language Models, accessed May 1, 2026, [https://arxiv.org/html/2402.03848v2](https://arxiv.org/html/2402.03848v2)  
52. \[Literature Review\] ANLS\* \-- A Universal Document Processing Metric for Generative Large Language Models \- Moonlight | AI Colleague for Research Papers, accessed May 1, 2026, [https://www.themoonlight.io/en/review/anls-a-universal-document-processing-metric-for-generative-large-language-models](https://www.themoonlight.io/en/review/anls-a-universal-document-processing-metric-for-generative-large-language-models)  
53. ANLS\* \-- A Universal Document Processing Metric for Generative Large Language Models \- arXiv, accessed May 1, 2026, [https://arxiv.org/pdf/2402.03848](https://arxiv.org/pdf/2402.03848)  
54. Where is this coming from? Making groundedness count in the evaluation of Document VQA models \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/390177225\_Where\_is\_this\_coming\_from\_Making\_groundedness\_count\_in\_the\_evaluation\_of\_Document\_VQA\_models](https://www.researchgate.net/publication/390177225_Where_is_this_coming_from_Making_groundedness_count_in_the_evaluation_of_Document_VQA_models)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAYCAYAAAAoG9cuAAAAmUlEQVR4XmNgGAXkAkUg9gViHSBmRJMDg0YgPgTEeUC8Eoh7UKUZGGSB+BMQC0L5k4H4CkIaAjKB+B8QFwGxNBC7ALEBigogEAPir0D8H4qnoEojgCkQtwLxTQaIwhhkyUlAfBuJLwnEP4E4AkmM4T0Qz0Pi2wDxLSAWQBJjSATiwwwQH4EU7wRiBWQFMMAMxEpAzIEuMbwBAMj1F+ENSt3lAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAZCAYAAAAIcL+IAAAAtUlEQVR4Xu3PvwqBURzG8d8gfxYlG6uMsllfLGRwA6wyG61mE8NrUO7BpkgxSrkDuQQlKcX3dM7RL5PFxFOf4Tzn6fS+Iv98K0VUEXu/8Mliizl6WGKBlh4lcUCouj4eqKtOprghrboB7mIfeeWjYRxnbHzhssZOFw2x32Je8DF/fMVQdRKIHdZUV3ZdEyV0TRnFBW03SmHvhjmMUXB30sERE8xQwQkrjPzIJ4G8OkeQUecfzBOzmSGqupZ2UQAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAZCAYAAADjRwSLAAAAoElEQVR4XmNgGAXkAmEg9gNia3QJEOAF4oVAfAiI86DsHUAsB1PADcSXgHgrELPABIHgNhD3wjhzgPg/EBvCpSFgAxDfgXEIKhIA4p9AvA9FGgIeAvFNEMOfAWJKE4o0A4MaVHwqiOMK5UQgqwCCBiB+D8RaIA4jEF8A4h4kBeZQBTZIYgyWQLwTiJcB8SIgPg3EjsgKkIEkEEugCw5vAAAqhx4Wq50fpgAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALoAAAAYCAYAAABN2ucUAAAHJklEQVR4Xu2aBawcVRSGD+7uFoqUYsEtTUMprsETNLRAcXcLFCsenAAptIQAQRIgELTQh0NwQtBCH5Ziwd3P17O37+7ZO29ntu3MI5kv+bPvnXt3dmfm3HtkVqSmpqampmY+1WzeWBJLeENNn2BxbyiJmVULeePUgBN6QTWPHyiJm1S7eWNNpWyiutsbS4IN9znVSn5gSphF9bxqmB8okcVUX6kG+YEKmUN1rDeWzD2qT1Wfqz5Trd08LLurPlF9oBqvek+1fNMMkZNVHzb0fvR6VjzJsaxqomoFP1Ai24l914X9QKdcrHpRNZ0fKJkTVRNUM/mBEiF9O0k1VvWb6ufm4UpYUfWd6l+xyOfBEb4Wi4iE/BTcW47B+eRJR15SXeKNFfCg6hZv7IS5VT+oBvqBCuAmsdMc5AdKBKc5XDVY9aT0DUfnehBZiHgsPp+7EnnecDYPi4WF8oAfSDBE9YVUl8bGDBA751X8QFGOVL3pjRVyuup1b6yIh6VvOPptqpVVI8Wc9ZTmYdlUdbWzeQ4Qe+/xfiABeflV3lghj6su98YUy4jlO6tKa3ryqFge2Bv9VFuqFnD2TiD321myc789VP9I665VBX3F0V9rvC6l+kv1sWqGnmE5W7VL9H8Kwj+Ovo4fcEyv+l11lB9wrCFWrFLfTQn447piPjG/GwtcLzk24zPFQvARqtvF8vEYChhvC5Ar36h6VXW0qks1WnVtNCcvFEjPqm5VHSa2Sq+U1gu1ntgN2dHZq6AvODqhm/sWuEvs+uwU2Z6Q9hsDBe330rxAUiwpdvxt/UAD2sBPq+5THaN6TPWIaq94Uk62ECueL1IdJ1YXpNLWE8S+U9ZCmLQDkH9TYAGOFa8MTvoPSR8ccGrmz9j4nzyeD7xg8ox8kF/xPeKQGy4okSaGk8F+qLN75hRbLNzklLpU4xpzEDdkTd5YgL7g6AeqDo7+p3bg+nBOMLvq5Z7hJP3F3nO/H0iwodhccnoP9RxpZbzRcU+Zv1Vky8M+qr+lucu2p+pP6fHXAIuaz8jM07lApAGsPFYiuRwhJ7C02AEIQZ69pdURmYdtm8jWDqLC22LtsVndGF2AUc4G36jO8MYKwNF/8cYM2AzoZuRRkXZZyM9jKDzDjeeeXtE83MJwsfnsmikWif4ODpjq3twgltbEKSxpE+kUiyAvpK1c13udHd/ke/rosFrDPtjZJ8MFZUdiEvIFxloNO68ewhEnxY4RGCF2EYpU41uLfYZ3XOoG7Nc5O9ATPscbKwBH/9UbMyBa4JR5NEZ6omQ7UoV5cNxrVOdKcxqTgnSR+aSFHjYfol2ATZHo62Ee9qecvUss5SjChZJ23GENO88GYshMsLOoMyHR52K8K62rJTibL2RwZFapP6lx0j5Mei4V+4yNnT10AXZ19pBOcTN7gyKGCLNZARUtpos4+rSAp4J3eqPY5kPU+0lsIbQ7r4mqHyW9uEgReW4QCM62YGQDojh2dvAA9RXXp2i/nZqPFMVHDRYkGYiPeCGd6u/skyCc0ZMO8OSRHTp+zD6X2AHiEwVWLydwfmTjkSw2HJe/2U0CdFJ8WhJgkfEZizp7l9iioQccw7Harl6xm32q6rQCWm7SO/ODo9PDrQqckJ5+ChoIXKfQkcmCxcK8h/yA2E5JB4fXAKkq8zeIbLBRw073LTCkYdtetb4013o+3Yp5RvWOs1GbsXhTEX5fyU6n5FuxjklgkFiFO29kA6rxUc4GtB3JyYCWE8fipPYXiwB0cYCowCokYqRggbFA4iqe3v14ac4NA5uLfQ4OXzVEMKILhW/ZcONJCXzEC3B9uPmX+QEH3TK/mRE1uR/dYucYQ3eM+XHkB5yMNJjaDSgYXxGby3tIi8mlYWjDnnJaoKPGdeUcgWPTWaLNneoKjRRbkEkIQaQedFpwUnanfvGEBmPEdlcPK3SCmLPT0iLXxvkpLHmgEHJ3ipC3xMJo1o45VKwlRRQIfft+0XjMIWJtsFSYLQPSACIh505Oitg02IFoh5UBrTucioKNTYLH4Cm4D9yXFBSeOAcpAk7HOXQ3bNwrNifG8BNPt1g95iHd7BZz4DFi6SjHC63iwECxdCnOKGJIO4lI+ARpDwuGB0JZWcEd0uaJLqujt7QC2J15tJyaw/sHSPNvTwhzfFEPTkw13RsshHbVOQuLk66pDnbnrFYkaWt8n0O3KcVYb3AE/yLXz4JsggUzxc9VOBA7Mvlup+D4qRywKOR11BHtFkzNtGVpsdqEdLdTaGeP9sYOGK76SNIpTWEoQKjK+XKdQB4YP9ToFFqa7X6zUVMOpBY0C9gIO4Fu0ereWBDqyS8lu07pCLoSN3tjTnDyVDpThB3EfqoQp0k11UFKQpTezw/kgPTUF7OdQKF9njdODUZIa/+0LKisq/rsmjS0n6eJo+WAZzn81KTTiFJTU1NT87/hPzbwjHFPxj5tAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB8AAAAYCAYAAAACqyaBAAABsElEQVR4Xu2UyytFURTGP4+IJEkmMhKSTCQZUfgDRIaGHpmJkhmZmFBiIgMT5TGWqWJAkYRIXjflUcojycDzW9Y53XPWPfdeBkbur37ds9e3975nn33OBhL8d+rpGb2mN3TWH3+zSUP0GNp3zJcC2fSInkP7nNBT5zrP0y8q8/SBftAik6XQQbpKC/yRjx76SUdpssliskv7oIPtyoQR2mqLBlmAjK+2QSxK6SLNoc/0nmb6egBrNN/ULFf0Efqkfkwn7Xaup6B33xGOkUG3Pe0giqHjlmwQjzla5lxXQCeRbXBpoJOedhDt0HGydb9ix7RXoBPVOu1h2hKOA5GvRMZU2SAW7n57aYZO5NZlv+N9MpeIvt9p0Pcpgi6E99tFJrigr7SEbvnjCKSP3OyyDRx6oQuKYIGW2yIZgE64TsdNZpEXVvr22wC66n2aboMkeuD8WuQxv0AnbTKZRRYg/WpsQIbotC0KbXQP0U+jGfpOc23gQVZ2S59oqqdeSCegN1XnqaMReo7LykQ5VuWMt1TSDVt0yKKH9A76B3KT8p6EoHO/OTU544OebIIECf6OL2EuXBqGqt6RAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAAeElEQVR4XmNgGAUDAeSAWBmIVZCwEhDzICsyAuKLQPwfB14MUygNxJeBOBOIbYC4AojXA7EtEFsDsSUQ88MUhwGxNowDBFOBOA2JjxOYAPEfIOZAl8AGmhkgTiIKXALiReiC2AA3EH8D4hR0CVxABogZ0QVHATYAAKTSEvZf02u5AAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAG8AAAAYCAYAAAD04qMZAAAE4ElEQVR4Xu2Zd6gdRRSHfxbsDRVbLNgbKooVS6z/2TuC8DCxooINKyQWUKyIBUVFjF3BAlYUe8VeUMTyothAsBfsns+z6509b/bu3htz86L3gx/ZPTN7szszp8w8aciQIUPmMS0ZjQNiQjQMac9cpvtNm8eGATHZdFk0zk5sb/rA9Jnpc9MN1ea/ecE03fSuvO+FlVZpEdM7plF5n/dM7xfX3bzqItN10ThA5pB/2wmxYXbjFtPXpj9Mq4Y2PGSq6Ql1DzXHmP40XWCaM7RFdjF9Z1o2NgwYvP5X01qxoYaJ8mixh5q/cWC8bjpePvjRs+Ac0z7RGGAB8PymsSHD46aTo3EWMU3+7m1Z3DTF9IxpkjxvzzLWNN1mWsz0g+kr0wKVHtKTpqWCLfKp6Ru5p3ZjA/kkdwupg2Qb0++mJWJDAwuajjU9L4863A+cQ01HFNdXygf2kE6z5je9nNznWF3+3D2xIcOp8hDdDQZyJ40N4f3Ab+1q2jI2FCwnf/e9Y0NL5pWPF5OIR+KZ/bKi/JtXS7SKaaG0U8rNprWL6/XkH0IYLdnBdGlyn+Ng+XOE3iauNr0YjQlHyYue00x3yhfUS5Ue7VhYXhCRq48urh+QD1AKhQsR5+Jg7xUizgGmp03nmpapNndlI9Nr8jHM6fpO1yqvhvtH5Q8QTuBM016d5ixUqTyzcWzI8Ijqc8yI6UfTysU9q5rChlXdC4QwFuC9prkTOxUwBVWEvrdHY5+QA/mtj+Xv3wRF4Bumw01bmU6SL9qt5dFiC9Oi//ROKPNdyp7yiSjt5Lum/PSJ6vMdH0M+LRmVL4gI4eEXVQeXkI3tvMTWBrybb9gw2O+Se3XkDvminRF4Vzychcb2A89vw76mdZN79p5p2qrlMHXyXQkT8JG8hF5D3UMc0IeBui82FBwnXxAl38oTfeQU+e8Qpku2LWw7J7YmWCg/yz088qF8Txq5xvRUNLYEr+Ddn5OP53zV5p4ghOOxm8SGHLeqOusllPEM2rNqzgUUPPQ9MTbIve5NVcMHA5jLoWyYGfS00p0irwRTz21iN/n7nBHs5SLLnaow0TdGYwNLm86WRyZyXS7q9App5ze1WADM8lvFvxHC5E/yj909tEVYAPTLHXOdbroq2Cg+KBwij8lXbwr7wVeKa0JhOUBUdHUFAVUq77N/sE+Vb4PWCXYg0pwVjTWsYLrE9LA8IuTGr19IJ+S/Rg6Ud6w7KbhWvuq7lb541hfyoiItDMoPZBAnJnaguMnlHbYQHKuVjMifpz8TlebmUfn/iTdFGEyKsPMT22byiaMgiLDKOVk6KDZkYIHerU4x929D4cShQS07ys8x8SzEnoszzgjla/SEEvYdb5u+lA8wk8zqnS7/bVwfG5MRV+aIPJ+mkw0cApDsKR7w5iPlnvu9vPpiG1PC9obJm5TYUqjQHjTdJB8MQvJ2lR4d8ES+oc3J0MyECplKe3JsGE+wKaaCrNtWrKRqjiO35EpuqrSmqoyz07rwWkIEYo81HlheYxf7uOMK00PR2CNs3sv9YL+wKIgObVc7aYLTpDaa0Xcbt7DCyEFUhv1AKd1rdZiDPEsB1VjdFawvL8Da6HKNTQ3/GcizVLu5kNjEfur9IDkyQf73SYqrIX3AsVtdITGz4Ty21WZ4yJAh/2f+AoUX+9oOzTnYAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAWCAYAAAD5Jg1dAAAAx0lEQVR4Xt3QMctBYRiH8TthkGTEzCAy2Egmq9XrI9hkYTLJF7BIJrIok7Kb2KwGs+H9AO/2FtfpuZ+6D5+Af/2W63nO6XREPmsRtDHFANnwsVscOxzQxBB31OylYD38ImFa8OYboqbJBXsbWAsPNHxIa1j5oKtqH/tQ0LDwQVfSPvch+OBQ0BW1b33Ia1j6oCtrn/mQxL+YJ3V1cRdHNh5xtoF1xV2s2PiDP+RMW4t7wdsmuKKPDU5IhW6YZdAR9ydiL2ffsyeA0yVvb/qBbwAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAG8AAAAYCAYAAAD04qMZAAAE1klEQVR4Xu2ZZ6xlUxTH/0r0Mkr0EkYZLUrUKKN+wwgGkYgXRi+JXsMoHyR6kBAmohsSfMAQicEQJqMTIsoboiUSvUS3ftY53j7LOfece+e63sucX/LPnLvWPvfd3dZae4/U0tLSsohpxWgcEKtHQ0tzFjLNMG0fHQNiiumGaBxL7G76wPSZ6XPTnUX338wxzTW9K297VcErLWN6xzQsb/Oe6f3sudOuutp0WzQOkAXkfTsjOsYa95q+Nv1hGh987JCppmfUOdScYvrTdKVpweCL7GP6zrRqdAwYdv2vpgnRMZZ43XS6fPDjzoLLTJOjMcAC4P1to6OEp03nROP/xO3y395vOkWdvrGh6T7TONMPpq9MSxRaSLNMKwVb5FPTN/Kd2onN5ZM8kM41YBfT76YVoqNH6N8dpvtNiwZf3znGdHz2fJN8YI8ecWtx08vJ5zLWl7/3cHSUcJ48RHeCgdxL/w7hvcB37WvaMToyVpP/9gOjo0smyvtP3bBp8DVlLXmf10u0rmmptFHKPaaNsufN5B0hjObsYbo++VzGUfL3CL113GJ6MRoTTpIXPeebHpQvqJcKLZqxtLwgIlefnD0/Jh+gFAoXIs61wd4E3iV/P2m6UT7QvbCV6TX5GJaJnVzKq+HzTPkLhBO4xHTAiLsUVhvvbB0dJdDRqhwzZPrRtE72mbBDYTM7b9CQJeUL8BHTwomdCpiCKkJbwlxTSA2Hmp4zXa55K7woAt8wHWfayXS2fNHuLI8WO5iW/ad1Qp7vUvaXT0RuJ9/V5adPVJ3vOIiTT3OG5Qsiwqr9RcXBJWRjY4C6gd1NH7YM9ofkuzrygHzR1kFfSCkvmC4wLVd098RBpk2Sz5w907RVybEayXc5TMBH8hJ6A3UOcUAbBurR6Mg4Tb4gcr41nZp8zjlX/j2E6ZxdM9veia0OFsrP8h0e+VB+Jo1MMz0bjSWwC1ioZ5oWC75+QBj+2LRNdJQxXcVZz6GMZ9CeV30uoOCh7VnRIV+pb6pYdTGAZTmUAzODnla6F8orwXTn1jFJ/nsuDvZ8kZXdqjDRd0VjBUzaCfJQTp+5pOgXpJ3f1GBhMMtvZf9GCJM/yTu7X/BFWAC0K7vmush0c7BRfFA4RJ6Sh6MUzoOvZM+EwjwsL29aJXuOUKXyew4J9qnyY9DGwQ5EmkujsQZy6eHyvEcaqEstTeB7yH+1HCZvWHUbcqt81TNQVbCzvpAXFWlhsKbpOvkgTkzsQHFTlnc4QnCtljMkf5/2TFSam4flf5PdFGExUoRdkdi2k08cBUGEVc7N0hHR0RD+Hgt8puka0xpFd1dQOHFpUMme8ntMdhbizMUdZ4TyNe6EHM4db5u+lA8wk8zqnSv/brY+NiYj7uwheT5NJxu4BCAUUTywm0+U79zv5dUXx5gcjjdM3pGJLYXc9LjpbvlgEJJ3K7QYgZ1IH5rcDNXBOHLWI0pw9u0GKmQq7SnRMZrgUEwFWXWsWFvFHLeyym8qqNLqqjJK+KrwmkME4ozVT7aQn/tqc1eAXRsX+6iDjj0RjV3C4T0/D/YKi4LoMKpX+2iDFUYOojLsBUrpptVhJ8izFFDd7pD5HvID1W5ZSKzjYM37RTI3G/z/JMVVSw9w7VZVSPzXcB/b6DDc0tIyP/MXGqnzYXnXyyoAAAAASUVORK5CYII=>