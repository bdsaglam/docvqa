## Background and Problem Definition

The field of visual document understanding has reached a critical juncture. While optical character recognition (OCR) systems have achieved remarkable maturity—with state-of-the-art vision-language models like MinerU2.5 and PaddleOCR-VL consistently reaching 90-95% accuracy on full-document parsing—a fundamental inefficiency persists in downstream applications. Traditional OCR operates under a "parse everything" paradigm, extracting complete document content regardless of specific information needs. This creates what the authors term a "paradox": while general-purpose document parsing is largely solved, there remains a significant gap for instruction-following or on-demand parsing in practical applications.

This inefficiency becomes particularly pronounced in visual Retrieval-Augmented Generation (RAG) systems, which have emerged as a crucial technology for processing complex documents while mitigating model hallucinations. Current visual RAG approaches typically employ page-level retrieval, feeding entire document pages directly to generative models. While this preserves layout semantics, it introduces substantial overhead: retrieving whole pages containing irrelevant headers, footers, and decorative elements overloads the generator's attention mechanism and dilutes salient evidence. Furthermore, compressing high-resolution pages into limited visual token budgets sacrifices fine-grained details, particularly problematic for complex elements like rotated tables or small annotations.

AgenticOCR addresses this fundamental limitation by transforming OCR from a static preprocessing step into a dynamic, query-driven process that extracts only the information necessary to answer specific questions.

## Core Methodology and Technical Framework

The AgenticOCR framework centers on a unified visual interaction tool called `image_zoom_and_ocr_tool`, which combines region localization, geometric correction, and content recognition into a single atomic operation. This tool takes as input an image (I), a bounding box (b) for the region to crop, a rotation angle (θ), and a semantic type (τ) such as "region", "text", "table", "image", or "equation".

The tool's behavior adapts based on the semantic type parameter:

$$
\text{Tool Output} = \begin{cases} 
\text{MinerU full pipeline} & \text{if } \tau = \text{region} \\
\text{Direct recognition} & \text{if } \tau \in \{\text{text}, \text{table}, \text{equation}\} \\
\text{Visual patch only} & \text{if } \tau = \text{image}
\end{cases}
$$

This design enables the model to actively decide where to focus attention, how to orient the content, and at what semantic granularity to parse—mimicking human visual attention during document reading.

The training methodology employs a two-stage approach combining supervised fine-tuning (SFT) with reinforcement learning (RL). The SFT phase establishes a robust prior for tool invocation through trajectory distillation from Gemini-3-Pro-Preview using rejection sampling. High-quality multi-turn reasoning traces are constructed from the ViDoRe-v3 benchmark, with a dual-threshold filtering strategy to handle inconsistent annotation granularity.

The RL phase uses Group Relative Policy Optimization (GRPO) with a carefully designed reward function that combines dual-recall metrics with behavioral penalties:

$$
R_{\text{pos}} = \frac{\text{Recall}_{\text{min}} + \text{Recall}_{\text{EM}}}{2} - P_{\text{over-pred}} - P_{\text{overlap}} - P_{\text{oversized}}
$$

where the penalties discourage spurious predictions, redundant overlaps, and lazy full-page parsing behaviors.

## Experimental Results and Performance Analysis

The experimental evaluation demonstrates AgenticOCR's effectiveness across multiple benchmarks. On MMLongBench-Doc, the AgenticOCR-8B model achieved an overall accuracy of 66.4 when providing "Evidence + OCR" input to Gemini-2.5-Pro, surpassing the human expert baseline of 65.8 and approaching the performance of DocLens (67.6). On FinRAGBench-V, designed for dense financial documents, AgenticOCR-8B achieved 78.6 accuracy, outperforming all prior agentic frameworks.

The system exhibited particular strengths across different modalities:
- Text (TXT): 67.4 accuracy
- Layout (LAY): 68.8 accuracy  
- Figure (FIG): 63.6 accuracy

These results are notable given that AgenticOCR uses a significantly smaller model backbone (Qwen3-VL-8B) compared to competing systems. The "Evidence+OCR" input format, which interleaves low-resolution page snapshots with cropped visual patches and extracted text, provides a rich yet compact representation supporting both layout-sensitive and content-heavy reasoning.

However, the evaluation also revealed specific limitations. Performance on table-related questions lagged behind top agentic frameworks, often returning incomplete extractions that missed necessary headers or context. Additionally, accuracy on unanswerable questions was lower than DocLens, attributed to lower retrieval precision in the current system.

## Efficiency Analysis and Token Optimization

A critical contribution of AgenticOCR is its potential for computational efficiency. Analysis of visual token consumption revealed that with Qwen3-VL-32B-Thinking as the generator, the "Evidence+OCR" setting actually reduced total input tokens from 14,517 to 13,238 compared to "Page+OCR" while maintaining or improving accuracy. This validates AgenticOCR's core premise of maximizing signal-to-token ratio by filtering irrelevant visual information.

The efficiency gains stem from selective attention to query-relevant regions rather than processing entire pages. Even when working with models that have fixed per-image token allocation policies (like Gemini), AgenticOCR improves the quality of information delivered to the generator, resulting in better reasoning outcomes despite similar token costs.

## Broader Implications and Future Directions

AgenticOCR represents a paradigm shift toward more intelligent document processing systems. By positioning itself as a "third building block" alongside existing Embedding and Reranking modules, it provides a complete visual RAG architecture that can be integrated into diverse applications without extensive modifications.

The implications extend beyond visual RAG to other document intelligence tasks including key information extraction, element-level evidence citation, and interactive document assistants. The open-sourcing of code, models, and datasets facilitates reproducibility and community development.

The authors acknowledge several areas for future improvement, including enhanced table extraction capabilities, improved handling of unanswerable questions through better retrieval precision, and exploration of tighter integration between visual agents and generative models. These challenges point toward continued innovation in agentic document intelligence systems.

AgenticOCR fundamentally demonstrates that query-driven, selective parsing can substantially improve both efficiency and accuracy in visual document understanding, establishing a new standard for how AI systems should interact with complex visual information.