## Framework Overview

DocDancer introduces an agentic framework for document-grounded information seeking that represents a fundamental shift from traditional Document Question Answering (DocQA) approaches. The system operates through a **ReAct (Reasoning and Acting)** framework that enables iterative interaction with documents through a cycle of thoughts, actions, and observations.

\x3Cimg src="https://paper-assets.alphaxiv.org/figures/2601.05163v1/img-0.jpeg" alt="DocDancer framework overview showing the agent's interaction with a document through Search and Read tools" width="800"/>

*Figure 1: DocDancer's agentic framework demonstrating iterative information seeking through tool-based document interaction.*

Unlike conventional approaches that rely on single-shot retrieval or prompt engineering, DocDancer treats DocQA as an **autonomous information-seeking problem**. The agent dynamically explores documents using a minimal but powerful toolkit consisting of only two tools: **Search** for global keyword-based retrieval and **Read** for fine-grained, goal-oriented content extraction. This design philosophy emphasizes the "Bitter Lesson" principle, favoring scalable learning over complex hand-engineered solutions.

The framework's key innovation lies in its **end-to-end trainable nature**. Rather than relying on pre-defined prompts or closed-source models, DocDancer learns agentic behaviors directly from data, enabling it to develop sophisticated document navigation and comprehension strategies autonomously.

## Document Processing and Tool Design

DocDancer's effectiveness stems from significant enhancements to document processing and a carefully designed tool ecosystem. The system leverages **MinerU2.5** for high-precision layout analysis, identifying 17 distinct element types including text, images, and tables. This processing goes beyond traditional XML-based hierarchical representations by:

- **Enhanced Content Accuracy**: Removing structurally irrelevant elements while preserving semantic information
- **Improved Structural Understanding**: Using visual clustering to infer hierarchical levels and enable fine-grained section segmentation
- **Multimodal Integration**: Generating descriptive captions for visual elements to improve cross-modal retrieval

The tool design reflects a minimalist approach that prioritizes effectiveness over complexity:

**Search Tool**: Performs keyword-based full-text searches across the entire document, returning relevant section IDs, page numbers, and contextual snippets. This provides global textual signals that guide the agent toward relevant document regions.

**Read Tool**: Conducts goal-oriented reading of specific sections, extracting both textual content and visual information (images, tables, page screenshots). A multimodal model then integrates these inputs to provide consolidated, goal-relevant summaries.

This two-tool design demonstrates superior performance compared to systems using five or more tools, validating the principle that well-designed simplicity often outperforms complex architectures.

## Exploration-then-Synthesis Data Generation

A critical innovation in DocDancer is the **Exploration-then-Synthesis** pipeline for generating high-quality training data. This addresses the fundamental challenge of data scarcity in training sophisticated DocQA agents.

\x3Cimg src="https://paper-assets.alphaxiv.org/figures/2601.05163v1/img-1.jpeg" alt="Two-stage data synthesis process showing exploration and synthesis phases" width="600"/>

*Figure 2: The Exploration-then-Synthesis pipeline for generating high-quality training data through iterative document exploration followed by QA synthesis.*

**Exploration Stage**: An LLM iteratively interacts with source documents from diverse datasets (LongDocURL, MMDocRAG, CUAD, DUDE). At each step, the model generates:
- An **exploration intent** that guides the agent's next action
- An **action** (either Search or Read)
- An **observation** from executing the action

This creates rich trajectories that capture structured evidence across different document modalities and sections.

**Synthesis Stage**: Using the accumulated exploration trajectory, a synthesis model performs multi-observation reasoning to generate document-grounded question-answer pairs. The process emphasizes:
- **Multi-hop reasoning** across different document sections
- **Cross-modal integration** of textual and visual information
- **Anti-shortcut learning** to prevent superficial pattern matching

A final rejection sampling step using strong open-source models ensures only the highest-quality training trajectories are retained. Remarkably, this pipeline generates synthetic data that **outperforms human-annotated datasets** in training effectiveness, demonstrating the potential for autonomous data generation in complex reasoning tasks.

## Training Methodology and Performance

DocDancer employs **Supervised Fine-Tuning (SFT)** on the synthetic data, with strategic loss masking that excludes observation tokens from gradient computation. This ensures the agent learns from its own decision-making processes rather than external feedback, enhancing robustness and performance.

The training demonstrates remarkable data efficiency, achieving strong performance with only **5,000 agent trajectories**. The framework has been instantiated with open-source LLM backbones including Qwen3-4B-Thinking-2507 and Qwen3-30B-A3B-Thinking-2507, proving that sophisticated agentic behaviors can be trained on relatively modest computational resources.

## Experimental Results and Validation

DocDancer's effectiveness is validated through comprehensive experiments on **MMLongBench-Doc** and **DocBench**, two challenging long-context multimodal document understanding benchmarks.

\x3Cimg src="https://paper-assets.alphaxiv.org/figures/2601.05163v1/img-5.jpeg" alt="Performance comparison showing DocDancer outperforming baseline methods" width="600"/>

*Figure 3: Performance comparison between DocDancer and baseline methods on document understanding benchmarks, demonstrating the effectiveness of synthetic training data.*

**State-of-the-art Performance**: When instantiated with proprietary LLMs (GPT-5.2), DocDancer achieves 56.8 F1 and 67.6 LasJ on MMLongBench-Doc, and 85.5 on DocBench - **exceeding human baseline performance by 4 points**. Even with smaller open-source models, DocDancer maintains competitive performance against closed-source alternatives.

**Synthetic Data Superiority**: Models trained on DocDancer's synthetic data consistently outperform those trained on equally-sized human-annotated datasets, validating the quality and complexity of the generated training examples.

**Component Analysis**: Ablation studies reveal that DocDancer's enhanced document processing provides consistent improvements over baseline methods, while the two-tool design proves more effective than systems using larger tool sets. The framework shows robust generalization across diverse document domains, with particularly strong performance on structurally complex documents.

\x3Cimg src="https://paper-assets.alphaxiv.org/figures/2601.05163v1/img-7.jpeg" alt="Qualitative analysis showing DocDancer's multi-step reasoning process" width="800"/>

*Figure 4: Qualitative analysis demonstrating DocDancer's ability to perform multi-step reasoning across different document sections and modalities to compute complex financial metrics.*

## Significance and Implications

DocDancer represents a significant advancement in document understanding AI by demonstrating that **end-to-end trained agents can outperform traditional pipeline-based approaches** for complex document reasoning tasks. The research makes several important contributions:

**Paradigm Shift**: Moving from prompt-engineered pipelines to learnable agentic behaviors opens new possibilities for autonomous document understanding systems that can adapt and improve through experience.

**Data Generation Innovation**: The Exploration-then-Synthesis pipeline provides a scalable solution for generating high-quality training data for complex reasoning tasks, potentially applicable beyond document understanding to other domains requiring multi-step reasoning.

**Open-Source Accessibility**: By achieving competitive performance with open-source models, DocDancer democratizes access to advanced document intelligence capabilities, enabling broader research and application development.

**Practical Impact**: The demonstrated ability to exceed human performance on challenging benchmarks indicates readiness for real-world deployment in domains such as legal document analysis, financial report processing, academic research, and regulatory compliance.

The research establishes DocDancer as the **first end-to-end trained open-source DocQA agent**, setting a new standard for autonomous document understanding and providing a foundation for future developments in agentic AI systems. The work's emphasis on simplicity, effectiveness, and accessibility positions it to significantly influence both academic research and practical applications in document intelligence.