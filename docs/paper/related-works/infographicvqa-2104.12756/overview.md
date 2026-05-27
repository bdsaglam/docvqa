## Introduction

InfographicVQA represents a significant advancement in Visual Question Answering (VQA) research by introducing a comprehensive dataset specifically designed for understanding infographics. Created through a collaboration between researchers at IIIT Hyderabad and the Computer Vision Center at Universitat Autònoma de Barcelona, this work addresses the growing need for AI systems that can comprehend complex visual documents containing rich combinations of text, graphics, and data visualizations.

![First infographic example](https://paper-assets.alphaxiv.org/figures/2104.12756v2/img-0.jpeg)

Infographics present unique challenges for machine comprehension due to their dense information content, diverse layouts, and the intricate relationships between textual and visual elements. Unlike traditional VQA datasets that focus on natural images or simpler documents, InfographicVQA captures the complexity of real-world infographics collected from thousands of web sources, requiring models to perform sophisticated reasoning across multiple modalities.

## Dataset Creation and Characteristics

The InfographicVQA dataset comprises 5,485 infographic images paired with 30,035 question-answer pairs, making it one of the largest datasets for document-based VQA. The creation process involved multiple stages of careful curation and annotation.

The researchers collected over 10,000 images using search queries and applied a rigorous two-stage de-duplication process. First, perceptual hashing identified visually similar duplicates, then Jaccard similarity of OCR tokens removed images with similar textual content. This resulted in approximately 7,000 unique images from 2,594 distinct web domains, ensuring remarkable diversity in design and content.

The annotation process was conducted by a trained team of 13 annotators using a custom web-based tool. This approach was chosen over crowdsourcing due to the complexity of infographics and the need for consistent, high-quality annotations. The process included two stages: initial question-answer pair collection and subsequent validation with QA type assignment.

![QA type distributions](https://paper-assets.alphaxiv.org/figures/2104.12756v2/img-1.jpeg)

The dataset exhibits several distinctive characteristics that set it apart from existing VQA benchmarks. Infographics contain an average of 217.89 text tokens per image, significantly higher than other text-rich datasets like TextVQA (12.17 tokens) or DocVQA (182.75 tokens). Questions are notably complex, with an average length of 11.54 tokens and 99.11% uniqueness, indicating sophisticated reasoning requirements.

![Token distribution comparison](https://paper-assets.alphaxiv.org/figures/2104.12756v2/img-11.jpeg)

The answers are categorized into four types based on their sources: Image-span (exact text from image), Question-span (text from question), Multi-span (multiple text pieces), and Non-extractive (computed numerical answers). This categorization reveals that while many answers can be extracted directly from text, a significant portion requires numerical reasoning or synthesis across multiple elements.

## Methodology and Evaluation Framework

The evaluation framework establishes both upper bounds and challenging baselines to assess model performance comprehensively. Upper bound analyses reveal that over 75% of questions have answers present either in common vocabularies or as subsequences of OCR text, indicating that the challenge lies not in answer availability but in the reasoning required to identify correct responses.

![Model architecture diagram](https://paper-assets.alphaxiv.org/figures/2104.12756v2/img-4.jpeg)

The researchers evaluated several state-of-the-art models, including M4C (a multimodal transformer for text-based VQA) and LayoutLM (a layout-aware document understanding model). M4C was adapted with different visual feature configurations, while LayoutLM was modified for span prediction tasks with specialized pretraining on InfographicVQA data.

The evaluation employed two primary metrics: Average Normalized Levenshtein Similarity (ANLS) and exact match accuracy. ANLS provides robustness against minor OCR errors, while accuracy measures precise matching. Human performance was established as a gold standard, achieving near-perfect scores (ANLS: 0.980, Accuracy: 95.70%).

## Main Findings and Model Performance

The experimental results reveal a substantial gap between human and machine performance, highlighting the challenging nature of infographic understanding. The best-performing model, LayoutLM, achieved only 0.272 ANLS and 19.74% accuracy, while M4C performed significantly worse with 0.147 ANLS and 6.64% accuracy.

![Performance comparison across different question types](https://paper-assets.alphaxiv.org/figures/2104.12756v2/img-5.jpeg)

Several key findings emerged from the evaluation:

**Visual Feature Limitations**: Surprisingly, generic object detection features from models trained on Visual Genome or document layout analysis provided minimal benefit. In some cases, models without visual features performed comparably to those with sophisticated visual encoders, suggesting that current visual feature extraction methods are inadequate for infographics.

**Layout Information Importance**: LayoutLM's superior performance over M4C demonstrates the critical role of spatial layout information in infographic understanding. The model's ability to incorporate 2D position embeddings alongside textual content proved essential for reasonable performance.

**Domain Adaptation Benefits**: In-domain pretraining on InfographicVQA data significantly improved LayoutLM performance, emphasizing the importance of domain-specific adaptation for transformer models.

**Reasoning Limitations**: Qualitative analysis revealed that current models struggle with elementary arithmetic operations, counting tasks, and multi-step reasoning. Models often fail on questions requiring subtraction, sorting, or synthesis of information across multiple visual elements.

## Question Analysis and Complexity

The dataset's question distribution reveals the sophisticated reasoning demands placed on models. Common question patterns include "How many..." queries requiring counting operations, percentage calculations needing arithmetic reasoning, and comparative questions demanding sorting capabilities.

![Most frequent question beginnings](https://paper-assets.alphaxiv.org/figures/2104.12756v2/img-12.jpeg)

![Most frequent answer words](https://paper-assets.alphaxiv.org/figures/2104.12756v2/img-14.jpeg)

The prevalence of numerical answers and color-related queries reflects the visual and quantitative nature of infographics. Many questions require understanding relationships between different chart elements, interpreting color coding schemes, or extracting specific data points from complex visualizations.

Evidence type analysis shows that questions draw from diverse sources: text (most common), figures, tables/lists, maps, and visual/layout elements. This multimodal evidence requirement distinguishes InfographicVQA from purely text-based or image-based VQA tasks.

## Significance and Research Impact

InfographicVQA establishes a crucial benchmark that exposes significant limitations in current multimodal AI systems. The substantial performance gap between humans and machines indicates fundamental challenges in joint reasoning across visual, textual, and spatial modalities.

The dataset's impact extends beyond academic research into practical applications. Enhanced infographic understanding capabilities could revolutionize information retrieval, enable automatic content summarization, improve accessibility tools for visually impaired users, and support business intelligence applications requiring data extraction from visual reports.

The work identifies several critical research directions for the community:

**Specialized Visual Encoders**: Generic object detectors prove inadequate for infographics, necessitating development of specialized visual feature extractors that understand charts, graphs, icons, and their semantic relationships.

**Advanced Multimodal Fusion**: Current concatenation-based approaches for combining visual and textual features are insufficient. More sophisticated architectures that can perform integrated reasoning across modalities are needed.

**Numerical Reasoning Integration**: The prevalence of arithmetic and counting questions highlights the need for models that can perform discrete mathematical operations within visual contexts.

The dataset serves as a catalyst for developing more capable multimodal AI systems that can truly understand and reason about complex visual information. By providing a challenging, diverse, and realistic benchmark, InfographicVQA pushes the field toward achieving human-level comprehension of information-dense visual documents.

The research demonstrates that while current state-of-the-art models excel at text extraction and simple pattern matching, they fall short of the sophisticated reasoning required for genuine document understanding. This gap presents both a challenge and an opportunity for the research community to develop more advanced AI systems capable of human-like visual reasoning and comprehension.