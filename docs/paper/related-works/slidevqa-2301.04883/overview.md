## Introduction

Document Visual Question Answering (DocVQA) has emerged as a critical challenge in artificial intelligence, requiring systems to understand and reason about complex, multi-modal documents that combine text, visual elements, and intricate layouts. While significant progress has been made in single-image document understanding, a fundamental limitation persists: most real-world documents, particularly presentation slides, reports, and manuals, distribute crucial information across multiple pages, requiring sophisticated reasoning capabilities to connect disparate facts.

![SlideVQA Dataset Examples](https://paper-assets.alphaxiv.org/figures/2301.04883v1/img-0.jpeg)
*Figure 1: Examples from the SlideVQA dataset showing multi-image reasoning scenarios where questions require information synthesis across multiple slides, demonstrating various reasoning types including multi-hop and numerical operations.*

The research presented by Tanaka et al. addresses this gap by introducing SlideVQA, the first large-scale dataset specifically designed for Document Visual Question Answering across multiple images, along with M3D (Multi-Modal Multi-image Document VQA), an end-to-end model capable of reasoning across slide decks. This work represents a significant step toward building AI systems that can genuinely comprehend real-world, multi-page documents.

## Problem Statement and Motivation

Current Document VQA systems face several critical limitations that prevent them from effectively handling real-world document understanding tasks. Existing datasets like DocVQA, VisualMRC, and InfographicVQA primarily focus on reasoning within single document images, which inadequately represents how humans interact with complex documents. In practice, understanding documents often requires synthesizing information distributed across multiple pages, performing multi-hop reasoning to connect related concepts, and executing numerical computations based on visual data.

The authors identified presentation slide decks as an ideal testbed for multi-image document understanding due to their widespread use and strategic arrangement of both visual and textual information across multiple slides to build coherent narratives. Unlike previous multi-modal QA datasets that focus on generic visual content, SlideVQA specifically targets the unique characteristics of document images, where layout understanding, OCR accuracy, and structured content interpretation are paramount.

The motivation extends beyond academic benchmarking to practical applications where intelligent systems need to automatically summarize presentations, facilitate information retrieval across large document repositories, and support professional tasks by quickly comprehending complex visual-textual information.

## Dataset Creation and Characteristics

The SlideVQA dataset construction involved a multi-stage process designed to ensure both scale and quality. The authors collected 25,327 slide decks from SlideShare, ultimately filtering down to 2,619 high-quality decks containing 52,480 images based on criteria including English language, comprehensibility, and the presence of graphs, tables, figures, or numerical data.

The annotation process was comprehensive and multi-faceted. Crowd workers provided detailed bounding box annotations categorizing content into nine semantic classes: Title, Page-text, Obj-text (text within figures/tables), Caption, Other-text, Diagram, Table, Image, and Figure. This resulted in 890,945 bounding boxes, providing rich layout and visual context for model training.

![Dataset Statistics](https://paper-assets.alphaxiv.org/figures/2301.04883v1/img-1.jpeg)
*Figure 2: Statistical analysis of the SlideVQA dataset showing (a) distribution of bounding box categories, (b) reasoning types required by questions, (c) numerical operation types, and (d) answer format types, demonstrating the dataset's complexity and diversity.*

Question-answer pair creation followed a two-pronged approach. Single-hop questions (12,466 pairs) were generated where answers could be found on individual slides, with emphasis on numerical reasoning capabilities. Multi-hop questions (2,018 pairs) were created by editing existing single-hop questions, replacing "bridge entities" with descriptive phrases from other related slides, thus requiring models to link information across multiple pages.

A unique contribution of SlideVQA is the annotation of arithmetic expressions for numerical answers. Rather than just providing final numbers, annotators supplied underlying computational expressions (e.g., "11% - 6%"), enabling models to learn interpretable reasoning processes rather than merely memorizing numerical outputs.

## M3D Model Architecture

The M3D (Multi-Modal Multi-image Document VQA) model represents an innovative approach to multi-image document understanding, building upon the Fusion-in-Decoder (FiD) architecture while incorporating specific adaptations for the SlideVQA task.

![M3D Architecture](https://paper-assets.alphaxiv.org/figures/2301.04883v1/img-4.jpeg)
*Figure 3: Overview of the M3D model architecture showing (a) the unified sequence-to-sequence framework for joint evidence selection and question answering, and (b) the multi-modal input representation incorporating text, layout, and visual features.*

The model employs a unified sequence-to-sequence framework that jointly performs evidence selection and question answering within a single architecture. This is achieved through task-specific prefixes added to inputs, allowing the model to learn both sub-tasks concurrently. The multi-modal input representation incorporates:

**Text Features**: OCR tokens extracted using Google Cloud Vision API, providing the textual content of slides.

**Layout Features**: Normalized bounding box coordinates embedded to capture spatial relationships and document structure.

**Visual Features**: Appearance features for semantic regions and OCR bounding boxes, extracted using a Faster-RCNN model trained on SlideVQA's annotations.

**Segment Embeddings**: Indicators connecting semantic regions to their corresponding OCR tokens, enabling the model to understand which text belongs to which visual element.

The architecture consists of a Transformer-based encoder that processes each slide sequence independently before concatenating representations for the decoder. The decoder performs two critical functions: generating answers or arithmetic expressions for question answering, and selecting relevant evidence slides. During training, the model minimizes a weighted combination of losses for both tasks.

A particularly innovative aspect is the arithmetic expression generation capability. For numerical questions, instead of directly predicting final numbers, the model generates computational expressions that are then evaluated by an external calculator. This approach encourages the model to learn interpretable reasoning steps and has been shown to significantly improve numerical reasoning performance.

## Experimental Results and Analysis

The experimental evaluation demonstrates both the challenging nature of the SlideVQA task and the effectiveness of the proposed M3D model compared to various baseline approaches.

M3D consistently outperformed all evaluated baselines across multiple metrics. On the main SlideVQA task, M3D achieved a Joint Exact Match (EM) of 28.0% and Joint F1 of 37.3% on the test set, surpassing the next best baseline (LayoutT5) by notable margins. However, when provided with ground-truth evidence slides (M3D_GT), performance improved substantially to 35.4% Joint EM and 44.7% Joint F1, indicating that evidence selection remains a significant bottleneck.

The results reveal several important insights about multi-image document reasoning:

**Multi-modality is Essential**: Ablation studies confirmed that all input modalities—text, layout, and visual features—contribute meaningfully to performance. Removing any component led to performance degradation, with text features being most critical, followed by visual and layout features.

**Generative Approaches Excel**: Generative models like M3D and LayoutT5 significantly outperformed extractive models, attributed to the prevalence of multi-span and non-span answers in SlideVQA (32.4% of total questions).

**Arithmetic Expression Generation Improves Numerical Reasoning**: The strategy of generating arithmetic expressions rather than direct numerical answers led to substantial improvements in numerical question performance, demonstrating the value of teaching models interpretable reasoning processes.

![Performance Analysis](https://paper-assets.alphaxiv.org/figures/2301.04883v1/img-5.jpeg)
*Figure 4: Detailed performance comparison across different question types and reasoning categories, highlighting the persistent gap between model performance and human capabilities, particularly for complex multi-hop and numerical reasoning tasks.*

Despite these achievements, a substantial gap remains between M3D and human performance (Joint EM: 28.0% vs. 88.6%, Joint F1: 37.3% vs. 91.9%). This gap is particularly pronounced for multi-hop reasoning questions, especially those requiring numerical computation, indicating significant room for future research.

## Technical Contributions and Innovations

SlideVQA introduces several technical innovations that advance the state-of-the-art in document understanding:

**Unified Task Formulation**: The joint learning framework for evidence selection and question answering represents a departure from traditional pipeline approaches, enabling end-to-end optimization and potentially better information flow between subtasks.

**Arithmetic Expression Supervision**: The explicit annotation and generation of computational expressions for numerical questions is unique among DocVQA datasets, providing a mechanism for interpretable numerical reasoning that could be applied broadly across quantitative reasoning tasks.

**Comprehensive Multi-modal Integration**: The systematic incorporation of text, layout, and visual features, along with segment embeddings that connect different modalities, provides a template for future multi-modal document understanding systems.

**Scalable Multi-hop Question Generation**: The approach of creating multi-hop questions by editing single-hop questions through bridge entity replacement offers a scalable methodology for generating complex reasoning scenarios, though the authors acknowledge this may produce somewhat artificial questions.

The evidence selection component uses neural generative approaches that outperformed traditional methods like BM25 and zero-shot approaches, suggesting that learned representations are superior to hand-crafted features for identifying relevant document segments.

## Implications and Future Directions

The research has significant implications for both academic research and practical applications. SlideVQA establishes a challenging benchmark that will drive research in multi-modal reasoning, complex question answering, and document understanding systems. The substantial gap between current AI performance and human capabilities indicates ample opportunity for algorithmic innovations.

From a practical perspective, the ability to automatically comprehend slide decks opens possibilities for intelligent document summarization, automated knowledge extraction, and advanced information retrieval systems. These capabilities could transform how professionals interact with large document repositories, enabling rapid analysis of complex presentations and reports.

The authors identify several areas for future work, including addressing scalability challenges for open-domain QA with many input images, developing more natural multi-hop question generation techniques, and improving robustness to diverse visual layouts and semantic categories. The evidence selection bottleneck suggests that hierarchical retrieval mechanisms and more sophisticated attention architectures could yield significant improvements.

The methodology and insights from SlideVQA could also be extended to other multi-page document types, such as research papers, technical manuals, and financial reports, broadening the impact of this research beyond presentation slides.

## Conclusion

SlideVQA represents a meaningful advance in document understanding research by introducing the first large-scale dataset for multi-image Document Visual Question Answering. The comprehensive annotation of slide decks, incorporation of arithmetic expression supervision, and development of the M3D model provide a foundation for future research in complex document reasoning.

While the proposed M3D model demonstrates clear improvements over existing approaches, the substantial performance gap compared to human capabilities underscores the complexity of multi-image document understanding and the need for continued research. The dataset's emphasis on real-world document types, combined with its challenging reasoning requirements, positions SlideVQA as a valuable benchmark for driving progress toward AI systems capable of genuinely understanding complex, multi-modal documents.

The work successfully bridges the gap between single-image document understanding and the multi-page reasoning required for real-world applications, establishing a foundation for future advances in document AI that could significantly impact how intelligent systems process and comprehend information-rich visual documents.