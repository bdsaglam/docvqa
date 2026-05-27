## Problem and Motivation

Document Visual Question Answering (DocVQA) has emerged as an important task combining computer vision and natural language processing to enable automated document understanding. However, current DocVQA research predominantly focuses on single-page documents, creating a significant gap with real-world applications where most business documents span multiple pages. Documents in industries like banking, insurance, legal services, and public administration typically contain multiple pages with interconnected information that cannot be understood in isolation.

![Multi-page document example showing pages with red and blue borders](https://paper-assets.alphaxiv.org/figures/2212.05935v2/img-0.jpeg)

The fundamental challenge lies in the computational limitations of existing Transformer-based models, which suffer from quadratic complexity with input sequence length. When processing multi-page documents that can contain thousands of OCR tokens, current approaches either truncate crucial information or become computationally prohibitive. This limitation has prevented the development of practical DocVQA systems for real-world document workflows.

## Dataset Contribution: MP-DocVQA

The authors address the lack of multi-page DocVQA benchmarks by introducing MP-DocVQA, a large-scale dataset specifically designed for multi-page document understanding. Built upon the existing SingleDocVQA dataset, MP-DocVQA extends single-page contexts by incorporating preceding and succeeding pages from the original UCSF-IDL document sources.

The dataset contains 46,176 questions posed over 47,952 page images from 5,928 multi-page documents, with documents limited to a maximum of 20 pages to focus on the most common real-world scenarios. A critical preprocessing step involved filtering questions that became ambiguous in multi-page contexts, such as questions asking about "the title of the document" when multiple pages could contain title elements.

The resulting dataset exhibits diverse document layouts including forms, tables, and diagrams, with both handwritten and typewritten text. Importantly, 85.95% of the documents are truly multi-page, providing a realistic benchmark for evaluating multi-page DocVQA systems.

## Hierarchical Architecture: Hi-VT5

To overcome the sequence length limitations of existing models, the authors propose Hi-VT5 (Hierarchical Visual T5), a novel hierarchical multimodal Transformer architecture. The key innovation lies in its two-level processing approach that avoids the quadratic complexity of processing entire multi-page documents as single sequences.

### Page-Level Encoding

Each page is processed independently by a T5-based encoder that combines:
- **Textual representation**: OCR tokens with 2D spatial embeddings encoding bounding box coordinates
- **Visual representation**: Patch embeddings from a pre-trained Document Image Transformer (DiT)
- **Question tokens**: The encoded natural language question

Crucially, Hi-VT5 introduces M learnable `[PAGE]` tokens (set to M=10) that act as summary representations for each page. During encoding, these tokens are designed to embed the most relevant information from each page, conditioned on the input question.

### Document-Level Decoding

After independent page encoding, the contextualized `[PAGE]` tokens from all pages are concatenated to form a compressed representation of the entire document. This hierarchical approach allows the T5 decoder to generate answers from a manageable sequence length regardless of the original document length.

The model architecture can be expressed as:

$$
h_i^{page} = \text{Encoder}(\text{OCR}_i \oplus \text{Visual}_i \oplus \text{Question} \oplus [\text{PAGE}]_M)
$$

$$
\text{Answer} = \text{Decoder}(\text{concat}(h_1^{page}[[\text{PAGE}]], h_2^{page}[[\text{PAGE}]], ..., h_N^{page}[[\text{PAGE}]]))
$$

where $\oplus$ denotes concatenation and $h_i^{page}[[\text{PAGE}]]$ represents the contextualized `[PAGE]` tokens from page $i$.

## Training Strategy and Answer Page Identification

Hi-VT5 employs a hierarchical layout-aware pre-training strategy that extends T5's denoising objective by incorporating spatial information about masked tokens. This pre-training helps align the multimodal representations and trains the `[PAGE]` tokens to effectively summarize page content.

Due to memory constraints, initial training uses shortened documents (typically two pages), followed by fine-tuning on full-length documents with frozen encoder weights. This staged approach allows the model to learn hierarchical representations efficiently.

Additionally, Hi-VT5 includes an answer page identification module that predicts which page contains the answer. This component serves dual purposes: providing explainability by indicating where information was found, and distinguishing between answers derived from document content versus those potentially generated from learned biases.

## Experimental Results and Performance Analysis

The evaluation on MP-DocVQA reveals significant insights about multi-page DocVQA challenges. Traditional models show substantial performance degradation when processing multiple pages, with performance particularly declining when answers appear on later pages in the document.

Hi-VT5 significantly outperforms all baselines in realistic multi-page settings, achieving 48.28% accuracy and 0.6201 ANLS (Average Normalized Levenshtein Similarity). Most importantly, Hi-VT5 demonstrates remarkable robustness across different answer page positions, maintaining consistent performance regardless of where the answer appears in the document.

The answer page identification module achieves 79.23% accuracy, providing valuable explainability. Analysis shows the model can correctly identify relevant pages even when failing to provide exact answers, and conversely, sometimes provides correct answers even with incorrect page predictions, indicating reliance on learned patterns.

Ablation studies confirm that hierarchical pre-training provides the most significant performance boost, while visual features and 2D position embeddings contribute positively to the multimodal understanding task.

## Significance and Impact

This work represents a crucial step toward practical DocVQA systems by addressing the multi-page challenge that has limited real-world applicability. The contributions extend beyond the specific technical achievements:

**Research Impact**: MP-DocVQA provides the research community with a much-needed benchmark for multi-page document understanding, while Hi-VT5's hierarchical approach offers a scalable solution for long document processing that could influence broader sequence modeling research.

**Practical Applications**: The ability to process multi-page documents efficiently opens possibilities for automation in document-intensive industries, potentially reducing manual effort in tasks like compliance checking, auditing, and information extraction.

**Architectural Innovation**: The hierarchical design with learnable summary tokens provides a general framework for handling long sequences while maintaining computational efficiency, with potential applications beyond document understanding.

**Explainability**: The integrated answer page identification enhances system transparency and trustworthiness, crucial for deployment in regulated industries where understanding model reasoning is essential.

By bridging the gap between academic DocVQA capabilities and real-world multi-page document complexity, this research establishes a foundation for more practical and deployable document intelligence systems.