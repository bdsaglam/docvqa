## Introduction

Visual Document Understanding (VDU) has traditionally followed a two-stage pipeline approach: first, an Optical Character Recognition (OCR) engine extracts text and layout information from document images, then a downstream model processes this structured data to extract relevant information. While this approach has achieved remarkable success, it suffers from computational overhead, limited flexibility across languages and document types, and error propagation from OCR preprocessing steps.

![OCR-free Document Understanding Pipeline](https://paper-assets.alphaxiv.org/figures/2111.15664v5/img-0.jpeg)
*Figure 1: Comparison of traditional OCR-dependent approaches versus the proposed OCR-free Donut model. The figure illustrates how Donut directly processes document images to generate structured JSON output, bypassing the need for separate OCR engines.*

The paper "OCR-free Document Understanding Transformer" introduces Donut, a paradigm-shifting approach that eliminates OCR dependency entirely. Instead of relying on separate text extraction engines, Donut employs an end-to-end Transformer architecture that directly processes document images and generates structured information in JSON format. This represents the first comprehensive attempt to achieve competitive VDU performance using a purely OCR-free, Transformer-only approach.

## Architecture and Methodology

Donut employs a streamlined encoder-decoder Transformer architecture designed specifically for document understanding tasks. The model consists of two primary components working in tandem to transform raw document images into structured outputs.

The **visual encoder** utilizes a Swin Transformer backbone, specifically Swin-B, which processes input document images by dividing them into non-overlapping patches and applying hierarchical attention mechanisms. This choice was motivated by preliminary studies showing superior performance of Swin Transformers on document parsing tasks compared to traditional CNN backbones like ResNet or other vision transformers.

The **textual decoder** is based on BART architecture, initialized with weights from a pre-trained multilingual BART model. During training, the decoder uses teacher forcing to learn the mapping from visual embeddings to structured text sequences. At inference time, it generates tokens autoregressively, conditioned on both the encoded visual features and previously generated tokens.

![Pipeline Architecture](https://paper-assets.alphaxiv.org/figures/2111.15664v5/img-1.jpeg)
*Figure 2: Comparison between traditional AS-IS OCR-dependent pipeline and the proposed Donut approach, along with system benchmarks showing Donut's superior memory efficiency and inference speed.*

The model's input-output paradigm represents a significant departure from conventional approaches. Instead of processing OCR-extracted text and coordinates, Donut receives raw document images and task-specific prompts (similar to GPT-3's prompt-based generation). The output is designed as a JSON-formatted token sequence, enabling representation of complex hierarchical document structures while maintaining parsing simplicity.

## Pre-training Strategy

Donut's pre-training methodology centers on teaching the model to "read" text directly from images through a pseudo-OCR objective. The training process minimizes cross-entropy loss for next-token prediction, where the model learns to generate text tokens in reading order (top-left to bottom-right) while conditioning on both the input image and preceding text context.

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t | x, y_{\x3Ct})
$$

where $x$ represents the input document image, $y_t$ is the target token at position $t$, and $y_{\x3Ct}$ denotes all previously generated tokens.

The pre-training leverages two distinct data sources. First, the IIT-CDIP dataset provides 11 million scanned English document images with pseudo-text labels generated using commercial OCR APIs. Second, to address multilingual capabilities and reduce dependence on real-world datasets, the authors developed SynthDoG (Synthetic Document Generator).

SynthDoG creates realistic synthetic documents by sampling background textures from ImageNet, extracting text from Wikipedia in multiple languages (Chinese, Japanese, Korean, English), and applying sophisticated rendering techniques including elastic distortion, Gaussian noise, perspective transformations, shadow effects, and compression artifacts. This synthetic approach generated 0.5 million samples per language, enabling robust multilingual pre-training without extensive real-world data collection.

## Training and Fine-tuning Process

![Training Process](https://paper-assets.alphaxiv.org/figures/2111.15664v5/img-13.jpeg)
*Figure 3: Detailed illustration of Donut's training and inference process, showing how the model learns to generate structured JSON outputs through token classification and conversion.*

The fine-tuning phase adapts the pre-trained model for specific VDU tasks by reformulating all downstream applications as JSON generation problems. Document classification, information extraction, and visual question answering are unified under this framework, with task-specific prompts guiding the decoder toward appropriate output formats.

For example, a document classification task uses prompts like `\x3Cclass>\x3C/class>\x3Cclassification>` to trigger classification behavior, while information extraction employs prompts such as `\x3Cparsing>` to initiate structured field extraction. Visual question answering incorporates the question directly into the prompt structure: `\x3Cvqa>\x3Cquestion>what is the price?\x3C/question>\x3Canswer>`.

![Task Examples](https://paper-assets.alphaxiv.org/figures/2111.15664v5/img-2.jpeg)
*Figure 4: Examples of Donut's versatile application across different document understanding tasks, showing unified JSON output format for classification, information extraction, and visual question answering.*

## Experimental Results

Donut demonstrates compelling performance across diverse VDU benchmarks, consistently matching or exceeding OCR-dependent baselines while offering significant practical advantages.

### Document Classification
On the RVL-CDIP dataset, Donut achieved 95.30% accuracy, surpassing LayoutLMv2's 95.25% performance while using fewer parameters (143M vs. 200M+OCR) and delivering twice the inference speed (752ms vs. 1489ms). This result establishes Donut's capability to perform high-level document understanding tasks without sacrificing efficiency.

### Information Extraction
Across four datasets (CORD, Ticket, Business Card, Receipt), Donut consistently delivered superior F1 scores and Tree Edit Distance-based accuracies compared to OCR-dependent models including BERT, LayoutLM, LayoutLMv2, BROS, SPADE, and WYVERN. The model particularly excelled at capturing complex nested structures and relationships within documents.

![Information Extraction Results](https://paper-assets.alphaxiv.org/figures/2111.15664v5/img-10.jpeg)
*Figure 5: Examples of Donut's information extraction capabilities on ticket documents, showing accurate extraction of structured information with high Tree Edit Distance (TED) accuracy scores.*

### Visual Question Answering
On DocVQA, Donut achieved 67.5% ANLS (Average Normalized Levenshtein Similarity), demonstrating competitive performance against specialized OCR-dependent models. Notably, Donut showed superior robustness on handwritten documents, achieving 72.1% ANLS compared to LayoutLMv2-Large-QG's 67.3%, highlighting its ability to overcome OCR limitations on challenging text.

![Receipt Processing](https://paper-assets.alphaxiv.org/figures/2111.15664v5/img-11.jpeg)
*Figure 6: Donut's performance on receipt understanding tasks, demonstrating accurate extraction of menu items, prices, and hierarchical structure information.*

## Analysis and Ablation Studies

The research includes comprehensive ablation studies examining key architectural and training decisions. The visual encoder comparison confirmed Swin Transformer and EfficientNetV2 as superior backbones for VDU tasks, with Swin Transformer selected for its scalability advantages.

Input resolution analysis revealed significant performance improvements with larger image sizes, particularly crucial for tasks involving small text like DocVQA. However, this comes with increased computational costs, highlighting the trade-off between accuracy and efficiency.

![Ablation Studies](https://paper-assets.alphaxiv.org/figures/2111.15664v5/img-6.jpeg)
*Figure 7: Comprehensive ablation study results showing the impact of different pre-training strategies, backbone architectures, and input resolutions on model performance.*

The pre-training strategy analysis demonstrated that synthetic data alone suffices for many information extraction tasks, while real images prove important for complex reasoning tasks like DocVQA. This finding validates the synthetic data generation approach while highlighting the complementary value of real-world document diversity.

Cross-attention visualizations revealed that Donut's decoder implicitly learns to attend to relevant text regions in images, even without explicit localization supervision. This emergent behavior provides interpretability and confirms the model's ability to perform spatial reasoning over document layouts.

## Significance and Impact

Donut represents a fundamental shift in Visual Document Understanding methodology, establishing the viability of OCR-free approaches for complex document processing tasks. The work's significance extends across multiple dimensions:

**Technical Innovation**: By demonstrating that end-to-end Transformer architectures can match OCR-dependent systems, Donut challenges established paradigms and opens new research directions toward more integrated document intelligence systems.

**Practical Advantages**: The elimination of separate OCR engines reduces computational overhead, simplifies system architecture, and removes error propagation bottlenecks. The 2x inference speed improvement over comparable OCR-dependent models addresses real-world deployment constraints.

**Generalization Capabilities**: Donut's architecture inherently avoids OCR-specific limitations on new languages, scripts, or document types. The synthetic data generation framework enables rapid adaptation to diverse domains without extensive data collection efforts.

**Research Impact**: The open-source release of code, pre-trained models, and synthetic data generation tools accelerates community research and development in OCR-free document understanding, fostering innovation and practical adoption.

The work establishes competitive baselines for future OCR-free VDU research while demonstrating immediate applicability to industrial document processing workflows. Its robust performance in low-resource scenarios (outperforming LayoutLMv2 with only 10% training data) particularly enhances its value for real-world applications with limited annotated data.

Donut's success in handling handwritten text and diverse document layouts showcases its potential to address long-standing OCR limitations, paving the way for more robust and flexible document intelligence systems across languages and domains.