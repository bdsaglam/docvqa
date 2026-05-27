## Understanding Scene Text in Visual Questions

![Example questions and answers from the ST-VQA dataset showing various scenarios where reading text is essential for answering questions about images](https://paper-assets.alphaxiv.org/figures/1905.13648v2/img-0.jpeg)

Visual Question Answering (VQA) systems have made significant progress in recent years, but they face a fundamental limitation: most existing models struggle to incorporate textual information present in images. This paper introduces Scene Text Visual Question Answering (ST-VQA), a specialized dataset and framework that addresses this critical gap by creating questions that can only be answered by reading and understanding text within images.

The core insight driving this research is that text appears in approximately 50% of images in large-scale computer vision datasets and carries crucial semantic information that cannot be inferred from visual cues alone. Despite this prevalence, existing VQA datasets like VQA 2.0 contain less than 1% of questions requiring text understanding, leaving a significant capability gap in current models.

## Dataset Construction and Methodology

The ST-VQA dataset was constructed through a systematic process designed to ensure high quality and diversity. The researchers collected 23,038 images from six different public datasets including ICDAR 2013/2015, ImageNet, VizWiz, IIIT Scene Text Retrieval, Visual Genome, and COCO-Text. This multi-source approach was specifically chosen to mitigate dataset bias and provide diverse visual contexts.

Images were selected using an automated process that ensured each contained at least two text instances, making the task more challenging than simple binary choices. The question-answer collection process involved Amazon Mechanical Turk workers who were instructed to create questions that could be unambiguously answered using only the text present in the image. A verification step involving separate workers helped ensure quality, with questions requiring alignment between generation and verification phases.

The final dataset comprises 31,791 question-answer pairs across the selected images, with a standard training-testing split. Importantly, the dataset design ensures that all questions necessitate reading scene text - unlike concurrent work such as TextVQA where only 61% of answers are grounded in OCR tokens.

## Task Formulation and Evaluation

The paper defines three tasks of increasing difficulty to simulate different levels of available contextual information:

**Strongly Contextualized Task**: Models receive a dictionary of 100 words per image, including ground truth answers plus distractors from OCR outputs and dynamic lexicon generation.

**Weakly Contextualized Task**: A single dictionary of 30,000 words covers the entire dataset, representing a more general but constrained vocabulary scenario.

**Open Vocabulary Task**: No external dictionary is provided, requiring models to generate answers freely in an unconstrained setting.

A key methodological contribution is the introduction of Average Normalized Levenshtein Similarity (ANLS) as the primary evaluation metric. Traditional exact-match accuracy proves too strict for text-based answers, as minor OCR errors would result in zero scores even when reasoning is correct. ANLS provides soft penalization for minor discrepancies:

$$
\text{ANLS} = \frac{1}{|Q|} \sum_{q \in Q} \max_{a \in A_q} s(a, o_q)
$$

where $s(a, o) = \max(0, 1 - \text{NL}(a, o))$ if $\text{NL}(a, o) \x3C 0.5$, otherwise $0$. The threshold ensures answers with more than half incorrect characters receive zero credit while providing partial credit for closer matches.

## Baseline Performance and Key Findings

The researchers evaluated several baseline approaches to establish initial performance benchmarks. Standard VQA models (SAAA and SAN) that do not explicitly incorporate scene text performed poorly, achieving ANLS scores between 0.084-0.102 and accuracy between 6.13%-7.78%. This quantitatively confirms that existing VQA architectures are largely incapable of text-dependent reasoning.

Interestingly, simple heuristic methods that inherently rely on scene text often outperformed sophisticated VQA models. The Scene Text Retrieval (STR) baseline achieved 0.171 ANLS on Task 1, while Scene Image OCR reached 0.132-0.145 ANLS - both surpassing standard VQA models despite ignoring the natural language questions entirely.

![Comparison of question and answer distributions between ST-VQA and TextVQA datasets](https://paper-assets.alphaxiv.org/figures/1905.13648v2/img-1.jpeg)

When textual features were fused with VQA models using PHOC (Pyramidal Histogram of Characters) representations, consistent but modest improvements emerged. For example, SAN+STR achieved 0.136 ANLS compared to SAN's 0.102, demonstrating the value of multimodal integration while highlighting the need for more sophisticated fusion mechanisms.

## Dataset Analysis and Characteristics

![Sunburst visualization showing the distribution of question types and patterns in the ST-VQA dataset](https://paper-assets.alphaxiv.org/figures/1905.13648v2/img-2.jpeg)

The dataset exhibits several important characteristics that distinguish it from existing VQA benchmarks. Questions primarily begin with "What," "Where," "Which," "How," and "Who," with a large percentage being "What" questions as expected for text-based inquiries. Crucially, many questions require prior world knowledge (e.g., "what is the brand," "what is the website"), indicating that simple text recognition is insufficient - contextual understanding is essential.

Unlike VQA v1, which suffered from strong language priors due to uneven answer distributions, ST-VQA exhibits a more uniform distribution for most answer types. This design choice effectively reduces the risk of models learning simple linguistic shortcuts rather than engaging in genuine visual and textual reasoning.

![Visualization showing the diversity of answers across different question categories in the dataset](https://paper-assets.alphaxiv.org/figures/1905.13648v2/img-3.jpeg)

## Performance Analysis Across Question Types

![Detailed performance breakdown of different baseline methods across various question categories](https://paper-assets.alphaxiv.org/figures/1905.13648v2/img-4.jpeg)

The analysis reveals significant performance variations across different question types. Models consistently struggle with specific categories such as license plates, "who" questions, and numerical queries. This difficulty stems partly from the classification-based nature of current VQA models, which cannot handle out-of-vocabulary answers effectively.

The paper suggests that morphological embeddings like PHOC are important for handling words not present in pre-trained vocabularies, such as numbers, prices, or proper names that lack semantic representations in standard word embeddings.

## Qualitative Results and Model Comparison

![Examples showing how different baseline methods perform on various ST-VQA questions, highlighting their strengths and limitations](https://paper-assets.alphaxiv.org/figures/1905.13648v2/img-5.jpeg)

Qualitative analysis of model outputs reveals distinct patterns in how different approaches handle text-dependent questions. Scene text-aware methods often provide more accurate answers even when they don't consider the question context, while VQA models enhanced with textual features show improved reasoning but still struggle with complex text understanding scenarios.

## Implications and Future Directions

This work fundamentally challenges the current VQA paradigm by demonstrating that comprehensive scene understanding requires explicit integration of textual information. The consistently low performance of even enhanced baseline models indicates significant room for improvement and points toward several critical research directions.

The results suggest that future VQA systems should move beyond classification-based approaches toward generative models capable of producing arbitrary string outputs. Additionally, more sophisticated multimodal fusion mechanisms are needed to effectively combine visual, linguistic, and textual information sources.

The ST-VQA dataset and ANLS metric provide the community with essential tools for advancing text-aware visual reasoning. By establishing clear benchmarks and evaluation protocols, this work creates a foundation for developing more capable AI systems that can truly understand complex visual scenes in their entirety, including the rich semantic information conveyed through text.