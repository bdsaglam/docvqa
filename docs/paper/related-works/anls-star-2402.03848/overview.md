## Introduction to Generative Document Processing

The landscape of document processing has evolved from discriminative token-labeling models to more versatile generative large language models (GLLMs). While traditional models like LayoutLMv3 were effective at identifying specific fields within predefined classes, they often required extensive fine-tuning and were limited to simple extraction tasks. In contrast, GLLMs like the GPT, Claude, and Gemini series exhibit zero-shot capabilities, allowing them to process documents without task-specific training data. These models can also perform complex text transformations, such as reformating dates or translating content, which go beyond the capabilities of older architectures.

However, this shift introduces a significant evaluation challenge. Standard metrics like the F1 score, which rely on binary true/false classifications, cannot effectively assess the nuances of generated text. A model might produce a minor typo or a slightly different formatting that is still semantically correct, or it might hallucinate information for an unanswerable question. The most widely used metric for these models, Average Normalized Levenshtein Similarity (ANLS), was designed primarily for string-to-string comparisons and simple lists. As GLLMs are increasingly tasked with generating structured outputs—such as JSON dictionaries or nested lists—the standard ANLS metric becomes insufficient.

To address these limitations, researchers have introduced ANLS*, a universal metric specifically designed for generative document processing. ANLS* treats both the ground truth and the model's prediction as tree structures, allowing for a systematic comparison of complex, nested data types.

![Tree representation of a ground truth structure in ANLS*](https://paper-assets.alphaxiv.org/figures-normalized/figures/2402.03848v10/g.png)

## The Limitations of Traditional Metrics

Evaluating GLLMs requires a metric that is both flexible and rigorous. The original ANLS metric was created for the DocVQA task to handle Optical Character Recognition (OCR) errors by using normalized Levenshtein distance. While this worked for single-string answers, it struggled with more complex scenarios.

1.  **Structural Complexity**: Modern document processing often involves extracting hierarchical information, such as line items from an invoice or nested attributes from a contract. Comparing a JSON-formatted prediction to a ground truth dictionary using string-based ANLS is brittle, as minor formatting changes (like whitespace or key order) can lead to artificially low scores.
2.  **Hallucinations**: GLLMs are prone to generating answers even when the information is missing from the document. Previous metrics did not always provide a consistent way to penalize these "hallucinations" while rewarding the model for correctly identifying unanswerable questions (predicting `None`).
3.  **Semantic Grouping**: Tasks often require "one-of" semantics (e.g., a date can be formatted as "31.12.2023" or "31-Dec-2023") or "all-of" semantics (e.g., a list of all products in a shipment). Existing metrics lacked a unified framework to handle these different requirements simultaneously.

ANLS* is designed as a "drop-in replacement" that extends the standard ANLS to accommodate these complexities while maintaining backward compatibility with simpler tasks.

## The ANLS* Metric Design

The core philosophy of ANLS* is recursive decomposition. By representing outputs as trees, the metric can evaluate the correctness of each node (leaf or branch) independently and then aggregate the results. This approach allows for a granular assessment of both content and structure.

![Example of a matching prediction tree structure](https://paper-assets.alphaxiv.org/figures-normalized/figures/2402.03848v10/p_correct.png)

The metric supports several fundamental data types:
*   **String**: Evaluated using normalized Levenshtein similarity, allowing for minor OCR errors.
*   **None**: Represents unanswerable questions.
*   **Tuple**: Implements "one-of" semantics, where a prediction is compared against several valid alternatives, and the best score is kept.
*   **List**: Implements "all-of" semantics. The metric uses the Hungarian matching algorithm to pair predicted items with ground truth items optimally, penalizing both missing and extra (hallucinated) elements.
*   **Dict**: Handles key-value pairs. It ensures that the model not only finds the correct value but also associates it with the correct key.

## Formal Definition and Mathematical Framework

The ANLS* score between a ground truth $g$ and a prediction $p$ is defined as the ratio of a similarity score $s(g, p)$ to an effective length $l(g, p)$:

$$ANLS^*(g, p) = \frac{s(g, p)}{l(g, p)}$$

The similarity score $s(g, p)$ is calculated recursively based on the data types involved. If the types of $g$ and $p$ do not match, the score is zero, which encourages models to adhere to the required output format. The formal definition for $s(g, p)$ is:

$$s(g, p) = \begin{cases} 1.0 & \text{if } g, p = \text{None} \\ 1.0 - \frac{\text{Levenshtein}(g, p)}{\max(|g|, |p|)} & \text{if } g, p \in \text{String and similarity} \geq \tau \\ \max_i ANLS^*(g_i, p) & \text{if } g \in \text{Tuple} \\ \sum_{(g_i, p_i) \in \psi} s(g_i, p_i) & \text{if } g, p \in \text{List} \\ \sum_{k \in \text{keys}(g) \cap \text{keys}(p)} s(g_k, p_k) & \text{if } g, p \in \text{Dict} \\ 0.0 & \text{otherwise} \end{cases}$$

Here, $\tau$ represents a similarity threshold (typically 0.5) to filter out noise, and $\psi$ represents the optimal matching found by the Hungarian algorithm for list comparisons. 

To normalize this score, the length $l(g, p)$ accounts for mismatches, such as missing or hallucinated fields:

$$l(g, p) = \begin{cases} 1 & \text{if } g, p \in \text{None, String} \\ l(g_{\text{best}}, p) & \text{if } g \in \text{Tuple} \\ \sum_{(g_i, p_i) \in \psi} l(g_i, p_i) + \sum_{g_u} l_t(g_u) + \sum_{p_u} l_t(p_u) & \text{if } g, p \in \text{List} \\ \sum_{k \in \text{keys}(g) \cap \text{keys}(p)} l(g_k, p_k) + \sum_{k \in \text{keys}(g) \setminus \text{keys}(p)} l_t(g_k) + \sum_{k \in \text{keys}(p) \setminus \text{keys}(g)} l_t(p_k) & \text{if } g, p \in \text{Dict} \\ \max(l_t(g), l_t(p)) & \text{otherwise} \end{cases}$$

The auxiliary function $l_t(x)$ calculates the total number of elements within a structure $x$, ensuring that complex nested hallucinations are penalized more heavily than simple ones:

$$l_t(x) = \begin{cases} 1 & \text{if } x \in \text{None, String} \\ \max_i l_t(x_i) & \text{if } x \in \text{Tuple} \\ \sum_i l_t(x_i) & \text{if } x \in \text{List} \\ \sum_k l_t(x_k) & \text{if } x \in \text{Dict} \end{cases}$$

![Example of a prediction tree with hallucinated fields and errors](https://paper-assets.alphaxiv.org/figures-normalized/figures/2402.03848v10/p_wrong.png)

## Benchmarking Generative Models

To demonstrate the utility of ANLS*, the researchers conducted an extensive benchmark involving over 20 GLLMs and multiple prompting strategies across seven document processing datasets. The datasets included Visual Question Answering (VQA) tasks like DocVQA and Information Extraction (IE) tasks like SROIE (receipts) and Kleister Charity (reports).

### Prompting Strategies
The evaluation compared three distinct ways of presenting document information to text-only GLLMs:
1.  **Simple**: The OCR text is provided as a direct sequence.
2.  **LATIN**: A layout-aware instruction method that encodes the spatial coordinates of text segments.
3.  **SFT (Proposed Approach)**: An advanced document representation technique developed by the researchers to better capture the two-dimensional nature of documents.

### Model Selection
The study included a wide array of state-of-the-art models, including:
*   **OpenAI**: GPT-4, GPT-4o, and GPT-4.5.
*   **Anthropic**: Claude 3 and 3.7.
*   **Google**: Gemini 1.5 and 2.5 series.
*   **Open Source**: Mistral-large and Llama-3.1-405B.

The benchmark also included vision-capable models (e.g., GPT-4-vision) that process document images directly, bypassing the need for an external OCR step.

## Insights from the Benchmarking Results

The results provide a comprehensive view of the current state of generative document processing. 

### Model Performance
The latest iterations of high-end models, specifically Claude-3.7 and Gemini-2.5-pro, emerged as the top performers across most datasets. These models consistently achieved higher ANLS* scores than their predecessors, indicating steady progress in the reasoning and extraction capabilities of GLLMs. Interestingly, the benchmark revealed that even without native vision, text-based GLLMs using advanced prompting could often outperform vision-native models on complex extraction tasks.

### The Power of Prompting
The proposed SFT prompting method achieved the highest ANLS* scores on 5 out of 7 datasets. This suggests that the way document layout is represented to a text-based LLM is just as critical as the model's underlying architecture. For documents with complex two-dimensional structures (like invoices and forms), layout-aware representations significantly outperformed simple text sequences. 

### Comparison with Specialized Models
The research compared these general-purpose GLLMs against specialized document models like DocLLM. While DocLLM is specifically trained for document tasks, it was outperformed by much larger general-purpose models like Claude-3.7 when combined with effective prompting. This highlights the competitive zero-shot performance of the latest GLLMs in the document processing domain.

### Handling Structured Output
ANLS* proved essential for evaluating information extraction tasks where models were required to output JSON. The metric correctly identified instances where models failed to adhere to the requested schema or hallucinated additional fields, providing a more reliable measure of performance than string-based metrics could offer.

## Future Impact and Standardization

ANLS* addresses a critical gap in the evaluation of generative AI. By providing a unified, recursive framework that handles strings, lists, dictionaries, and unanswerable questions, it offers a more nuanced understanding of model capabilities. The researchers have made their evaluation scripts publicly available, encouraging the community to adopt ANLS* as a standard for document processing tasks.

As generative models continue to move toward processing more complex, multi-page, and structured documents, the need for robust metrics will only grow. ANLS* provides the necessary mathematical foundation to ensure that model improvements are measured accurately, paving the way for more reliable and capable document AI systems.