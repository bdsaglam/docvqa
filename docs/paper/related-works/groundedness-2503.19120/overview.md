## The Challenge of Evaluation in Document Visual Question Answering

Visual Question Answering (VQA) on multimodal documents requires models to perform complex reasoning across text, layout, and visual elements. As generative multimodal models advance, their performance on standard benchmarks has improved significantly, sometimes rivaling human accuracy. However, a critical gap remains in how these models are evaluated. Traditional metrics primarily focus on surface-level similarity—checking if the predicted string of text matches the ground truth. This approach often fails to account for "groundedness," or whether the model's answer is actually derived from the document rather than being a plausible hallucination.

In real-world applications, particularly in domains like finance or legal services, the provenance of an answer is as important as the answer itself. If a model provides a correct number but cannot locate it on the page, the reliability of the output is compromised. Current benchmarks using metrics like Normalized Levenshtein Similarity (NLS) or Average Normalized Levenshtein Similarity (ANLS) treat hallucinated answers and well-grounded answers similarly if they share enough characters. This limitation masks the true reasoning capabilities of models and their readiness for high-stakes deployment.

![Model rank shifts across different answer types](https://paper-assets.alphaxiv.org/figures-normalized/figures/2503.19120v1/info_reranking.png)
*Figure 1: Comparison of model rankings across different semantic answer types, illustrating how rankings shift when accounting for groundedness and semantic alignment.*

## Addressing the Groundedness Gap with SMuDGE

To address these limitations, researchers have introduced a new evaluation framework called Semantics and MUltimodal Document Grounded Evaluation (SMuDGE). Unlike previous metrics, SMuDGE explicitly incorporates the concept of multimodal groundedness and semantic type awareness into its scoring mechanism. The goal is to ensure that a model is rewarded not just for producing the right characters, but for finding the correct information in the document and respecting the data type of the expected answer.

The motivation for SMuDGE is two-fold. First, existing metrics often ignore whether a model's output aligns with the expected semantic category. For instance, if the ground truth is a specific number, a model should be penalized more heavily for a minor digit error than for a minor spelling error in a textual response, as the former can completely change the factual meaning. Second, there is a distinct lack of verification regarding the spatial provenance of the answer. SMuDGE attempts to bridge this by verifying if the predicted answer can be located within the source document's coordinates.

## The Technical Framework of SMuDGE

The SMuDGE score for an individual question $i$ is a composite of two primary components: a type-aware surface similarity score $m_i$ and a multimodal grounding score $g_i$. These are combined using a weighting parameter $\alpha$, which allows users to adjust the relative importance of textual match versus spatial grounding:

$$s_i = \alpha \cdot m_i + (1 - \alpha) \cdot g_i$$

Where:
- $s_i$ is the final SMuDGE score.
- $m_i$ represents the type-aware similarity.
- $g_i$ represents the multimodal grounding.
- $\alpha$ is a parameter between 0 and 1.

### Multimodal Grounding ($g_i$)
The grounding score $g_i$ measures the spatial distance between the predicted answer and the ground truth. The process begins by locating both the ground truth $t_i$ and the predicted answer $a_i$ within the document using OCR-extracted word coordinates. If the predicted answer cannot be found in the document with sufficient similarity, it is considered ungrounded.

If both are located, a distance $d_i$ is calculated between their respective bounding box centroids $b_{t_i}$ and $b_{a_i}$. This distance is normalized by the page dimensions. To convert this distance into a score, an exponential decay function is used:

$$g_i = e^{-\frac{d_i}{1 - d_i}}$$

This function ensures that scores are high when the predicted answer is very close to the ground truth location and decreases rapidly as the distance increases.

### Type-Aware Surface Similarity ($m_i$)
The similarity score $m_i$ is tailored to the data type of the ground truth:
1.  **Textual Answers:** For standard text, the framework uses traditional NLS.
2.  **Numeric Answers:** For numbers, the framework requires an exact match (with some flexibility for scaling, like "100" vs "100.0"). This prevents high scores for numbers that are "close" in string similarity but represent different values.
3.  **Hybrid Answers:** For answers containing both text and numbers, the framework calculates separate scores for the numeric $num_{t_i}$ and textual $str_{t_i}$ parts. These are then combined using a weighted harmonic mean, where the numeric accuracy is weighted significantly higher (often 10 times more) than the textual accuracy.

## Experimental Impact on Benchmarks

The researchers applied SMuDGE to re-evaluate top-performing models on major Document VQA benchmarks, including DocVQA, InfographicVQA, and MP-DocVQA. The results demonstrated that accounting for groundedness and semantic type significantly alters the competitive landscape.

On the DocVQA leaderboard, while the very top models generally maintained their lead, nearly all other models experienced rank shifts. The volatility was most pronounced in categories involving handwritten text or complex "Other" question types. Interestingly, human performance, while superior in textual and hybrid answers, was slightly lower in numeric categories because humans tend to rephrase or scale numbers (e.g., writing "1.2k" instead of "1200"), which SMuDGE's strict numeric matching penalizes.

![Kendall's tau correlation across alpha values](https://paper-assets.alphaxiv.org/figures-normalized/figures/2503.19120v1/calibration.png)
*Figure 2: The relationship between the $\alpha$ parameter and the correlation between SMuDGE scores and model calibration error.*

The parameter $\alpha$ plays a crucial role in these rankings. When $\alpha = 0$, the metric focuses entirely on whether the model found the right location. When $\alpha = 1$, it focuses entirely on the text match. The analysis found that at $\alpha = 0.25$, the metric provides the best balance, minimizing the correlation with Expected Calibration Error (ECE) and providing the most stable assessment of model quality.

## Model Calibration and Robustness

A significant finding of this work is the relationship between SMuDGE and model calibration. Calibration refers to whether a model's predicted confidence accurately reflects its likelihood of being correct. By analyzing the DUDE dataset, the researchers found that models with higher SMuDGE scores (at lower $\alpha$ values) tended to have lower calibration errors. This suggests that models that are better grounded—meaning they can locate the information they are citing—are also more reliable in their internal confidence estimates.

Furthermore, SMuDGE appears to be a better indicator of model robustness. Robustness is defined as a model's ability to maintain consistent performance across different datasets and question subsets. SMuDGE-based rankings showed a stronger correlation between score stability and rank stability compared to traditional ANLS rankings. This implies that SMuDGE is more effective at identifying models that truly understand the multimodal document structure rather than those that have simply learned to exploit linguistic patterns in the text.

## Aligning with Human Preferences

To validate that SMuDGE is a more meaningful metric, a human judgment study was conducted. Researchers compared instances where traditional NLS and SMuDGE disagreed on which of two models performed better. Human annotators were asked to choose the superior answer based on document evidence.

![Human judgment study results](https://paper-assets.alphaxiv.org/figures-normalized/figures/2503.19120v1/userstudy.png)
*Figure 3: Results of the human preference study, showing that SMuDGE scores align with human judgment in the vast majority of cases across different datasets.*

The results were compelling: human annotators agreed with the SMuDGE ranking in over 80% of the cases across DocVQA and MP-DocVQA. This high level of agreement, paired with a high inter-annotator reliability ($\kappa = 0.82$), suggests that SMuDGE successfully captures the qualitative aspects of "correctness" that humans value—namely, that an answer should be both factually accurate and derived from the correct part of the source material.

## Conclusion and Significance

The introduction of SMuDGE represents a shift in how Document VQA systems are assessed. By moving beyond surface-level string matching, this framework forces a consideration of two critical factors:
- **Multimodal Provenance:** Can the model show where the information came from?
- **Semantic Integrity:** Does the answer respect the specific requirements of its data type (e.g., numeric precision)?

This approach is particularly valuable for enterprise-grade AI, where the cost of a hallucinated number is high and the need for data lineage is paramount. By providing a configurable parameter $\alpha$, SMuDGE allows researchers to tailor their evaluation to specific needs—whether they prioritize literal string accuracy or spatial grounding. Ultimately, this work suggests that making groundedness count is essential for developing trustworthy, robust, and truly multimodal document intelligence.

![Impact of alpha on Handwritten question rankings](https://paper-assets.alphaxiv.org/figures-normalized/figures/2503.19120v1/Handwritten.png)
*Figure 4: Analysis of rank volatility for "Handwritten" question types as the importance of grounding versus similarity is adjusted.*