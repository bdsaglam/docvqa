## Introduction: From Black-Box Predictions to Auditable Reasoning

The deployment of Artificial Intelligence (AI) in clinical radiology has reached a critical juncture. While Vision-Language Models (VLMs) have demonstrated an ability to detect lesions and generate diagnostic reports, their widespread adoption is often hindered by their "black-box" nature. In most current systems, an AI model receives a medical image and produces a diagnosis in a single inference pass. This single-pass approach provides no insight into the model's internal reasoning, making it difficult for clinicians to verify the findings or for healthcare organizations to comply with emerging regulations like the EU AI Act and the NIST AI Risk Management Framework. These regulations emphasize that high-risk AI systems must be transparent, auditable, and allow for meaningful human oversight.

![RVLM Pipeline Overview](https://paper-assets.alphaxiv.org/figures-normalized/figures/2603.24224v1/fig1.png)
*Figure 1: The RVLM pipeline transitions from raw multi-sequence imaging data to an iterative, code-based reasoning loop, ultimately producing a structured, auditable radiological report.*

A significant advancement in this field is the development of Recursive Vision-Language Models (RVLM). Unlike traditional models that offer a one-shot prediction, the RVLM framework operates through an iterative "generate-execute" loop. By embedding a vision-capable language model within a persistent environment, the system explicitly unfolds its reasoning process into executable code. This shift from opaque prediction to transparent procedure allows every diagnostic claim to be grounded in verifiable steps. Furthermore, the framework addresses a common efficiency problem in iterative AI: the "fixed-budget" limitation. Most reasoning agents use a pre-set number of steps, which can waste resources on simple cases and provide insufficient depth for complex ones. RVLM introduces an adaptive depth mechanism that dynamically scales its computational effort to the diagnostic difficulty of the task at hand.

## The RVLM Architecture: Recursive Vision-Language Integration

The RVLM architecture is built upon the Recursive Language Model (RLM) paradigm, extended to include native multimodal support. At its core, the system consists of three main components: a vision-capable root language model ($M_V$), a visual Read-Eval-Print Loop ($E_V$) environment, and a mechanism for recursive vision calls.

The process begins when the model receives an initial multimodal prompt containing clinical images and a diagnostic task. Instead of outputting a final report immediately, the controller ($M_V$) generates Python code designed to inspect specific features of the image. This code is executed within the visual REPL ($E_V$), which acts as a stateful "sandbox." This environment treats images as first-class objects, allowing the model to store intermediate analysis artifacts, such as cropped regions of interest or contrast-enhanced versions of the original scan, in its "visual working memory."

![RVLM Architecture and REPL Loop](https://paper-assets.alphaxiv.org/figures-normalized/figures/2603.24224v1/x1.png)
*Figure 2: The RVLM architectural paradigm. The multimodal controller generates executable Python code that interacts with the Visual REPL, enabling multi-scale inspection and stateful reasoning.*

One of the unique features of RVLM is its ability to perform recursive sub-calls to the vision model. Using specialized functions like $\text{describe\_image}(\text{index}, \text{prompt})$, the controller can "ask" itself specific questions about localized image segments or different imaging modalities. This prevents "attention dilution," where a model might miss subtle details because it is trying to process too much information at once. For example, in a brain MRI study involving multiple sequences—such as T1-weighted native ($T1_n$), T1-weighted contrast-enhanced ($T1_{ce}$), T2-weighted ($T2_w$), and Fluid-Attenuated Inversion Recovery ($FLAIR$)—the model can systematically describe each sequence individually before synthesizing a final conclusion.

The loop continues until the model produces a $\text{FINAL}$ or $\text{FINAL\_VAR}$ signal. Only the initial user message is fully multimodal (containing high-resolution images); subsequent iterations rely on the persistent state of the REPL and lower-cost text-based or selective visual queries, which significantly reduces the computational overhead compared to passing full images in every turn of a long conversation.

## RECURSIONROUTER: Solving the Fixed-Budget Problem

A persistent challenge in iterative reasoning systems is determining the optimal number of steps. Simple diagnostic cases might be resolved in three steps, while complex ones might require twelve. A fixed budget of twelve iterations for every patient is computationally expensive and introduces unnecessary latency. Conversely, a budget of three might lead to an incomplete diagnosis for a complex tumor.

To solve this, the framework incorporates a module called RECURSIONROUTER, which treats the recursion depth as a dynamic variable determined by task complexity. For neuroradiology tasks, the router extracts four scalar features from initial image data or segmentation masks:
1.  **Label entropy ($H$):** Measures the spatial complexity and distribution of different tumor regions.
2.  **Total tumor volume ($V$):** Higher volumes often imply more extensive infiltration or multi-focal spread.
3.  **Present sub-region count ($R$):** Indicates the variety of radiological components (e.g., necrotic core vs. enhancing tumor).
4.  **Tiny region indicator ($T$):** Signals the presence of small, easily missed features that require higher-resolution inspection.

These features are combined into a composite complexity score $s$ using the following weighted linear combination:
$$s = w_1 H + w_2 V + w_3 R + w_4 T$$

The score $s$ is then mapped to a recommended iteration budget $n^*$ through a piecewise-constant function. This allows the system to pre-allocate an appropriate amount of reasoning depth. Furthermore, the router employs a "stall detection" mechanism. If the model completes two consecutive iterations without making progress—defined by a lack of new sub-calls or meaningful output—and has exceeded its suggested budget $n^*$, the router terminates the process early. This ensures that the system does not get stuck in repetitive loops, further optimizing resource allocation.

## Clinical Performance: Benchmarking on MRI and X-Rays

The RVLM system was evaluated on two distinct clinical benchmarks: the BraTS 2023 Meningioma dataset (brain MRI) and the MIMIC-CXR dataset (chest X-rays). These datasets were chosen because they represent very different diagnostic workflows, allowing researchers to test the framework's generalizability.

### Neuroradiology (BraTS)
In the BraTS experiments, the model was tasked with characterizing tumor sub-regions. A representative case (BraTS-MEN-00008-000) demonstrated the effectiveness of the adaptive depth mechanism. The RECURSIONROUTER assigned a budget of three iterations ($n^*=3$) based on a low complexity score ($s=0.14$). The model successfully completed its analysis in exactly three iterations, identifying the enhancing tumor and confirming the absence of a necrotic core.

![Radiological Report for BraTS MRI](https://paper-assets.alphaxiv.org/figures-normalized/figures/2603.24224v1/brats_report-1.png)
*Figure 3: A sample neuroradiology report generated by RVLM. The report includes quantitative statistics from segmentation masks alongside structured qualitative analysis.*

An important finding was RVLM's ability to detect cross-modal discrepancies autonomously. In one instance, the model noted that while the $FLAIR$ sequence showed hyperintense signals suggesting peritumoral edema, the provided segmentation masks did not label that region as edema. The model was able to reason through this discrepancy by comparing its independent descriptions of each modality, a level of critical assessment typically absent in single-pass models.

### Chest Radiography (MIMIC-CXR)
The system demonstrated its flexibility by adapting to chest X-rays using only a change in its prompting protocol, without any task-specific fine-tuning. For MIMIC-CXR, the model implemented a checklist-based reasoning pattern: first describing individual views (e.g., AP portable or lateral), then performing a comparative cross-view analysis, and finally synthesizing the report.

![Radiological Report for MIMIC-CXR](https://paper-assets.alphaxiv.org/figures-normalized/figures/2603.24224v1/mimic_report-1.png)
*Figure 4: A structured chest X-ray report. RVLM identified specific findings like reticulonodular infiltrates and appropriately qualified cardiac size estimates based on the projection type.*

In a test case involving an AP portable projection, RVLM correctly identified the projection type and noted that cardiac size estimates might be exaggerated due to magnification artifacts—a nuance well-known to radiologists but often missed by simpler AI models. It also identified reticulonodular infiltrates that were consistent with the visual evidence, providing a detailed breakdown that matched the findings of human-written ground truth reports.

## Interpretability and the Future of Trustworthy AI

The significance of RVLM lies not just in its diagnostic accuracy, but in its "trust-by-design" architecture. By forcing the reasoning process into the open through code, the system provides three layers of interpretability:
*   **Procedural Transparency:** Every step of the analysis is logged as a REPL trajectory. A human reviewer can see exactly which images the model looked at and what code it ran.
*   **Visual Provenance:** Diagnostic claims are linked to specific visual evidence, such as crops or difference maps (e.g., $T1_{ce} - T1_n$) created during the reasoning loop.
*   **Scientific Grounding:** The model can test visual hypotheses. If it suspects a dural tail sign, it can generate a zoomed-in crop of that specific anatomical region to verify its presence.

This level of transparency is essential for meeting the stringent requirements of clinical AI governance. The framework's ability to generate formal, structured PDF reports—complete with execution statistics and AI disclaimers—makes it a practical tool for real-world clinical environments.

Future developments for this framework include moving from hand-weighted complexity scores to a fully learned RECURSIONROUTER that can improve its budget predictions over time. Additionally, researchers aim to extend the system to handle full 3D volumetric reasoning and incorporate "uncertainty-gated recursion," where the model would automatically increase its reasoning depth whenever its internal confidence in a diagnosis drops below a certain threshold. By bridging the gap between high-performance vision models and verifiable, iterative reasoning, RVLM provides a path toward AI systems that are both clinically effective and foundationally trustworthy.