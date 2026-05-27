The shift toward visually rich documents—such as slide decks, financial reports with complex charts, and scanned PDFs—has presented a significant challenge for traditional Retrieval-Augmented Generation (RAG). While standard RAG systems focus on retrieving text snippets, Visual RAG (VRAG) systems operate directly on document images to preserve the spatial layout and visual cues that text extraction often loses. However, existing VRAG systems frequently struggle with multi-step reasoning and long-horizon tasks.

The VISOR (Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning) framework addresses these limitations. Most current systems use a "retrieve-then-read" pipeline, which is often insufficient for complex queries that require evidence from multiple pages. VISOR introduces a single-agent framework that performs iterative searches and manages visual evidence through a structured memory space, ensuring the model remains focused on the user's intent even as the conversation history grows.

![Figure 1: Illustration of the challenges in agentic VRAG, including visual evidence sparsity and search drift during long-horizon interaction.](https://paper-assets.alphaxiv.org/figures-normalized/figures/2604.09508v1/x1.png)

## Addressing Visual Evidence Sparsity

A primary difficulty in visual document analysis is that relevant information is often sparse. It might be scattered across several different pages (across-image sparsity) or buried within a tiny, high-density region of a single image, like a specific cell in a large table (within-image sparsity).

To solve the across-image problem, VISOR implements a persistent **Evidence Space** $E$. Instead of forcing the model to remember every detail of every retrieved image, the agent generates concise text summaries for each page it visits. For a given page $I_k$, the agent produces an initial summary $e_{\text{pre},k}$ upon first glance. If the agent decides to zoom in on a specific region, it produces a second, more detailed summary $e_{\text{post},k}$. These summaries are stored in the Evidence Space, allowing the agent to synthesize clues from multiple pages without needing to re-process thousands of visual tokens at every step.

For within-image sparsity, VISOR utilizes a "crop-and-zoom" action. Rather than cropping indiscriminately—which can introduce noise or waste computation—the agent follows a visual action evaluation protocol. It only triggers a crop if it determines the current resolution is insufficient for a confident answer. If a crop turns out to be uninformative, a correction mechanism redirects the agent's attention back to its previous reasoning, preventing the system from getting "lost in the details."

## Overcoming Search Drift in Long-Horizon Reasoning

As an agent interacts with a document over many turns, it faces "Search Drift." In the visual domain, document images are token-heavy. A single high-resolution image can consume a large portion of a model's context window. In a multi-turn interaction, the accumulation of these tokens quickly leads to context saturation. This "buries" earlier evidence and causes the agent to lose sight of the original user query $q$.

VISOR manages this through a **Dynamic Trajectory** approach using a sliding window. At any turn $t$, the input context $C_t$ is reconstructed to include:
1.  The initial system prompt and user query $P$.
2.  The current Evidence Space $E_t$ (the distilled text summaries).
3.  Only the last $W$ turns of raw interaction history $(r_i, o_i)$.

By setting a sliding window (typically $W=2$), VISOR ensures the model always sees the most recent observations while relying on the Evidence Space for long-term memory. To further combat drift, the system uses **Intent Injection**. Every time the environment returns a new image or crop, the system automatically appends a prompt that restates the original query and directs the model to consult its Evidence Space. This keeps the agent's reasoning anchored to the user's initial goal.

## The Agentic Loop and Action Space

VISOR operates in a `think`-`action`-`observation` cycle. In the `⟨think⟩` block, the agent performs internal reasoning and summarizes its findings for the Evidence Space. It then chooses one of three primary actions:

1.  **Search:** The agent generates a text query to retrieve a new document page. VISOR uses a multi-vector retrieval engine (ColQwen2.5) to find the most relevant visual content.
2.  **Crop:** The agent provides a bounding box to zoom into a specific area of the current page.
3.  **Answer:** Once sufficient evidence is gathered, the agent provides the final response and terminates the loop.

To ensure reliability, VISOR employs a "verification search" strategy. Before providing a final answer, the agent is encouraged to perform one last search using the original question to confirm that no contradictory information exists on other pages.

![Figure 2: Overview of the VISOR framework, showing the Evidence Space, the action space, and the dynamic context management.](https://paper-assets.alphaxiv.org/figures-normalized/figures/2604.09508v1/x2.png)

## Training via Trajectory Distillation and RL

The authors use a two-stage training pipeline to refine the agent's behavior.

### Stage 1: Supervised Fine-Tuning (SFT)
The model is first trained on high-quality agentic trajectories distilled from a larger model (Qwen3-VL-235B). These trajectories are filtered for correctness and structural validity. During this stage, the model learns the basic syntax of the agentic loop, such as how to format search queries and when to use crop actions.

### Stage 2: Reinforcement Learning (RL)
To optimize decision-making and stopping criteria, VISOR undergoes Reinforcement Learning using Group Relative Policy Optimization (GRPO). The agent generates its own trajectories and receives rewards based on the final outcome. The total reward $r$ is calculated as:

$$
r = I_{\text{format}} \cdot (r_{\text{ans}} + r_{\text{ret}})
$$

Where:
*   $I_{\text{format}}$ is a penalty for invalid output formats.
*   $r_{\text{ans}}$ is the answer reward, determined by comparing the predicted answer $\hat{a}$ to the ground truth $a^*$. If the agent correctly identifies that the information is missing rather than hallucinating, it receives a partial "honesty" reward.
*   $r_{\text{ret}}$ is the retrieval reward, which encourages the agent to retrieve all necessary pages while penalizing excessive or redundant searches.

## Performance and Efficiency

VISOR was evaluated on three major benchmarks: SlideVQA (multimodal slides), ViDoSeek (visually rich documents), and MMLongBench (long-context multimodal tasks).

The results showed that VISOR consistently outperformed existing agentic VRAG baselines. On SlideVQA, VISOR achieved an accuracy of 72.37% using a 7B backbone, significantly higher than previous iterative methods like VRAG-RL (which scored 60.31%). The improvement was most notable in multi-hop questions, which require synthesizing information from several pages—a direct result of the Evidence Space design.

| Method | SlideVQA (Acc) | ViDoSeek (Acc) |
| :--- | :---: | :---: |
| EVisRAG (One-shot) | 60.10% | 61.35% |
| VRAG-RL (Iterative) | 60.31% | 63.85% |
| **VISOR (Ours)** | **72.37%** | **74.87%** |

In terms of efficiency, although VISOR is an iterative model, the sliding window context management prevents the linear growth of computational costs. It remains faster than multi-agent systems like ViDoRAG, which require coordination between multiple specialized models.

![Figure 4: Average latency comparison between VISOR and other VRAG methods.](https://paper-assets.alphaxiv.org/figures-normalized/figures/2604.09508v1/x4.png)

## Conclusion

VISOR represents a systematic approach to the core challenges of visual retrieval-augmented generation. By introducing a structured Evidence Space, it allows models to bridge reasoning gaps across multiple pages. Simultaneously, its context management and intent injection mechanisms prevent the model from drifting during long interactions. The combination of targeted visual actions and a robust RL-based training pipeline enables the system to handle complex, visually dense documents with higher accuracy and reliability than traditional fixed-pipeline or naive agentic approaches. This framework provides a blueprint for more capable document intelligence systems that can reason effectively over the vast amount of visual information found in real-world professional and scientific documents.