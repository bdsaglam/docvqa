Title: ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering

URL Source: https://arxiv.org/html/2603.02438

Published Time: Wed, 04 Mar 2026 01:11:21 GMT

Markdown Content:
ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering
===============

##### Report GitHub Issue

×

Title: 
Content selection saved. Describe the issue below:

Description: 

Submit without GitHub Submit in GitHub

[![Image 1: arXiv logo](https://arxiv.org/static/browse/0.3.4/images/arxiv-logo-one-color-white.svg)Back to arXiv](https://arxiv.org/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html)[Report Issue](https://arxiv.org/html/2603.02438# "Report an Issue")[Back to Abstract](https://arxiv.org/abs/2603.02438v1 "Back to abstract page")[Download PDF](https://arxiv.org/pdf/2603.02438v1 "Download PDF")[](javascript:toggleNavTOC(); "Toggle navigation")[](javascript:toggleReadingMode(); "Disable reading mode, show header and footer")[](javascript:toggleColorScheme(); "Toggle dark/light mode")
1.   [Abstract](https://arxiv.org/html/2603.02438#abstract1 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
2.   [1 Introduction](https://arxiv.org/html/2603.02438#S1 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
3.   [2 Related Work](https://arxiv.org/html/2603.02438#S2 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
4.   [3 Methodology](https://arxiv.org/html/2603.02438#S3 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    1.   [3.1 Stage 1: Context Understanding](https://arxiv.org/html/2603.02438#S3.SS1 "In 3 Methodology ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    2.   [3.2 Stage 2: Collaborative Agent Execution](https://arxiv.org/html/2603.02438#S3.SS2 "In 3 Methodology ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    3.   [3.3 Stage 3: Stress Testing Session](https://arxiv.org/html/2603.02438#S3.SS3 "In 3 Methodology ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    4.   [3.4 Stage 4: Multi-turn Conversation](https://arxiv.org/html/2603.02438#S3.SS4 "In 3 Methodology ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    5.   [3.5 Stage 5: Answer Refinement](https://arxiv.org/html/2603.02438#S3.SS5 "In 3 Methodology ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")

5.   [4 Experiments](https://arxiv.org/html/2603.02438#S4 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    1.   [4.1 Experiment Setup](https://arxiv.org/html/2603.02438#S4.SS1 "In 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    2.   [4.2 Main Results](https://arxiv.org/html/2603.02438#S4.SS2 "In 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    3.   [4.3 Inference Latency and Cost Analysis](https://arxiv.org/html/2603.02438#S4.SS3 "In 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    4.   [4.4 Ablation Studies](https://arxiv.org/html/2603.02438#S4.SS4 "In 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    5.   [4.5 Case Study](https://arxiv.org/html/2603.02438#S4.SS5 "In 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    6.   [4.6 Error Analysis](https://arxiv.org/html/2603.02438#S4.SS6 "In 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")

6.   [5 Conclusion](https://arxiv.org/html/2603.02438#S5 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
7.   [References](https://arxiv.org/html/2603.02438#bib "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
8.   [A Implementation and Training Details](https://arxiv.org/html/2603.02438#A1 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    1.   [A.1 Datasets.](https://arxiv.org/html/2603.02438#A1.SS1 "In Appendix A Implementation and Training Details ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    2.   [A.2 Baselines.](https://arxiv.org/html/2603.02438#A1.SS2 "In Appendix A Implementation and Training Details ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    3.   [A.3 Evaluation Metrics.](https://arxiv.org/html/2603.02438#A1.SS3 "In Appendix A Implementation and Training Details ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")

9.   [B Router](https://arxiv.org/html/2603.02438#A2 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    1.   [B.1 Training Data and Augmentation](https://arxiv.org/html/2603.02438#A2.SS1 "In Appendix B Router ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    2.   [B.2 Model Architecture and Optimization](https://arxiv.org/html/2603.02438#A2.SS2 "In Appendix B Router ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    3.   [B.3 Turbo DFS Decoding for Multi-Label Prediction](https://arxiv.org/html/2603.02438#A2.SS3 "In Appendix B Router ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")

10.   [C Algorithms](https://arxiv.org/html/2603.02438#A3 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
11.   [D Inference Latency and Cost Analysis](https://arxiv.org/html/2603.02438#A4 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    1.   [D.1 Optimization Details](https://arxiv.org/html/2603.02438#A4.SS1 "In Appendix D Inference Latency and Cost Analysis ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    2.   [D.2 ORCA-Lite Configuration](https://arxiv.org/html/2603.02438#A4.SS2 "In Appendix D Inference Latency and Cost Analysis ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")

12.   [E Additional Results](https://arxiv.org/html/2603.02438#A5 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    1.   [E.1 Detailed DocVQA Performance Breakdown](https://arxiv.org/html/2603.02438#A5.SS1 "In Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    2.   [E.2 Detailed Infographics VQA Performance Breakdown](https://arxiv.org/html/2603.02438#A5.SS2 "In Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    3.   [E.3 Prompt Settings](https://arxiv.org/html/2603.02438#A5.SS3 "In Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    4.   [E.4 Experiments on different model backbones in ORCA](https://arxiv.org/html/2603.02438#A5.SS4 "In Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    5.   [E.5 Additional case studies](https://arxiv.org/html/2603.02438#A5.SS5 "In Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")

13.   [F Extended Error Analysis](https://arxiv.org/html/2603.02438#A6 "In ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    1.   [F.1 Failure Mode Breakdown](https://arxiv.org/html/2603.02438#A6.SS1 "In Appendix F Extended Error Analysis ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    2.   [F.2 Error Propagation Analysis](https://arxiv.org/html/2603.02438#A6.SS2 "In Appendix F Extended Error Analysis ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    3.   [F.3 Failure Case Examples](https://arxiv.org/html/2603.02438#A6.SS3 "In Appendix F Extended Error Analysis ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")
    4.   [F.4 Implications for Future Work](https://arxiv.org/html/2603.02438#A6.SS4 "In Appendix F Extended Error Analysis ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")

[License: CC BY-NC-SA 4.0](https://info.arxiv.org/help/license/index.html#licenses-available)

 arXiv:2603.02438v1 [cs.CV] 02 Mar 2026

ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering
=============================================================================================

 Aymen Lassoued 1,2, Mohamed Ali Souibgui 3, Yousri Kessentini 1

1 Digital Research Center of Sfax, SMARTS Laboratory, Sfax, Tunisia. 

2 École Polytechnique de Tunisie, University of Carthage 

3 Computer Vision Center, Universitat Autònoma de Barcelona 

{aymen.lassoued@ept.ucar.tn, msouibgui@cvc.uab.cat, yousri.kessentini@crns.rnrt.tn}

###### Abstract

Document Visual Question Answering (DocVQA) remains challenging for existing Vision-Language Models (VLMs), especially under complex reasoning and multi-step workflows. Current approaches struggle to decompose intricate questions into manageable sub-tasks and often fail to leverage specialized processing paths for different document elements. We present ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering, a novel multi-agent framework that addresses these limitations through strategic agent coordination and iterative refinement. ORCA begins with a reasoning agent that decomposes queries into logical steps, followed by a routing mechanism that activates task-specific agents from a specialized agent dock. Our framework leverages a set of specialized AI agents, each dedicated to a distinct modality, enabling fine-grained understanding and collaborative reasoning across diverse document components. To ensure answer reliability, ORCA employs a debate mechanism with stress-testing, and when necessary, a thesis-antithesis adjudication process. This is followed by a sanity checker to ensure format consistency. Extensive experiments on three benchmarks demonstrate that our approach achieves significant improvements over state-of-the-art methods, establishing a new paradigm for collaborative agent systems in vision-language reasoning.

1 Introduction
--------------

![Image 2: Refer to caption](https://arxiv.org/html/2603.02438v1/x1.png)

Figure 1: Comparison of different approaches for DocVQA. Single-model VLMs and reasoning-enhanced VLMs lack critical capabilities such as adaptivity and self-verification. In contrast, ORCA introduces a feature-oriented, multi-agent design achieving improved DocVQA performance as well as the missing capabilities in one unified framework. 

Answering questions based on single-page document images (DocVQA) [[39](https://arxiv.org/html/2603.02438#bib.bib39)] requires more than simple information extraction. Questions often span multiple document modalities (e.g., text, tables, figures, handwritten content) and demand complex reasoning that current Vision Language Models (VLMs) struggle to perform reliably [[66](https://arxiv.org/html/2603.02438#bib.bib66)]. While VLMs have demonstrated impressive capabilities in document visual understanding [[5](https://arxiv.org/html/2603.02438#bib.bib5), [18](https://arxiv.org/html/2603.02438#bib.bib18), [65](https://arxiv.org/html/2603.02438#bib.bib65)], they frequently fall short when faced with multi-step reasoning, coordination across document elements, or specialized handling of diverse content types.

Most existing DocVQA approaches typically rely on a single model to handle all aspects of document understanding, leading to suboptimal performance when questions involve heterogeneous information sources. For example, a question about data in a table with handwritten annotations requires expertise in both structured data extraction and OCR/HTR capabilities, skills that general-purpose models execute inconsistently [[28](https://arxiv.org/html/2603.02438#bib.bib28)]. Moreover, these models typically produce answers directly, without planning or exposing the reasoning steps behind their predictions.

Recently, some work has explored chain-of-thought (CoT) prompting and related techniques [[61](https://arxiv.org/html/2603.02438#bib.bib61)], which encourage models to articulate intermediate reasoning steps before generating answers [[71](https://arxiv.org/html/2603.02438#bib.bib71), [72](https://arxiv.org/html/2603.02438#bib.bib72)]. These methods have improved both interpretability and accuracy on complex question answering tasks. However, they still rely on a single model to handle all reasoning steps and document modalities, lacking mechanisms for content-aware specialization, self-verification, or adaptive agent selection based on document components. They do not employ specialized agents tailored to specific document elements (tables, charts, handwritten text), nor do they incorporate debate mechanisms to stress-test predictions or resolve conflicts between competing interpretations. Furthermore, without iterative refinement or cross-validation mechanisms, these models often produce answers without adequate confidence assessment [[7](https://arxiv.org/html/2603.02438#bib.bib7)], limiting their reliability for complex DocVQA scenarios that require coordination across diverse document components.

To address these limitations, we introduce ORCA, a multi-agent framework that integrates explicit reasoning with collaborative execution for DocVQA and operates through five key stages. (1) Context Understanding: A thinking agent analyzes the document and question to generate a structured reasoning path and initial hypotheses. (2) Collaborative Agent Execution: Guided by the reasoning path, specialized agents tailored to document components such as tables, figures, forms, and handwritten text are dynamically activated to generate the answers. This introduces both reasoning capability and content-aware specialization. (3) Debate Session and (4) Multi-turn Conversation: To ensure reliability, ORCA incorporates a self-verification mechanism in which agents engage in iterative debate and reflection to stress-test and reconcile divergent responses. (5) Answer Refinement: A validation stage attends to fine-grained details and formatting consistency to refine the final output.

As illustrated in Figure[1](https://arxiv.org/html/2603.02438#S1.F1 "Fig. 1 ‣ 1 Introduction ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering"), our design delivers key advantages over single-model and reasoning-enhanced DocVQA approaches, including transparency, query decomposition, adaptivity, self-verification, and fine-grained attention to detail. Our contributions can be summarized as follows:

*   •We propose a multi-agent framework that integrates explicit reasoning, specialized document understanding, and adversarial verification for robust single-page DocVQA. 
*   •We achieved top performance on nearly all standard benchmarks compared to current state-of-the-art methods, demonstrating that our collaborative architecture with built-in debate and verification mechanisms produces more accurate and reliable answers for complex document question answering. 
*   •We perform ablation studies to validate the contribution of each component, particularly the reasoning-guided agent selection and multi-turn conversation stages. 

2 Related Work
--------------

Vision-Language Models for Document Understanding. Document Visual Question Answering (DocVQA) has progressed from processing simple text-based documents to handling visually rich documents containing diverse content types[[39](https://arxiv.org/html/2603.02438#bib.bib39), [40](https://arxiv.org/html/2603.02438#bib.bib40), [8](https://arxiv.org/html/2603.02438#bib.bib8), [58](https://arxiv.org/html/2603.02438#bib.bib58)]. Early approaches employed specialized multimodal transformers such as LayoutLM[[67](https://arxiv.org/html/2603.02438#bib.bib67), [21](https://arxiv.org/html/2603.02438#bib.bib21)], TILT[[45](https://arxiv.org/html/2603.02438#bib.bib45)], and Donut[[26](https://arxiv.org/html/2603.02438#bib.bib26)], which jointly model textual content, spatial layout, and visual features. Recent Vision-Language Models (VLMs), including BLIP-2[[31](https://arxiv.org/html/2603.02438#bib.bib31)], Qwen3-VL[[51](https://arxiv.org/html/2603.02438#bib.bib51)], InternVL[[74](https://arxiv.org/html/2603.02438#bib.bib74), [11](https://arxiv.org/html/2603.02438#bib.bib11)], and GLM-4.5V[[57](https://arxiv.org/html/2603.02438#bib.bib57)], have demonstrated strong capabilities by integrating language understanding with visual perception[[35](https://arxiv.org/html/2603.02438#bib.bib35), [36](https://arxiv.org/html/2603.02438#bib.bib36), [13](https://arxiv.org/html/2603.02438#bib.bib13)]. These models process document images directly, preserving layout and visual context. However, questions spanning multiple content modalities—such as extracting data from tables or interpreting figures alongside textual explanations—remain challenging, as they require coordination across different types of document elements[[38](https://arxiv.org/html/2603.02438#bib.bib38), [20](https://arxiv.org/html/2603.02438#bib.bib20), [10](https://arxiv.org/html/2603.02438#bib.bib10)].

Reasoning and Verification in Language Models. Explicit reasoning mechanisms have become increasingly important for complex question answering tasks. In[[61](https://arxiv.org/html/2603.02438#bib.bib61)], Chain-of-Thought (CoT) prompting introduced the concept of generating intermediate reasoning steps to improve complex question answering. Building on this, Self-Consistency[[60](https://arxiv.org/html/2603.02438#bib.bib60)] demonstrated the benefits of sampling multiple reasoning paths. Extensions such as ReAct[[69](https://arxiv.org/html/2603.02438#bib.bib69)] and Reflexion[[53](https://arxiv.org/html/2603.02438#bib.bib53)] incorporate external tool use and iterative refinement. More recently, models with extended reasoning capabilities like DeepSeek-R1[[14](https://arxiv.org/html/2603.02438#bib.bib14)] and OpenAI’s o1[[43](https://arxiv.org/html/2603.02438#bib.bib43)] have shown the value of longer thinking time for problem-solving. Parallel to these advances, verification mechanisms emerged to address the trustworthiness of model outputs, including those based on debate and argumentation[[24](https://arxiv.org/html/2603.02438#bib.bib24), [16](https://arxiv.org/html/2603.02438#bib.bib16), [9](https://arxiv.org/html/2603.02438#bib.bib9), [34](https://arxiv.org/html/2603.02438#bib.bib34)], which explore how multiple models can challenge and refine predictions through structured interaction. Together, these developments inspire designing systems where reasoning transparency and cross-validation are essential.

![Image 3: Refer to caption](https://arxiv.org/html/2603.02438v1/x2.png)

Figure 2: Overview of ORCA: A reasoning-guided multi-agent framework for Document Visual Question Answering operating through five stages: (1) Context Understanding: A thinker agent analyzes the question and document to generate both a reasoning path and initial answer (a T a_{T}). (2) Collaborative Agent Execution: A router selects relevant specialized agents from a dock of nine expert types (OCR, Layout, Table/List, Figure/Diagram, Form, Free Text, Image/Photo, Yes/No, and General), which an orchestrator sequences for optimal execution to produce an expert answer (a E a_{E}). (3) Stress Testing: When a E a_{E} differs from a T a_{T}, a debate agent generates challenging questions to stress-test the specialized agent’s confidence, with an evaluation agent assessing the responses to produce a D a_{D}. (4) Multi-turn Conversation: If stress testing indicates uncertainty, thesis and antithesis agents engage in structured three-turn debate under judge supervision to resolve conflicts and generate a C a_{C}. (5) Answer Refinement: A sanity checker performs final formatting corrections to ensure consistency with document conventions, producing the final answer (a F a_{F}).

Multi-Agent Frameworks. Multi-agent systems coordinate specialized components to tackle complex tasks[[63](https://arxiv.org/html/2603.02438#bib.bib63), [30](https://arxiv.org/html/2603.02438#bib.bib30), [27](https://arxiv.org/html/2603.02438#bib.bib27), [19](https://arxiv.org/html/2603.02438#bib.bib19)]. Architectures such as Visual ChatGPT[[62](https://arxiv.org/html/2603.02438#bib.bib62)] and HuggingGPT[[52](https://arxiv.org/html/2603.02438#bib.bib52)] use language models as controllers to orchestrate specialized vision and language modules. In document understanding contexts, different agents can focus on specific aspects such as table extraction, OCR, or layout analysis[[54](https://arxiv.org/html/2603.02438#bib.bib54), [25](https://arxiv.org/html/2603.02438#bib.bib25), [33](https://arxiv.org/html/2603.02438#bib.bib33)]. The challenge lies in effectively routing questions to appropriate specialized models and coordinating their outputs when multiple modalities are involved.Our work explores how explicit reasoning can guide agent selection, how sequential orchestration can enable information flow between specialists, and how debate mechanisms can reconcile conflicting predictions when the thinker and expert agents reach divergent conclusions. Unlike prior orchestration frameworks such as Visual ChatGPT[[62](https://arxiv.org/html/2603.02438#bib.bib62)] and HuggingGPT[[52](https://arxiv.org/html/2603.02438#bib.bib52)], which rely on hand-crafted routing rules, ORCA introduces a VLM trained specifically for document-type routing via constrained generation with Turbo DFS decoding. In contrast to chain-of-thought and tool-use approaches, our reasoning path masking mechanism explicitly prevents confirmation bias in downstream agents. Finally, rather than applying verification universally, ORCA employs conditional activation, engaging debate in only 8.3% of instances, concentrating computational overhead on cases where genuine uncertainty arises.

3 Methodology
-------------

This section details our proposed framework for Document Visual Question Answering. Our approach employs a five-stage pipeline as illustrated in Figure [2](https://arxiv.org/html/2603.02438#S2.F2 "Fig. 2 ‣ 2 Related Work ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering"). At its core, our approach integrates explicit reasoning, specialized agent collaboration, and adversarial verification to handle the diverse and complex nature of document questions. We describe each stage in detail below.

Problem Formulation. Given a single-page document 𝒟\mathcal{D} and a natural language question q q, our goal is to generate an accurate answer a a by reasoning over the document’s multimodal content, which may include text, tables, figures, forms, and handwritten elements.

### 3.1 Stage 1: Context Understanding

The first stage establishes the reasoning foundation for our framework. We employ a thinker agent A think A_{\text{think}} based on GLM-4.5V-9B with thinking capabilities to analyze the question and document image jointly. The thinker agent generates two critical outputs: (1) a structured reasoning path ℛ\mathcal{R} that decomposes the question into logical steps, and (2) an initial answer a T a_{T} based on this reasoning process.

(ℛ,a T)=A think​(q,𝒟)(\mathcal{R},a_{T})=A_{\text{think}}(q,\mathcal{D})(1)

The reasoning path ℛ={r 1,r 2,…,r n}\mathcal{R}=\{r_{1},r_{2},\ldots,r_{n}\} consists of n n intermediate reasoning steps that describe the cognitive process required to answer the question. For example, for a question ”What is the total revenue in Q3?”, the reasoning path might specify: r 1 r_{1}: ”Locate the quarterly revenue table”, r 2 r_{2}: ”Find the Q3 column”, r 3 r_{3}: ”Extract the total revenue value”. This explicit reasoning serves as a guide for subsequent agent selection and orchestration.

### 3.2 Stage 2: Collaborative Agent Execution

The second stage dynamically selects and sequences specialized agents based on the reasoning path. This stage consists of three components: the agent dock, router, and orchestrator.

Agent Dock. We maintain nine specialized agents, each designed to handle specific document content types:

*   •A figure A_{\text{figure}}: Handles diagrams and charts 
*   •A yesno A_{\text{yesno}}: Processes yes/no questions 
*   •A table A_{\text{table}}: Extracts information from tables and lists 
*   •A layout A_{\text{layout}}: Analyzes document layout structure 
*   •A image A_{\text{image}}: Interprets photographs and images 
*   •A ocr A_{\text{ocr}}: Recognizes handwritten and difficult text 
*   •A text A_{\text{text}}: Processes free-form textual content 
*   •A form A_{\text{form}}: Handles structured forms 
*   •A other A_{\text{other}}: Addresses miscellaneous content types 

All specialized agents are based on variants of Qwen3-VL-8B, fine-tuned for their respective tasks.

Router. The router agent A route A_{\text{route}}, plays a critical role in our framework by determining which specialized agents should process each document-question pair. We formulate this as a multi-label classification problem over nine agent types and employ several optimization strategies to ensure efficient and accurate routing. Given the reasoning path ℛ\mathcal{R}, question q q, and document 𝒟\mathcal{D}, the router must predict a binary activation vector 𝐯∈{0,1}9\mathbf{v}\in\{0,1\}^{9} indicating which agents should be activated. Note that multiple agents can be simultaneously relevant for a single document-question pair.

We train the router on the Single-Page Document VQA dataset with ground-truth agent annotations. To improve model robustness and generalization, we incorporate data augmentation techniques. We employ Qwen2.5-VL-7B as the base architecture for A route A_{\text{route}}, unlike standard classification approaches that apply a sigmoid threshold to output logits, we treat routing as a constrained generation task. Thus, we use Turbo DFS (Depth-First Search with score-guided pruning) for decoding. Given the ranked candidate sequences from Turbo DFS, we apply a union strategy to extract the final agent activation set. More details about training the router are available in Appendix [B.1](https://arxiv.org/html/2603.02438#A2.SS1 "B.1 Training Data and Augmentation ‣ Appendix B Router ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering").

Our trained router can be used, therefore, to analyze the reasoning path ℛ\mathcal{R}, question q q, and document 𝒟\mathcal{D} to determine which specialized agents are required. It outputs a binary activation vector 𝐯∈{0,1}9\mathbf{v}\in\{0,1\}^{9}, where v i=1 v_{i}=1 indicates that agent i i should be activated.

𝐯=A route​(q,𝒟,ℛ)\mathbf{v}=A_{\text{route}}(q,\mathcal{D},\mathcal{R})(2)

Thus, from the router we have 𝒜 active={A i∣v i=1}\mathcal{A}_{\text{active}}=\{A_{i}\mid v_{i}=1\} denote the set of activated agents.

Orchestrator. The orchestrator determines the optimal execution order for activated agents. Given n n activated agents, it produces a sequence σ=(σ 1,σ 2,…,σ n)\sigma=(\sigma_{1},\sigma_{2},\ldots,\sigma_{n}) where σ i∈𝒜 active\sigma_{i}\in\mathcal{A}_{\text{active}} represents the agent to execute at step i i.

σ=Orchestrate​(𝒜 active,ℛ,q,𝒟)\sigma=\text{Orchestrate}(\mathcal{A}_{\text{active}},\mathcal{R},q,\mathcal{D})(3)

Agents execute sequentially, with each agent receiving the output from its predecessor. For agent σ i\sigma_{i}, the input consists of the question q q, document 𝒟\mathcal{D}, and the answer a i−1 a_{i-1} from the previous agent:

a i=σ i​(q,𝒟,a i−1)a_{i}=\sigma_{i}(q,\mathcal{D},a_{i-1})(4)

The final agent σ n\sigma_{n} additionally receives a masked version of the reasoning path ℛ∗\mathcal{R}^{*}, where occurrences of the answer are masked if they appear frequently in the reasoning steps. Specifically, if the answer a T a_{T} appears more than the threshold τ\tau we mask all occurrences.

a E=σ n​(q,𝒟,a n−1,ℛ∗)a_{E}=\sigma_{n}(q,\mathcal{D},a_{n-1},\mathcal{R}^{*})(5)

So, a E a_{E} is the expert answer, representing the specialized agent’s final answer.

### 3.3 Stage 3: Stress Testing Session

In this stage, we compare the expert answer a E a_{E} with the thinker’s answer a T a_{T}. If a E=a T a_{E}=a_{T}, we proceed directly to Stage 5 (sanity checking). If not, we initiate a stress testing session to assess the confidence of the expert system. The stress testing session consists of three agents (debate, specialized and evaluation) working together to stress-test the specialized agent’s answer. This process is detailed in what follows.

Debate Agent. The goal of the debate agent A debate A_{\text{debate}} is to probe the reasoning behind the answer and identify potential weaknesses. Hence, it generates challenging follow-up questions q debate q_{\text{debate}} based on the document 𝒟\mathcal{D}, original question q q, and expert answer a E a_{E}.

q debate=A debate​(q,𝒟,a E)q_{\text{debate}}=A_{\text{debate}}(q,\mathcal{D},a_{E})(6)

Specialized Agent. The specialized agent σ n\sigma_{n} is the final agent from Stage 2 (with the final answer), in this stage, it receives the debate question and must provide: (1) a response r debate r_{\text{debate}} to the debate question, and (2) a potentially revised answer a E′a^{\prime}_{E} to the original question.

(r debate,a E′)=σ n​(q debate,q,𝒟,a E)(r_{\text{debate}},a^{\prime}_{E})=\sigma_{n}(q_{\text{debate}},q,\mathcal{D},a_{E})(7)

Evaluation Agent. The evaluation agent A eval A_{\text{eval}} is an LLM-based model that assesses whether the specialized agent: (1) provided a coherent response, (2) stayed on-topic, and (3) maintained its original answer. The evaluation produces a binary decision d∈{pass,fail}d\in\{\text{pass},\text{fail}\}.

d=A eval​(q debate,r debate,a E,a E′)d=A_{\text{eval}}(q_{\text{debate}},r_{\text{debate}},a_{E},a^{\prime}_{E})(8)

This stress testing repeats for two turns. If the specialized agent passes both turns (maintaining a E a_{E} consistently with coherent responses), we set a D=a E a_{D}=a_{E} and proceed to Stage 5. If it fails in either turn, we proceed to Stage 4 (multi-turn conversation).

### 3.4 Stage 4: Multi-turn Conversation

This stage is executed if the stress testing session indicates uncertainty. At this point, we engage a multi-turn communication protocol engaging three distinct agents: thesis, antithesis, and judge. Here,the thesis agent A thesis A_{\text{thesis}} (same backbone as the specialized agents) advocates for answer a E a_{E}, while the antithesis agent A anti A_{\text{anti}} (InternVL3-8B-hf) generates an alternative answer a alt a_{\text{alt}} and argues against a E a_{E}.

a alt=A anti​(q,𝒟,a E)a_{\text{alt}}=A_{\text{anti}}(q,\mathcal{D},a_{E})(9)

If A anti A_{\text{anti}} cannot generate a distinct alternative (i.e., a alt=a E a_{\text{alt}}=a_{E} or a alt a_{\text{alt}} contains a E a_{E}), we accept a E a_{E} and proceed to Stage 5. Otherwise, we initiate a three-turn debate with a defined protocol.

Conversation Protocol. Each turn t t consists of structured exchanges between the two agents. The antithesis agent presents its argument in a structured format containing three components:

*   •[REFERENCE]: Evidence from the document supporting its position 
*   •[CRITICISM]: Critique of the thesis agent’s answer 
*   •[CONCLUSION]: Its proposed answer and reasoning 

arg anti(t)=A anti​(q,𝒟,a E,summary(t−1))\text{arg}_{\text{anti}}^{(t)}=A_{\text{anti}}(q,\mathcal{D},a_{E},\text{summary}^{(t-1)})(10)

The thesis agent receives only the [REFERENCE] and [CRITICISM] components and responds by defending its position and addressing the criticism:

arg thesis(t)=A thesis​(q,𝒟,a E,arg anti(t)​[REF, CRIT],summary(t−1))\text{arg}_{\text{thesis}}^{(t)}=A_{\text{thesis}}(q,\mathcal{D},a_{E},\text{arg}_{\text{anti}}^{(t)}[\text{REF, CRIT}],\text{summary}^{(t-1)})(11)

During this debate, the judge agent A judge A_{\text{judge}} (LLM-based) performs three functions: (1) Evaluates the [CONCLUSION] sections to determine if either agent has been convinced to change its position, (2) Generates a summary of the discussion for the next turn and (3) If no convincement occurs after three turns, analyzes the full debate transcript linguistically to determine which agent demonstrated greater confidence. Thus, after each turn t t:

(convinced,summary(t))=A judge​(arg thesis(t),arg anti(t))(\text{convinced},\text{summary}^{(t)})=A_{\text{judge}}(\text{arg}_{\text{thesis}}^{(t)},\text{arg}_{\text{anti}}^{(t)})(12)

The final answer a C a_{C} is determined when one agent is convinced or after three turns, based on the judge’s confidence assessment.

### 3.5 Stage 5: Answer Refinement

The final stage ensures formatting consistency between the predicted answer and the source document. A sanity checker agent A sanity A_{\text{sanity}} receives the question q q, document 𝒟\mathcal{D}, and the answer from the previous stage (either a E a_{E}, a D a_{D}, or a C a_{C}). It performs two specific operations: Correction of missing spaces that appear in the document but not in the answer and Adjustment of punctuation to match the document’s formatting conventions

a F=A sanity​(q,𝒟,a prev)a_{F}=A_{\text{sanity}}(q,\mathcal{D},a_{\text{prev}})(13)

Hence, a F a_{F} is the final output answer. This stage ensures that the answer maintains fidelity to the document’s original formatting, which is particularly important for DocVQA evaluation metrics.

More detailed implementation details can be found in Appendix [C](https://arxiv.org/html/2603.02438#A3 "Appendix C Algorithms ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")

4 Experiments
-------------

We evaluate our framework on three document understanding benchmarks to answer the following questions: (1) Does our reasoning-guided multi-agent approach improve DocVQA accuracy compared to existing VLMs? and (2) What is the contribution of each stage in our pipeline?

### 4.1 Experiment Setup

Implementation Details. Our framework consists of multiple specialized components: a thinker agent, nine specialized agents in the agent dock, a router agent, debate agents, thesis and antithesis agents, an evaluation agent (LLM-based), a judge agent (LLM-based), and a sanity checker. For ORCA(Qwen3VL-8B-Instruct), the thinker agent uses GLM-4.5V-9B[[57](https://arxiv.org/html/2603.02438#bib.bib57)] with thinking capabilities, while all other agents (specialized agents, debate agents, thesis agent, and sanity checker) are based on Qwen3VL-8B-Instruct[[50](https://arxiv.org/html/2603.02438#bib.bib50)], and the antithesis agent is based on InternVL3-8B-hf[[74](https://arxiv.org/html/2603.02438#bib.bib74), [23](https://arxiv.org/html/2603.02438#bib.bib23)]. The evaluation and judge agents use Qwen3-1.7B[[48](https://arxiv.org/html/2603.02438#bib.bib48)] for robust assessment. The multi-agent debate mechanism enables collaborative reasoning, while the thinking agent provides enhanced chain-of-thought reasoning. All experiments are conducted on 4 NVIDIA L4 GPUs, each with 24GB VRAM (96GB total VRAM) and 175GB RAM. Details of agent configurations and hyperparameters are provided in the supplementary materials. We evaluate our approach on three challenging document understanding benchmarks (Single-Page DocVQA[[39](https://arxiv.org/html/2603.02438#bib.bib39)], InfographicsVQA[[40](https://arxiv.org/html/2603.02438#bib.bib40)] and OCRBench-v2 (en)[[17](https://arxiv.org/html/2603.02438#bib.bib17)]), Following standard evaluation protocols[[39](https://arxiv.org/html/2603.02438#bib.bib39), [40](https://arxiv.org/html/2603.02438#bib.bib40)], we report ANLS scores for Single-Page DocVQA and InfographicsVQA. For OCRBench-v2, we employ its official multi-dimensional evaluation suite with six task-specific metrics. The final score represents the average across all dimensions. More details about implementation details are available in Appendix[A](https://arxiv.org/html/2603.02438#A1 "Appendix A Implementation and Training Details ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering").

### 4.2 Main Results

The overall performance of our method compared to other approaches is available in Table[1](https://arxiv.org/html/2603.02438#S4.T1 "Table 1 ‣ 4.2 Main Results ‣ 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering") and Table[2](https://arxiv.org/html/2603.02438#S4.T2 "Table 2 ‣ 4.2 Main Results ‣ 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering"), where we test the performance for DocVQA and OCRBench, respectively.

Table 1: Performance comparison across DocVQA benchmarks. Models are categorized by architectural paradigm.

| Model | Open-source | DocVQA | InfoVQA | Avg. |
| --- | --- |
| Document Understanding Models |
| LayoutLMv2 LARGE[[68](https://arxiv.org/html/2603.02438#bib.bib68)] | ✓ | 86.7 | 28.3 | 57.5 |
| Text-Monkey[[32](https://arxiv.org/html/2603.02438#bib.bib32)] | ✓ | 66.7 | 28.6 | 47.7 |
| DocOwl-1.5[[20](https://arxiv.org/html/2603.02438#bib.bib20)] | ✓ | 82.2 | 50.7 | 66.5 |
| General-Purpose VLMs |
| Qwen2-VL[[5](https://arxiv.org/html/2603.02438#bib.bib5)] | ✓ | 96.7 | 84.7 | 90.7 |
| InternVL2-Pro[[12](https://arxiv.org/html/2603.02438#bib.bib12)] | ✓ | 95.1 | 83.3 | 89.2 |
| Molmo-72B[[15](https://arxiv.org/html/2603.02438#bib.bib15)] | ✓ | 93.5 | 81.9 | 87.7 |
| DeepSeek-VL2[[64](https://arxiv.org/html/2603.02438#bib.bib64)] | ✓ | 93.3 | 78.1 | 85.7 |
| LLaVA-One-Vision-8B[[29](https://arxiv.org/html/2603.02438#bib.bib29)] | ✓ | 94.8 | 78.4 | 86.6 |
| Gemini Pro 1.5[[55](https://arxiv.org/html/2603.02438#bib.bib55)] | × | 86.5 | 72.7 | 79.6 |
| Claude-3.7 Sonnet[[3](https://arxiv.org/html/2603.02438#bib.bib3)] | × | 94.1 | 65.5 | 79.8 |
| Qwen-VL-Max (single)[[4](https://arxiv.org/html/2603.02438#bib.bib4)] | × | 93.1 | 73.4 | 83.3 |
| GPT-4 Turbo + Textract[[1](https://arxiv.org/html/2603.02438#bib.bib1)] | × | 87.4 | 71.9 | 79.7 |
| GPT-4o[[1](https://arxiv.org/html/2603.02438#bib.bib1)] | × | 93.0 | 82.1 | 87.6 |
| Reasoning-Enhanced OS Models |
| MiMo-VL-7B-RL[[65](https://arxiv.org/html/2603.02438#bib.bib65)] | ✓ | 95.0 | 88.1 | 91.6 |
| VideoLLaMA3-7B[[70](https://arxiv.org/html/2603.02438#bib.bib70)] | ✓ | 95.0 | 78.9 | 87.0 |
| Gemma-3-27B-IT[[56](https://arxiv.org/html/2603.02438#bib.bib56)] | ✓ | 86.6 | 70.6 | 78.6 |
| Baseline VLMs (Single-Model) |
| Qwen2.5-VL-7B-Instruct[[47](https://arxiv.org/html/2603.02438#bib.bib47)] | ✓ | 95.7 | 77.7 | 86.7 |
| Qwen3VL-4B-Instruct[[49](https://arxiv.org/html/2603.02438#bib.bib49)] | ✓ | 95.3 | 80.3 | 87.8 |
| Qwen3VL-8B-Instruct[[50](https://arxiv.org/html/2603.02438#bib.bib50)] | ✓ | 96.1 | 83.1 | 89.6 |
| ORCA (Multi-Agent Framework) |
| ORCA (Qwen2.5-VL-7B) | ✓ | 96.4 (+0.7) | 86.9 (+9.2) | 91.7 (+5.0) |
| ORCA (Qwen3VL-4B) | ✓ | 96.0 (+0.7) | 85.4 (+5.1) | 90.7 (+2.9) |
| ORCA (Qwen3VL-8B) | ✓ | 97.2(+1.1) | 88.0(+4.9) | 92.6(+3.0) |
| Relative Improvements |
| Average Gain | – | +0.8% | +6.4% | +3.6% |

Performance on DocVQA Tasks. As shown in Table[1](https://arxiv.org/html/2603.02438#S4.T1 "Table 1 ‣ 4.2 Main Results ‣ 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering"), we evaluate across two document visual question answering benchmarks: DocVQA[[39](https://arxiv.org/html/2603.02438#bib.bib39)] and InfographicVQA[[40](https://arxiv.org/html/2603.02438#bib.bib40)]. We categorize the models into five groups: (1) Document Understanding Models, designed for document-centric tasks; (2) General-Purpose VLMs, vision-language models for broad multimodal tasks including both open-source and closed-source variants; (3) Reasoning-Enhanced Open-Source Models, approaches that incorporate explicit reasoning capabilities; (4) Baseline VLMs (Single-Model), the base models upon which our framework builds; and (5) ORCA (Multi-Agent Framework), our proposed models.

As can be seen, the DocVQA benchmark has become highly competitive, with several models achieving strong results. We begin by document understanding models. These models perform well on single-page DocVQA tasks, where answers are simply extractive. However, they struggle on the InfographicVQA benchmark, which requires higher-level reasoning, cross-modal grounding, and complex layout understanding. Next, for General-purpose VLMs, they demonstrate much stronger generalization capabilities. Models such as Qwen2-VL[[5](https://arxiv.org/html/2603.02438#bib.bib5)] and InternVL2-Pro[[12](https://arxiv.org/html/2603.02438#bib.bib12)] exhibit remarkable performance gains across both benchmarks. These models can be further enhanced when equipped with explicit reasoning mechanisms, as shown by recent reasoning-oriented open-source systems such as MiMo-VL-7B-RL[[65](https://arxiv.org/html/2603.02438#bib.bib65)], which achieve notable improvements, particularly on InfographicVQA. Finally, it is clear that our proposed ORCA model achieves substantial improvements on both benchmarks. On DocVQA, our framework attains the best results among all compared approaches, delivering consistent yet modest average gains of +0.8%. This corresponds to a 28.2% relative error reduction (3.9%→\rightarrow 2.8%), reflecting substantial progress in a low-error regime. On InfographicVQA, we observe a significant improvement of +6.4% on average. Notably, InfographicVQA demands sophisticated integration of textual and visual information across intricate infographic layouts. This validates our hypothesis that orchestrated multi-agent collaboration with specialized reasoning capabilities excels in complex scenarios where single models often struggle.

Table 2: OCRBench-v2 performance comparison between baseline VLMs and ORCA across model scales. The results marked with * are not reproduced, they are given by the papers’ authors.

| Model | Rec | Ref | Spo | Ext | Par | Cal | Und | Rea | Avg |
| --- |
| Open-source LMMs |
| LLaVA-Next-8B* [[37](https://arxiv.org/html/2603.02438#bib.bib37)] | 41.3 | 18.8 | 0 | 49.5 | 21.2 | 17.3 | 55.2 | 48.9 | 31.5 |
| LLaVA-OV-7B* [[29](https://arxiv.org/html/2603.02438#bib.bib29)] | 46.0 | 20.8 | 0.1 | 58.3 | 25.3 | 23.3 | 64.4 | 53.0 | 36.4 |
| TextMonkey* [[32](https://arxiv.org/html/2603.02438#bib.bib32)] | 39.1 | 0.7 | 0 | 19.0 | 12.2 | 19.0 | 61.1 | 40.2 | 23.9 |
| Molmo-7B* [[15](https://arxiv.org/html/2603.02438#bib.bib15)] | 52.4 | 21.3 | 0.1 | 45.5 | 7.6 | 28.5 | 65.3 | 55.0 | 34.5 |
| Cambrian-34B* [[59](https://arxiv.org/html/2603.02438#bib.bib59)] | 45.3 | 21.5 | 0 | 53.6 | 19.2 | 19.5 | 63.5 | 55.5 | 34.7 |
| Pixtral-12B* [[2](https://arxiv.org/html/2603.02438#bib.bib2)] | 48.9 | 21.6 | 0 | 66.3 | 35.5 | 29.8 | 66.9 | 53.7 | 40.3 |
| Nemotron Nano V2 VL* [[42](https://arxiv.org/html/2603.02438#bib.bib42)] | 67.6 | 54.5 | 36.2 | 92.0 | 26.6 | 80.4 | 75.5 | 57.0 | 61.2 |
| Llama Nemotron VL 8B* [[41](https://arxiv.org/html/2603.02438#bib.bib41)] | 70.2 | 69.1 | 61.8 | 81.4 | 39.2 | 31.9 | 73.1 | 54.7 | 60.2 |
| InternVL-3-14B* [[22](https://arxiv.org/html/2603.02438#bib.bib22)] | 67.3 | 36.9 | 11.2 | 89.0 | 38.4 | 38.4 | 79.2 | 60.5 | 52.6 |
| Qwen2.5-VL-7B* [[47](https://arxiv.org/html/2603.02438#bib.bib47)] | 68.8 | 25.7 | 1.2 | 80.2 | 30.4 | 38.2 | 73.2 | 56.2 | 46.7 |
| Qwen3-Omni-A3B* [[46](https://arxiv.org/html/2603.02438#bib.bib46)] | 72.3 | 62.0 | 45.6 | 93.5 | 20.8 | 67.0 | 74.1 | 55.3 | 61.3 |
| Qwen3-VL-4B-Instruct* [[49](https://arxiv.org/html/2603.02438#bib.bib49)] | - | - | - | - | - | - | - | - | 63.7 |
| Qwen3-VL-8B-Instruct* [[50](https://arxiv.org/html/2603.02438#bib.bib50)] | - | - | - | - | - | - | - | - | 65.4 |
| ORCA (Multi-Agent Framework) |
| ORCA (Qwen2.5-VL-7B) | 72.3 | 28.6 | 3.4 | 84.7 | 34.2 | 41.9 | 75.8 | 61.5 | 50.3 (+3.6) |
| ORCA (Qwen3VL-4B) | 81.1 | 48.2 | 24.3 | 89.3 | 60.1 | 57.2 | 86.6 | 71.6 | 64.8 (+1.1) |
| ORCA (Qwen3VL-8B) | 83.2 | 51.3 | 28.4 | 89.8 | 62.3 | 59.4 | 88.7 | 73.7 | 67.1(+1.7) |

Performance on OCRBench-v2 Multi-Task. We further evaluate the models on OCRBench-v2 across eight critical OCR subtasks: Recognition (Rec), Referring (Ref), Spotting (Spo), Extraction (Ext), Parsing (Par), Calculation (Cal), Understanding (Und), and Reasoning (Rea). The results are presented in Table[2](https://arxiv.org/html/2603.02438#S4.T2 "Table 2 ‣ 4.2 Main Results ‣ 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering"). As can be seen, ORCA achieves consistent improvements across all model scales, with gains inversely correlated to model capacity: +3.6 points for Qwen2.5-VL-7B versus +1.7 for Qwen3VL-8B. This suggests that our approach of multi-agent collaboration provides greater benefits for smaller models by compensating for individual capacity limitations through specialized agent expertise. Notably, our framework demonstrates especially strong improvements in challenging tasks such as Understanding, Reasoning, and Spotting for ORCA over its single-model baseline, directly reflecting ORCA’s design objectives of structured reasoning and modality-specific specialization. These gains are concentrated precisely where specialized agent collaboration is most effective. ORCA with Qwen3VL-8B configuration reaches 67.1% average performance, demonstrating that our framework

addresses the inherent complexity of document understanding tasks that challenge monolithic vision-language models.

Generalization to Chart-Centric Reasoning. To evaluate generalization beyond document VQA, we additionally assess ORCA on ChartQA, a benchmark requiring visual and numerical reasoning over charts and figures. ORCA (Qwen3VL-8B) achieves 90.1%, improving over the single-model baseline of 85.7% (+4.4%). This improvement is consistent with ORCA’s design, as chart understanding benefits directly from the dedicated figure/diagram agent and the multi-turn debate mechanism for resolving numerical ambiguities. Beyond document-centric tasks, ORCA also improves performance on VQAv2 by +4.7% over the single-model baseline, suggesting that the orchestrated reasoning pipeline extends to broader vision-language question answering scenarios.

### 4.3 Inference Latency and Cost Analysis

Table[3](https://arxiv.org/html/2603.02438#S4.T3 "Table 3 ‣ 4.3 Inference Latency and Cost Analysis ‣ 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering") reports inference latency measured on 4×\times NVIDIA L4 GPUs using vLLM acceleration. ORCA introduces moderate overhead relative to single-model baselines, mitigated by three key optimizations: (1) vLLM inference acceleration providing approximately 5×\times speedup over standard sequential execution; (2) conditional execution that bypasses the debate stages when the thinker and expert agents agree, which occurs in 77% of cases; and (3) backbone reuse across stages, avoiding redundant model loading. In the early-termination regime (Stages 1–2 only), ORCA incurs approximately 4–6×\times overhead while delivering +2–3% improvement on complex tasks, making it suitable for latency-sensitive deployments. Full pipeline execution is recommended for accuracy-critical applications such as processing higly sensitive documents, where the accuracy-latency trade-off is favorable. Notably, scaling monolithic models beyond 100B parameters achieves comparable accuracy at substantially higher memory and deployment cost, making ORCA a compute-efficient alternative that improves reasoning quality through structured orchestration rather than brute-force parameter scaling. Furthermore, intermediate reasoning traces and routing decisions generated during inference can be logged and reused as supervision signals for training on downstream tasks, further amortizing the compute cost over time.

Table 3: Inference latency comparison on 4×\times NVIDIA L4 GPUs with vLLM optimization. Early-termination applies when thinker and expert agents agree, bypassing Stages 3–5 (77% of cases).

| Configuration | Latency (s) | DocVQA | InfoVQA |
| --- | --- | --- | --- |
| Baseline (Qwen3VL-8B) | 0.3–0.8 | 96.1 | 83.1 |
| ORCA Early-Termination | 2.9–4.5 | 96.7 | 86.9 |
| ORCA Full Pipeline | 9.6–13.1 | 97.2 | 88.0 |

### 4.4 Ablation Studies

We systematically evaluate each component’s contribution using the three benchmarks. We start by ablating individual stages in Table[4](https://arxiv.org/html/2603.02438#S4.T4 "Table 4 ‣ 4.4 Ablation Studies ‣ 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering") and report the following findings.

Table 4: Stage-wise ablation study on Qwen3VL-8B. Each row isolates the impact of removing a specific stage from the complete ORCA framework.

| Configuration | DocVQA | InfoVQA | OCRBench-v2 |
| --- |
| ORCA (Full) | 97.2 | 88.0 | 67.1 |
| Individual Component Ablations |
| w/o Stage 1 (Reasoning) | 96.5 (-0.7) | 84.9 (-3.1) | 66.1 (-1.0) |
| w/o Stage 2 (Collaborative Agents) | 96.3 (-0.9) | 84.1 (-3.9) | 66.0 (-1.1) |
| w/o Stage 3 (Stress Testing) | 97.0 (-0.2) | 87.5 (-0.5) | 66.9 (-0.2) |
| w/o Stage 4 (Multi-turn Debate) | 96.9 (-0.3) | 87.2 (-0.8) | 66.7 (-0.4) |
| w/o Stage 5 (Answer Refinement) | 97.1 (-0.1) | 87.9 (-0.1) | 67.0 (-0.1) |
| Cumulative Ablations |
| w/o Stages 4+5 | 96.8 (-0.4) | 87.6 (-0.4) | 66.9 (-0.2) |
| w/o Stages 3+4+5 (All Verification) | 96.7 (-0.5) | 86.9 (-1.1) | 66.3 (-0.8) |
| w/o Stages 2–5 | 96.2 (-1.0) | 84.1 (-3.9) | 65.6 (-1.5) |
| Baseline (Single Model) | 96.1 | 83.1 | 65.4 |

Core Components Drive Performance. Stages 1 (Reasoning) and 2 (Collaborative Agents) constitute the framework’s core components. Removing Stage 1 causes substantial drops across all benchmarks, demonstrating that strategic reasoning paths provide essential task decomposition. Stage 2 ablation shows even larger impact, with performance approaching baseline levels, confirming that multi-agent specialization represents our core architectural innovation.

Verification Stages Provide Targeted Refinement. Stages 3–5 contribute incrementally with smaller magnitude improvements. Stage 3 (Stress Testing) yields 0.2–0.5 point gains, Stage 4 (Multi-turn Debate) adds 0.3–0.8 points, while Stage 5’s minimal impact primarily addresses formatting inconsistencies rather than semantic accuracy. Cumulatively removing all verification stages causes slight degradation, indicating complementary rather than foundational roles.

![Image 4: Refer to caption](https://arxiv.org/html/2603.02438v1/x3.png)

Figure 3: A case study demonstrating ORCA’s multi-agent reasoning pipeline on a complex visual document question. Better viewed with zoom. Question: What publication detail accompanies the Genealogical Society entry?. GT answer: “GSU, 1977”.

Selective Activation Explains Modest Verification Impact. Following the previous findings, we investigate the cause of the complementary roles of stages 3 and 4. An analysis of 500 randomly sampled test instances from the validation sets of the single-page DocVQA and InfographicsVQA datasets reveals that Stage 3 activates only when expert and thinker agents disagree (a E≠a T a_{E}\neq a_{T}), occurring in 23.4% of cases. Among these, 35.7% fail stress testing and proceed to Stage 4 debate. Consequently, multi-turn debate is engaged in merely 8.3% of instances. This low activation frequency accounts for modest aggregate contributions, though these stages potentially enhance reliability in ambiguous edge cases where disagreement signals genuine uncertainty.

Table 5: Reasoning path masking ablation on Qwen3VL-8B. Masking prevents confirmation bias while preserving strategic guidance.

| Component Variation | DocVQA | InfoVQA | OCRBench-v2 |
| --- | --- | --- | --- |
| No reasoning masking | 96.5 | 87.6 | 66.4 |
| Reasoning masking (Full) | 97.2 | 88.0 | 67.1 |
| Performance gain | +0.7 | +0.4 | +0.7 |

Answer Masking Mitigates Confirmation Bias. Finally, we investigate the role of the thinker’s answer masking. Table[5](https://arxiv.org/html/2603.02438#S4.T5 "Table 5 ‣ 4.4 Ablation Studies ‣ 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering") demonstrates that exposing the thinker’s answer to specialized agents causes consistent degradation across benchmarks. Without masking, later agents exhibit pronounced anchoring bias toward the thinker’s preliminary conclusion rather than conducting independent analysis. By masking the answer while preserving the reasoning path, we ensure agents concentrate on designated subtasks without premature influence.

### 4.5 Case Study

Figure[3](https://arxiv.org/html/2603.02438#S4.F3 "Fig. 3 ‣ 4.4 Ablation Studies ‣ 4 Experiments ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering") presents an example to demonstrate the workflow of ORCA. The visualization illustrates transparent reasoning, self-verification, confidence assessment, and attention to fine-grained details. The multi-agent pipeline systematically decomposes the problem, with each specialist contributing domain-specific expertise while maintaining interpretable intermediate outputs. Thus, providing several advantages over single-VLM approaches in terms of reasoning, robustness, and explainability. More details about executing ORCA on similar cases are available in Appendix[E](https://arxiv.org/html/2603.02438#A5 "Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering").

### 4.6 Error Analysis

We analyzed 100 incorrect predictions from ORCA on the Single-Page DocVQA and InfographicsVQA validation sets. Each error was traced to the first pipeline stage responsible for the failure. The main error sources are: (1) Reasoning errors (43%), where the thinker agent produces an incorrect reasoning path that misguides downstream agents; (2) Router errors (27%), caused by incorrect agent selection that omits relevant document elements or activates unsuitable specialists; (3) Agent coordination failures (18%), where early-stage errors propagate through sequential execution; and (4) Over-refinement (12%), in which verification stages over-analyze or override initially correct answers.

A key advantage of ORCA’s modular architecture is that individual components can be upgraded independently as stronger models become available, enabling continuous system improvement without architectural changes.

5 Conclusion
------------

We present ORCA, a multi-agent framework for Document Visual Question Answering that integrates explicit reasoning decomposition with specialized agent collaboration to address the limitations of single-model VLMs. Our modular architecture enables component upgrades as foundation models evolve. In future, to further improve ORCA,we plain to optimize the router via reinforcement learning with task-specific rewards to learn agent selection beyond the current supervised labels; orchestration ordering can be learned through policy gradients with intermediate answer quality as the state representation; and the debate mechanism can be refined via multi-agent PPO with adversarial rewards. We also plan to extend ORCA to multi-page document undersranding, introducing long-context routing and inter-page agentic reasoning.

Acknowledgements
----------------

The contribution of Mohamed Ali Souibgui to this work has been supported by Consolidated Research Group 2021 SGR 01559, by project PID2023-146426NB-100 funded by MCIU/AEI/10.13039/501100011033 and FSE+, and by the European Union’s Horizon Europe programme under grant agreement No 101070617 (Project ELSA)and No 101214398 (Project ELLIOT).

References
----------

*   Achiam et al. [2023] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_, 2023. 
*   Agrawal et al. [2024] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baudouin Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. Pixtral 12b. _arXiv preprint arXiv:2410.07073_, 2024. 
*   Anthropic [2025] Anthropic. Claude. [https://www.anthropic.com/claude](https://www.anthropic.com/claude), 2025. Accessed: 11 November 2025. 
*   Bai et al. [2023] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. _arXiv preprint arXiv:2308.12966_, 2023. 
*   Bai et al. [2025] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. _arXiv preprint arXiv:2502.13923_, 2025. 
*   Banerjee and Lavie [2005] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization_, pages 65–72, 2005. 
*   Banerjee et al. [2025] Sourav Banerjee, Ayushi Agarwal, and Saloni Singla. Llms will always hallucinate, and we need to live with this. In _Intelligent Systems Conference_, pages 624–648. Springer, 2025. 
*   Biten et al. [2019] Ali Furkan Biten, Rubèn Tito, Andrés Mafla, Lluis Gomez, Marçal Rusiñol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In _2019 International Conference on Document Analysis and Recognition (ICDAR)_, pages 1291–1296. IEEE, 2019. 
*   Chan et al. [2023] Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. Chateval: Towards better llm-based evaluators through multi-agent debate. _arXiv preprint arXiv:2308.07201_, 2023. 
*   Chen et al. [2024a] Jiaqi Chen, Zeyu Zhang, Chengcheng Xu, and Zhou Zhao. Multimodal large language models: A survey. _arXiv preprint arXiv:2405.07538_, 2024a. 
*   Chen et al. [2023] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. _arXiv preprint arXiv:2312.14238_, 2023. 
*   Chen et al. [2024b] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages 24185–24198, 2024b. 
*   Dai et al. [2023] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In _Advances in Neural Information Processing Systems_, pages 49250–49267, 2023. 
*   DeepSeek-AI [2025] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025. 
*   Deitke et al. [2024] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. _arXiv e-prints_, pages arXiv–2409, 2024. 
*   Du et al. [2023] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In _International Conference on Machine Learning_, pages 8633–8656. PMLR, 2023. 
*   Fu et al. [2025] Ling Fu, Zhebin Kuang, Jiajun Song, Mingxin Huang, Biao Yang, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu, Zhang Li, Guozhi Tang, Bin Shan, Chunhui Lin, Qi Liu, Binghong Wu, Hao Feng, Hao Liu, Can Huang, Jingqun Tang, Wei Chen, Lianwen Jin, Yuliang Liu, and Xiang Bai. Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. _arXiv preprint arXiv:2501.00321_, 2025. Version 2, revised 5 Jun 2025. 
*   Guo et al. [2025] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1.5-vl technical report. _arXiv preprint arXiv:2505.07062_, 2025. 
*   Hong et al. [2024] Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, et al. Metagpt: Meta programming for a multi-agent collaborative framework. In _International Conference on Learning Representations_, 2024. 
*   Hu et al. [2024] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. _arXiv preprint arXiv:2403.12895_, 2024. 
*   Huang et al. [2022] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In _Proceedings of the 30th ACM International Conference on Multimedia_, pages 4083–4091, 2022. 
*   InternVL Team [2024] InternVL Team. Internvl-3: Scaling up vision-language models. [https://internvl.github.io/](https://internvl.github.io/), 2024. Accessed: 2024-12-29. 
*   InternVL Team [2025] InternVL Team. Internvl3-8b-hf. [https://huggingface.co/OpenGVLab/InternVL3-8B-hf](https://huggingface.co/OpenGVLab/InternVL3-8B-hf), 2025. Accessed: 2025-01-01. 
*   Irving et al. [2018] Geoffrey Irving, Paul Christiano, and Dario Amodei. Ai safety via debate. _arXiv preprint arXiv:1805.00899_, 2018. 
*   Kannan et al. [2024] Sai Kannan, Homanga Bharadhwaj, Aishwarya Jain, and Dinesh Jayaraman. Smart: Scalable multi-agent real-time simulation via next-token prediction. _arXiv preprint arXiv:2405.15677_, 2024. 
*   Kim et al. [2022] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In _European Conference on Computer Vision_, pages 498–517. Springer, 2022. 
*   Kim et al. [2024] Yubin Kim, Chanwoo Cho, Hyewon Kim, Sik Song, and Edward Choi. Mdagents: An adaptive collaboration of llms for medical decision-making. _arXiv preprint arXiv:2404.15155_, 2024. 
*   Lamm and Keuper [2024] Bianca Lamm and Janis Keuper. Can visual language models replace ocr-based visual question answering pipelines in production? a case study in retail. _arXiv preprint arXiv:2408.15626_, 2024. 
*   Li et al. [2024a] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. _arXiv preprint arXiv:2408.03326_, 2024a. 
*   Li et al. [2023a] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for “mind” exploration of large language model society. In _Advances in Neural Information Processing Systems_, pages 51991–52008, 2023a. 
*   Li et al. [2023b] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International Conference on Machine Learning_, pages 19730–19742. PMLR, 2023b. 
*   Li et al. [2024b] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 26763–26773, 2024b. 
*   Li et al. [2025] Zhiwei Li, Xiaoqiang Wang, Shuo Zhang, and Yong Chen. Metal: Towards multi-agent large language models for legal case retrieval. _arXiv preprint arXiv:2501.09234_, 2025. 
*   Liang et al. [2023] Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, and Shuming Shi. Encouraging divergent thinking in large language models through multi-agent debate. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_, pages 9006–9021, 2023. 
*   Liu et al. [2023a] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 26296–26306, 2023a. 
*   Liu et al. [2023b] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _Advances in Neural Information Processing Systems_, pages 34892–34916, 2023b. 
*   Liu et al. [2024] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 
*   Luo et al. [2024] Masato Luo, Brian Sibue, Xin Chen, Huaxiu Tang, Simeng Huang, and Yiheng Feng. Layoutllm: Large language model instruction tuning for visually-rich document understanding. _arXiv preprint arXiv:2403.05252_, 2024. 
*   Mathew et al. [2021] Minesh Mathew, Dimosthenis Karatzas, and C.V. Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_, pages 2200–2209, 2021. 
*   Mathew et al. [2022] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Infographicvqa. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_, pages 1697–1706, 2022. 
*   NVIDIA [2024a] NVIDIA. Llama nemotron vl 8b. [https://developer.nvidia.com/](https://developer.nvidia.com/), 2024a. Accessed: 2024-12-29. 
*   NVIDIA [2024b] NVIDIA. Nemotron nano v2 vl. [https://developer.nvidia.com/](https://developer.nvidia.com/), 2024b. Accessed: 2024-12-29. 
*   OpenAI [2024] OpenAI. Learning to reason with llms. Technical report, OpenAI, 2024. Available at: [https://openai.com/index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/). 
*   Papineni et al. [2002] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of Annual Meeting of the Association for Computational Linguistics_, pages 311–318, 2002. 
*   Powalski et al. [2021] Rafał Powalski, Łukasz Borchmann, Dawid Jurkiewicz, Tomasz Dwojak, Michał Pietruszka, and Gabriela Pałka. Going full-tilt boogie on document understanding with text-image-layout transformer. In _International Conference on Document Analysis and Recognition_, pages 732–747. Springer, 2021. 
*   Qwen Team [2024] Qwen Team. Qwen3-omni: Multimodal large language model. [https://qwenlm.github.io/](https://qwenlm.github.io/), 2024. Accessed: 2024-12-29. 
*   Qwen Team [2025a] Qwen Team. Qwen2.5-vl-7b-instruct. [https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), 2025a. Accessed: 2025-01-01. 
*   Qwen Team [2025b] Qwen Team. Qwen3-1.7b. [https://huggingface.co/Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), 2025b. Accessed: 2025-01-01. 
*   Qwen Team [2025c] Qwen Team. Qwen3-vl-4b-instruct. [https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), 2025c. Accessed: 2025-01-01. 
*   Qwen Team [2025d] Qwen Team. Qwen3-vl-8b-instruct. [https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), 2025d. Accessed: 2025-01-01. 
*   Qwen Team [2025e] Qwen Team. Qwen3-vl: The most powerful vision-language model in the qwen series. [https://github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), 2025e. Accessed: 2025-11-08. 
*   Shen et al. [2023] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. _Advances in Neural Information Processing Systems_, 36:40075–40093, 2023. 
*   Shinn et al. [2023] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. _Advances in Neural Information Processing Systems_, 36:8634–8652, 2023. 
*   Su et al. [2020] Tong Su, Yifei Zhong, Kai Zhang, Qiyuan Chen, Xiang Li, and Stephen Lin. Adapting document images to display constraints. In _Proceedings of the 28th ACM International Conference on Multimedia_, pages 1617–1625, 2020. 
*   Team et al. [2024] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_, 2024. 
*   Team et al. [2025a] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. _arXiv preprint arXiv:2503.19786_, 2025a. 
*   Team et al. [2025b] V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025b. 
*   Tito et al. [2023] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for visual question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 10495–10504, 2023. 
*   Tong et al. [2024] Shengbang Tong, Ellis L Brown II, Penghao Wu, Sanghyun Woo, Arjun Jyoti IYER, Sai Charith Akula, Shusheng Yang, Jihan Yang, Megha Middepogu, Zichen Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In _Advances in Neural Information Processing Systems_, 2024. 
*   Wang et al. [2023] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _International Conference on Learning Representations_, 2023. 
*   Wei et al. [2022] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. _arXiv preprint arXiv:2201.11903_, 2022. Version 6, revised 10 Jan 2023. 
*   Wu et al. [2023a] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. _arXiv preprint arXiv:2303.04671_, 2023a. 
*   Wu et al. [2023b] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversation. _arXiv preprint arXiv:2308.08155_, 2023b. 
*   Wu et al. [2024] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yuxiang You, Kai Dong, Xingkai Yu, Haowei Zhang, Liang Zhao, Yisong Wang, and Chong Ruan. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding, 2024. 
*   Xiaomi LLM-Core Team [2025] Xiaomi LLM-Core Team. Mimo-vl technical report. [https://github.com/XiaomiMiMo/MiMo-VL](https://github.com/XiaomiMiMo/MiMo-VL), 2025. 
*   Xu et al. [2025] Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 2087–2098, 2025. 
*   Xu et al. [2020] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. Layoutlm: Pre-training of text and layout for document image understanding. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_, pages 1192–1200, 2020. 
*   Xu et al. [2021] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, et al. Layoutlmv2: Multi-modal pre-training for visually-rich document understanding. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_, pages 2579–2591, 2021. 
*   Yao et al. [2023] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In _International Conference on Learning Representations_, 2023. 
*   Zhang et al. [2025] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. _arXiv preprint arXiv:2501.13106_, 2025. 
*   Zhang et al. [2024a] Jinxu Zhang, Qiyuan Fan, and Yu Zhang. Read and think: An efficient step-wise multimodal language model for document understanding and reasoning. _arXiv preprint arXiv:2403.00816_, 2024a. 
*   Zhang et al. [2024b] Ruohong Zhang, Bowen Zhang, Yanghao Li, Haotian Zhang, Zhiqing Sun, Zhe Gan, Yinfei Yang, Ruoming Pang, and Yiming Yang. Improve vision language model chain-of-thought reasoning. _arXiv preprint arXiv:2410.16198_, 2024b. 
*   Zhong et al. [2020] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: data, model, and evaluation. In _Proceedings of European Conference on Computer Vision_, pages 564–580. Springer, 2020. 
*   Zhu et al. [2025] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. _arXiv preprint arXiv:2504.10479_, 2025. 

\thetitle

Supplementary Material

Appendix A Implementation and Training Details
----------------------------------------------

### A.1 Datasets.

We evaluate our approach on three challenging document understanding benchmarks: (1) Single-Page DocVQA[[39](https://arxiv.org/html/2603.02438#bib.bib39)]: a standard benchmark for single-page document question answering covering diverse document types; (2) InfographicsVQA[[40](https://arxiv.org/html/2603.02438#bib.bib40)]: a dataset requiring integration of textual and visual cues to answer questions about infographics; and (3) OCRBench-v2 (en)[[17](https://arxiv.org/html/2603.02438#bib.bib17)]: a comprehensive benchmark for OCR and document understanding. These datasets span a variety of document structures, including forms, tables, charts, and mixed content, ensuring a comprehensive evaluation.

### A.2 Baselines.

We compare our method against state-of-the-art vision-language models (VLMs), including Qwen2.5-VL-7B-Instruct[[5](https://arxiv.org/html/2603.02438#bib.bib5)], Qwen3VL-4B-Instruct, and Qwen3VL-8B-Instruct[[51](https://arxiv.org/html/2603.02438#bib.bib51)]. These baselines represent single-model systems without multi-agent collaboration or explicit reasoning decomposition.

### A.3 Evaluation Metrics.

Following standard evaluation protocols[[39](https://arxiv.org/html/2603.02438#bib.bib39), [40](https://arxiv.org/html/2603.02438#bib.bib40)], we report ANLS scores for Single-Page DocVQA and InfographicsVQA. For OCRBench-v2, we employ its official multi-dimensional evaluation suite with six task-specific metrics. The final score represents the average across all dimensions:

*   •Parsing: TEDS[[73](https://arxiv.org/html/2603.02438#bib.bib73)] for structural similarity in format conversion. 
*   •Localization: IoU for spatial overlap of text regions. 
*   •Extraction: F1 score for relation and key information extraction. 
*   •Long Reading: BLEU[[44](https://arxiv.org/html/2603.02438#bib.bib44)], METEOR[[6](https://arxiv.org/html/2603.02438#bib.bib6)], F1, and edit distance for long-form comprehension. 
*   •Counting: Normalized L1 distance for text instance enumeration. 
*   •Basic VQA: Exact match for multiple-choice; substring matching (≤5\leq 5 words) or ANLS for open-ended questions. 

Appendix B Router
-----------------

In this Section, we provide further details on the router agent used in ORCA.

### B.1 Training Data and Augmentation

We train the router on the Single-Page Document VQA dataset with ground-truth agent annotations. To enhance model robustness and generalization, we apply some data augmentation techniques:

*   •Back-translation: Questions are translated through intermediate languages (French and Chinese) and then back to English, generating paraphrased variants while preserving semantic meaning 
*   •Document perturbations: Minor transformations to document images (rotation, contrast adjustment) simulate real-world scanning variations 

To ensure robust evaluation and prevent data leakage in the multi-label setting, we employ Multilabel Stratified K-Fold cross-validation with n splits=8 n_{\text{splits}}=8. This stratification strategy preserves the distribution of label combinations across folds, which is critical given that some agent combinations are significantly rarer than others. We train the router on seven folds and validate on the remaining fold.

### B.2 Model Architecture and Optimization

We employ Qwen2.5-VL-7B as the base architecture for A route A_{\text{route}}, fine-tuned on our augmented dataset. To optimize training efficiency for our English-focused benchmark evaluation, we apply several key techniques:

*   •

Vocabulary Shrinking. We reduce the tokenizer vocabulary by identifying and removing tokens unused in our training corpus. This process:

    *   –Analyzes the actual token distribution in our DocVQA datasets 
    *   –Removes unused tokens while preserving special tokens and model configuration tokens 
    *   –Shrinks the embedding layers accordingly 

This vocabulary reduction yields substantial benefits: reduced memory footprint (enabling larger batch sizes), faster training convergence, and decreased inference latency—critical for real-time routing decisions. For English-centric benchmarks, this approach typically reduces vocabulary size with no loss in task performance.

*   •Efficient Training Infrastructure. We leverage Unsloth’s optimized training framework combined with Flash Attention 2 for memory-efficient attention computation. Flash Attention 2 reduces memory complexity from O​(N 2)O(N^{2}) to O​(N)O(N) for sequence length N N, enabling us to process high-resolution document images with longer context windows during training. 

### B.3 Turbo DFS Decoding for Multi-Label Prediction

Unlike standard classification approaches that apply a sigmoid threshold to output logits, we treat routing as a constrained generation task and employ Turbo DFS (Depth-First Search with score-guided pruning) for decoding. This choice addresses fundamental limitations of traditional multi-label decoding:

*   •Sampling-based methods introduce non-determinism and may miss valid label combinations across runs 
*   •Greedy decoding returns a single sequence, potentially missing alternative valid agent combinations 
*   •Beam search explores only a fixed number of sequences without explicit probability thresholds 

Turbo DFS offers several advantages for our multi-label routing task:

Algorithm Overview. Turbo DFS performs score-guided enumeration over token continuations, pruning branches whose cumulative negative log-likelihood exceeds a configurable threshold. Starting from the model’s output logits:

1.   1.Compute token-level negative log-likelihoods (NLL) after temperature scaling: NLL​(t)=−log⁡P​(t∣context)\text{NLL}(t)=-\log P(t\mid\text{context}) 
2.   2.For each candidate token t t, calculate cumulative score: s new=s prev+NLL​(t)s_{\text{new}}=s_{\text{prev}}+\text{NLL}(t) 
3.   3.Prune branches where s new>−log⁡(min_prob)s_{\text{new}}>-\log(\text{min\_prob}), with special handling for the greedy token (most probable continuation) 
4.   4.Recursively explore unpruned branches up to max_new_tokens depth 
5.   5.Return all valid sequences as ranked candidates with their cumulative scores 

Deterministic Multi-Label Extraction. Given the ranked candidate sequences from Turbo DFS, we employ a union strategy to extract the final agent activation set:

𝒜 active=⋃(s,τ)∈candidates e−s≥min_prob DecodeAgents​(τ)\mathcal{A}_{\text{active}}=\bigcup_{\begin{subarray}{c}(s,\tau)\in\text{candidates}\\ e^{-s}\geq\text{min\_prob}\end{subarray}}\text{DecodeAgents}(\tau)(14)

where s s is the cumulative score, τ\tau is the token sequence, and DecodeAgents​(τ)\text{DecodeAgents}(\tau) maps token sequences to agent identifiers by decoding tokens and parsing agent labels from the resulting text. This union approach ensures high recall: any agent appearing in a high-probability candidate sequence is included in the final routing decision. Hyperparameters. We configure Turbo DFS with:

*   •min_prob=0.02\text{min\_prob}=0.02 (accept sequences with probability ≥2%\geq 2\%) 
*   •max_new_tokens=3\text{max\_new\_tokens}=3 (agent labels are short) 
*   •temperature=0.9\text{temperature}=0.9 (slight smoothing of probability distribution) 

This decoding strategy provides deterministic, ranked agent selections with explicit confidence scores, enabling principled multi-label thresholding and supporting downstream reranking if needed.

Appendix C Algorithms
---------------------

Algorithm 1 Collaborative Agent Execution

1:Question q q, Document 𝒟\mathcal{D}, Reasoning path ℛ\mathcal{R}, Initial answer a T a_{T}, Agent dock {A 1,…,A 9}\{A_{1},\ldots,A_{9}\}

2:Expert answer a E a_{E}

3:𝐯←A route​(q,𝒟,ℛ)\mathbf{v}\leftarrow A_{\text{route}}(q,\mathcal{D},\mathcal{R})⊳\triangleright Activate agents

4:𝒜 active←{A i∣v i=1}\mathcal{A}_{\text{active}}\leftarrow\{A_{i}\mid v_{i}=1\}

5:σ←Orchestrate​(𝒜 active,ℛ,q,𝒟)\sigma\leftarrow\text{Orchestrate}(\mathcal{A}_{\text{active}},\mathcal{R},q,\mathcal{D})⊳\triangleright Determine order

6:a 0←∅a_{0}\leftarrow\emptyset⊳\triangleright Initialize

7:for i=1 i=1 to |σ|−1|\sigma|-1 do

8:a i←σ i​(q,𝒟,a i−1)a_{i}\leftarrow\sigma_{i}(q,\mathcal{D},a_{i-1})⊳\triangleright Sequential execution

9:end for

10:ℛ∗←MaskAnswer​(ℛ,a T,τ)\mathcal{R}^{*}\leftarrow\text{MaskAnswer}(\mathcal{R},a_{T},\tau)⊳\triangleright Mask reasoning

11:a E←σ n​(q,𝒟,a n−1,ℛ∗)a_{E}\leftarrow\sigma_{n}(q,\mathcal{D},a_{n-1},\mathcal{R}^{*})⊳\triangleright Final agent

12:return a E a_{E}

Algorithm 2 Stress Testing Session

1:Question q q, Document 𝒟\mathcal{D}, Expert answer a E a_{E}, Specialized agent σ n\sigma_{n}

2:Debate answer a D a_{D}, Proceed to Stage 4: flag comm\text{flag}_{\text{comm}}

3:flag comm←False\text{flag}_{\text{comm}}\leftarrow\text{False}

4:for t=1 t=1 to 2 2 do⊳\triangleright Two debate turns

5:q debate←A debate​(q,𝒟,a E)q_{\text{debate}}\leftarrow A_{\text{debate}}(q,\mathcal{D},a_{E})

6:(r debate,a E′)←σ n​(q debate,q,𝒟,a E)(r_{\text{debate}},a^{\prime}_{E})\leftarrow\sigma_{n}(q_{\text{debate}},q,\mathcal{D},a_{E})

7:d←A eval​(q debate,r debate,a E,a E′)d\leftarrow A_{\text{eval}}(q_{\text{debate}},r_{\text{debate}},a_{E},a^{\prime}_{E})

8:if d=fail d=\text{fail}then

9:flag comm←True\text{flag}_{\text{comm}}\leftarrow\text{True}

10:break

11:end if

12:end for

13:if flag comm=False\text{flag}_{\text{comm}}=\text{False}then

14:a D←a E a_{D}\leftarrow a_{E}⊳\triangleright Agent passed stress test

15:else

16:a D←None a_{D}\leftarrow\text{None}⊳\triangleright Proceed to Stage 4

17:end if

18:return a D a_{D}, flag comm\text{flag}_{\text{comm}}

Algorithm 3 Multi-turn Communication

1:Question q q, Document 𝒟\mathcal{D}, Expert answer a E a_{E}

2:Communication answer a C a_{C}

3:a alt←A anti​(q,𝒟,a E)a_{\text{alt}}\leftarrow A_{\text{anti}}(q,\mathcal{D},a_{E})

4:if a alt=a E a_{\text{alt}}=a_{E} or a E⊂a alt a_{E}\subset a_{\text{alt}}then

5:return a E a_{E}⊳\triangleright No alternative found

6:end if

7:summary(0)←∅\text{summary}^{(0)}\leftarrow\emptyset

8:transcript←[]\text{transcript}\leftarrow[\,]

9:for t=1 t=1 to 3 3 do⊳\triangleright Three-turn debate

10:arg anti(t)←A anti​(q,𝒟,a E,summary(t−1))\text{arg}_{\text{anti}}^{(t)}\leftarrow A_{\text{anti}}(q,\mathcal{D},a_{E},\text{summary}^{(t-1)})

11:arg thesis(t)←A thesis(q,𝒟,a E,\text{arg}_{\text{thesis}}^{(t)}\leftarrow A_{\text{thesis}}(q,\mathcal{D},a_{E},

12:arg anti(t)[REF, CRIT],summary(t−1))\text{arg}_{\text{anti}}^{(t)}[\text{REF, CRIT}],\text{summary}^{(t-1)})

13:(convinced,summary(t))←A judge​(arg thesis(t),arg anti(t))(\text{convinced},\text{summary}^{(t)})\leftarrow A_{\text{judge}}(\text{arg}_{\text{thesis}}^{(t)},\text{arg}_{\text{anti}}^{(t)})

14:transcript.append​(arg anti(t),arg thesis(t))\text{transcript}.\text{append}(\text{arg}_{\text{anti}}^{(t)},\text{arg}_{\text{thesis}}^{(t)})

15:if convinced≠None\text{convinced}\neq\text{None}then

16:a C←convinced.answer a_{C}\leftarrow\text{convinced.answer}

17:return a C a_{C}⊳\triangleright Early termination

18:end if

19:end for

20:a C←A judge.FinalDecision​(transcript)a_{C}\leftarrow A_{\text{judge}}.\text{FinalDecision}(\text{transcript})⊳\triangleright Linguistic analysis

21:return a C a_{C}

Appendix D Inference Latency and Cost Analysis
----------------------------------------------

### D.1 Optimization Details

Three optimizations reduce ORCA’s effective latency: (1) vLLM acceleration provides approximately 5×\times throughput improvement over naive Hugging Face inference via continuous batching and PagedAttention. (2) Conditional execution bypasses Stages 3 and 4 when the thinker and expert agents produce identical answers, occurring in 77% of test instances. (3) Backbone reuse shares model weights across agents of the same architecture, reducing GPU memory overhead and eliminating redundant model initialization.

### D.2 ORCA-Lite Configuration

For latency-sensitive scenarios, ORCA-Lite restricts the pipeline to Stages 1, 2 and 5 only, incurring approximately 4–7×\times the latency of a single-model baseline while delivering +2–3% improvement on complex reasoning tasks.

Table 6: ORCA-Lite vs. full pipeline accuracy–latency trade-off.

| Configuration | Latency | DocVQA | InfoVQA | OCRBench-v2 |
| --- | --- | --- | --- | --- |
| Single Model | 0.3–0.8s | 96.1 | 83.1 | 65.4 |
| ORCA-Lite | 3.2–5.3s | 96.8 | 87.0 | 66.4 |
| ORCA Full | 9.6–13.1s | 97.2 | 88.0 | 67.1 |

Appendix E Additional Results
-----------------------------

### E.1 Detailed DocVQA Performance Breakdown

Table[7](https://arxiv.org/html/2603.02438#A5.T7 "Table 7 ‣ E.2 Detailed Infographics VQA Performance Breakdown ‣ Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering") presents a comprehensive performance breakdown on the DocVQA benchmark, evaluating all open-source models across different document types and question categories. The analysis covers nine distinct categories: Figures/Diagrams, Forms, Tables/Lists, Layout, Free Text, Images/Photos, Handwriting, Yes/No questions, and Other question types. This granular evaluation provides insights into model capabilities across diverse document understanding tasks.

Our multi-agent framework, ORCA, demonstrates consistent improvements across all categories when compared to baseline open-source models. Notably, ORCA with Qwen3VL-8B achieves the highest overall score of 97.2%, with particularly strong performance on Yes/No questions (100%) and Tables/Lists (97.8%). The framework shows robust performance across challenging categories such as handwriting recognition (96.7%) and form understanding (98.2%), indicating its effectiveness in handling complex document layouts and varying text modalities.

### E.2 Detailed Infographics VQA Performance Breakdown

Table[8](https://arxiv.org/html/2603.02438#A5.T8 "Table 8 ‣ E.2 Detailed Infographics VQA Performance Breakdown ‣ Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering") provides an in-depth analysis of model performance on the Infographics VQA benchmark, which presents unique challenges due to the complex visual and textual information typical of infographics. The evaluation is structured across three dimensions: Answer Type (including image span, question span, multiple spans, and non-span answers), Evidence Source (table/list, textual, visual object, figure, and map), and Required Operations (comparison, arithmetic, and counting tasks). This multi-faceted categorization enables a comprehensive understanding of how models handle different aspects of infographic comprehension.

The results reveal that ORCA maintains strong performance across all three evaluation dimensions. With Qwen3VL-8B, our framework achieves an overall score of 88.0%, demonstrating particularly notable capabilities in visual object recognition (94.1%) and counting operations (91.4%). Our approach shows consistent improvements across answer types and evidence sources. The framework excels at multi-span answers (83.1%), a particularly challenging task requiring integration of information from multiple locations, and demonstrates solid performance on arithmetic operations (82.8%), indicating robust reasoning capabilities. These results underscore the effectiveness of our multi-agent approach in handling the diverse and complex reasoning requirements inherent in infographic understanding.

Table 7: Detailed performance breakdown on DocVQA benchmark for open-source models. Results are shown across different document types and question categories.

| Model | Fig/Diag | Form | Table/List | Layout | Free Text | Img/Photo | Handwr. | Yes/No | Others | Score |
| --- |
| Open-source models results |
| LayoutLMv2 LARGE | 65.7 | 89.5 | 87.7 | 87.9 | 87.1 | 72.9 | 67.3 | 55.2 | 81.0 | 86.7 |
| Qwen2-VL | 92.1 | 98.2 | 97.0 | 96.8 | 96.2 | 91.4 | 94.4 | 96.6 | 95.4 | 96.7 |
| InternVL2-Pro | 88.9 | 97.1 | 94.9 | 95.8 | 94.5 | 89.1 | 92.8 | 96.6 | 94.1 | 95.1 |
| Molmo-72B | 88.2 | 95.5 | 93.9 | 94.1 | 91.0 | 86.9 | 92.0 | 92.0 | 92.3 | 93.5 |
| DeepSeek-VL2 | 88.5 | 95.8 | 93.6 | 93.1 | 92.1 | 86.9 | 89.9 | 89.7 | 90.1 | 93.3 |
| LLaVA-One-Vision-8B | 90.0 | 96.7 | 95.3 | 95.3 | 92.7 | 85.1 | 92.1 | 93.1 | 94.4 | 94.8 |
| MiMo-VL-7B-RL | 91.6 | 97.1 | 96.6 | 93.9 | 93.4 | 86.0 | 94.6 | 95.4 | 92.9 | 95.0 |
| VideoLLaMA3-7B | 88.4 | 96.9 | 95.0 | 95.3 | 94.3 | 88.4 | 92.9 | 93.1 | 93.1 | 95.0 |
| ORCA (Multi-Agent Framework) |
| ORCA (Qwen2.5-VL-7B) | 91.8 | 97.8 | 97.2 | 96.9 | 95.2 | 91.0 | 95.8 | 96.6 | 95.4 | 96.4 |
| ORCA (Qwen3VL-4B) | 91.2 | 97.4 | 96.8 | 96.4 | 94.6 | 90.2 | 95.2 | 96.6 | 94.7 | 96.0 |
| ORCA (Qwen3VL-8B) | 93.2 | 98.2 | 97.8 | 97.6 | 95.6 | 91.4 | 96.7 | 100.0 | 96.6 | 97.2 |

Table 8: Detailed performance breakdown on Infographics VQA benchmark for open-source models. Results are categorized by answer type, evidence source, and required operations.

|  |  | Answer type | Evidence | Operation |
| --- |
| Method | Score | Image span | Question span | Multiple spans | Non span | Table/List | Textual | Visual object | Figure | Map | Comparison | Arithmetic | Counting |
| Open-source models results |
| LayoutLMv2 LARGE | 28.3 | 34.3 | 27.6 | 6.4 | 11.1 | 24.5 | 38.6 | 14.4 | 26.0 | 31.1 | 19.0 | 11.3 | 11.6 |
| Qwen2-VL | 84.7 | 87.4 | 87.1 | 77.8 | 74.2 | 86.0 | 94.3 | 78.3 | 81.7 | 75.9 | 73.0 | 89.8 | 57.9 |
| InternVL2-Pro | 83.3 | 86.8 | 89.3 | 73.5 | 69.7 | 83.4 | 92.6 | 77.6 | 80.9 | 71.9 | 73.0 | 85.8 | 53.7 |
| Molmo-72B | 81.9 | 85.1 | 88.3 | 68.2 | 70.4 | 81.8 | 91.4 | 80.6 | 79.5 | 69.6 | 70.5 | 81.9 | 59.3 |
| DeepSeek-VL2 | 78.1 | 81.9 | 80.1 | 69.9 | 63.6 | 79.4 | 90.4 | 73.7 | 74.3 | 63.3 | 62.1 | 72.8 | 53.3 |
| LLaVA-One-Vision-8B | 78.4 | 82.2 | 83.1 | 64.6 | 65.0 | 79.6 | 89.9 | 70.6 | 74.7 | 62.6 | 62.7 | 75.8 | 54.5 |
| MiMo-VL-7B-RL | 88.1 | 90.1 | 89.5 | 84.5 | 80.9 | 90.1 | 93.3 | 83.7 | 85.8 | 73.0 | 82.6 | 89.1 | 74.4 |
| VideoLLaMA3-7B | 78.9 | 82.7 | 83.6 | 68.5 | 64.5 | 79.4 | 91.7 | 74.5 | 75.0 | 66.6 | 64.1 | 77.9 | 51.8 |
| ORCA (Multi-Agent Framework) |
| ORCA (Qwen2.5-VL-7B) | 86.9 | 89.0 | 90.3 | 77.7 | 75.4 | 81.6 | 87.8 | 81.5 | 84.3 | 69.4 | 78.3 | 89.8 | 59.8 |
| ORCA (Qwen3VL-4B) | 85.4 | 87.4 | 88.7 | 74.1 | 74.5 | 80.4 | 86.3 | 79.4 | 82.9 | 64.1 | 77.0 | 88.6 | 57.9 |
| ORCA (Qwen3VL-8B) | 88.0 | 90.1 | 91.4 | 83.1 | 79.5 | 88.9 | 94.1 | 84.3 | 85.8 | 73.5 | 82.8 | 91.4 | 68.4 |

### E.3 Prompt Settings

![Image 5: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x4.png)

![Image 6: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x5.png)![Image 7: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x6.png)

![Image 8: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x7.png)

![Image 9: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x8.png)

![Image 10: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x9.png)

![Image 11: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x10.png)

![Image 12: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x11.png)

![Image 13: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x12.png)

![Image 14: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x13.png)

![Image 15: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x14.png)

![Image 16: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x15.png)

![Image 17: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x16.png)

![Image 18: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x17.png)

![Image 19: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x18.png)

![Image 20: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x19.png)

![Image 21: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x20.png)

![Image 22: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x21.png)

![Image 23: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x22.png)

![Image 24: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x23.png)

![Image 25: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x24.png)

![Image 26: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x25.png)

![Image 27: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x26.png)

![Image 28: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x27.png)

![Image 29: [Uncaptioned image]](https://arxiv.org/html/2603.02438v1/x28.png)

### E.4 Experiments on different model backbones in ORCA

To evaluate the effectiveness and generalizability of our multi-agent framework across different vision-language model architectures, we conduct comprehensive experiments using three distinct backbone models: Qwen2.5-VL-7B, Qwen3VL-4B, and Qwen3VL-8B. These experiments demonstrate the framework’s ability to consistently improve performance regardless of the underlying model capacity and architecture.

Overall Performance Trends. As shown in Tables[7](https://arxiv.org/html/2603.02438#A5.T7 "Table 7 ‣ E.2 Detailed Infographics VQA Performance Breakdown ‣ Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering") and[8](https://arxiv.org/html/2603.02438#A5.T8 "Table 8 ‣ E.2 Detailed Infographics VQA Performance Breakdown ‣ Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering"), ORCA achieves substantial improvements across all tested backbones. On DocVQA, the framework boosts performance from 96.0% (Qwen3VL-4B) to 97.2% (Qwen3VL-8B), representing a 1.2% absolute improvement. Similarly, on Infographics VQA, we observe a consistent scaling pattern with scores of 85.4%, 86.9%, and 88.0% for the 4B, 7B, and 8B parameter models respectively. This trend suggests that our multi-agent architecture effectively leverages increased model capacity while maintaining robust performance even with smaller backbones.

Model Scaling Behavior. The results reveal interesting scaling characteristics across different backbone sizes. While Qwen3VL-8B achieves the highest overall scores, the performance gap between the 4B and 8B variants is relatively modest (1.2% on DocVQA, 2.6% on Infographics VQA), indicating that ORCA maintains high effectiveness even with resource-constrained models. Notably, the Qwen2.5-VL-7B backbone, despite having fewer parameters than the 8B variant, achieves competitive results (96.4% on DocVQA, 86.9% on Infographics VQA), demonstrating the framework’s ability to extract strong performance from different architectural designs.

These experiments validate that ORCA is architecture-agnostic and can consistently enhance document understanding capabilities across diverse backbone models. The framework’s ability to maintain strong performance with smaller models while scaling effectively with larger variants.

### E.5 Additional case studies

We present two qualitative case studies that illustrate typical successes and failure modes of ORCA compared with baselines (Figures[4](https://arxiv.org/html/2603.02438#A5.F4 "Fig. 4 ‣ E.5 Additional case studies ‣ Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering") and[5](https://arxiv.org/html/2603.02438#A5.F5 "Fig. 5 ‣ E.5 Additional case studies ‣ Appendix E Additional Results ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering")).

![Image 30: Refer to caption](https://arxiv.org/html/2603.02438v1/x29.png)

Figure 4: ORCA demonstrates robust multi-stage reasoning on a document containing ambiguous textual references and visually challenging OCR content. While baseline VLMs fail due to misidentification and shallow pattern matching, ORCA decomposes the task into OCR parsing, cell-level localization, cross-reference verification, and answer consistency checking. Through iterative agent collaboration and critical evidence consolidation, ORCA resolves ambiguity, corrects earlier misinterpretations, and converges on the correct entity with high confidence

![Image 31: Refer to caption](https://arxiv.org/html/2603.02438v1/x30.png)

Figure 5: ORCA successfully handles a structurally complex form where precise line indexing, noisy OCR text, and subtle vocabulary variations mislead baseline VLMs. By combining layout-aware processing, content-aware sequence reasoning, and downstream sanity validation, ORCA incrementally narrows the search space and suppresses earlier incorrect hypotheses. The multi-agent pipeline enables reliable disambiguation and robust extraction even under OCR artifacts and positional uncertainty.

Appendix F Extended Error Analysis
----------------------------------

### F.1 Failure Mode Breakdown

Table[9](https://arxiv.org/html/2603.02438#A6.T9 "Table 9 ‣ F.1 Failure Mode Breakdown ‣ Appendix F Extended Error Analysis ‣ ORCA: Orchestrated Reasoning with Collaborative Agents for Document Visual Question Answering") summarizes the failure modes observed across 100 incorrect predictions from ORCA (Qwen3VL-8B) on the Single-Page DocVQA and InfographicsVQA validation sets. Each error was traced to its originating stage.

Table 9: Error attribution by originating stage across 100 analyzed failure cases.

| Failure Mode | Proportion | Description |
| --- | --- | --- |
| Reasoning errors | 43% | Thinker agent generates incorrect reasoning path, misleading all subsequent agents |
| Router errors | 27% | Incorrect agent selection causes missing evidence or mismatched specialist |
| Agent coordination failures | 18% | Error propagation through sequential execution from early agents |
| Over-refinement | 12% | Verification stages introduce errors by over-analyzing initially correct answers |

### F.2 Error Propagation Analysis

Only 18% of failures involve cross-stage error propagation, while 70% originate from a single component (43% reasoning, 27% routing). This is partly by design: the stress testing and multi-turn debate stages generate new candidate answers rather than modifying existing ones, actively limiting rather than amplifying errors from earlier stages. The remaining 12% of over-refinement errors are concentrated in questions with short, ambiguous answers where the debate mechanism incorrectly identifies uncertainty.

### F.3 Failure Case Examples

Reasoning error: For questions involving indirect spatial references (e.g., “What is the value in the row above the highlighted cell?”), the thinker agent occasionally misidentifies spatial relationships, producing a flawed reasoning path that directs specialized agents to the wrong document region.

Router error: Questions involving handwritten annotations embedded within printed tables are sometimes misrouted exclusively to the OCR agent, missing the table agent’s structural extraction capability, resulting in partial answers that lack the necessary table context.

Over-refinement error: For yes/no questions where the expert answer is already correct, the antithesis agent occasionally generates a spurious alternative, initiating a debate that produces an incorrect final answer. Adding a confidence threshold for initiating debate on binary questions is a planned improvement.

### F.4 Implications for Future Work

Given that 43% of failures originate in the thinker agent, improving reasoning path generation represents the highest-leverage opportunity. Fine-tuning the thinker on document-specific reasoning traces is expected to yield the largest accuracy improvements. Router errors (27%) suggest the routing training set would benefit from more diverse annotation of edge cases involving mixed-modality content.

 Experimental support, please [view the build logs](https://arxiv.org/html/2603.02438v1/__stdout.txt) for errors. Generated by [L A T E xml![Image 32: [LOGO]](blob:http://localhost/70e087b9e50c3aa663763c3075b0d6c5)](https://math.nist.gov/~BMiller/LaTeXML/). 

Instructions for reporting errors
---------------------------------

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

*   Click the "Report Issue" () button, located in the page header.

**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

BETA

[](javascript:toggleReadingMode(); "Disable reading mode, show header and footer")

