Title: VideoAtlas: Navigating Long-Form Video in Logarithmic Compute

URL Source: https://arxiv.org/html/2603.17948

Markdown Content:
Mohamed Eltahir 1 Ali Habibullah 1 1 1 1 Equal Contribution Yazan Alshoibi 1 1 1 1 Equal Contribution Lama Ayash 1,2 Tanveer Hussain 3 2 2 2 Corresponding Author Naeemullah Khan 1 3 3 3 Principal Investigator (PI)

1 King Abdullah University of Science and Technology (KAUST), Thuwal, Saudi Arabia 

2 Department of Computer Science, King Khalid University (KKU), Abha, Saudi Arabia 

3 Department of Computer Science, Edge Hill University, Ormskirk, England 

{mohamed.hamid, ali.habibullah, yazen.shaebi, lama.ayash}@kaust.edu.sa

hussaint@edgehill.ac.uk, naeemullah.khan@kaust.edu.sa

###### Abstract

Extending language models to video introduces two challenges: _representation_, where existing methods rely on lossy approximations such as uniform sampling, and _long-context_, where caption- or agent-based pipelines collapse video into text and lose visual fidelity. To overcome this, we introduce VideoAtlas, a task-agnostic environment to represent video as a hierarchical grid that is simultaneously lossless, navigable, scalable, caption- and preprocessing-free. An overview of the video is available at a glance, and any region can be recursively zoomed into, with the same visual representation used uniformly for the video, intermediate investigations, and the agent’s memory, eliminating lossy text conversion end-to-end. This hierarchical structure ensures access depth grows only logarithmically with video length. For long-context, Recursive Language Models (RLMs) recently offered a powerful solution for long text, but extending them to visual domain requires a structured environment to recurse into, which VideoAtlas provides. VideoAtlas as a Markov Decision Process unlocks Video-RLM: a parallel Master-Worker architecture where a Master coordinates global exploration while Workers concurrently drill into assigned regions to accumulate lossless visual evidence. We demonstrate three key findings: (1)logarithmic compute growth with video duration, in contrast to the linear cost of baselines, further amplified by a 30-60% multimodal cache hit rate arising from the grid’s structural reuse. (2)environment budgeting, where bounding the maximum exploration depth provides a principled compute-accuracy hyperparameter. (3)emergent adaptive compute allocation that scales with question granularity. When scaling from 1-hour to 10-hour benchmarks, Video-RLM remains the most duration-robust method with minimal accuracy degradation while baselines degrade significantly, demonstrating that structured environment navigation is a viable and scalable paradigm for video understanding.

4 4 footnotetext: Code: [github.com/mohammad2012191/VideoAtlas](https://arxiv.org/html/2603.17948v1/github.com/mohammad2012191/VideoAtlas)
## 1 Introduction

Understanding long-form video requires locating sparse, task-relevant evidence within a massive temporal space: an hour video has 90,000 frames at 25 fps, yet the answer to a query often resides in a few seconds. When a movies editor faces the same challenge, the solution is well-established: a _contact sheet_ (a single composite image showing sampled shots) to identify promising regions at a glance before zooming into only those clips. This loop of _overview, identify, zoom_ is the key to efficient visual navigation, and it is precisely what current VLMs lack.

![Image 1: Refer to caption](https://arxiv.org/html/2603.17948v1/x1.png)

Figure 1: Logarithmic compute scaling with video duration. Video-RLM’s hierarchical grid grows sub-linearly (O​(log⁡T)O(\log T)), requiring up to 9.7×\times fewer tokens than linear-scaling baselines. A uniform VLM maxes out its 256K context trading off sampled frame count with resolution.

Existing approaches to long-form video understanding can be broadly categorized into four paradigms: uniform sampling, composite grids, caption-based, and agentic-based approaches. Uniform sampling[[20](https://arxiv.org/html/2603.17948#bib.bib1 "Longvideobench: a benchmark for long-context interleaved video-language understanding"), [6](https://arxiv.org/html/2603.17948#bib.bib2 "Video-mme: the first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis")] introduces severe temporal sparsity, i.e., at practical budgets, frames are sampled minutes apart, resulting in short events being systematically missed. Moreover, within a fixed context window, increasing the number of sampled frames forces a proportional decrease in per-frame resolution, creating a fundamental coverage-vs-fidelity tradeoff. Composite grids[[10](https://arxiv.org/html/2603.17948#bib.bib3 "An image grid can be worth a video: zero-shot video question answering using a vlm"), [5](https://arxiv.org/html/2603.17948#bib.bib6 "Vote-in-context: turning vlms into zero-shot rank fusers")] pack frames into a single representative image, improving token efficiency but remaining a fixed, lossy snapshot. Caption-based and agentic approaches[[17](https://arxiv.org/html/2603.17948#bib.bib9 "Videoagent: long-form video understanding with large language model as agent"), [25](https://arxiv.org/html/2603.17948#bib.bib10 "Deep video discovery: agentic search with tool use for long-form video understanding"), [22](https://arxiv.org/html/2603.17948#bib.bib11 "VideoARM: agentic reasoning over hierarchical memory for long-form video understanding")] rely on text as their primary reasoning medium (captioning clips, storing text summaries, or converting visual observations into language before planning). Even when these systems adaptively sample frames, their intermediate memory and decision-making operate over text, not over a structured visual space. Any visual detail overlooked during transcription or abstraction cannot be recovered by subsequent reasoning. These paradigms also face distinct scalability bottlenecks i.e., standard VLM pipelines[[2](https://arxiv.org/html/2603.17948#bib.bib22 "Qwen3-vl technical report")] must decode the video, extract frames, and perform visual tokenization on CPU before any reasoning begins. For long videos, this preprocessing alone can exhaust hundreds of gigabytes of system RAM. Caption-based and agentic methods avoid this by converting video to text first, but incur a different cost: an offline captioning stage that scales linearly with video duration and irreversibly discards visual fidelity. While some agentic methods[[22](https://arxiv.org/html/2603.17948#bib.bib11 "VideoARM: agentic reasoning over hierarchical memory for long-form video understanding")] perform this conversion online, they still rely on text as the intermediate representation, inheriting the same information loss.

We claim that a useful video representation must be simultaneously _lossless_ (frame-level access at any resolution), _navigable_ (agent-directed), _scalable_ (no context ceiling), _caption-free_ (native visual reasoning), and _preprocessing-free_ (no offline decoding). As detailed in [Tab.1](https://arxiv.org/html/2603.17948#S2.T1 "In Agentic, Hierarchical, and Memory Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), current approaches typically optimize for a subset of these properties at the expense of others.

_VideoAtlas._ We propose a task-agnostic environment that represents any video as a navigable, hierarchical K×K K\times K image grid ([Fig.2](https://arxiv.org/html/2603.17948#S2.F2 "In Long Context as the Core Challenge. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")). The root grid renders the full video as a contact sheet. By invoking Expand (a recursive descent action that generates a new, finer-resolution sub-grid for a selected cell) an agent achieves sub-second temporal precision in O​(log⁡T)O(\log T) steps, where T T is the video duration in seconds. The design is uniform throughout: the video, intermediate investigations, and the agent’s internal _evidence scratchpad_ (a lossless multimodal memory that stores collected frames, subtitles, timestamps, and descriptions) are all rendered as grids. This completely eliminates captioning, offline preprocessing, and context-window ceilings, satisfying all properties in the aforementioned paragraph. Crucially, VideoAtlas also escapes the _coverage-vs-fidelity tradeoff_ inherent to uniform VLMs: within a fixed context window, sampling more frames forces lower per-frame resolution, and vice versa. VideoAtlas sidesteps this entirely (each grid image is always rendered at full resolution, and the agent zooms only where needed, never sacrificing visual fidelity for temporal coverage). Structurally, the hierarchy yields _logarithmic_ compute growth: as video length increases, only a few additional depth layers are needed rather than linearly more frames. Moreover, the fixed hierarchical grid is inherently _cache-friendly_: root grids and overlapping sub-grids are naturally reused across exploration rounds, achieving 30-60% multimodal cache hit rates that further reduce effective GPU compute (see Appendix Sec. C.1).

#### From representation to reasoning.

With a lossless and navigable video representation in hand, a crucial observation follows: the long-video problem reduces to a _long-context_ problem. The video is the context, and what is needed is a mechanism for agents to explore it recursively without compressing it. Recursive Language Models (RLMs)[[23](https://arxiv.org/html/2603.17948#bib.bib12 "Recursive language models")] provide exactly this mechanism for text, allowing agents to query arbitrarily long contexts through recursive subagent calls and accumulate exact symbolic variables. RLMs, however, require a structured environment to recurse into. VideoAtlas is precisely that structure. We deploy Master-Worker Agents (Video-RLM) within this environment to extend RLMs to the video domain, yielding depth-controlled compute budgeting and logarithmic cost growth. Following are our main contributions.

1.   1.
VideoAtlas. We formulate video understanding as navigation within a formally defined geometric environment. The hierarchical grid is lossless, caption-free, preprocessing-free, and strategy-agnostic, with logarithmic access depth, parallelizable subgrids, and a structural cache-friendliness.

2.   2.
Video-RLM. A parallel Master-Worker architecture extending Recursive Language Models to video. Workers explore grid subtrees concurrently and accumulate evidence in a lossless Visual Scratchpad, while a Master steers exploration via uncertainty analysis.

3.   3.
Configurable Traversal Strategies. Breadth-First and Depth-First instantiations plus a query-adaptive policy that selects traversal order automatically, all composable with the environment without modification.

4.   4.
Environment Budgeting. We budget the _environment_, not the agent: bounding exploration depth d d directly controls temporal resolution and compute, providing a principled compute-accuracy hyperparameter.

Beyond these architectural contributions, experiments reveal that the formulation produces emergent scaling behaviors (adaptive compute allocation and logarithmic cost growth) that we detail in [Sec.4](https://arxiv.org/html/2603.17948#S4 "4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute").

## 2 Related Work

#### Long-Form Video Understanding.

Standard Video-Language Models process videos by uniformly sampling a fixed number of frames in a single forward pass[[20](https://arxiv.org/html/2603.17948#bib.bib1 "Longvideobench: a benchmark for long-context interleaved video-language understanding"), [6](https://arxiv.org/html/2603.17948#bib.bib2 "Video-mme: the first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis")]. This introduces two structural problems. First, at any practical budget (e.g., 64 frames in an hour), the temporal stride is ∼\sim 56 seconds per frame, so short events, fine-grained visual details, and scene transitions are easily missed. Second, the context window imposes a hard ceiling: beyond a few hundred high-resolution frames, the model truncates input or degrades. One practical workaround is to pack multiple frames into a single K×K K\times K composite image (a contact-sheet grid)[[10](https://arxiv.org/html/2603.17948#bib.bib3 "An image grid can be worth a video: zero-shot video question answering using a vlm"), [5](https://arxiv.org/html/2603.17948#bib.bib6 "Vote-in-context: turning vlms into zero-shot rank fusers")], which improves token efficiency. However, a single-resolution grid is still fundamentally _lossy_: it represents the video with a fixed sample of moments and cannot recover the events in between. Grids alleviate the context-packing problem, but they do not resolve the _coverage_ problem.

#### Caption-Based Approaches.

A prominent line of work avoids the frame-count limit by first transcribing the video into text captions and then reasoning over them. LLoVi[[24](https://arxiv.org/html/2603.17948#bib.bib13 "A simple llm framework for long-range video question-answering")] converts densely sampled short clips into text summaries and aggregates them with an LLM. MR.Video[[12](https://arxiv.org/html/2603.17948#bib.bib16 "MR. video: mapreduce as an effective principle for long video understanding")] scales this with a MapReduce design: clips are captioned in parallel, standardized, and then synthesized into a final answer by a reducer LLM. Video to Text conversion is standard practice, although systems that explicitly observe video frames at a coarse step immediately convert those observations into text before any planning or memory update. Pang _et al_.[[12](https://arxiv.org/html/2603.17948#bib.bib16 "MR. video: mapreduce as an effective principle for long video understanding")] explicitly acknowledge that video-to-text modality transitions cause reasoning failures on scene transitions and fine-grained visual details.

#### Agentic, Hierarchical, and Memory Approaches.

Another set of approaches treat long-video understanding as agentic search. DVD[[25](https://arxiv.org/html/2603.17948#bib.bib10 "Deep video discovery: agentic search with tool use for long-form video understanding")] constructs a multi-granular database (global summaries, clip captions/embeddings, and indexed raw frames) and queries it with tools (Global Browse, Clip Search, Frame Inspect). VideoARM [[22](https://arxiv.org/html/2603.17948#bib.bib11 "VideoARM: agentic reasoning over hierarchical memory for long-form video understanding")] performs on-the-fly coarse-to-fine search via a set of predefined tools (e.g., captioning, temporal localization, visual QA) over a hierarchical multimodal memory, avoiding exhaustive preprocessing. VideoTree[[18](https://arxiv.org/html/2603.17948#bib.bib17 "Videotree: adaptive tree-based video representation for llm reasoning on long videos")] builds a query-adaptive hierarchical representation to guide efficient exploration. On the memory side, WorldMM[[21](https://arxiv.org/html/2603.17948#bib.bib5 "WorldMM: dynamic multimodal memory agent for long video reasoning")] organizes long-video memory into episodic, semantic, and visual components, retrieved adaptively per query[[8](https://arxiv.org/html/2603.17948#bib.bib19 "Visual sketchpad: sketching as a visual chain of thought for multimodal language models")].

Despite their diversity, these systems share a common limitation: intermediate evidence is stored as captions, text summaries, or compressed embeddings, never as raw visual frames, meaning none provide lossless, navigable access to any arbitrary video moment by construction.

Table 1: Comparison of long-video QA methods. “Caption-Free” = no text captions used as intermediate representation. “Lossless” = no information lost between input and reasoning. “∞\infty Context” = can handle arbitrarily long videos without context overflow. “Parallel” = workers explore concurrently.

#### Long Context as the Core Challenge.

Recursive Language Models (RLMs)[[23](https://arxiv.org/html/2603.17948#bib.bib12 "Recursive language models")] address long text contexts by letting agents access context through recursive subagent calls, storing results in lossless symbolic variables rather than compressing them into the model’s context window. The RLM insight transfers naturally to video, but only if an _environment_ is defined in which agents can navigate the video visually. Existing video “environments” are built around clip databases and text-based retrieval[[25](https://arxiv.org/html/2603.17948#bib.bib10 "Deep video discovery: agentic search with tool use for long-form video understanding"), [22](https://arxiv.org/html/2603.17948#bib.bib11 "VideoARM: agentic reasoning over hierarchical memory for long-form video understanding")]. No visual, lossless, recursively navigable environment for video has been proposed. VideoAtlas fills precisely this gap.

![Image 2: Refer to caption](https://arxiv.org/html/2603.17948v1/figures/VideoAtlas.png)

Figure 2: The VideoAtlas Environment. (Left) The state space is a hierarchical grid stack S 0,S 1,…,S D S_{0},S_{1},\ldots,S_{D}, where S 0 S_{0} is the root grid covering the entire video of duration T T. Each grid has K 2 K^{2} cells. Deeper levels d d provide finer temporal resolution Δ​t d=T/K 2​(d+1)\Delta t_{d}=T/K^{2(d+1)}. (Top Right) The discrete action space 𝒜\mathcal{A} is divided into navigation (e.g., Expand to S t+1 S_{t+1}), perception, and commit actions. (Bottom Right) The visual scratchpad memory ℳ+\mathcal{M}^{+} accumulates multimodal evidence (images, timestamps, QA pairs) across exploration rounds.

#### Environment Budgeting vs. Prior Compute Adaptation.

Chain-of-thought reasoning[[19](https://arxiv.org/html/2603.17948#bib.bib20 "Chain-of-thought prompting elicits reasoning in large language models")] and adaptive test-time compute allocation[[15](https://arxiv.org/html/2603.17948#bib.bib21 "Scaling llm test-time compute optimally can be more effective than scaling parameters for reasoning")] have shown that allocating more inference compute consistently improves performance on language and reasoning tasks. In the video domain, the closest analog is VideoARM[[22](https://arxiv.org/html/2603.17948#bib.bib11 "VideoARM: agentic reasoning over hierarchical memory for long-form video understanding")], which adaptively chooses how many frames N 1 N_{1} to sample per localized interval, a form of density adaptation that improves efficiency. However, this controls sampling _quantity_ (how many frames), not structural _resolution_ (how fine the temporal decomposition is): within each interval, sampling remains uniform, and events falling between sample points can still be missed regardless of N 1 N_{1}. MR.Video[[12](https://arxiv.org/html/2603.17948#bib.bib16 "MR. video: mapreduce as an effective principle for long video understanding")] offers no such control at all. Its captioning cost is fixed by video duration regardless of the query. A fundamentally different form of budgeting is absent from prior works: controlling the _temporal resolution of the environment itself_, where each depth level geometrically subdivides time, providing formal precision guarantees calibrated to video length and query granularity. We introduce exactly this form of budgeting with VideoAtlas.

#### What Is Missing?

[Tab.1](https://arxiv.org/html/2603.17948#S2.T1 "In Agentic, Hierarchical, and Memory Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") summarizes the key properties of representative methods. In the next section, we introduce VideoAtlas, which addresses all the aforementioned gaps.

## 3 Methodology

We present our methodology in two parts. First, we introduce VideoAtlas ([Sec.3.1](https://arxiv.org/html/2603.17948#S3.SS1 "3.1 VideoAtlas ‣ 3 Methodology ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")): a task-agnostic environment that renders any video as a navigable, hierarchical grid with formally defined state, action, and observation spaces ([Fig.2](https://arxiv.org/html/2603.17948#S2.F2 "In Long Context as the Core Challenge. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")). Second, we describe Video-RLM ([Sec.3.2](https://arxiv.org/html/2603.17948#S3.SS2 "3.2 Video-RLM: Master-Worker Architecture ‣ 3 Methodology ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")): a parallel Master-Worker agent architecture that operates within VideoAtlas to answer questions about arbitrarily long videos ([Fig.3](https://arxiv.org/html/2603.17948#S3.F3 "In 3 Methodology ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")).

![Image 3: Refer to caption](https://arxiv.org/html/2603.17948v1/figures/grid2.png)

Figure 3: Video-RLM overview. The query is converted into a search task. In each round r r, the Master examines the root grid S 0 S_{0} (with dead zones masked) and the scratchpad ℳ+\mathcal{M}^{+}, then assigns promising cells to Workers. Each Worker autonomously explores its assigned region via navigation, perception, and commit actions. After all Workers return, ℳ+\mathcal{M}^{+} and ℳ−\mathcal{M}^{-} are updated. The Master performs an uncertainty analysis: if evidence is sufficient, the final answer is produced. Otherwise, a new round begins.

### 3.1 VideoAtlas

#### Hierarchical Grid.

At the core of VideoAtlas is a recursive K×K K\times K image grid (default K=8 K{=}8, yielding 64 cells). Given a video of duration T T seconds, the root grid S 0 S_{0} assigns each cell c i c_{i} to a contiguous temporal interval [t i start,t i end][t_{i}^{\text{start}},t_{i}^{\text{end}}] and displays a representative frame sampled at the interval midpoint, providing a “bird’s-eye view” of the entire video ([Fig.3](https://arxiv.org/html/2603.17948#S3.F3 "In 3 Methodology ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")). Every cell is _addressable_: applying Expand to cell c i c_{i} deterministically generates a child grid S d+1 S_{d+1} for that cell’s sub-interval, increasing temporal resolution by a factor of K 2 K^{2}. At depth d d, the temporal resolution is Δ​t d=T/K 2​(d+1)\Delta t_{d}=T/K^{2(d+1)}, and reaching any frame requires at most D max=⌈log K 2⁡(T⋅fps)⌉D_{\max}=\lceil\log_{K^{2}}(T\cdot\mathrm{fps})\rceil steps, achieving sub-second precision even for 10-hour videos. Sub-grids are generated on-the-fly with no offline preprocessing. Agents interact with raw frames at every level.

#### Action Space.

Unlike agentic methods[[22](https://arxiv.org/html/2603.17948#bib.bib11 "VideoARM: agentic reasoning over hierarchical memory for long-form video understanding")] whose actions perform _video-processing_ operations (captioning, translating), VideoAtlas exposes _environment-navigation_ actions grouped into three categories ([Fig.2](https://arxiv.org/html/2603.17948#S2.F2 "In Long Context as the Core Challenge. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), right):

Navigation (move through the hierarchy): Expand(c i)(c_{i}) descends into cell c i c_{i}, generating a child grid. Backtrack()() returns to the parent grid. MarkPromising(c i,c j,…)(c_{i},c_{j},\ldots) flags cells for later exploration via a FIFO queue (BFS mode only).

Perception (sense the environment): Zoom(c i)(c_{i}) returns a full-resolution frame for cell c i c_{i}. Investigate(c i,direction (before/after))(c_{i},\text{direction (before/after)}) generates a temporal context scan of the frames immediately before or after a cell, used when an anchor event is found but the answer lies in neighboring frames.

Commit (record evidence): AddToScratchpad(items)(\text{items}) stores evidence tuples to the scratchpad. Finished()() declares the current region fully explored.

The available action set is state-dependent: Expand is removed when cell span drops below a threshold (e.g., <1{<}1 s), Backtrack is removed at the root, and BFS and DFS workers receive different action sets. The agent cannot select what it cannot see, eliminating invalid actions by construction, while deciding its own explore-exploit balance from visual cues.

#### Memory.

Positive memory (ℳ+\mathcal{M}^{+}, Visual Scratchpad): a lossless multimodal memory that stores evidence as tuples (I img,s,τ,c,d)(I_{\text{img}},s,\tau,c,d) representing image patch, subtitle, timestamp, confidence score, and a text description relating the evidence to the query. When presented to the VLM, ℳ+\mathcal{M}^{+} is rendered as a grid image with timestamps, subtitles, and indices burned into pixel space, enabling unambiguous cross-referencing. Negative memory (ℳ−\mathcal{M}^{-}, Dead Zones): intervals explored with no relevant findings are marked as dead zones. The grid renderer enforces this _visually_ by blacking out overlapping cells, physically preventing the VLM from hallucinating details in already-explored regions.

#### Formal Environment Definition.

At any step, the environment state S S comprises five components: the current temporal position p=(center,span)p=(\text{center},\text{span}), the depth d d in the hierarchy, the positive and negative memories ℳ+\mathcal{M}^{+} and ℳ−\mathcal{M}^{-}, and the navigation stack σ\sigma for backtracking. The observation is the grid image rendered for the interval defined by p p at depth d d, together with aligned subtitle context filtered for the current temporal window.

This state definition, together with the action space, formally defines a Markov Decision Process (MDP). The reward is task-defined (e.g., answer correctness for QA, temporal IoU for grounding) making VideoAtlas a general substrate for any task reducible to “find relevant moments in a video.” In this work we solve it via zero-shot VLM reasoning, but the formal MDP opens a direct path to reinforcement learning. The environment exhibits four structural properties: (1)_Parallelizable_: the grid decomposes into independent subtrees explorable concurrently. (2)_Traversal-agnostic_: BFS, DFS, beam search, or learned policies can govern expansion order without modifying the environment. (3)_Depth-controlled compute_: bounding d d yields a principled compute-accuracy hyperparameter. (4)_Logarithmic overhead_: as video duration grows, the hierarchy adds depth levels logarithmically, yielding 𝒪​(log⁡T)\mathcal{O}(\log T) scaling rather than 𝒪​(T)\mathcal{O}(T). Notably, the depth parameter d d interpolates between uniform sampling (d=0 d{=}0, equivalent to a single composite grid of K 2 K^{2} frames) and full recursive exploration (d=D max d{=}D_{\max}); prior uniform-sampling and composite-grid methods are thus degenerate cases of VideoAtlas with no exploration.

### 3.2 Video-RLM: Master-Worker Architecture

We extend Recursive Language Models[[23](https://arxiv.org/html/2603.17948#bib.bib12 "Recursive language models")] to videos by deploying agents in VideoAtlas. Agents access video context through recursive subagents (workers) and store outputs in the Visual Scratchpad ℳ+\mathcal{M}^{+}. Exploration proceeds in discrete _rounds_: in each round r r, the Master assigns cells to Workers, Workers explore in parallel, and results are merged into ℳ+\mathcal{M}^{+} and ℳ−\mathcal{M}^{-} before the next round begins ([Fig.3](https://arxiv.org/html/2603.17948#S3.F3 "In 3 Methodology ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")).

#### Search Task Extraction.

Before visual exploration, a text-only step converts the raw query into a concrete search task. For example: _“What treaty was signed after the London conference?”_→\rightarrow _“Find the London conference scene. Look immediately after for treaty names in text overlays or subtitles.”_ This search task guides all subsequent prompts.

#### Master Agent.

The Master holds the global view: it examines the root grid S 0 S_{0} (with dead zones masked) and the current scratchpad ℳ+\mathcal{M}^{+}, then selects promising cells for the next round (_Global Probing_). A priority queue with _Virtual Loss_[[3](https://arxiv.org/html/2603.17948#bib.bib4 "Parallel monte-carlo tree search")] ensures that cells already assigned to workers are deprioritized, preventing redundant exploration. After each round, the Master performs _Uncertainty Analysis_: (a)a sufficiency check, (b)temporal interpolation to suggest targeted search bounds from gaps between evidence anchors, and (c)dynamic memory pruning.

#### Worker Agents.

Each worker receives one cell from the frontier and explores it autonomously. Two modes are supported: Depth-First Search (DFS) mode where the worker Expand s deeper into the timeline with a multi-step budget, ideal for localizing specific details. Breadth-First Search (BFS) mode where the worker scans one level with a single-step budget, ideal for evidence spread across the video. The traversal queue is re-prioritized via the Master’s visual scoring.

#### Query-Adaptive Traversal.

The Master selects the traversal strategy before any frames are processed by analyzing the query’s linguistic traits: DFS for specific detail localization, BFS for sequence or flow understanding.

#### Sufficiency, Stopping, and Final Decision.

Exploration stops at three levels: (1)worker-level (budget exhausted or Finished), (2)master-level (sufficiency check passes after round r r), (3)global (total compute budget reached). Once exploration terminates, the Master synthesizes the answer from ℳ+\mathcal{M}^{+}: it sees the actual collected evidence frames (rendered as a grid with burned-in labels), not text summaries, and evaluates each candidate against the visual evidence.

## 4 Experiments

### 4.1 Experimental Setup

#### Benchmarks.

We evaluated VideoRLM on the long subsets of two benchmarks: LongVideoBench[[20](https://arxiv.org/html/2603.17948#bib.bib1 "Longvideobench: a benchmark for long-context interleaved video-language understanding")] (LVB, 15-60 min videos) and Video-MME[[6](https://arxiv.org/html/2603.17948#bib.bib2 "Video-mme: the first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis")] (VMME, without subtitles). To stress-test scalability beyond VLM context limits, we constructed 10-hour variants by concatenating multiple videos from each benchmark. Each query targeted a single source video placed at a random position among distractors. Subtitle tracks were merged with correct temporal offsets. This isolates the “needle in a haystack” challenge at extreme durations. For VMME, we evaluated the system _without subtitles_ to verify that the system genuinely understands visual content rather than relying on textual cues.

#### Model.

Our primary experiments used Qwen3.5-35B-A3B[[13](https://arxiv.org/html/2603.17948#bib.bib15 "Qwen3.5: towards native multimodal agents")] (35B total parameters, 3B active per forward pass) for both Master and Workers, differentiated via separate system prompts and action sets, served through vLLM[[11](https://arxiv.org/html/2603.17948#bib.bib7 "Efficient memory management for large language model serving with pagedattention")] on 4×\times A100 80 GB GPUs. Each image in the grids used throughout the system was rendered at a unified resolution of 320×320 320\times 320 pixels. We use grid size K=8 K=8. To demonstrate VLM-agnosticism, we additionally evaluate with Gemini-3-Flash as the backbone ([Tab.2](https://arxiv.org/html/2603.17948#S4.T2 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")).

#### Baselines.

We compared against five categories. (1)Proprietary Models: GPT-4o, GPT-5, Gemini-3-Flash, and Claude-Opus-4.5. (2)Open-Source VLMs InternVL3.5 241B (28B active) and GLM-4.5V-106B (12B active), uniform-sampling baselines with significantly larger active parameters than ours. (3)Caption Reliant Agentic Methods: DVD[[25](https://arxiv.org/html/2603.17948#bib.bib10 "Deep video discovery: agentic search with tool use for long-form video understanding")], MR.Video[[12](https://arxiv.org/html/2603.17948#bib.bib16 "MR. video: mapreduce as an effective principle for long video understanding")], and VideoARM[[22](https://arxiv.org/html/2603.17948#bib.bib11 "VideoARM: agentic reasoning over hierarchical memory for long-form video understanding")], reported from their original papers. (4)Uniform Sampling: Qwen3.5-35B-A3B with 160 uniformly sampled frames at a resolution of 320×320 320\times 320 pixels (similar to our framework) along with their temporally aligned subtitles, representing the strongest single-pass baseline within our hardware budget. (5)LLM over Captions: following LLoVi[[24](https://arxiv.org/html/2603.17948#bib.bib13 "A simple llm framework for long-range video question-answering")], GPT-4o captions (from the MR.Video repository) are concatenated temporally and answered by Qwen3.5-35B-A3B, isolating the benefit of visual exploration over textual summarization.

### 4.2 Main Results

[Tab.2](https://arxiv.org/html/2603.17948#S4.T2 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") compares all methods on the standard long-video benchmarks. [Tab.3](https://arxiv.org/html/2603.17948#S4.T3 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") evaluates the 10-hour extended-duration variants, alongside average token consumption per question.

Table 2: Video QA accuracy (%) on the standard (Long) subsets. LVB: LongVideoBench. VMME: Video-MME (no subs). 

⋆ GPT-4o captions (MR.Video repo), answered by Qwen3.5-35B-A3B.

Table 3: 10-hour variant: accuracy (%) and average tokens per question. Δ\Delta: accuracy drop from standard benchmarks ([Tab.2](https://arxiv.org/html/2603.17948#S4.T2 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")). 

∗ GPT-4o captions + Qwen3.5 (exceeds 256K context in many samples). † Effective tokens after vLLM multimodal prefix cache (avg. 36-42% hit rate). ‡ QA tokens only, excludes GPT-4o captioning cost.

#### Standard benchmarks.

At standard durations ([Tab.2](https://arxiv.org/html/2603.17948#S4.T2 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")), Video-RLM (3B active parameters) achieves accuracy competitive with substantially larger open-source VLMs (12-28B active) and proprietary baselines. Importantly, Qwen 3.5 is video-finetuned, whereas Video-RLM assumes a purely _zero-shot_ agent with no video-specific training, achieving these results without any intermediate text representation or captioning. The accuracy gap between zero-shot navigation and finetuned uniform sampling narrows with a stronger backbone: Video-RLM (Gemini) reaches 72.0% on LVB, within 2.5 points of Gemini-3-Flash’s direct performance (74.5%), confirming VideoAtlas is VLM-agnostic and performance scales with backbone capability.

#### Extended duration (10 hours).

The 10-hour variants ([Tab.3](https://arxiv.org/html/2603.17948#S4.T3 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")) reveal a more significant comparison. LLM over Captions _collapses to 36.0%_ on VMME-10hr: concatenated captions exceed Qwen’s 256K context window, forcing truncation and information loss, demonstrating linear captioning fails beyond context limits. Notably, captions degrade only −0.3%-0.3\% on LVB-10hr (where subtitle tracks are available) but −28.2%-28.2\% on VMME-10hr (no subtitles), exposing a heavy reliance on textual cues rather than genuine visual understanding. Uniform Qwen degrades moderately (63.8%→\to 50.6% VMME) as sampling becomes prohibitively sparse. Video-RLM maintains highly stable performance across durations (e.g., Qwen VMME drops only 0.7% vs uniform 13.2% and Captions 28.2%), validating that VideoAtlas buffers the agent against duration scaling. On VMME-10hr, the absence of subtitles forces the purely zero-shot Qwen agent into extensive visual exploration (403K effective tokens), while the stronger Gemini backbone zeroes in on the answer faster (390K tokens), an emergent adaptive compute property where weaker perception necessitates more search steps. Crucially, the recursive nature of the environment is inherently _cache-friendly_: workers re-examine the same grid view across multiple reasoning steps that do not change the navigation state, creating repeated visual token prefixes. For self-hosted Qwen (vLLM), automatic multimodal prefix caching exploits this redundancy transparently, achieving 36-42% hit rates at 10 hours (up to 61% for shorter videos, see Appendix Sec. C1). Video-RLM (Gemini) achieves 70.1% on LVB-10hr with near-zero degradation (-1.9%) from the standard benchmark.

#### Error analysis.

Manual inspection of failure cases reveals three dominant error modes: VLM perception errors (misreading text overlays, confusing visually similar scenes), premature sufficiency (the Master declares evidence sufficient despite contradictions), and text latching (anchoring on phrases in evidence that superficially match a candidate answer). All three are model-dependent, as confirmed by the substantial accuracy improvement when switching from Qwen (3B active) to Gemini-3-Flash without any changes to VideoAtlas. We discuss these errors in more detail in the Appendix Sec. A.

### 4.3 Logarithmic Compute Scaling

[Fig.1](https://arxiv.org/html/2603.17948#S1.F1 "In 1 Introduction ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") demonstrates the fundamental efficiency advantage of hierarchical navigation using LVB-10hr. As video duration grows from 1 minute to 10 hours (a 600×\times increase), Video-RLM’s compute cost increases sub-linearly: the hierarchy adds depth levels logarithmically (⌈log K 2⁡(T⋅fps)⌉\lceil\log_{K^{2}}(T\cdot\text{fps})\rceil), and the sufficiency mechanism halts exploration once evidence is found. Caption-based pipelines scale linearly, where every clip must be captioned regardless of the query, requiring over 1.4M tokens per query at 10 hours. Video-RLM achieves comparable accuracy using only 148K effective tokens (a 9.7×\times reduction), and unlike uniform VLMs whose cost is fixed, depth can always be extended to accommodate longer videos.

### 4.4 Environment Budgeting and Adaptive Compute

![Image 4: Refer to caption](https://arxiv.org/html/2603.17948v1/x2.png)

Figure 4: (a)Environment budgeting: accuracy and tokens vs. max depth on subset of LVB-10hr (temporal span annotated). Green: optimal depth (first sub-second layer). (b)Adaptive compute: average tokens scale with evidence spread without ground-truth supervision.

[Fig.4](https://arxiv.org/html/2603.17948#S4.F4 "In 4.4 Environment Budgeting and Adaptive Compute ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")(a) shows accuracy vs. maximum exploration depth on 30 questions sampled from LVB-10hr. Accuracy rises from 30% (d=0 d{=}0, root grid only) to 43.3% (d=2 d{=}2, 137 ms span), then plateaus at depths 3-4 where the finest resolution drops below one millisecond, well beyond any meaningful visual granularity. In practice, we set the maximum depth to the first sub-second layer, which automatically adapts to video duration (a 1-minute video reaches sub-second at d=1 d{=}1, a 10-hour video at d=2 d{=}2). Depth d d is thus a principled compute-accuracy hyperparameter that directly controls temporal _resolution_ rather than frame _quantity_.

[Fig.4](https://arxiv.org/html/2603.17948#S4.F4 "In 4.4 Environment Budgeting and Adaptive Compute ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")(b) verifies that compute allocation adapts to question difficulty without explicit supervision. Grouping LVB questions by the number of ground-truth temporal positions containing answer evidence, scattered answers (3+ positions) consume 40% more tokens (322K vs. 230K) than localized ones. This emergent behavior arises from the interaction between the Master’s uncertainty analysis, the sufficiency mechanism, and the hierarchical structure.

### 4.5 Worker Scaling

![Image 5: Refer to caption](https://arxiv.org/html/2603.17948v1/x3.png)

Figure 5: Wall-clock time (normalized to equal workload) vs. number of workers 30 questions sampled from LVB-10hr. Accuracy (annotated) remains stable across all configurations.

We vary the number of workers ∈{1,3,5,7}\in\{1,3,5,7\} on 30 questions sampled from LVB-10hr ([Fig.5](https://arxiv.org/html/2603.17948#S4.F5 "In 4.5 Worker Scaling ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")). After normalizing for workload differences, wall-clock time decreases from 588s (1 workers) to 257s (7 workers), a 2.25×\times speedup, while accuracy remains stable (40-47%). This is a structural property of the environment: each subtree is self-contained, so adding workers improves throughput without modifying the search protocol.

## 5 Limitations

We identify four principal limitations. _(1)VLM perception bottleneck_: the system’s perceptual ceiling is set entirely by the backbone VLM. Our error analysis reveals three dominant failure modes, all VLM-dependent rather than environment-dependent: (a)perception errors (misreading text overlays, confusing visually similar scenes), (b)premature sufficiency, where the Master declares evidence sufficient despite contradictions rather than directing further exploration, and (c)text latching, where the agent over-relies on subtitle cues when available. Crucially, switching from Qwen (3B active) to Gemini-3-Flash eliminates a substantial portion of these errors without any architectural changes ([Tab.2](https://arxiv.org/html/2603.17948#S4.T2 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute")), confirming that performance scales directly with backbone capability and will improve as VLMs advance. _(2)No-anchor exploration overhead_: when the root grid S 0 S_{0} contains no visually obvious anchor for the query, the agent may require additional exploration rounds before finding relevant regions. The Master mitigates this progressively, as each round’s newly collected evidence refines subsequent cell assignments. Developing methods to surface semantically relevant information into upper depth layers could substantially improve efficiency and is a promising direction for future work. _(3)Evaluation scope_: we validate on multiple-choice QA. The MDP formulation supports temporal grounding, summarization, and anomaly detection (only the reward signal changes. The environment does not), but these remain to be demonstrated empirically. _(4)Zero-shot only_: we solve the MDP entirely via zero-shot VLM reasoning, the weakest possible agent. The discrete, finite action space makes VideoAtlas directly compatible with RL training (PPO, DQN), which would likely improve exploration efficiency, but we leave this to future work.

## 6 Conclusion

We introduced VideoAtlas, a formulation that reframed video understanding as navigation within a formally defined hierarchical environment, and Video-RLM, a parallel Master-Worker agent that operates within it. Three properties emerged from the formulation: logarithmic compute growth with duration, principled environment budgeting via depth control, and emergent adaptive compute allocation. This environment defines the state, action, and observation spaces of a complete MDP opening a direct path from zero-shot reasoning to learned exploration policies, and from question answering to any task reducible to “find relevant moments in a video.”

## Acknowledgements

We are grateful to the KAUST Academy for its generous support, and especially to Prof. Sultan Albarakati that made this work possible. For computer time, this research used Ibex managed by the Supercomputing Core Laboratory at King Abdullah University of Science & Technology (KAUST) in Thuwal, Saudi Arabia.

## References

*   [1] (2024)The claude 3 model family: opus, sonnet, haiku. Claude-3 Model Card 1 (1),  pp.4. Cited by: [Table 2](https://arxiv.org/html/2603.17948#S4.T2.1.7.6.1 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [2]S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. (2025)Qwen3-vl technical report. arXiv preprint arXiv:2511.21631. Cited by: [§1](https://arxiv.org/html/2603.17948#S1.p2.1 "1 Introduction ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [3]G. M.J-B. Chaslot, M. H.M. Winands, and H. J. van den Herik (2008)Parallel monte-carlo tree search. In Computers and Games, Lecture Notes in Computer Science, Vol. 5131, Berlin, Heidelberg,  pp.60–71. Cited by: [§3.2](https://arxiv.org/html/2603.17948#S3.SS2.SSS0.Px2.p1.2 "Master Agent. ‣ 3.2 Video-RLM: Master-Worker Architecture ‣ 3 Methodology ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [4]Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu, et al. (2024)Internvl: scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,  pp.24185–24198. Cited by: [Table 2](https://arxiv.org/html/2603.17948#S4.T2.1.9.8.1 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [5]M. Eltahir, A. Habibullah, L. Ayash, T. Hussain, and N. Khan (2025)Vote-in-context: turning vlms into zero-shot rank fusers. arXiv preprint arXiv:2511.01617. Cited by: [§1](https://arxiv.org/html/2603.17948#S1.p2.1 "1 Introduction ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px1.p1.2 "Long-Form Video Understanding. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [6]C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang, et al. (2025)Video-mme: the first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,  pp.24108–24118. Cited by: [§1](https://arxiv.org/html/2603.17948#S1.p2.1 "1 Introduction ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px1.p1.2 "Long-Form Video Understanding. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§4.1](https://arxiv.org/html/2603.17948#S4.SS1.SSS0.Px1.p1.1 "Benchmarks. ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [7]T. Glm, A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin, D. Zhang, D. Rojas, G. Feng, H. Zhao, et al. (2024)Chatglm: a family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793. Cited by: [Table 2](https://arxiv.org/html/2603.17948#S4.T2.1.10.9.1 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [8]Y. Hu, W. Shi, X. Fu, D. Roth, M. Ostendorf, L. Zettlemoyer, N. A. Smith, and R. Krishna (2024)Visual sketchpad: sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems 37,  pp.139348–139379. Cited by: [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px3.p1.1 "Agentic, Hierarchical, and Memory Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [9]A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et al. (2024)Gpt-4o system card. arXiv preprint arXiv:2410.21276. Cited by: [Table 2](https://arxiv.org/html/2603.17948#S4.T2.1.4.3.1 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [10]W. Kim, C. Choi, W. Lee, and W. Rhee (2024)An image grid can be worth a video: zero-shot video question answering using a vlm. IEEE Access 12,  pp.193057–193075. Cited by: [§1](https://arxiv.org/html/2603.17948#S1.p2.1 "1 Introduction ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px1.p1.2 "Long-Form Video Understanding. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [11]W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica (2023)Efficient memory management for large language model serving with pagedattention. External Links: 2309.06180, [Link](https://arxiv.org/abs/2309.06180)Cited by: [§4.1](https://arxiv.org/html/2603.17948#S4.SS1.SSS0.Px2.p1.3 "Model. ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [12]Z. Pang and Y. Wang MR. video: mapreduce as an effective principle for long video understanding. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Cited by: [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px2.p1.1 "Caption-Based Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px5.p1.2 "Environment Budgeting vs. Prior Compute Adaptation. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [Table 1](https://arxiv.org/html/2603.17948#S2.T1.3.4.3.1 "In Agentic, Hierarchical, and Memory Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§4.1](https://arxiv.org/html/2603.17948#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [Table 2](https://arxiv.org/html/2603.17948#S4.T2.1.12.11.1 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [13]Qwen Team (2026-02)Qwen3.5: towards native multimodal agents. External Links: [Link](https://qwen.ai/blog?id=qwen3.5)Cited by: [§4.1](https://arxiv.org/html/2603.17948#S4.SS1.SSS0.Px2.p1.3 "Model. ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [14]A. Singh, A. Fry, A. Perelman, A. Tart, A. Ganesh, A. El-Kishky, A. McLaughlin, A. Low, A. Ostrow, A. Ananthram, et al. (2025)Openai gpt-5 system card. arXiv preprint arXiv:2601.03267. Cited by: [Table 2](https://arxiv.org/html/2603.17948#S4.T2.1.5.4.1 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [15]C. V. Snell, J. Lee, K. Xu, and A. Kumar (2025)Scaling llm test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, Cited by: [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px5.p1.2 "Environment Budgeting vs. Prior Compute Adaptation. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [16]G. Team, R. Anil, S. Borgeaud, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. (2023)Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805. Cited by: [Table 2](https://arxiv.org/html/2603.17948#S4.T2.1.6.5.1 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [17]X. Wang, Y. Zhang, O. Zohar, and S. Yeung-Levy (2024)Videoagent: long-form video understanding with large language model as agent. In European Conference on Computer Vision,  pp.58–76. Cited by: [§1](https://arxiv.org/html/2603.17948#S1.p2.1 "1 Introduction ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [18]Z. Wang, S. Yu, E. Stengel-Eskin, J. Yoon, F. Cheng, G. Bertasius, and M. Bansal (2025)Videotree: adaptive tree-based video representation for llm reasoning on long videos. In Proceedings of the Computer Vision and Pattern Recognition Conference,  pp.3272–3283. Cited by: [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px3.p1.1 "Agentic, Hierarchical, and Memory Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [19]J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. (2022)Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35,  pp.24824–24837. Cited by: [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px5.p1.2 "Environment Budgeting vs. Prior Compute Adaptation. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [20]H. Wu, D. Li, B. Chen, and J. Li (2024)Longvideobench: a benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems 37,  pp.28828–28857. Cited by: [§1](https://arxiv.org/html/2603.17948#S1.p2.1 "1 Introduction ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px1.p1.2 "Long-Form Video Understanding. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§4.1](https://arxiv.org/html/2603.17948#S4.SS1.SSS0.Px1.p1.1 "Benchmarks. ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [21]W. Yeo, K. Kim, J. Yoon, and S. J. Hwang (2025)WorldMM: dynamic multimodal memory agent for long video reasoning. External Links: 2512.02425, [Link](https://arxiv.org/abs/2512.02425)Cited by: [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px3.p1.1 "Agentic, Hierarchical, and Memory Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [22]Y. Yin, Q. Meng, M. Chen, J. Ding, Z. Shao, and Z. Yu (2025)VideoARM: agentic reasoning over hierarchical memory for long-form video understanding. arXiv preprint arXiv:2512.12360. Cited by: [§1](https://arxiv.org/html/2603.17948#S1.p2.1 "1 Introduction ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px3.p1.1 "Agentic, Hierarchical, and Memory Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px4.p1.1 "Long Context as the Core Challenge. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px5.p1.2 "Environment Budgeting vs. Prior Compute Adaptation. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [Table 1](https://arxiv.org/html/2603.17948#S2.T1.3.5.4.1 "In Agentic, Hierarchical, and Memory Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§3.1](https://arxiv.org/html/2603.17948#S3.SS1.SSS0.Px2.p1.1 "Action Space. ‣ 3.1 VideoAtlas ‣ 3 Methodology ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§4.1](https://arxiv.org/html/2603.17948#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [Table 2](https://arxiv.org/html/2603.17948#S4.T2.1.13.12.1 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [Table 2](https://arxiv.org/html/2603.17948#S4.T2.1.14.13.1 "In 4.2 Main Results ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [23]A. L. Zhang, T. Kraska, and O. Khattab (2025)Recursive language models. arXiv preprint arXiv:2512.24601. Cited by: [§1](https://arxiv.org/html/2603.17948#S1.SS0.SSS0.Px1.p1.1 "From representation to reasoning. ‣ 1 Introduction ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px4.p1.1 "Long Context as the Core Challenge. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§3.2](https://arxiv.org/html/2603.17948#S3.SS2.p1.4 "3.2 Video-RLM: Master-Worker Architecture ‣ 3 Methodology ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [24]C. Zhang, T. Lu, M. M. Islam, Z. Wang, S. Yu, M. Bansal, and G. Bertasius (2024)A simple llm framework for long-range video question-answering. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,  pp.21715–21737. Cited by: [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px2.p1.1 "Caption-Based Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [Table 1](https://arxiv.org/html/2603.17948#S2.T1.3.3.2.1 "In Agentic, Hierarchical, and Memory Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§4.1](https://arxiv.org/html/2603.17948#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 
*   [25]X. Zhang, Z. Jia, Z. Guo, J. Li, B. Li, H. Li, and Y. Lu (2025)Deep video discovery: agentic search with tool use for long-form video understanding. arXiv preprint arXiv:2505.18079. Cited by: [§1](https://arxiv.org/html/2603.17948#S1.p2.1 "1 Introduction ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px3.p1.1 "Agentic, Hierarchical, and Memory Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§2](https://arxiv.org/html/2603.17948#S2.SS0.SSS0.Px4.p1.1 "Long Context as the Core Challenge. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [Table 1](https://arxiv.org/html/2603.17948#S2.T1.3.6.5.1 "In Agentic, Hierarchical, and Memory Approaches. ‣ 2 Related Work ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [§4.1](https://arxiv.org/html/2603.17948#S4.SS1.SSS0.Px3.p1.1 "Baselines. ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). 

## Appendix

We present a complete end-to-end trace of Video-RLM answering “How many yellow cards were given in this video?” on a 25-minute FIFA World Cup Final highlight reel (90,117 frames). [Figures 6](https://arxiv.org/html/2603.17948#Sx2.F6 "In Appendix ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [8](https://arxiv.org/html/2603.17948#Sx2.F8 "Figure 8 ‣ Appendix ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") and[7](https://arxiv.org/html/2603.17948#Sx2.F7 "Figure 7 ‣ Appendix ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") show the three stages of the pipeline.

![Image 6: Refer to caption](https://arxiv.org/html/2603.17948v1/figures/trace_grid_initial.jpg)

(a) Initial root grid (Round 0)

⇓\boldsymbol{\Downarrow}

![Image 7: Refer to caption](https://arxiv.org/html/2603.17948v1/figures/trace_grid_round8.jpg)

(b) After 8 rounds of exploration

Figure 6: Navigation grid before and after exploration.(a)The Master’s initial 8×\times 8 root grid provides a temporal overview of the full 25-minute video; each cell covers ∼\sim 23 s. (b)After 8 DFS rounds, explored regions are blacked out (24 of 64 cells), visually showing the coverage pattern. The system explored 4.7% of total frames.

![Image 8: Refer to caption](https://arxiv.org/html/2603.17948v1/figures/trace_scratchpad.jpg)

Figure 7: Final evidence scratchpad. The system’s lossless visual memory after 8 rounds of exploration: 51 collected frames with burned-in labels [A]-[AY], each paired with a timestamp and natural-language description. Representative entries include “[D] @86.3 s: Referee shows yellow card to Marcus Thuram (Minute 18),”“[O] @371.1 s: Referee gives yellow card to Enzo Fernández (39′),” Not all items are yellow-card events: the scratchpad also captures contextual evidence such as match score overlays, player close-ups, and celebration scenes, enabling the Master to cross-reference events against the full match timeline. This grid image is passed directly to the Master for the final decision.

![Image 9: Refer to caption](https://arxiv.org/html/2603.17948v1/figures/reasoning_trace.png)

Figure 8: Exploration trace. Condensed log of the 8-round DFS exploration. The Master probes the root grid and dispatches 3 parallel Workers (W) per round to the highest-scored Cells (C). Workers use Zoom, Investigate, Expand, and Backtrack to drill into their assigned regions and collect evidence via Add_to_Scratchpad. After each round, the Master runs an uncertainty analysis to decide whether to continue or declare sufficiency. Evidence grows from 0 to 51 items across 8 rounds (97 VLM calls, 449K tokens), with the Master declaring Final_Decision after identifying 8 distinct yellow card events, which is the correct answer.

## A Detailed Error Analysis

We extend the error analysis from [Sec.4](https://arxiv.org/html/2603.17948#S4 "4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") with a systematic characterization of failure modes. To isolate environment failures from backbone limitations, we analyze all questions where our two backbone configurations (Qwen3.5-35B-A3B, 3B active; Gemini-3-Flash) _disagree_ under identical VideoAtlas configurations. Across LongVideoBench-Long and VideoMME-Long combined, we observe 522 disagreement cases; the stronger backbone is correct in 423 of these (81%), confirming that the dominant failure modes are backbone-dependent rather than architectural. We identify three systematic patterns from these cases, described below. These patterns are not mutually exclusive: a single failure may exhibit more than one.

### A.1 VLM Perception Errors

The agent navigates to the correct temporal region but misperceives the visual content. Two sub-patterns emerge.

(i)Attribute confusion. The agent correctly identifies the scene and entities but misreads fine-grained attributes like colors, materials, spatial relationships, or on-screen text. This is especially common when the question hinges on a single distinguishing visual feature (e.g., the color of a specific object, the label on a chart axis).

(ii)Cross-frame inconsistency. The backbone produces contradictory descriptions of the same scene across different frames, then arbitrarily selects one rather than reconciling the conflict. For example, describing the same object as “purple/pink” in one frame, “white against blue” in another, and “blue” in a third.

### A.2 Surface-Text Latching

A reasoning failure where the agent anchors on a phrase in the evidence/subtitle that superficially matches a candidate answer, without verifying whether the match is contextually correct. The agent’s reasoning frequently contains high-confidence language (“the evidence _explicitly states_…,” “this _directly supports_ candidate X”) but that confidence is built on literal pattern-matching rather than understanding. This is particularly problematic in documentary and educational videos, where narrators use rhetorical phrasings that contain candidate-answer keywords without implying them as the correct answer.

### A.3 Early Evidence Anchoring

The agent commits to an answer based on the first plausible evidence item it encounters, failing to integrate later evidence that would contradict or refine its conclusion. This failure mode interacts with the system’s sufficiency mechanism: the Master may declare evidence “sufficient” after a single supporting item, rather than verifying coverage across all candidate answers.

### A.4 Impact of Backbone Quality

All three failure patterns are _backbone-dependent_: switching to a stronger VLM under identical VideoAtlas configuration, prompts, and exploration budget resolves the majority of these errors without any architectural changes. The 4:1 to 5:1 win ratio across both benchmarks reflects this consistently. [Figures 9](https://arxiv.org/html/2603.17948#S2.F9 "In B Per-Category Accuracy Breakdown ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") and[9](https://arxiv.org/html/2603.17948#S2.F9 "Figure 9 ‣ B Per-Category Accuracy Breakdown ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") illustrate two representative cases.

These results reinforce the main paper’s finding that VideoAtlas performance scales directly with backbone capability, and suggest that the framework is well-positioned to benefit from future advances in VLM quality.

## B Per-Category Accuracy Breakdown

LongVideoBench annotates questions along three axes: _question type_, _reasoning level_, and _topic_. [Tabs.4](https://arxiv.org/html/2603.17948#S2.T4 "In B Per-Category Accuracy Breakdown ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"), [5](https://arxiv.org/html/2603.17948#S2.T5 "Table 5 ‣ B Per-Category Accuracy Breakdown ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") and[6](https://arxiv.org/html/2603.17948#S2.T6 "Table 6 ‣ B Per-Category Accuracy Breakdown ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") report Video-RLM (Qwen3.5, 3B active) accuracy on both LVB-Long and LVB-10hr.

Several patterns emerge: (1)_Sequence-type questions are hardest_: SSS (Scene Sequence Summary, 21.4%) and SAA (33.3%) require ordering multiple events across the full video, demanding both broad coverage and temporal precision. (2)_Perception degrades more than relation at 10 hours_: L1-Perception drops 9.6 points (59.4→\to 49.8) vs. L2-Relation dropping 2.7 points (47.6→\to 44.9), suggesting that the additional temporal distance primarily harms low-level visual recognition rather than relational reasoning. (3)_Life-Vlogs are consistently hardest_: at 35.3% (1hr) and 26.7% (10hr), these videos feature rapid visual changes, informal framing, and minimal subtitles, stressing the VLM’s perception most severely.

Table 4: Accuracy (%) by question type. Categories present only in one split are marked with –.

Table 5: Accuracy (%) by reasoning level.

Table 6: Accuracy (%) by topic category.

![Image 10: Refer to caption](https://arxiv.org/html/2603.17948v1/figures/answer_for_position.png)

Failure mode: VLM Perception Error Question:“Which direction is the narrator in red facing relative to the narrator in green?”Correct answer: Right front (A)

![Image 11: Refer to caption](https://arxiv.org/html/2603.17948v1/figures/answer_for_desert.png)

Failure mode: Surface-Text Latching Question:“What is the primary reason for the appearance of giraffe images on the rocks?”Correct answer: Climate change (C)

Figure 9: Representative backbone failure cases.Top:Perception error: the lightweight backbone cannot perceive a clearly visible host wearing red and concludes the entity does not exist; the stronger backbone identifies both hosts and reasons about their spatial relationship. Bottom:Surface-text latching: the lightweight backbone matches a rhetorical narrator phrase to a candidate answer with high confidence; the stronger backbone synthesizes broader evidence to identify the underlying scientific explanation. Both runs use identical VideoAtlas configurations.

## C Compute Breakdown

[Tab.7](https://arxiv.org/html/2603.17948#S3.T7 "In C Compute Breakdown ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") decomposes the average token consumption per question into Master and Worker contributions. The Master accounts for a small fraction of total tokens (global probing, uncertainty analysis, final decision), while the bulk of compute is spent on Worker exploration. This confirms that the parallel architecture efficiently distributes work: the Master acts as a lightweight coordinator while Workers perform the heavy visual exploration.

Table 7: Average per-question compute breakdown for Video-RLM (Qwen3.5, 3B active).

At 10 hours, Worker tokens increase by 80% (121K→\to 219K) while Master tokens increase by only 10% (27.5K→\to 30.4K), confirming that the Master’s coordination overhead scales minimally with video duration. The additional Worker cost reflects deeper exploration (2.3 vs. 2.0 rounds) needed to locate evidence in a 10×\times longer video, yet the increase is far sub-linear relative to the 10×\times duration increase, consistent with the logarithmic scaling property described in [Sec.4.3](https://arxiv.org/html/2603.17948#S4.SS3 "4.3 Logarithmic Compute Scaling ‣ 4 Experiments ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute"). Evidence items increase modestly (6.2→\to 7.2), suggesting the system explores more but collects evidence at a similar density.

### C.1 Multimodal Token Efficiency

During DFS exploration, each worker re-examines the same grid view across multiple reasoning steps (e.g., Add_to_Scratchpad steps that do not change the navigation state), creating inherent redundancy in visual token processing. For Qwen (self-hosted via vLLM), this is handled transparently: vLLM’s automatic multimodal prefix caching detects repeated image token prefixes across requests and serves KV cache hits without any code changes. [Tab.8](https://arxiv.org/html/2603.17948#S3.T8 "In C.1 Multimodal Token Efficiency ‣ C Compute Breakdown ‣ VideoAtlas: Navigating Long-Form Video in Logarithmic Compute") reports the measured hit rates across video durations.

Table 8: vLLM multimodal prefix cache hit rates for Qwen3.5 across video durations.

## D Prompt Templates

We include the complete prompt templates used by VideoAtlas. All prompts are zero-shot (no in-context examples). The same templates are used across all benchmarks and video durations without modification.

### D.1 Search Task Extraction

A text-only call that converts the raw query and answer candidates into a concrete visual search task.

Convert this question + choices into
a concrete SEARCH TASK for exploring a video.

Question: "{query}"
Choices:
{candidates}

Describe EXACTLY what to look for visually.
Be specific about scenes, objects,
text overlays, or transitions
that would confirm each choice.
Output only the search task, no preamble.

### D.2 Master: Global Probing

The Master examines the root grid (with dead zones blacked out) and ranks the top-N N cells for worker assignment.

You are analyzing a KxK grid of frames sampled
from a SINGLE video in chronological order
(left-to-right, top-to-bottom).

**QUERY:** "{query}"

**GRID CELLS:**
{context_str}

Pick EXACTLY {top_n} cells (no more, no fewer)
most likely to help answer the query.

**OUTPUT (raw JSON, EXACTLY {top_n} entries):**
{"top": [{"id": <cell_id>}, ...]}

### D.3 Master: Uncertainty Analysis

After each round with new evidence, the Master performs three tasks in one call: sufficiency check, explore suggestions, and noise erasure.

**UNCERTAINTY ANALYSIS**

You are the MASTER coordinator analyzing
search progress for a video question.

**QUERY:** "{query}"
**ANSWER CHOICES:**
{candidates}
**EVIDENCE COLLECTED SO FAR:**
{evidence_text}
**EXPLORATION PROGRESS:**
{progress_text}
**NAVIGATION GRID (blacked-out = explored):**
{context_str}

**YOUR 3 TASKS (do all in one response):**

1. **UNCERTAINTY CHECK:** For each answer choice,
   do you have sufficient evidence?
2. **EXPLORE SUGGESTIONS:** Suggest up to {N}
   regions. ONLY suggest non-blacked-out cells.
   - Cell IDs from the grid
   - Custom time ranges {"start","end"} (<60s)
3. **ERASE NOISE:** ONLY erase evidence completely
   unrelated to query, task, and ALL choices.
   Keep partial evidence. When in doubt, keep it.

**If sufficient:** {"action": "FINAL_DECISION",...}
**Otherwise:** {"action": "CONTINUE",
  "reasoning": "...", "explore": [...],
  "erase": [...]}

### D.4 Worker: Exploration Step

Each worker receives a grid view of its assigned region and the available tool set. The prompt includes a 1-sentence summary of the previous step’s outcome (conversation history).

SEARCH TASK: "{search_task}"
QUERY: "{query}"

You are exploring [{start}-{end}s]
({pct}% through a {duration}s video, depth {d}).
Grid: {K}x{K}, chronological L-to-R, T-to-B.

**CELLS:**
{context_str}

**PREVIOUS:** {prev_summary}

**RULES:**
- EXPAND into promising cells to zoom in
- Use ZOOM only when you found a relevant scene
  and need a closer high-resolution look
- Use INVESTIGATE only when you found the anchor
  scene and need to check what happens before/after
- ADD_TO_SCRATCHPAD with timestamp, description,
  and confidence when you find evidence
- FINISHED when region has no relevant content

Pick ONE action. Be precise with timestamps.

### D.5 Master: Final Decision

After exploration terminates, the Master sees the evidence scratchpad grid and evaluates each candidate.

You are making a FINAL DECISION based on all
collected visual evidence.

QUERY: "{query}"
ANSWER CHOICES:
{candidates}

EVIDENCE (see grid image with burned-in labels):
{evidence_descriptions}

For EACH choice: state which evidence supports
or contradicts it. Then select the best-supported
answer.

**OUTPUT:**
{"answer": <choice index>, "reasoning": "..."}

