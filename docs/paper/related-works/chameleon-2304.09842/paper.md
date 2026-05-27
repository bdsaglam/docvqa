# Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models

Pan Lu<sup>1</sup>, Baolin Peng<sup>2</sup>, Hao Cheng<sup>2</sup>, Michel Galley<sup>2</sup>  
 Kai-Wei Chang<sup>1</sup>, Ying Nian Wu<sup>1</sup>, Song-Chun Zhu<sup>1</sup>, Jianfeng Gao<sup>2</sup>  
<sup>1</sup>University of California, Los Angeles <sup>2</sup>Microsoft Research, Redmond

<https://chameleon-llm.github.io>

Figure 1 illustrates the Chameleon approach for multi-modal question answering. It shows three examples of questions and the reasoning paths used to answer them:

- **Example 1:** A baseball player pushing a bat. The question is "What is the direction of this push?". The reasoning path involves an Image Captioner, Knowledge Retrieval, Solution Generator, and Answer Generator.
- **Example 2:** An advertisement for Earthzy paper plates. The question is "Which is the main persuasive appeal used in this ad?". The reasoning path involves a Text Detector, Knowledge Retrieval, Solution Generator, and Answer Generator. The reasoning includes:
  - Text: "[194, 21]: Earthzy paper plates now carry the Sierra Club seal of approval. [172, 112]: carry the Sierra Club seal of approval [231, 159]: Club seal of approval [275, 198]: approval"
  - Analysis: "Analyze the advertisement to determine which persuasive appeal is used. The ad mentions that Earthzy paper plates carry the Sierra Club seal of approval. This implies that the product is environmentally friendly and has been endorsed by a reputable organization. This appeal is primarily based on the credibility and authority of the Sierra Club, which is an example of ethos (character)."
  - Conclusion: "ethos (character)"
- **Example 3:** A polar bear. The question is "Which animal's skin is adapted for survival in cold places?". The reasoning path involves an Image Captioner, Query Generator, Bing Search, Solution Generator, and Answer Generator.

The central Chameleon model synthesizes these reasoning paths to provide the final answer.

Figure 1: Examples from our **Chameleon** approach with GPT-4 on ScienceQA [32], a multi-modal question answering benchmark in scientific domains. **Chameleon** is adaptive to different queries by synthesizing programs to compose various tools and executing them sequentially to get final answers.

## Abstract

Large language models (LLMs) have achieved remarkable progress in solving various natural language processing tasks due to emergent reasoning abilities. However, LLMs have inherent limitations as they are incapable of accessing up-to-date information (stored on the Web or in task-specific knowledge bases), using external tools, and performing precise mathematical and logical reasoning. In this paper, we present **Chameleon**, an AI system that mitigates these limitations by augmenting LLMs with *plug-and-play* modules for compositional reasoning. **Chameleon** synthesizes programs by composing various tools (e.g., LLMs, off-the-shelf vision models, web search engines, Python functions, and heuristic-based modules) for accomplishing complex reasoning tasks. At the heart of **Chameleon** is an LLM-based planner that assembles a sequence of tools to execute to generate the final response. We showcase the effectiveness of **Chameleon** on two multi-modal knowledge-intensive reasoning tasks: ScienceQA and TabMWP. **Chameleon**, powered by GPT-4, achieves an 86.54% overall accuracy on ScienceQA, improving the best published few-shot result by 11.37%. On TabMWP, GPT-4-powered **Chameleon** improves the accuracy by 17.0%, lifting the state of the art to 98.78%. Our analysis also shows that the GPT-4-powered planner exhibits more consistent and rational tool selection via inferring potential constraints from instructions, compared to a ChatGPT-powered planner.

This title draws inspiration from the *chameleon*'s ability to adapt and blend into its surroundings, which parallels the adaptability and versatility of large language models in compositional reasoning tasks with external tools.# 1 Introduction

Remarkable progress has been observed in recent large language models (LLMs) for various natural language processing tasks, with prominent examples such as GPT-3 [4], PaLM [8], LLaMA [64], ChatGPT [41], and the recently developed GPT-4 [42]. LLMs have demonstrated emergent abilities, including in-context learning and chain-of-thought (CoT) reasoning [56]. These models are capable of solving diverse tasks in a zero-shot fashion [25] or with the aid of a few examples [57], and they show great potential in planning and decision-making akin to human beings [17, 16]. Despite these capabilities, LLMs face inherent limitations, such as an inability to access up-to-date information [26], perform precise mathematical reasoning [44, 35], or utilize specialized models [49]. Therefore, enhancing current LLMs with the capability to automatically *compose* external tools for real-world task solving is critical to address these drawbacks.

Consider the example ② in Figure 1: *Which is the main persuasive appeal used in this ad?*. To answer this question, one needs to: 1) infer that there is an ad image containing text context and call a text decoder to understand the semantics; 2) retrieve background knowledge about *persuasive appeals* and the differences among three persuasive appeals; 3) generate a solution based on the input query and intermediate results from previous steps; and 4) finally produce the answer in a task-specific format. On the other hand, when answering *Which animal’s skin is adapted for survival in cold places* (③), one might need to call modules such as an image captioner to decipher image information and a web search engine to retrieve domain knowledge to understand scientific terminologies. However, current tool-augmented LLMs still face challenges when addressing these real-world queries across various scenarios. Most existing approaches are either limited to a small number of tools [39, 6, 55, 18, 43, 49] or relying on domain-specific tools [40, 60, 13, 59, 52], and thus are not easy to generalize to queries of new domains (see sections 2 and A.1 for further discussion). In this work, we study how to enable LLMs to synthesize programs to capture the logic of composing heterogeneous tools.

To address the challenges of existing work, we introduce **Chameleon**, a *plug-and-play compositional reasoning framework* that leverages LLMs to synthesize programs and compose various tools for a wide range of tasks. Unlike existing tool-augmented LLMs [49, 40, 60, 13, 59, 52], **Chameleon** uses a richer set of tools, including LLMs, off-the-shelf vision models, web search engines, Python functions, and heuristics-based modules. Moreover, **Chameleon** leverages the in-context learning capabilities of LLMs and builds on an LLM as a natural language planner, without requiring any training or carefully curated rules. Prompted by tool descriptions and usage examples, the planner infers a program composed of a sequence of tools to execute in order to generate the final response for a user query. Instead of generating programs in domain-specific languages [40, 52, 13], **Chameleon** generates natural-language-like (NL) programs (e.g., [Text\_Detector, Knowledge\_Retrieval, Solution\_Generator, Answer\_Generator] for the second query in Figure 1). The NL-like programs are easy to understand and debug by users with limited programming experience, and easily extendable to new modules. During each module’s execution, the module processes the query and cached context, returns a result determined by the module itself, and updates the query and context for subsequent execution. Composing modules as a sequential program allows subsequent modules to leverage prior cached context and updated queries.

We showcase the adaptability and effectiveness of **Chameleon** on two tasks: ScienceQA [32] and TabMWP [33]. ScienceQA is a multi-modal question answering benchmark spanning multiple context formats and various scientific topics, while TabMWP is a mathematical benchmark involving diverse tabular contexts. These two benchmarks serve as a good testbed to evaluate **Chameleon**’s ability to coordinate diverse tools across different types and domains. Notably, **Chameleon** with GPT-4 achieves an 86.54% accuracy on ScienceQA, significantly improving upon the best published few-shot model by 11.37%. On TabMWP, using GPT-4 as the underlying LLM, **Chameleon** achieves an improvement of 7.97% over chain-of-thought (CoT) prompted GPT-4 [57] and a 17.0% increase over the best-published model [6], lifting the state of the art to 98.78%. Further studies suggest that using GPT-4 as a planner exhibits more consistent and rational tool selection and is able to infer potential constraints given the instructions, compared to other LLMs like ChatGPT.

Our contributions are as follows: (1) We develop a plug-and-play compositional reasoning framework, **Chameleon**, that effectively composes external tools to address inherent limitations of LLMs and tackle a broad range of reasoning tasks. (2) Relying on an LLM as a natural language planner to generate programs, **Chameleon** successfully integrates various tools, including LLMs, off-the-shelf vision models, web search engines, Python functions, and rule-based modules, to build a versatile and**Miles hiked**

<table border="1">
<thead>
<tr>
<th>Day</th>
<th>Miles</th>
</tr>
</thead>
<tbody>
<tr>
<td>Sunday</td>
<td>10</td>
</tr>
<tr>
<td>Monday</td>
<td>9</td>
</tr>
<tr>
<td>Tuesday</td>
<td>10</td>
</tr>
<tr>
<td>Wednesday</td>
<td>5</td>
</tr>
<tr>
<td>Thursday</td>
<td>9</td>
</tr>
</tbody>
</table>

Wanda went on a camping trip and logged the number of miles she hiked each day. What is the **median** of the numbers?

**Location**

<table border="1">
<thead>
<tr>
<th>Location</th>
<th>Arrive</th>
<th>Depart</th>
</tr>
</thead>
<tbody>
<tr>
<td>stadium</td>
<td>10:20 A.M.</td>
<td>10:25 A.M.</td>
</tr>
<tr>
<td>park</td>
<td>10:35 A.M.</td>
<td>10:45 A.M.</td>
</tr>
<tr>
<td>hotel</td>
<td>11:10 A.M.</td>
<td>11:15 A.M.</td>
</tr>
<tr>
<td>airport</td>
<td>12:05 P.M.</td>
<td>12:10 P.M.</td>
</tr>
<tr>
<td>train station</td>
<td>12:25 P.M.</td>
<td>12:35 P.M.</td>
</tr>
<tr>
<td>bus station</td>
<td>1:10 P.M.</td>
<td>1:10 P.M.</td>
</tr>
</tbody>
</table>

Look at the following schedule. When does the bus **depart** from the **train station**?  
(A) 12:35 P.M. (B) 1:10 P.M.  
(C) 1:10 P.M. (D) 10:45 A.M.

**Knowledge Retrieval**  
This table shows the number of miles Wanda hiked each day on her camping trip. The **median** is the middle value in a set of data when the data is **arranged in order**. To find the **median**, the data must be **arranged in order from least to greatest (or greatest to least)**, and then the **middle value(s) is/are determined**.

**Program Generator**  
`miles_hiked = [10, 9, 10, 5, 9]`  
`miles_hiked = sorted(miles_hiked)`  
`middle1 = (len(miles_hiked) - 1) // 2`  
`middle2 = len(miles_hiked) // 2`  
`ans = (miles_hiked[middle1] + miles_hiked[middle2]) / 2`

**Program Verifier**  
`ans = 9.0`

**Program Executor**  
`ans = 9.0`

**Answer Generator**  
`9`

**Row Lookup**  

<table border="1">
<thead>
<tr>
<th>Location</th>
<th>Arrive</th>
<th>Depart</th>
</tr>
</thead>
<tbody>
<tr>
<td>train station</td>
<td>12:25 P.M.</td>
<td>12:35 P.M.</td>
</tr>
</tbody>
</table>

**Solution Generator**  
(Step 1) Find the train station on the schedule. Find the **departure** time for the **train station**.  
(Step 2) **Train station**: 12:35 P.M. The bus **departs** from the **train station** at 12:35 P.M.  
(Step 3) **The answer is 12:35 P.M.**

**Answer Generator**  
`12:35 P.M.`

Figure 2: Two examples from our **Chameleon** approach with GPT-4 on TabMWP [33], a mathematical reasoning benchmark with tabular contexts. **Chameleon** demonstrates flexibility and efficiency in adapting to different queries that require various reasoning abilities.

adaptable AI system capable of answering real-world queries. (3) We demonstrate **Chameleon**’s effectiveness on two challenging benchmarks, significantly surpassing the state of the art.

## 2 Related Work

**Compositional Reasoning** Neural modular and compositional approaches have been explored to automatically perform desired sub-task decomposition, enhancing interpretability and adaptability across various reasoning tasks. Early work [2, 3] posits that complex reasoning tasks are fundamentally compositional and proposes neural module networks (NMN) to decompose them into subtasks. However, these methods rely on brittle off-the-shelf parsers and are limited by module configurations. Some later work [19, 15, 14, 21], takes a step further by predicting instance-specific network layouts in an end-to-end manner, without relying on parsers, using reinforcement learning [58] and weak supervised learning. In visual reasoning, models comprising a program generator and an execution engine have been proposed to combine deep representation learning and symbolic program execution [19, 61]. In the domain of mathematical reasoning, an interpretable solver has been developed to incorporate theorem knowledge as conditional rules and perform symbolic reasoning step by step [31]. Our work takes inspiration from neural module networks, yet it offers several distinct advantages. First, **Chameleon** does not require expensive supervision of task-specific programs for modeling training. Instead, it generates sequential programs, consisting of modules, that are easy to generalize to various domains and tasks, allowing the extension to new modules in a plug-and-play manner. Second, **Chameleon** does not require any training, but uses the in-context learning capabilities of LLMs to generate programs prompted by natural language instruction and demonstrations.

**Tool-Augmented Language Models** In recent years, the development of large language models (LLMs) [48, 8, 9, 53, 4, 41, 42] has made tremendous progress and has stimulated research in prompt learning [57, 33, 22] and instruction learning [53, 64, 46, 11]. Despite the impressive performance of LLMs, they suffer from inherent limitations, such as the inability to access up-to-date information [26], utilize external tools [49], or perform precise mathematical reasoning [44, 35]. Recent benchmarks, such as ScienceQA and TabMWP [32, 33, 7, 54, 51, 30], have emerged to evaluate the capability of LLMs to tackle intricate reasoning challenges, especially those emphasizing the use of external tools. Concurrently, there has been a growing interest in harnessing external tools and modular approaches to augment LLMs. These augmented LLMs can access real-time information aided by web search engines [40] and leverage domain-specific knowledge from external resources [62]. Some work leverages the Python interpreter to generate complex programs to employ powerful computational resources, and execute logical reasoning tasks more effectively [55, 10, 6, 39, 18, 43, 36]. For example, Toolformer [49] constructs tool-use augmented data to train language models to select five<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="6">Tool Use</th>
<th colspan="5">Skill Dimension</th>
<th colspan="3">Inference &amp; Extension</th>
</tr>
<tr>
<th>Size</th>
<th>OpenAI</th>
<th>Hugging Face</th>
<th>Github</th>
<th>Web search</th>
<th>code</th>
<th>Image</th>
<th>Web</th>
<th>Know.</th>
<th>Math</th>
<th>Table</th>
<th>Composition</th>
<th>Planning</th>
<th>Plug-n-Play</th>
</tr>
</thead>
<tbody>
<tr>
<td>CoT [57]</td>
<td>1</td>
<td>✓</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>Lila [39]</td>
<td>1</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>PoT [6]</td>
<td>2</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>Code4Struct [55]</td>
<td>1</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>PAL [10]</td>
<td>2</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>MathPrompter [18]</td>
<td>2</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>ART [43]</td>
<td>4</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td>Toolformer [49]</td>
<td>5</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>natural lang.</td>
<td>✗</td>
</tr>
<tr>
<td>WebGPT [40]</td>
<td>10</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>program</td>
<td>✗</td>
</tr>
<tr>
<td>MM-ReAct [60]</td>
<td>&gt;10</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>word match</td>
<td>✓</td>
</tr>
<tr>
<td>Visual ChatGPT [59]</td>
<td>&gt;10</td>
<td>✓</td>
<td>-</td>
<td>-</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>natural lang.</td>
<td>✓</td>
</tr>
<tr>
<td>ViperGPT [52]</td>
<td>&gt;10</td>
<td>✓</td>
<td>-</td>
<td>-</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✓</td>
<td>program</td>
<td>✓</td>
</tr>
<tr>
<td>VisProg [13]</td>
<td>&gt;10</td>
<td>✓</td>
<td>-</td>
<td>-</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>program</td>
<td>✓</td>
</tr>
<tr>
<td>HuggingGPT [50]</td>
<td>&gt;10</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>-</td>
<td>✗</td>
<td>-</td>
<td>✓</td>
<td>natural lang.</td>
<td>✓</td>
</tr>
<tr>
<td><b>Chameleon (ours)</b></td>
<td>&gt;10</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>natural lang.</td>
<td>✓</td>
</tr>
</tbody>
</table>

Table 1: A comparison of work that augments large language models with tool usage. We report the tool size and tool types, including OpenAI (🤖), Hugging Face (🐼), Github (🐙), Web search (🔍), and code (🐹). We compare the skills each method possesses, such as image understanding, browser search, knowledge retrieval, mathematical reasoning, and table understanding. Some models can compose various tools, propose a planner to infer the relevant tools for execution, or are inherently extendable to new tools. The label “-” refers to uncertain information in the literature.

tools. In the realm of visual tools, various approaches have been proposed to enhance the capabilities of large language models in handling visual tasks [60, 59, 52, 13, 50], augmented with Hugging Face models [50], Azure models [60], visual foundation models [59].

We compare **Chameleon** with other tool-augmented language models in Table 1. Many of these approaches are either constrained to a small set of tools or limited to task-specific tools, which reduces their capabilities across various skill dimensions and hampers their generalizability to new tasks. A recent line of work relies on large amounts of supervision [49, 26] and focuses on generating commands [40] and programs [52, 13] to infer the choice of tools. However, this approach needs to carefully tailored prompts to specific tasks and particular tools, and is neither flexible nor adaptive. In contrast, **Chameleon** instructs LLMs with natural language instructions that simply describe the roles of each module and provide a few calling examples, eliminating the need for additional training or tool-specific prompts when learning to compose different tools. More importantly, **Chameleon** offers users flexibility in terms of tool types and sources, updating the underlying LLMs, adding new tools, and adapting to new tasks. Our work shares the same spirit of AutoGPT [47], an autonomous GPT-4 agent with the artificial general intelligence (AGI) ambition to incorporate numerous tools to achieve user-defined goals. While AutoGPT is still under development, our work is the first to instantiate the idea and verify its effectiveness on well-studied benchmarks.

### 3 General Framework: **Chameleon**

To address the limitations of current LLMs in utilizing diverse tools, we propose **Chameleon**, a novel *plug-and-play compositional* reasoning framework, synthesizing the composition of various tools to accommodate a wide range of problems. **Chameleon** is comprised of a *module inventory* that defines different types of tools and an LLM-based *planner*, whose purpose is to decompose the original problem into sub-tasks that can be effectively solved by task-specific tools. Unlike existing tool-augmented LLM approaches [49, 13, 59, 50], our module inventory features multiple tool types as illustrated in Table 2, enabling **Chameleon** to exhibit various reasoning abilities, including image understanding, knowledge retrieval, web search, complex mathematical reasoning, and table understanding. Instead of generating domain-specific programs [40, 13, 52], **Chameleon** employs an LLM-based planner to create natural-language-like programs that follow natural language instructions, which is less error-prone, easily expandable to new modules, and user-friendly.

We formalize our planner as follows: given the input query  $x_0$ , the module inventory  $\mathcal{M}$ , and constraints  $\mathcal{G}$ , the natural language planner  $\mathcal{P}$  selects a set of modules that can be executed sequentially to answer the query via generating a program in a natural-language-like format. The module inventory  $\mathcal{M}$  consists of a set of pre-built modules:  $\{M_i\}$ , each corresponding to a tool of various types (Table 2).  $\mathcal{G}$  are the constraints for the plan generation, for example, the concurrent relations and sequenceorders of modules. In our work, the planner  $\mathcal{P}$  is an LLM prompted to generate a sequence of module names in a few-shot setup. The planner is prompted in natural language with a planning task instruction  $\mathcal{I}$ , the descriptions of modules in  $\mathcal{M}$  with corresponding constraints  $\mathcal{G}$ , as well as a few demonstration examples  $\mathcal{D}$ . A  $T$ -length plan sampled from  $\mathcal{P}$  can be denoted as  $p = M^1, \dots, M^T$ , where  $M^t$  represents the  $t$ -th element in the generated plan and  $M^t \in \mathcal{M}$ . Formally, given an input query (problem statement)  $x_0$ , a plan  $p$  is generated as follows:

$$p \leftarrow \mathcal{P}(x_0; \mathcal{I}, \mathcal{M}, \mathcal{G}, \mathcal{D}). \quad (1)$$

Given the generated plan, the corresponding modules for each step are then executed sequentially. The plan is a natural-language program where each module is bound simply via string matching. When evaluating the module  $M^t$  at time step  $t$ , the output of the execution  $y^t$  is calculated by:

$$y^t \leftarrow M^t(x^{t-1}; c^{t-1}), \quad (2)$$

where  $x^{t-1}$  is the input for the current module  $M^t$ , and  $c^{t-1}$  is the cached information (e.g., image semantics, retrieved knowledge, generated programs) resulting from the execution history of modules. Both the problem input  $x^t$  and cache  $c^t$  for the next module  $M^{t+1}$  are updated, respectively, by:

$$x^t \leftarrow \text{update\_input}(x^{t-1}, y^t), \quad (3)$$

$$c^t \leftarrow \text{update\_cache}(c^{t-1}, y^t). \quad (4)$$

The `update_input` and `update_cache` functions are hand-designed for each  $M_i$ . Specifically, `update_input` is applied to elements in the input query, including the question, table context, and image. These elements are updated after module execution. `update_cache` corresponds to the generation of new information, such as a description for the input image or retrieved knowledge from external resources. Finally, the response  $r$  to the query is generated by the last module  $M^T$ :

$$r = y^T \leftarrow M^T(x^{T-1}; c^{T-1}). \quad (5)$$

## 4 Applications of Chameleon

We demonstrate the applications of **Chameleon** on two challenging tasks: ScienceQA [32] (section 4.2) and TabMWP [33] (section 4.3), using the module inventory introduced in section 4.1. Further experimental details can be found in appendix A.2.

### 4.1 Module Inventory

To accommodate various reasoning capabilities over a diverse range of queries, our system utilizes a rich module inventory of various external tools. We provide a high-level overview of this inventory here, with detailed implementations in specific experiments. The complete module inventory,  $\mathcal{M}$ , is presented in Table 2. Each tool within the inventory is defined as follows:

**Knowledge Retrieval:** This module retrieves additional background knowledge crucial for tackling complex problems. It is especially beneficial for specialized domains like science and mathematics, providing context for the task. For example, if a query is about a tax form table, this module could generate knowledge about tax procedures, offering valuable context.

**Bing Search:** Like “Knowledge Retrieval”, the “Bing Search” module aims to provide wide-ranging task-relevant knowledge. In contrast, it excels when broader or up-to-date information from multiple sources is required. Using the search engine API, this module returns relevant search results based on the input query, which are then parsed and used by subsequent modules to gather richer context information from diverse sources, enhancing problem-solving effectiveness.

**Query Generator:** Since the original problem typically lacks a tailored query for retrieving task-relevant information, this module creates search engine queries based on the problem, which are then

<table border="1">
<thead>
<tr>
<th>Tool Types</th>
<th>Tools</th>
</tr>
</thead>
<tbody>
<tr>
<td> OpenAI</td>
<td>Knowledge Retrieval, Query Generator, Row Lookup, Column Lookup, Table Verbalizer, Program Generator, Solution Generator</td>
</tr>
<tr>
<td> Hugging Face</td>
<td>Image Captioner</td>
</tr>
<tr>
<td> Github</td>
<td>Text Detector</td>
</tr>
<tr>
<td> Web Search</td>
<td>Bing Search</td>
</tr>
<tr>
<td> Python</td>
<td>Program Verifier, Program Executor</td>
</tr>
<tr>
<td> Rule-based</td>
<td>Answer Generator</td>
</tr>
</tbody>
</table>

Table 2: Different tools in our module inventory.used by the “Bing Search” module. Mostly, it is a good strategy to use the “Query Generator” module before the “Bing Search”. Coupled with the search engine tool, generating more targeted queries generally facilitates both the recall and precision of retrieved information.

**Image Captioner:** Designed to generate captions for images, this module provides crucial supplementary context for queries. It is particularly valuable when understanding an image semantically, like identifying objects and interactions in a scene. Using pre-trained models, it translates visual data into language, facilitating effective comprehension and reasoning about image content.

**Text Detector:** This module is designed to identify text within a given image. Typically, the “Text Detector” is employed when a question requires the extraction of textual information from images containing diagrams, charts, tables, maps, or other visual elements. By effectively detecting text in various formats, this module aids in the analysis and understanding of image-based content.

**Row Lookup:** This module is crucial when queries involve tabular context, as locating relevant cells is often required. Large tables can distract the system, so “Row Lookup” simplifies the table by retaining only the rows relevant to the query. If all rows are pertinent, it returns the original table.

**Column Lookup:** Like the “Row Lookup” module, “Column Lookup” addresses questions involving tabular context by focusing on relevant columns. It simplifies the table by retaining only pertinent columns, or returns the original table if all columns are relevant.

**Table Verbalizer:** Converting structured tables into text is likely to enhance the comprehension of tabular information by various downstream modules as shown by [37] for open-domain question answering, making this module a vital part of our system. It translates tables into easily understandable descriptions for modules like “Program Generator” and “Solution Generator”, particularly useful for small, domain-specific tables like stem-and-leaf plots or function tables.

**Program Generator:** Program-aided approaches are shown to enhance the logical and mathematical reasoning abilities of LLMs [55, 10, 6, 39, 18, 43]. The “Program Generator” generates Python programs to solve queries effectively, which is particularly beneficial for queries requiring complex computations or intricate logical operations, such as “if-else” statements.

**Program Verifier:** Recent studies highlight the importance of verification to reduce hallucination [45, 38]. Hence, “Program Verifier” ensures the validity and error-free nature of programs generated by “Program Generator”. It checks for syntax and logical errors, and potential execution issues, enhancing the reliability and accuracy of the solutions.

**Program Executor:** This module executes the program generated by “Program Generator” and produces the result, bridging the gap between program generation and final solution derivation.

**Solution Generator:** This module generates a detailed solution to the input query using all the cached information. Employing a chain-of-thought prompting approach [57], it ensures coherent and well-structured responses. The planner can directly employ this module instead of other functional modules if it can solve the query independently, especially for simpler ones.

**Answer Generator:** This task-specific module uses a rule-based approach to extract and normalize answers from the results of the “Program Executor” or “Solution Generator”. Unlike the Solution Generator that provides detailed multi-step solutions, “Answer Generator” serves as the final module in the pipeline, providing concise and task-specific answers.

## 4.2 Science Question Answering

Science Question Answering (ScienceQA [32]) is a diverse benchmark for multi-modal question answering over a range of scientific topics and contexts. As examples illustrated in Figure 1, answering these questions requires various tools and skills like image captioning, text detection, knowledge retrieval, online resource search, and multi-clue visual reasoning. When generating programs for using tools, we limit the search space to the relevant inventory subset (Table 6 in the appendix). Programs are deemed invalid and default to a “Solution Generator” and “Answer Generator” sequence if these are not the final two elements, following the chain-of-thought prompting baseline [57]. See Table 8 in the appendix for the constructed natural language planner prompt. The prompts for LLM-based modules like “Knowledge Retrieval”, “Query Generator”, and “Solution Generator” are shown in Table 10, 11, and 12, respectively, in the appendix.<table border="1">
<thead>
<tr>
<th>Model</th>
<th>#Tuned<br/>Params</th>
<th>ALL</th>
<th>NAT</th>
<th>SOC</th>
<th>LAN</th>
<th>TXT</th>
<th>IMG</th>
<th>NO</th>
<th>G1-6</th>
<th>G7-12</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="11"><i>Heuristic baselines</i></td>
</tr>
<tr>
<td>Random Choice [32]</td>
<td>-</td>
<td>39.83</td>
<td>40.28</td>
<td>46.13</td>
<td>29.25</td>
<td>47.45</td>
<td>40.08</td>
<td>33.66</td>
<td>39.35</td>
<td>40.67</td>
</tr>
<tr>
<td>Human [32]</td>
<td>-</td>
<td>88.40</td>
<td>90.23</td>
<td>84.97</td>
<td>87.48</td>
<td>89.60</td>
<td>87.50</td>
<td>88.10</td>
<td>91.59</td>
<td>82.42</td>
</tr>
<tr>
<td colspan="11"><i>Fine-tuned models</i></td>
</tr>
<tr>
<td>MCAN [63]</td>
<td>95M</td>
<td>54.54</td>
<td>56.08</td>
<td>46.23</td>
<td>58.09</td>
<td>59.43</td>
<td>51.17</td>
<td>55.40</td>
<td>51.65</td>
<td>59.72</td>
</tr>
<tr>
<td>Top-Down [1]</td>
<td>70M</td>
<td>59.02</td>
<td>59.50</td>
<td>54.33</td>
<td>61.82</td>
<td>62.90</td>
<td>54.88</td>
<td>59.79</td>
<td>57.27</td>
<td>62.16</td>
</tr>
<tr>
<td>BAN [23]</td>
<td>112M</td>
<td>59.37</td>
<td>60.88</td>
<td>46.57</td>
<td>66.64</td>
<td>62.61</td>
<td>52.60</td>
<td>65.51</td>
<td>56.83</td>
<td>63.94</td>
</tr>
<tr>
<td>DFAF [12]</td>
<td>74M</td>
<td>60.72</td>
<td>64.03</td>
<td>48.82</td>
<td>63.55</td>
<td>65.88</td>
<td>54.49</td>
<td>64.11</td>
<td>57.12</td>
<td>67.17</td>
</tr>
<tr>
<td>ViLT [24]</td>
<td>113M</td>
<td>61.14</td>
<td>60.48</td>
<td>63.89</td>
<td>60.27</td>
<td>63.20</td>
<td>61.38</td>
<td>57.00</td>
<td>60.72</td>
<td>61.90</td>
</tr>
<tr>
<td>Patch-TRM [34]</td>
<td>90M</td>
<td>61.42</td>
<td>65.19</td>
<td>46.79</td>
<td>65.55</td>
<td>66.96</td>
<td>55.28</td>
<td>64.95</td>
<td>58.04</td>
<td>67.50</td>
</tr>
<tr>
<td>VisualBERT [27, 28]</td>
<td>111M</td>
<td>61.87</td>
<td>59.33</td>
<td>69.18</td>
<td>61.18</td>
<td>62.71</td>
<td>62.17</td>
<td>58.54</td>
<td>62.96</td>
<td>59.92</td>
</tr>
<tr>
<td>UnifiedQA [20]</td>
<td>223M</td>
<td>70.12</td>
<td>68.16</td>
<td>69.18</td>
<td>74.91</td>
<td>63.78</td>
<td>61.38</td>
<td>77.84</td>
<td>72.98</td>
<td>65.00</td>
</tr>
<tr>
<td>UnifiedQA CoT [32]</td>
<td>223M</td>
<td>74.11</td>
<td>71.00</td>
<td>76.04</td>
<td>78.91</td>
<td>66.42</td>
<td>66.53</td>
<td>81.81</td>
<td>77.06</td>
<td>68.82</td>
</tr>
<tr>
<td>MM-COT<sub>T</sub> [65]</td>
<td>223M</td>
<td>70.53</td>
<td>71.09</td>
<td>70.75</td>
<td>69.18</td>
<td>71.16</td>
<td>65.84</td>
<td>71.57</td>
<td>71.00</td>
<td>69.68</td>
</tr>
<tr>
<td>MM-COT [65]</td>
<td>223M</td>
<td>84.91</td>
<td>87.52</td>
<td>77.17</td>
<td>85.82</td>
<td>87.88</td>
<td>82.90</td>
<td>86.83</td>
<td>84.65</td>
<td>85.37</td>
</tr>
<tr>
<td>MM-COT<sub>Large</sub> [65]</td>
<td>738M</td>
<td>91.68</td>
<td>95.91</td>
<td>82.00</td>
<td>90.82</td>
<td>95.26</td>
<td>88.80</td>
<td>92.89</td>
<td>92.44</td>
<td>90.31</td>
</tr>
<tr>
<td>LLaMA-Adapter<sub>T</sub> [64]</td>
<td>1.2M</td>
<td>78.31</td>
<td>79.00</td>
<td>73.79</td>
<td>80.55</td>
<td>78.30</td>
<td>70.35</td>
<td>83.14</td>
<td>79.77</td>
<td>75.68</td>
</tr>
<tr>
<td>LLaMA-Adapter [64]</td>
<td>1.8M</td>
<td>85.19</td>
<td>84.37</td>
<td>88.30</td>
<td>84.36</td>
<td>83.72</td>
<td>80.32</td>
<td>86.90</td>
<td>85.83</td>
<td>84.05</td>
</tr>
<tr>
<td colspan="11"><i>Few-shot GPT-3</i></td>
</tr>
<tr>
<td>GPT-3 [4]</td>
<td>0M</td>
<td>74.04</td>
<td>75.04</td>
<td>66.59</td>
<td>78.00</td>
<td>74.24</td>
<td>65.74</td>
<td>79.58</td>
<td>76.36</td>
<td>69.87</td>
</tr>
<tr>
<td>GPT-3 CoT [32]</td>
<td>0M</td>
<td>75.17</td>
<td>75.44</td>
<td>70.87</td>
<td>78.09</td>
<td>74.68</td>
<td>67.43</td>
<td>79.93</td>
<td>78.23</td>
<td>69.68</td>
</tr>
<tr>
<td colspan="11" style="text-align: center;">Published results (Above) ▲</td>
</tr>
<tr>
<td colspan="11"><i>Few-shot ChatGPT</i></td>
</tr>
<tr>
<td>ChatGPT CoT</td>
<td>0M</td>
<td>78.31</td>
<td>78.82</td>
<td>70.98</td>
<td>83.18</td>
<td>77.37</td>
<td>67.92</td>
<td>86.13</td>
<td>80.72</td>
<td>74.03</td>
</tr>
<tr>
<td><b>Chameleon (ChatGPT)</b></td>
<td>0M</td>
<td>79.93</td>
<td>81.62</td>
<td>70.64</td>
<td>84.00</td>
<td>79.77</td>
<td>70.80</td>
<td>86.62</td>
<td>81.86</td>
<td>76.53</td>
</tr>
<tr>
<td colspan="11"><i>Few-shot GPT-4</i></td>
</tr>
<tr>
<td>GPT-4 CoT</td>
<td>0M</td>
<td>83.99</td>
<td>85.48</td>
<td>72.44</td>
<td>90.27</td>
<td>82.65</td>
<td>71.49</td>
<td>92.89</td>
<td>86.66</td>
<td>79.04</td>
</tr>
<tr>
<td><b>Chameleon (GPT-4)</b></td>
<td>0M</td>
<td>86.54</td>
<td>89.83</td>
<td>74.13</td>
<td>89.82</td>
<td>88.27</td>
<td>77.64</td>
<td>92.13</td>
<td>88.03</td>
<td>83.72</td>
</tr>
</tbody>
</table>

Table 3: **QA accuracy (%) on the test set of ScienceQA [32]**. We report the number of tuned parameters for this task and the overall accuracy, along with accuracy scores for different question types, including natural, social, and language sciences, text, image, and no context, as well as grades 1-6 and 7-12. The highest scores among models in each section and overall are highlighted in **blue** and **red**, respectively, and the results of our best model are marked in **bold**.

### 4.3 Tabular Mathematical Reasoning

TabMWP [33] is a mathematical reasoning task involving diverse tabular contexts like schedules, prices, tax forms, plots, and function relations (Figure 2). It requires AI systems to understand various table formats and perform precise numerical or symbolic computations. Like ScienceQA, we constrain the program search space to focus on two tool types: 1) those helping LLMs better digest tabular information (e.g., “Row Lookup”, “Column Lookup”, and “Table Verbalizer”) and 2) those performing faithful symbolic computations (e.g., “Program Generator”, “Program Verifier”, and “Program Executor”) as listed in Table 6. The generated programs must meet certain constraints, such as including “Answer Generator” and placing “Program Generator” prior to both “Program Verifier” and “Program Executor”. Non-compliant programs default to a sequence of “Program Generator”, “Program Verifier”, “Program Executor”, and “Answer Generator”, aligning with the program-of-thought prompting baseline [6] with added verification.

## 5 Experiments

We assess **Chameleon**’s effectiveness and adaptability on two complex reasoning tasks, ScienceQA [32] and TabMWP [33]. See experimental details in appendix A.2.

### 5.1 Experimental Results

**ScienceQA**. Table 3 presents the results of existing baselines and our approach **Chameleon**, with key results highlighted in Figure 3 (a). Employing ChatGPT [41] as the base LLM, **Chameleon**<table border="1">
<thead>
<tr>
<th>Model</th>
<th>#Tuned<br/>Params</th>
<th>ALL</th>
<th>FREE</th>
<th>MC</th>
<th>INT</th>
<th>DEC</th>
<th>EXTR</th>
<th>BOOL</th>
<th>OTH</th>
<th>G1-6</th>
<th>G7-8</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="12"><i>Heuristic baselines</i></td>
</tr>
<tr>
<td>Heuristic guess</td>
<td>-</td>
<td>15.29</td>
<td>6.71</td>
<td>39.81</td>
<td>8.37</td>
<td>0.26</td>
<td>30.80</td>
<td>51.22</td>
<td>26.67</td>
<td>17.55</td>
<td>12.27</td>
</tr>
<tr>
<td>Human performance</td>
<td>-</td>
<td>90.22</td>
<td>84.61</td>
<td>93.32</td>
<td>84.95</td>
<td>83.29</td>
<td>97.18</td>
<td>88.69</td>
<td>96.20</td>
<td>94.27</td>
<td>81.28</td>
</tr>
<tr>
<td colspan="12"><i>Fine-tuned models</i></td>
</tr>
<tr>
<td>UnifiedQA<sub>SMALL</sub> [20]</td>
<td>41M</td>
<td>29.79</td>
<td>22.27</td>
<td>51.31</td>
<td>27.27</td>
<td>2.83</td>
<td>52.28</td>
<td>48.11</td>
<td>69.52</td>
<td>35.85</td>
<td>21.71</td>
</tr>
<tr>
<td>UnifiedQA<sub>BASE</sub> [20]</td>
<td>223M</td>
<td>43.52</td>
<td>34.02</td>
<td>70.68</td>
<td>40.74</td>
<td>7.90</td>
<td>84.09</td>
<td>55.67</td>
<td>73.33</td>
<td>53.31</td>
<td>30.46</td>
</tr>
<tr>
<td>UnifiedQA<sub>LARGE</sub> [20]</td>
<td>738M</td>
<td>57.35</td>
<td>48.67</td>
<td>82.18</td>
<td>55.97</td>
<td>20.26</td>
<td>94.63</td>
<td>68.89</td>
<td>79.05</td>
<td>65.92</td>
<td>45.92</td>
</tr>
<tr>
<td>TAPEX<sub>BASE</sub> [29]</td>
<td>139M</td>
<td>48.27</td>
<td>39.59</td>
<td>73.09</td>
<td>46.85</td>
<td>11.33</td>
<td>84.19</td>
<td>61.33</td>
<td>69.52</td>
<td>56.70</td>
<td>37.02</td>
</tr>
<tr>
<td>TAPEX<sub>LARGE</sub> [29]</td>
<td>406M</td>
<td>58.52</td>
<td>51.00</td>
<td>80.02</td>
<td>59.92</td>
<td>16.31</td>
<td>95.34</td>
<td>64.00</td>
<td>73.33</td>
<td>67.11</td>
<td>47.07</td>
</tr>
<tr>
<td colspan="12"><i>Zero-shot GPT-3</i></td>
</tr>
<tr>
<td>GPT-3 [4]</td>
<td>0M</td>
<td>56.96</td>
<td>53.57</td>
<td>66.67</td>
<td>55.55</td>
<td>45.84</td>
<td>78.22</td>
<td>55.44</td>
<td>54.29</td>
<td>63.37</td>
<td>48.41</td>
</tr>
<tr>
<td>GPT-3 CoT [57]</td>
<td>0M</td>
<td>57.61</td>
<td>54.36</td>
<td>66.92</td>
<td>55.82</td>
<td>48.67</td>
<td>78.82</td>
<td>55.67</td>
<td>51.43</td>
<td>63.62</td>
<td>49.59</td>
</tr>
<tr>
<td colspan="12"><i>Few-shot GPT-3</i></td>
</tr>
<tr>
<td>GPT-3 [4]</td>
<td>0M</td>
<td>57.13</td>
<td>54.69</td>
<td>64.11</td>
<td>58.36</td>
<td>40.40</td>
<td>75.95</td>
<td>52.41</td>
<td>53.02</td>
<td>63.10</td>
<td>49.16</td>
</tr>
<tr>
<td>GPT-3 CoT [57]</td>
<td>0M</td>
<td>62.92</td>
<td>60.76</td>
<td>69.09</td>
<td>60.04</td>
<td>63.58</td>
<td>76.49</td>
<td>61.19</td>
<td>67.30</td>
<td>68.62</td>
<td>55.31</td>
</tr>
<tr>
<td>GPT-3 CoT-PromptPG [33]</td>
<td>0M</td>
<td>68.23</td>
<td>66.17</td>
<td>74.11</td>
<td>64.12</td>
<td>74.16</td>
<td>76.19</td>
<td>72.81</td>
<td>65.71</td>
<td>71.20</td>
<td>64.27</td>
</tr>
<tr>
<td>Codex* [5]</td>
<td>0M</td>
<td>59.4</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Codex PoT* [6]</td>
<td>0M</td>
<td>73.2</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Codex PoT-SC* [6]</td>
<td>0M</td>
<td>81.8</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
</tbody>
</table>

Published results (Above) ▲

<table border="1">
<tbody>
<tr>
<td colspan="12"><i>Few-shot ChatGPT</i></td>
</tr>
<tr>
<td>ChatGPT CoT</td>
<td>0M</td>
<td>82.03</td>
<td>78.43</td>
<td>92.32</td>
<td>75.38</td>
<td>90.30</td>
<td>92.30</td>
<td>92.89</td>
<td>87.62</td>
<td>83.06</td>
<td>80.66</td>
</tr>
<tr>
<td>ChatGPT PoT</td>
<td>0M</td>
<td>89.49</td>
<td>90.24</td>
<td>87.35</td>
<td>89.31</td>
<td>93.82</td>
<td>92.10</td>
<td>85.89</td>
<td>55.24</td>
<td>90.60</td>
<td>88.00</td>
</tr>
<tr>
<td><b>Chameleon (ChatGPT)</b></td>
<td>0M</td>
<td>93.28</td>
<td>93.13</td>
<td>93.72</td>
<td>92.71</td>
<td>94.76</td>
<td>91.29</td>
<td>98.11</td>
<td>78.85</td>
<td>93.37</td>
<td>93.17</td>
</tr>
<tr>
<td colspan="12"><i>Few-shot GPT-4</i></td>
</tr>
<tr>
<td>GPT-4 CoT</td>
<td>0M</td>
<td>90.81</td>
<td>88.48</td>
<td>97.49</td>
<td>86.16</td>
<td>97.51</td>
<td>96.86</td>
<td>99.11</td>
<td>89.52</td>
<td>92.40</td>
<td>88.70</td>
</tr>
<tr>
<td>GPT-4 PoT</td>
<td>0M</td>
<td>96.93</td>
<td>97.40</td>
<td>95.58</td>
<td>98.48</td>
<td>93.22</td>
<td>96.25</td>
<td>98.00</td>
<td>68.57</td>
<td>96.97</td>
<td>96.87</td>
</tr>
<tr>
<td><b>Chameleon (GPT-4)</b></td>
<td>0M</td>
<td><b>98.78</b></td>
<td><b>98.95</b></td>
<td><b>98.29</b></td>
<td><b>99.34</b></td>
<td><b>97.42</b></td>
<td><b>98.58</b></td>
<td><b>98.56</b></td>
<td><b>93.33</b></td>
<td><b>98.95</b></td>
<td><b>98.54</b></td>
</tr>
</tbody>
</table>

Table 4: **QA accuracy (%) on the test set of TabMWP [33]**. We report the number of tuned parameters for this task and the overall accuracy, and accuracy of different question types, including free-text questions, multi-choice questions, integer answers, decimal answers, extractive answers, Boolean answers, other text answers, grades 1-6, and grades 7-8. \* refers to a subset of results.

Figure 3: Results of main baselines and **Chameleon**. Dashed lines represent human performance.

achieves a 79.93% accuracy, a 1.62% improvement over Chain-of-Thought (CoT) [57] prompted ChatGPT. Notably, **Chameleon** is a generalized form of CoT, where the generated program is a sequence of “Solution Generator” and “Answer Generator”. **Chameleon** benefits from additional tool usage, such as “Knowledge Retrieval”, “Bing Search”, “Image Captioner”, and “Text Detector”. When built upon GPT-4 [42], our model attains an accuracy of 86.54%, outperforming GPT-4 CoT [32] by 2.55% and GPT-3 CoT by 11.37%, creating the new state of the art in few-shot settings.

**TabMWP.** Table 4 presents results with key models in Figure 3 (b). Similarly, significant improvements are observed for **Chameleon** over both fine-tuned and few-shot models. It is worth noting that both CoT and Program-of-Thought (PoT) [6] can be viewed as special cases of **Chameleon**. Apart from “Solution Generator” and “Answer Generator”, CoT doesn’t utilize any tool, while PoTFigure 4: Tools called in the generated programs from **Chameleon** on ScienceQA.

Figure 5: Tools called in the generated programs from **Chameleon** on TabMWP.

only relies on symbolic programming tools like “Program Generator” and “Program Executor”. **Chameleon** (ChatGPT) outperforms ChatGPT CoT and ChatGPT PoT by 11.25% and 3.79%, respectively, emphasizing the advantage of our enriched tool set. With GPT-4, **Chameleon** gains an additional 5.50%, reaching a 98.78% accuracy. Notably, **Chameleon** (GPT-4) surpasses Codex PoT-SC [6], the best-published model, by 17.0% and human performance by 8.56%.

## 5.2 Qualitative Analysis

**Tool use planning.** The proportions of key tools called in the programs from **Chameleon** on ScienceQA and TabMWP are visualized in Figure 4 and Figure 5, respectively. Interestingly, ChatGPT and GPT-4 exhibit different planning behaviors. Generally, ChatGPT has a strong bias toward using or not using certain tools, highly influenced by in-context examples. For instance, ChatGPT calls “Knowledge Retrieval” in 72% of queries but only calls “Bing Search” in 3% of cases on ScienceQA; on TabMWP, ChatGPT heavily relies on “Row Lookup” (47%) but calls “Column Lookup” less frequently (4%). However, GPT-4 acts more *objectively* and *rationally* in tool selection. For example, GPT-4 calls “Knowledge Retrieval” more frequently (81% vs. 72%) and calls “Bing Search” more than ChatGPT (11% vs. 3%) when answering scientific questions on ScienceQA. Impressively, GPT-4 consistently calls “Query Generator” and “Bing Search” simultaneously by observing the tool usage descriptions, while ChatGPT lacks such reasoning capability.

**Ablation study with disabled modules.** We study the accuracy decline of **Chameleon** when key modules in the generated programs are disabled (Table 5), using ChaptGPT as the underlying LLMs and 500 test examples. The results reveal that “Knowledge Retrieval” plays a vital role in both tasks. Domain-specific tools, such as the search engine and vision models for ScienceQA, and program tools for TabMWP, also prove to be important.

<table border="1">
<thead>
<tr>
<th>Module</th>
<th><math>\Delta</math> (ScienceQA)</th>
<th><math>\Delta</math> (TabMWP)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Knowledge Retrieval</td>
<td>-7.8%</td>
<td>-2.2%</td>
</tr>
<tr>
<td>Bing Search</td>
<td>-7.4%</td>
<td>-</td>
</tr>
<tr>
<td>Text Detector</td>
<td>-8.4%</td>
<td>-</td>
</tr>
<tr>
<td>Image Captioner</td>
<td>-6.0%</td>
<td>-</td>
</tr>
<tr>
<td>Program Generator</td>
<td>-</td>
<td>-7.4%</td>
</tr>
<tr>
<td>Table Verbalizer</td>
<td>-</td>
<td>-0.2%</td>
</tr>
</tbody>
</table>

Table 5: Score drop with disabled modules.

**Module transitions.** We visualize the transition graphs of modules for generated programs by **Chameleon** (GPT-4) on ScienceQA and TabMWP in Figure 7 and 8, respectively. The transition probabilities in these graphs are computed from the tool transitions observed on the test sets. These graphs show that the GPT-4 planner is able to make good decisions on how to sequence tools in a few-shot setup. For example, on ScienceQA, **Chameleon** often decides to rely on either “Knowledge Retriever” or “Bing Search”, but rarely both. On TabMWP, we observe two main modes: either going through the solution generator module or via the program generator, verifier, and executor.

## 5.3 Case Study

**Visualization examples of ScienceQA.** Examples from **Chameleon** (GPT-4) on ScienceQA are visualized in Figure 1. **Chameleon** (GPT-4) is able to adapt to different input queries by generating programs that compose various tools and executing them sequentially to obtain accurate responses. For instance, to answer the first question (①), *What is the direction of this push?*, the system calls the image captioner model to extract semantic information from the image and employs the knowledge retrieval model to gather background knowledge for multi-modal reasoning. In the second example (②), the natural language planner infers that a text detector tool is needed to understand the contextof the ad. The third query (③; more details provided in Figure 9 in the appendix), *Which animal’s skin is adapted for survival in cold places?*, involves scientific terminology related to animal survival. The planner decides to call the Bing search engine to access domain-specific knowledge, benefiting from the numerous online resources.

**Visualization examples of TabMWP.** The adaptability and versatility of **Chameleon** for various queries are also observed on TabMWP, as illustrated in the examples in Figure 2. The first example (①) involves mathematical reasoning on a tax form. **Chameleon** (1) calls the knowledge retrieval model to recall basic knowledge that assists in understanding this domain-specific table, (2) describes the table in a more readable natural language format, and (3) finally relies on program-aided tools to perform precise computations. In the second example (②), the system generates Python code that closely aligns with the background knowledge provided by the knowledge retrieval model. The third example (③) requires the system to locate the cell in a large tabular context given the input query. **Chameleon** calls the row lookup model to help accurately locate the relevant rows and generate the language solution via an LLM model, instead of relying on program-based tools.

**Failure cases and limitations.** Failure examples from **Chameleon** (GPT-4) are illustrated in Tables 19 to 24 in the appendix. Inaccurate responses may arise from the limitations of the current modules or from suboptimal programs generated by the planner. Additionally, the module inventory may lack tools capable of addressing specific abilities. Future directions could involve upgrading the modules and the planner, or expanding the module inventory to support a broader range of capabilities. Further limitations and broader impacts are respectively discussed in sections B and C of the appendix.

## 5.4 Error Analysis

To examine the error sources of the base large language models and understand how our model reduces mistakes from different aspects, we conduct an error analysis, as shown in Figure 6. We select 50 mistake examples from the ChatGPT baseline on ScienceQA as the evaluation set. We count the number of mistake examples and analyze their corresponding mistake type categories for ChatGPT, our **Chameleon** (ChatGPT) approach, and **Chameleon** (GPT-4).

The results show that our **Chameleon** approach can substantially reduce the number of mistakes compared to ChatGPT. Our model features tools for image captioning and knowledge retrieval, thus the mistakes made by ChatGPT in the category of image understanding are reduced to 10 and 19 from 32 by **Chameleon** (ChatGPT) and **Chameleon** (GPT-4); while the mistakes made by ChatGPT in the category of knowledge understanding are reduced to 6 and 3 from 37 by **Chameleon** (ChatGPT) and **Chameleon** (GPT-4). Benefiting from the sequential execution of tools, the mistakes caused by solution generation are significantly reduced as well. Additionally, we find that the task planning of GPT-4 outperforms ChatGPT by a large margin.

Figure 6: # of mistake examples in different categories on ScienceQA. Image: image captioning, Knowledge: knowledge understanding, Solution: solution generation.

## 6 Conclusion

In conclusion, we introduce a novel *plug-and-play compositional* reasoning framework, **Chameleon**, that addresses the limitations of current large language models by augmenting them with external tools in a plug-and-play manner. Our approach employs a diverse set of tools and demonstrates impressive adaptability and effectiveness on two challenging benchmarks, ScienceQA and TabMWP. By achieving significant improvements in accuracy over existing state-of-the-art models, **Chameleon** showcases its potential for addressing real-world queries across various domains.## Acknowledgment

We would like to thank Chunyuan Li, Qiuyuan Huang, and other members of the Deep Learning group at Microsoft Research for their valuable discussions. We also thank Fan Yin from University of California, Los Angeles, and Mingyang Sun from University of Electronic Science and Technology of China for their thorough review of our paper and constructive feedback. Pan Lu’s research for this work was financially supported by Microsoft during his visit at Microsoft Research, and was also partially supported by the Amazon PhD Fellowship, Bloomberg PhD Fellowship, Qualcomm Innovation Fellowship, and UCLA Dissertation Year Fellowship. Kai-Wei was supported an ONR grant N00014-23-1-2780 and as a Sloan Fellow.

## References

- [1] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018.
- [2] Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Learning to compose neural networks for question answering. In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 1545–1554, 2016.
- [3] Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*, pages 39–48, 2016.
- [4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.
- [5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.
- [6] Wenhui Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. *Transactions on Machine Learning Research*, 2023.
- [7] Wenhui Chen, Ming Yin, Max Ku, Pan Lu, Elaine Wan, Xueguang Ma, Jianyu Xu, Tony Xia, and Xinyi Wang. TheoremQA: A theorem-driven question answering dataset. In *The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2023.
- [8] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrman, et al. PaLM: Scaling language modeling with pathways. *arXiv preprint arXiv:2204.02311*, 2022.
- [9] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. *arXiv preprint arXiv:2210.11416*, 2022.
- [10] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In *International Conference on Machine Learning*, pages 10764–10799. PMLR, 2023.
- [11] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, Hongsheng Li, and Yu Qiao. LLaMA-Adapter V2: Parameter-efficient visual instruction model. *arXiv preprint arXiv:2304.15010*, 2023.
- [12] Peng Gao, Zhengkai Jiang, Haoxuan You, Pan Lu, Steven CH Hoi, Xiaogang Wang, and Hongsheng Li. Dynamic fusion with intra-and inter-modality attention flow for visual question answering. In *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 6639–6648, 2019.
- [13] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14953–14962, 2023.
- [14] Ronghang Hu, Jacob Andreas, Trevor Darrell, and Kate Saenko. Explainable neural computation via stack neural module networks. In *Proceedings of the European conference on computer vision (ECCV)*, pages 53–69, 2018.- [15] Ronghang Hu, Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Kate Saenko. Learning to reason: End-to-end module networks for visual question answering. In *Proceedings of the IEEE international conference on computer vision*, pages 804–813, 2017.
- [16] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In *International Conference on Machine Learning*, pages 9118–9147. PMLR, 2022.
- [17] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. In *6th Annual Conference on Robot Learning*, 2022.
- [18] Shima Imani, Liang Du, and Harsh Shrivastava. Mathprompter: Mathematical reasoning using large language models. In *ICLR 2023 Workshop on Trustworthy and Reliable Large-Scale Machine Learning Models*, 2023.
- [19] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Judy Hoffman, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Inferring and executing programs for visual reasoning. In *Proceedings of the IEEE international conference on computer vision (ICCV)*, pages 2989–2998, 2017.
- [20] Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, and Hannaneh Hajishirzi. UnifiedQA: Crossing format boundaries with a single QA system. In *Findings of the Association for Computational Linguistics (EMNLP)*, pages 1896–1907, 2020.
- [21] Tushar Khot, Daniel Khashabi, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Text modular networks: Learning to decompose tasks in the language of existing models. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 1264–1279, 2021.
- [22] Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks. In *International Conference on Learning Representations (ICLR)*, 2023.
- [23] Jin-Hwa Kim, Jaehyun Jun, and Byoung-Tak Zhang. Bilinear attention networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, pages 1571–1581, 2018.
- [24] Wonjae Kim, Bokyung Son, and Ildoo Kim. ViLT: Vision-and-language transformer without convolution or region supervision. In *Proceedings of the 38th International Conference on Machine Learning (ICML)*, pages 5583–5594, 2021.
- [25] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. *Advances in neural information processing systems*, 35:22199–22213, 2022.
- [26] Mojtaba Komeili, Kurt Shuster, and Jason Weston. Internet-augmented dialogue generation. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 8460–8478, 2022.
- [27] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. VisualBERT: A simple and performant baseline for vision and language. *arXiv preprint arXiv:1908.03557*, 2019.
- [28] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. What does BERT with vision look at? In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, pages 5265–5275, 2020.
- [29] Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, and Jian-Guang Lou. TAPEX: Table pre-training via learning a neural sql executor. In *International Conference on Learning Representations (ICLR)*, 2022.
- [30] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. *arXiv preprint arXiv:2310.02255*, 2023.
- [31] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. InterGPS: Interpretable geometry problem solving with formal language and symbolic reasoning. In *The 59th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2021.- [32] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In *The 36th Conference on Neural Information Processing Systems (NeurIPS)*, 2022.
- [33] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In *International Conference on Learning Representations (ICLR)*, 2023.
- [34] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. IconQA: A new benchmark for abstract diagram understanding and visual language reasoning. In *The 35th Conference on Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks*, 2021.
- [35] Pan Lu, Liang Qiu, Wenhao Yu, Sean Welleck, and Kai-Wei Chang. A survey of deep learning for mathematical reasoning. In *The 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, 2023.
- [36] Yujie Lu, Pan Lu, Zhiyu Chen, Wanrong Zhu, Xin Eric Wang, and William Yang Wang. Multimodal procedural planning via dual text-image prompting. *arXiv preprint arXiv:2305.01795*, 2023.
- [37] Kaixin Ma, Hao Cheng, Xiaodong Liu, Eric Nyberg, and Jianfeng Gao. Open domain question answering with a unified knowledge interface. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1605–1620, Dublin, Ireland, May 2022. Association for Computational Linguistics.
- [38] Aman Madaan, Niket Tandon, Prakash Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-Refine: Iterative refinement with self-feedback. *arXiv preprint arXiv:2303.17651*, 2023.
- [39] Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, and Ashwin Kalyan. Lila: A unified benchmark for mathematical reasoning. In *The 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2022.
- [40] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. WebGPT: Browser-assisted question-answering with human feedback. *arXiv preprint arXiv:2112.09332*, 2021.
- [41] OpenAI. Chatgpt, 2022.
- [42] OpenAI. GPT-4 technical report. *ArXiv*, abs/2303.08774, 2023.
- [43] Bhargavi Paranjape, Scott Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, and Marco Tulio Ribeiro. ART: Automatic multi-step reasoning and tool-use for large language models. *arXiv preprint arXiv:2303.09014*, 2023.
- [44] Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP models really able to solve simple math word problems? In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2080–2094, 2021.
- [45] Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, and Jianfeng Gao. Check your facts and try again: Improving large language models with external knowledge and automated feedback. *arXiv preprint arXiv:2302.12813*, 2023.
- [46] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with GPT-4. *arXiv preprint arXiv:2304.03277*, 2023.
- [47] Toran Bruce Richards. Auto-GPT: An experimental open-source attempt to make GPT-4 fully autonomous. <https://github.com/Significant-Gravitas/Auto-GPT>, 2023.
- [48] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model. *arXiv preprint arXiv:2211.05100*, 2022.
- [49] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. *arXiv preprint arXiv:2302.04761*, 2023.- [50] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. HuggingGPT: Solving ai tasks with chatgpt and its friends in huggingface. *arXiv preprint arXiv:2303.17580*, 2023.
- [51] Liangtai Sun, Yang Han, Zihan Zhao, Da Ma, Zhennan Shen, Baocai Chen, Lu Chen, and Kai Yu. SciEval: A multi-level large language model evaluation benchmark for scientific research. *arXiv preprint arXiv:2308.13149*, 2023.
- [52] Dídac Surís, Sachit Menon, and Carl Vondrick. ViperGPT: Visual inference via python execution for reasoning. *arXiv preprint arXiv:2303.08128*, 2023.
- [53] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.
- [54] Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. SciBench: Evaluating college-level scientific problem-solving abilities of large language models. *arXiv preprint arXiv:2307.10635*, 2023.
- [55] Xingyao Wang, Sha Li, and Heng Ji. Code4Struct: Code generation for few-shot structured prediction from natural language. *arXiv preprint arXiv:2210.12810*, 2022.
- [56] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. *Transactions on Machine Learning Research*, 2022.
- [57] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35:24824–24837, 2022.
- [58] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Reinforcement learning*, pages 5–32, 1992.
- [59] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual ChatGPT: Talking, drawing and editing with visual foundation models. *arXiv preprint arXiv:2303.04671*, 2023.
- [60] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. MM-REACT: Prompting ChatGPT for multimodal reasoning and action. *arXiv preprint arXiv:2303.11381*, 2023.
- [61] Kexin Yi, Jiajun Wu, Chuang Gan, Antonio Torralba, Pushmeet Kohli, and Josh Tenenbaum. Neural-symbolic VQA: Disentangling reasoning from vision and language understanding. *Advances in neural information processing systems (NeurIPS)*, 31, 2018.
- [62] Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. Generate rather than retrieve: Large language models are strong context generators. In *International Conference on Learning Representations (ICLR)*, 2023.
- [63] Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and Qi Tian. Deep modular co-attention networks for visual question answering. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 6281–6290, 2019.
- [64] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Qiao Yu. LLaMA-Adapter: Efficient fine-tuning of language models with zero-init attention. *arXiv preprint arXiv:2303.16199*, 2023.
- [65] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. *arXiv preprint arXiv:2302.00923*, 2023.---

# Supplementary Materials for Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models

---

## A Appendix

### A.1 Current Tool-Augmented LLMs

To address the limitations of LLMs, an active research direction involves augmenting language models with access to external tools and resources, as well as exploring the integration of external tools and plug-and-play modular approaches. For example, aided by web search engines and external knowledge resources, LLMs are able to access real-time information and leverage domain-specific knowledge [40]. To enhance mathematical reasoning abilities, recent work uses LLMs [5] to generate complex programs to exploit powerful computational resources, and execute logical reasoning tasks more effectively [55, 10, 6, 39, 18, 43]. Another line of recent work, such as ViperGPT [52], Visual ChatGPT [59], VisProg [13], and HuggingGPT [50] incorporates a collection of foundation computer vision models to equip LLMs with the abilities to perform visual reasoning tasks.

### A.2 Experimental Details

**Module search space.** The inventory subsets for ScienceQA and TabMWP are shown in Table 6.

<table border="1"><thead><tr><th>Tool Types</th><th>Tools used on ScienceQA</th><th>Tools used on TabMWP</th></tr></thead><tbody><tr><td> OpenAI</td><td><b>Knowledge Retrieval</b>, Query Generator, <b>Solution Generator</b></td><td><b>Knowledge Retrieval</b>, Row Lookup, Column Lookup, Table Verbalizer, Program Generator, <b>Solution Generator</b></td></tr><tr><td> Hugging Face</td><td>Image Captioner</td><td></td></tr><tr><td> Github</td><td>Text Detector</td><td></td></tr><tr><td> Web Search</td><td>Bing Search</td><td></td></tr><tr><td> Python</td><td></td><td>Program Verifier, Program Executor</td></tr><tr><td> Rule-based</td><td><b>Answer Generator</b></td><td><b>Answer Generator</b></td></tr></tbody></table>

Table 6: Tools used on ScienceQA and TabMWP, respectively. Reusable tools are marked in **green**.

**Planner implementations.** We choose the *gpt-3.5-turbo* engine for ChatGPT and the *gpt-4* engine for GPT-4 when constructing the LLM-based planner. The maximum length for generated programs is set to 128, and the temperature is set to 0 for the most deterministic generation. The planner prompts for the ScienceQA and TabMWP are illustrated in Table 8 and Table 9, respectively.

**Module implementations for ScienceQA.** By default, the LLM-based models use four in-context examples as demonstrations, have a temperature setting of 0, and allow a maximum of 512 tokens for completion. Additional specific implementation details are provided as follows:

- • **Knowledge Retrieval:** The prompt consists of 3 demonstration examples and the template is shown in Table 10.
- • **Query Generator:** The prompt template is shown in Table 11. The maximum number of tokens for completion is set as 64.
- • **Solution Generator:** The prompt consists of 2 demonstration examples and the template is shown in Table 12.- • **Image Captioner:** We use the captioning model<sup>1</sup> to generate textual descriptions for input images. The maximum length of generated captions is set to 16, the number of beams is 4, and the maximum number of output tokens is 512.
- • **Text Detector:** This module is based on the github model<sup>2</sup> to extract the text contents with coordinates in the image.
- • **Bing Search:** This module calls the Bing Search API<sup>3</sup> and returns the top three responses for the text query.
- • **Answer Generator:** This module extracts the answer snippet from the result provided by the “Solution Generator” and selects the most similar option from the given choices.

**Module implementations for TabMWP.** Similar to ScienceQA, the LLM-based modules by default use four in-context examples as demonstrations, have a temperature setting of 0, and allow a maximum of 512 tokens for completion. Additional implementation details are provided as follows:

- • **Knowledge Retrieval:** The prompt consists of 5 demonstration examples and the template is shown in Table 13.
- • **Row Lookup:** It is enabled only when there are more than three rows and 18 table cells, in order to accelerate inference. The prompt consists of 7 demonstration examples and the template is shown in Table 14. The maximum number of tokens for completion is set as 256.
- • **Column Lookup:** Similarly, this module is enabled with two or more columns and 18 or more table cells. The prompt consists of 6 demonstration examples and the template is shown in Table 15. The maximum number of tokens for completion is set as 256.
- • **Table Verbalizer:** The prompt consists of 7 demonstration examples and the template is shown in Table 16.
- • **Program Generator:** The prompt template is shown in Table 17. The maximum number of tokens for completion is set as 256.
- • **Solution Generator:** The prompt consists of 16 demonstration examples and the template is shown in Table 18.
- • **Answer Generator:** It is used to normalize answers with two-place precision for questions with numerical answers and select the most similar option for multiple-choice questions.

**Implementations of update\_input and update\_cache.** update\_input is triggered by the execution of specific tools, like ‘Row\_Lookup’, which alter or replace elements in the input to reflect the updated state. Tools such as ‘Image\_Captioner’, ‘Text\_Detector’, ‘Knowledge\_Retrieval’, ‘Web\_Search’, and ‘Program\_Generation’ generate new elements. update\_cache stores these new elements in the cache, making them accessible for later tools’ execution.

### A.3 Experimental Results

**Generated program statistics.** Chameleon utilizes the LLM-based natural language planner to generate programs, i.e., sequences of used modules (tools). We report the statistics of the number of unique generated programs and the average length of corresponding tool sequences by Chameleon in Table 7. On both ScienceQA and TabMWP, using GPT-4 as the base LLM generates fewer distinct programs, i.e., more consistent programs, than using ChatGPT, even when given the exact same prompt in the planning model. Our results are consistent with the findings in [42], which observes that GPT-4 has a superior capability of understanding long contexts, aligning with human instructions, and performing high-level reasoning compared to other LLMs such as ChatGPT.

## B Limitations

While Chameleon represents a significant stride in exploiting large language models (LLMs) for compositional reasoning in a plug-and-play manner, there are a few areas that could benefit from

<sup>1</sup><https://huggingface.co/nlpconnect/vit-gpt2-image-captioning>

<sup>2</sup><https://github.com/JaidedAI/EasyOCR>

<sup>3</sup><https://www.microsoft.com/bing><table border="1">
<thead>
<tr>
<th>Task</th>
<th>Model</th>
<th># of different programs</th>
<th>Average program length</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">ScienceQA</td>
<td>Chain-of-thought (CoT)</td>
<td>1</td>
<td>2</td>
</tr>
<tr>
<td><b>Chameleon</b> (ChatGPT)</td>
<td>14</td>
<td>3.03</td>
</tr>
<tr>
<td><b>Chameleon</b> (GPT-4)</td>
<td>11</td>
<td>3.40</td>
</tr>
<tr>
<td rowspan="4">TabMWP</td>
<td>Chain-of-thought (CoT)</td>
<td>1</td>
<td>2</td>
</tr>
<tr>
<td>Program-of-thought (PoT)</td>
<td>1</td>
<td>3</td>
</tr>
<tr>
<td><b>Chameleon</b> (ChatGPT)</td>
<td>28</td>
<td>4.17</td>
</tr>
<tr>
<td><b>Chameleon</b> (GPT-4)</td>
<td>19</td>
<td>4.09</td>
</tr>
</tbody>
</table>

Table 7: The statistics of the number of different generated programs and the average length of generated programs by **Chameleon**, respectively. Chain-of-thought (CoT) prompting and Program-of-thought (PoT) prompting are also compared as they are the special cases of **Chameleon**.

further refinement. One such area is the expansion of its adaptability to a wider variety of tasks and domains, beyond the benchmarks presented. The LLM-based planner, responsible for synthesizing programs and determining the sequence of tools, introduces an innovative approach, yet it also raises intriguing research questions about optimizing the process for tool selection and sequence. It is plausible in the current system design that the quality of the LLM-based planner could impact overall performance. Moreover, **Chameleon** generates the program at one step, without incorporating a re-planning mechanism as the modules in the program are processed. Furthermore, we make the assumption that the list of modules and their descriptions will fit within the context window of LLMs, which may not always be the case. As the task complexity increases and the module inventory expands, there might be a corresponding surge in computational demands or limitations due to the context limit, indicating potential areas for future optimization. However, these potential areas for enhancement don’t detract from the paper’s central achievements, but instead provide valuable directions for future work and research.

## C Broader Impacts

The work presented in this paper, **Chameleon**, has significant potential for positive societal impact. By augmenting large language models (LLMs) with plug-and-play modules for compositional reasoning, **Chameleon** can provide more accurate responses to complex, multi-modal tasks, making it a potentially valuable framework for various applications, including but not limited to education, finance, and decision support systems. Additionally, the system’s ability to synthesize programs without requiring any training could democratize access to AI technology, enabling non-experts to leverage the power of AI in diverse fields. As research continues to advance in large language models and tool integration, we anticipate that our framework will serve as a foundation for further innovations in pursuing more generalizable and efficient solutions to complex reasoning tasks.

While there might be negative societal impacts associated with the **Chameleon**, such as misinformation and privacy concerns if data sources and external tools it utilizes are not curated meticulously, we believe these risks can be carefully managed and minimized. There’s also a risk that excessive reliance on **Chameleon**’s increased autonomy may undermine critical thinking skills or job functions. To effectively mitigate these issues, careful curation of data sources and external tools, along with a strong commitment to user data protection, are essential. Additionally, **Chameleon**’s autonomy should be viewed as a means to augment, not replace, human capabilities. Therefore, the development of robust ethical guidelines, transparency mechanisms, and safeguards is critical, underlying our commitment to the socially responsible deployment of AI.

<sup>4</sup><https://www.usgs.gov/geology-and-ecology-of-national-parks/ecology-death-valley-national-park-0>▷ *Instruction for the planner model*

You need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question.

The modules are defined as follows:

**Query\_Generator:** This module generates a search engine query for the given question. Normally, we consider using "Query\_Generator" when the question involves domain-specific knowledge.

**Bing\_Search:** This module searches the web for relevant information to the question. Normally, we consider using "Bing\_Search" when the question involves domain-specific knowledge.

**Image\_Captioner:** This module generates a caption for the given image. Normally, we consider using "Image\_Captioner" when the question involves the semantic understanding of the image, and the "has\_image" field in the metadata is True.

**Text\_Detector:** This module detects the text in the given image. Normally, we consider using "Text\_Detector" when the question involves the unfolding of the text in the image, e.g., diagram, chart, table, map, etc., and the "has\_image" field in the metadata is True.

**Knowledge\_Retrieval:** This module retrieves background knowledge as the hint for the given question. Normally, we consider using "Knowledge\_Retrieval" when the background knowledge is helpful to guide the solution.

**Solution\_Generator:** This module generates a detailed solution to the question based on the information provided. Normally, "Solution\_Generator" will incorporate the information from "Query\_Generator", "Bing\_Search", "Image\_Captioner", "Text\_Detector", and "Knowledge\_Retrieval".

**Answer\_Generator:** This module extracts the final answer in a short form from the solution or execution result. This module normally is the last module in the prediction pipeline.

Below are some examples that map the problem to the modules.

▷ *In-context example(s)*

**Question:** Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?

**Context:** The diagrams below show two pure samples of gas in identical closed, rigid containers. Each colored ball represents one gas particle. Both samples have the same number of particles.

**Options:** (A) neither; the samples have the same temperature (B) sample A (C) sample B

**Metadata:** 'pid': 19, 'has\_image': True, 'grade': 8, 'subject': 'natural science', 'topic': 'physics', 'category': 'Particle motion and energy', 'skill': 'Identify how particle motion affects temperature and pressure'

**Modules:** ["Text\_Detector", "Knowledge\_Retrieval", "Solution\_Generator", "Answer\_Generator"]

Table 8: The prompt constructed for the planner model on the ScienceQA task. The prompt consists of the instruction that describes the role of the planner model, the in-context examples that map the problem to the module sequence, and the test example.▷ *Instruction for the planner model*

You need to act as a policy model, that given a question and a modular set, determines the sequence of modules that can be executed sequentially can solve the question.

The modules are defined as follows:

**Program\_Generator:** This module generates a Python program that can solve the given question. It takes in the question and possible context and produces a program that can be executed by the "Program\_Executor" module. Normally, we consider using "Program\_Generator" when the questions and contexts involve complex computation, such as arithmetic operations over multiple numbers, or when the questions involve complex logical operations, such as "if-else" statements.

**Program\_Verifier:** This module verifies whether the generated program from "Program\_Generator" is valid and error-free. It checks for syntax errors, logical errors, and other potential issues that may arise during program execution.

**Program\_Executor:** This module executes the generated program from "Program\_Generator" and produces an output that can be further processed by other modules, such as "Question\_Answering".

**Row\_Lookup:** This module returns the simplified table that only remains the rows that are relevant to the question. It takes in the question and a table and returns the simplified table. If all rows are relevant or there are only three rows or fewer, return the original table. Normally, we only consider using "Row\_Lookup" when the table involves more than three rows and the question only requires a small number of rows to answer the question.

**Column\_Lookup:** This module returns the simplified table that only remains the columns that are relevant to the question. It takes in the question and a table and returns the simplified table. If all columns are relevant or there are only two columns, return the original table. Normally, we consider using "Column\_Lookup" when the table involves more than two columns and the question only requires a small number of columns to answer the question.

**Table\_Verbalizer:** This module converts the table to a description that can be easily understood by the downstream modules, like "Program\_Generator", "Solution\_Generator", "Question\_Answering". Normally, we consider using "Table\_Verbalizer" when the table involves a small number of rows and columns and the table is domain-specific, such as steam-and-leaf plots, function tables, etc.

**Knowledge\_Retrieval:** This module retrieves domain-specific knowledge for the given question and table. Normally, we consider using "Knowledge\_Retrieval" when the question and table involve domain-specific knowledge, such as "steam-and-leaf plots", "function tables", "tax forms", etc.

**Solution\_Generator:** This module generates a detailed solution to the question based on the information provided. Normally, we use "Solution\_Generator" when the question and table involve simple computation, such as arithmetic operations over a single number.

**Answer\_Generator:** This module extracts the final answer in a short form from the solution or execution result. This module normally follows the "Solution\_Generator" or "Problem\_Executor" module.

Below are some examples that map the problem to the modules.

▷ *In-context example(s)*

**Table:**

designer watch | \$8,141  
designer coat | \$6,391

**Question:** How much more does a designer watch cost than a designer coat? (unit: \$)

**Modules:** ["Program\_Generator", "Program\_Verifier", "Program\_Executor", "Answer\_Generator"]

Table 9: The prompt constructed for the planner model on the TabMWP task. Similarly, the prompt consists of the instruction, the in-context examples, and the test example.<table border="1">
<thead>
<tr>
<th style="text-align: center;">▷ <i>Instruction</i></th>
</tr>
</thead>
<tbody>
<tr>
<td>Read the following question, and generate the background knowledge as the context information that could be helpful for answering the question.</td>
</tr>
<tr>
<th style="text-align: center;">▷ <i>In-context example(s)</i></th>
</tr>
<tr>
<td><b>Question:</b> Which property do these three objects have in common?</td>
</tr>
<tr>
<td><b>Options:</b> (A) hard (B) soft (C) yellow</td>
</tr>
<tr>
<td><b>Metadata:</b> ‘pid’: 43, ‘has_image’: True, ‘grade’: 4, ‘subject’: ‘natural science’, ‘topic’: ‘physics’, ‘category’: ‘Materials’, ‘skill’: ‘Compare properties of objects’</td>
</tr>
<tr>
<td><b>Detected text in the image:</b> [‘handkerchief’, ‘slippers’, ‘leisure suit’]</td>
</tr>
<tr>
<td><b>Knowledge:</b></td>
</tr>
<tr>
<td>
<ul>
<li>- This question is about comparing the properties of three objects: a handkerchief, slippers, and a leisure suit.</li>
<li>- The objects are related to the topic of physics and the skill of comparing properties of objects.</li>
<li>- Properties of objects can include physical characteristics such as color, texture, shape, size, weight, and material.</li>
</ul>
</td>
</tr>
</tbody>
</table>

Table 10: The prompt constructed for the “Knowledge Retrieval” module on the ScienceQA task.

<table border="1">
<thead>
<tr>
<th style="text-align: center;">▷ <i>Instruction</i></th>
</tr>
</thead>
<tbody>
<tr>
<td>Read the following question and metadata, and generate the query for browser search as the context information that could be helpful for answering the question.</td>
</tr>
<tr>
<th style="text-align: center;">▷ <i>In-context example(s)</i></th>
</tr>
<tr>
<td><b>Question:</b> Which property do these two objects have in common?</td>
</tr>
<tr>
<td><b>Options:</b> (A) hard (B) bendable</td>
</tr>
<tr>
<td><b>Metadata:</b> ‘pid’: 329, ‘has_image’: True, ‘grade’: 2, ‘subject’: ‘natural science’, ‘topic’: ‘physics’, ‘category’: ‘Materials’, ‘skill’: ‘Compare properties of objects’</td>
</tr>
<tr>
<td><b>Detected text in the image:</b> [[[[41, 183], [131, 183], [131, 199], [41, 199]], ‘rubber gloves’), ([245, 183], [313, 183], [313, 197], [245, 197]), ‘rain boots’]]</td>
</tr>
<tr>
<td><b>Search Query:</b> Common material properties of jump rope and rubber gloves</td>
</tr>
</tbody>
</table>

Table 11: The prompt constructed for the “Query Generator” module on the ScienceQA task.▷ *Instruction*

Given the question (and the context), select the answer from the options ["A", "B", "C", "D", "E"]. You should give concise and step-by-step solutions. Finally, conclude the answer in the format of "the answer is [ANSWER]", where [ANSWER] is one from the options ["A", "B", "C", "D", "E"]. For example, "the answer is A", "the answer is B", "the answer is C", "the answer is D", or "the answer is E". If the answer is not in the options, select the most possible option.

▷ *In-context example(s)*

**Question:** Which property do these two objects have in common?

**Context:** Select the better answer.

**Options:** (A) hard (B) bendable

**Metadata:** ‘pid’: 6493, ‘has\_image’: True, ‘grade’: 2, ‘subject’: ‘natural science’, ‘topic’: ‘physics’, ‘category’: ‘Materials’, ‘skill’: ‘Compare properties of objects’

**Image caption:** A pair of scissors next to a pair of scissors.

**Detected text with coordinates in the image:** [([53, 185], [121, 185], [121, 199], [53, 199]), ‘jump rope’), ([233, 183], [323, 183], [323, 201], [233, 201]), ‘rubber gloves’)]

**Retrieved knowledge:**

- - This question is about comparing the properties of two objects: rubber gloves and rain boots.
- - The objects are related to the topic of physics and the skill of comparing properties of objects.
- - Properties of objects can include physical characteristics such as color, texture, shape, size, weight, and material. In this case, the two objects have the property of being bendable in common.

**Bing search response:** The most common materials used for disposable gloves are Latex, Vinyl and Nitrile. Each material has its benefits and drawbacks. Latex Gloves are constructed from Natural Rubber Latex and are the most popular type of disposable glove.

**Solution:** An object has different properties. A property of an object can tell you how it looks, feels, tastes, or smells. Different objects can have the same properties. You can use these properties to put objects into groups. Look at each object. For each object, decide if it has that property. A bendable object can be bent without breaking. Both objects are bendable. A hard object keeps its shape when you squeeze it. The rubber gloves are not hard. The property that both objects have in common is bendable. Therefore, the answer is B.

Table 12: The prompt constructed for the “Solution Generator” module on the ScienceQA task.

▷ *Instruction*

Read the following table and question, and generate the domain-specific knowledge as the context information that could be helpful for answering the question.

▷ *In-context example(s)*

**Table:**

<table><thead><tr><th>x</th><th>|</th><th>y</th></tr></thead><tbody><tr><td>10</td><td>|</td><td>15</td></tr><tr><td>11</td><td>|</td><td>9</td></tr><tr><td>12</td><td>|</td><td>2</td></tr></tbody></table>

**Question:** The table shows a function. Is the function linear or nonlinear?

**Knowledge:**

- - A linear function is a function whose graph is a straight line.
- - A nonlinear function is a function whose graph is not a straight line.
- - The equation of a linear function is  $y = mx + b$ , where  $m$  is the slope and  $b$  is the  $y$ -intercept.
- - The equation of a nonlinear function is not  $y = mx + b$ .

Table 13: The prompt constructed for the “Knowledge Retrieval” module on the TabMWP task.▷ *Instruction*

Read the following question and table. Each row is separated by a newline (“\n”) and each column is separated by a vertical bar (“|”). Return the simplified table that only remains the rows that are relevant to the question. If all rows are relevant, or the number of rows is fewer than three, return the original table.

▷ *In-context example(s)*

**Question:** In preparation for graduation, some teachers and students volunteered for the various graduation committees. How many people are on the music committee?

**Table:**

Committee | Students | Teachers  
Program | 5 | 17  
Ticket | 20 | 5  
Music | 20 | 15  
Schedule | 15 | 20  
Food | 18 | 2

**Simplified Table:**

Committee | Students | Teachers  
Music | 20 | 15

Table 14: The prompt constructed for the “Row Lookup” module on the TabMWP task.

▷ *Instruction*

Read the following question and table. Each row is separated by a newline (“\n”) and each column is separated by a vertical bar (“|”). Return the simplified table that only remains the columns that are relevant to the question. If all columns are relevant, return the original table.

▷ *In-context example(s)*

**Question:** Look at the following schedule. When does Recess end?

**Table:**

Subject | Begin | End  
Recess | 6:15 A.M. | 7:20 A.M.  
Orchestra | 7:30 A.M. | 8:40 A.M.  
Art | 8:45 A.M. | 9:35 A.M.  
Handwriting | 9:45 A.M. | 10:20 A.M.  
Gym | 10:30 A.M. | 11:15 A.M.  
Choir | 11:20 A.M. | 12:25 P.M.  
Science | 12:35 P.M. | 1:35 P.M.  
Reading | 1:40 P.M. | 2:50 P.M.

**Simplified Table:**

Subject | End  
Recess | 7:20 A.M.  
Orchestra | 8:40 A.M.  
Art | 9:35 A.M.  
Handwriting | 10:20 A.M.  
Gym | 11:15 A.M.  
Choir | 12:25 P.M.  
Science | 1:35 P.M.  
Reading | 2:50 P.M.

Table 15: The prompt constructed for the “Column Lookup” module on the TabMWP task.▷ *Instruction*

Read the following question and table. Write a textual description of the table. The description should keep the critical information in the table for answering the question. The description should not answer the question.

▷ *In-context example(s)*

**Table:**

Committee | Students | Teachers  
Program | 5 | 17  
Ticket | 20 | 5  
Music | 20 | 15  
Schedule | 15 | 20  
Food | 18 | 2

**Table description:** The table shows the number of students and teachers on each of the four graduation committees: Program, Ticket, Music, and Schedule. The Music committee has 20 students and 15 teachers.

Table 16: The prompt constructed for the “Table Verbalizer” module on the TabMWP task.

▷ *Instruction*

Read the following table and then write Python code to answer a question.

▷ *In-context example(s)*

**Table:**

Price | Quantity demanded | Quantity supplied  
\$895 | 21,000 | 3,400  
\$945 | 17,200 | 7,400  
\$995 | 13,400 | 11,400  
\$1,045 | 9,600 | 15,400  
\$1,095 | 5,800 | 19,400

**Questions:** Look at the table. Then answer the question. At a price of \$995, is there a shortage or a surplus? Please select from the following options: ['shortage', 'surplus'].

**Code:**

```
# Python Code, return 'ans'. Make sure that 'ans' is a string selected
    from the options in the question
quantity_demanded_at_price_955 = 13400
quantity_supplied_at_price_955 = 11400
if quantity_demanded_at_price_955 > quantity_supplied_at_price_955:
    ans = 'shortage'
else:
    ans = 'surplus'
```

Table 17: The prompt constructed for the “Program Generator” module on the TabMWP task.▷ *Instruction*

Read the following table and then answer a question.

▷ *In-context example(s)*

**Table:**

<table border="1">
<thead>
<tr>
<th>Price</th>
<th>Quantity demanded</th>
<th>Quantity supplied</th>
</tr>
</thead>
<tbody>
<tr>
<td>$895</td>
<td>21,000</td>
<td>3,400</td>
</tr>
<tr>
<td>$945</td>
<td>17,200</td>
<td>7,400</td>
</tr>
<tr>
<td>$995</td>
<td>13,400</td>
<td>11,400</td>
</tr>
<tr>
<td>$1,045</td>
<td>9,600</td>
<td>15,400</td>
</tr>
<tr>
<td>$1,095</td>
<td>5,800</td>
<td>19,400</td>
</tr>
</tbody>
</table>

**Question:** Look at the table. Then answer the question. At a price of \$995, is there a shortage or a surplus? Please select from the following options: ['shortage', 'surplus'].

**Solution:** At the price of \$995, the quantity demanded is greater than the quantity supplied. There is not enough of the good or service for sale at that price. So, there is a shortage. The answer is shortage.

Table 18: The prompt constructed for the “Solution Generator” module on the TabMWP task.

```

graph TD
    START([START]) -- 0.11 --> query_generator([query_generator])
    START -- 0.3 --> text_detector([text_detector])
    START -- 0.53 --> knowledge_retrieval([knowledge_retrieval])
    START -- 0.06 --> image_captioner([image_captioner])
    query_generator -- 1.0 --> bing_search([bing_search])
    bing_search -- 0.01 --> image_captioner
    bing_search -- 0.02 --> knowledge_retrieval
    bing_search -- 0.97 --> solution_generator([solution_generator])
    text_detector -- 0.78 --> knowledge_retrieval
    text_detector -- 0.22 --> solution_generator
    image_captioner -- 0.01 --> knowledge_retrieval
    image_captioner -- 0.79 --> solution_generator
    image_captioner -- 0.21 --> solution_generator
    knowledge_retrieval -- 1.0 --> solution_generator
    solution_generator -- 1.0 --> answer_generator([answer_generator])
    answer_generator -- 1.0 --> END([END])
  
```

Figure 7: Transitions between modules in programs generated by **Chameleon** (GPT-4) on ScienceQA. START is the start symbol, END is a terminal symbol and the others are non-terminal symbols.```

graph TD
    START([START]) -- 0.26 --> row_lookup([row_lookup])
    START -- 0.05 --> column_lookup([column_lookup])
    START -- 0.19 --> knowledge_retrieval([knowledge_retrieval])
    START -- 0.11 --> solution_generator([solution_generator])
    START -- 0.39 --> program_generator([program_generator])
    START -- 0.27 --> table_verbalizer([table_verbalizer])
    START -- 0.24 --> program_generator
    START -- 0.2 --> program_generator
    START -- 0.74 --> program_generator
    
    row_lookup -- 0.03 --> column_lookup
    row_lookup -- 0.02 --> knowledge_retrieval
    row_lookup -- 0.24 --> table_verbalizer
    row_lookup -- 0.27 --> program_generator
    
    column_lookup -- 0.02 --> knowledge_retrieval
    column_lookup -- 0.02 --> table_verbalizer
    
    knowledge_retrieval -- 0.25 --> solution_generator
    knowledge_retrieval -- 0.36 --> program_generator
    knowledge_retrieval -- 0.63 --> program_generator
    knowledge_retrieval -- 0.02 --> table_verbalizer
    
    solution_generator -- 1.0 --> answer_generator
    
    table_verbalizer -- 1.0 --> program_generator
    
    program_generator -- 1.0 --> program_verifier([program_verifier])
    program_verifier -- 1.0 --> program_executor([program_executor])
    program_executor -- 1.0 --> answer_generator
    
    answer_generator -- 1.0 --> END([END])
  
```

Figure 8: Transitions between modules in programs generated by **Chameleon** (GPT-4) on TabMWP. START is the start symbol, END is a terminal symbol and the others are non-terminal symbols.**Image Captioner** → A polar bear is standing in the snow.

**Query Generator** → Animal skin adaptations for cold environments

**Bing Search**

**Solution Generator**

**Answer Generator** → Eurasian lynx

**Input:** Which animal's skin is adapted for survival in cold places?  
(A) Eurasian lynx  
(B) Thorny Devil

**Response 1:** Birds use feathers, and most mammals use hair or fur, to trap a layer of air next to the skin and reduce heat transfer to the environment. Marine mammals like whales use blubber, a thick layer of fat, as a heavy-duty form of insulation. In cold weather, birds fluff their feathers and animals raise their fur to thicken the insulating layer.

**Response 2:** Adaptations: Have a thick fur coat for insulation against the cold. In winter, the coat is white to provide camouflage against the ice and snow. In summer, when the snow has melted, they have a thin, dark grey or brown coat which provides camouflage against rock and vegetation.

**Response 3:** The polar bear is adapted to life in a cold climate. Polar bears are well adapted for survival in the Arctic. Their adaptations include: a white appearance - as camouflage from prey on the snow...

**Context Analysis:** The context mentions that polar bears live in and around the cold Arctic Ocean, and their skin is adapted to help them survive in cold places. Out of the given options, neither the Eurasian lynx nor the thorny devil is mentioned in the context. However, the polar bear is mentioned as an animal with skin adapted for survival in cold places. Since the polar bear is not listed as an option, we must choose the most possible option from the given choices. The Eurasian lynx lives in colder regions compared to the thorny devil, which is a desert-dwelling reptile. Therefore, the answer is A.

Figure 9: One more example from our Chameleon (GPT-4) approach on ScienceQA.

**Knowledge Retrieval**

**Table Verbalizer**

**Program Generator**

**Program Verifier**

**Program Executor**

**Answer Generator** → 913.46

**Table:**

<table border="1">
<thead>
<tr>
<th>Employee</th>
<th>Pay period</th>
</tr>
</thead>
<tbody>
<tr>
<td>Bronson Le</td>
<td>March 16-31</td>
</tr>
<tr>
<td>Total earnings</td>
<td>$1,200.00</td>
</tr>
<tr>
<td>Federal income tax</td>
<td>$133.44</td>
</tr>
<tr>
<td>State income tax</td>
<td>$62.10</td>
</tr>
<tr>
<td>Other taxes</td>
<td>$91.00</td>
</tr>
<tr>
<td>Pay after taxes</td>
<td>?</td>
</tr>
</tbody>
</table>

**Input:** Look at Bronson's pay stub. Bronson lives in a state that has state income tax. How much did Bronson make after taxes?

**Knowledge Retrieval:** This table shows Bronson's pay stub for the pay period of March 16-31. Total earnings represent the amount of money Bronson made before taxes. Federal income tax, state income tax, and other taxes are the amounts deducted from Bronson's total earnings for various taxes. Total taxes can be calculated by adding the federal income tax, state income tax, and other taxes. Pay after taxes is the amount of money Bronson made after all taxes have been deducted from his total earnings. This can be calculated by subtracting the total taxes from the total earnings.

**Table Verbalizer:** The table shows Bronson Le's pay stub for the pay period of March 16-31. It includes his total earnings of \$1,200.00, federal income tax of \$133.44, state income tax of \$62.10, and other taxes of \$91.00. The total taxes and pay after taxes are not provided but can be calculated by adding the tax amounts and subtracting them from the total earnings, respectively.

**Program Generator:**

```

total_earnings = 1200
federal_income_tax = 133.44
state_income_tax = 62.10
other_taxes = 91.00
total_taxes = federal_income_tax + state_income_tax + other_taxes
ans = total_earnings - total_taxes

```

**Program Verifier:**

**Program Executor:** ans = 913.46

**Answer Generator:** 913.46

Figure 10: One more example from our Chameleon (GPT-4) approach on TabMWP.▷ Query

**Question:** Which better describes the Death Valley ecosystem?

**Context:** Figure: Death Valley. Death Valley is a desert ecosystem in eastern California.

**Options:** (A) It has dry, thin soil. It also has many different types of organisms. (B) It has a small amount of rain. It also has only a few types of organisms.

**Metadata:** ‘pid’: ‘18973’, ‘has\_image’: True, ‘grade’: 3, ‘subject’: ‘natural science’, ‘topic’: ‘biology’, ‘category’: ‘Ecosystems’, ‘skill’: ‘Describe ecosystems’

▷ Response

**Generated Program:** [ "Query\_Generator", "Bing\_Search", "Solution\_Generator", "Answer\_Generator" ]

**Query Generator:**

Characteristics of Death Valley ecosystem

**Bing Search:** (the first one from response candidates)

Visitors come to Death Valley to experience the stark and lonely vastness of the valley, the panorama of rugged canyons and mountains; the pleasures of the dry, moderate winter climate, the challenge of the hot, arid summer; the relief of the cooler mountains; and the reminders of frontier and Native American ways of life.

**Solution Generator:**

Death Valley is a desert ecosystem, which means it has specific characteristics related to its climate and the organisms that live there. A desert ecosystem typically has dry, thin soil and a small amount of rain. This environment supports only a few types of organisms that are adapted to survive in such harsh conditions. Therefore, the answer is B.

**Answer Generator (Wrong prediction):**

(B) It has a small amount of rain. It also has only a few types of organisms.

**Correct Answer (Ground truth):**

(A) It has dry, thin soil. It also has many different types of organisms.

Table 19: An example of failure cases in ScienceQA from Chameleon (GPT-4) where some modules perform inaccurately. The generated query from the “Query Generator” module is insufficiently specific, leading to a response from “Bing Search” that does not adequately detail the characteristics of the Death Valley ecosystem, particularly with regard to the types of organisms present. (From U.S. Geological Survey: *Contrary to its name, Death Valley National Park is very much alive ... Despite the low precipitation and extreme summer temperatures, Death Valley is home to many diverse species, several of which are endemic (found nowhere else in the world).*<sup>4</sup>▷ Query

**Question:** Which property matches this object?

**Context:** Select the better answer.

**Options:** (A) stretchy (B) yellow

**Metadata:** ‘pid’: ‘10338’, ‘has\_image’: True, ‘grade’: 2, ‘subject’: ‘natural science’, ‘topic’: ‘physics’, ‘category’: ‘Materials’, ‘skill’: ‘Identify properties of an object’

helium balloons

▷ Response

**Generated Program:** ["Image\_Captioner", "Solution\_Generator", "Answer\_Generator"]

**Image Captioner:**  
A colorful kite is being flown in the sky.

**Solution Generator:**  
Look at the image of the object, **which is a colorful kite**. A stretchy object can be stretched without breaking. **Kites are typically made of materials that are not stretchy, such as nylon or polyester**. A yellow object has a yellow color. **The kite in the image is colorful and includes the color yellow**. **Therefore, the answer is B.**

**Answer Generator (Wrong prediction):** (B) yellow

**Correct Answer (Ground truth):** (A) stretchy

Table 20: An example of failure cases in ScienceQA from **Chameleon** (GPT-4) where *the generated programs are suboptimal*. The LLM planner does not call the “Text Detector” and “Knowledge Retrieval” tools, resulting in an absence of semantic information from the image and a lack of related domain knowledge. Additionally, the employed image captioning model in providing an accurate textual description of the given visual context.▷ Query

**Question:** Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?

**Context:** Below is a food web from an ocean ecosystem in Monterey Bay, off the coast of California. A food web models how the matter eaten by organisms moves through an ecosystem. The arrows in a food web represent how matter moves between organisms in an ecosystem.

**Options:** (A) bat star (B) phytoplankton (C) kelp bass

**Metadata:** {‘pid’: ‘15175’, ‘has\_image’: True, ‘grade’: 6, ‘subject’: ‘natural science’, ‘topic’: ‘biology’, ‘category’: ‘Ecological interactions’, ‘skill’: ‘Interpret food webs II’}

▷ Response

**Generated Program:** ["Text\_Detector", "Knowledge\_Retrieval", "Solution\_Generator", "Answer\_Generator"]

**Text Detector:**

[[([297, 91], ‘orca’), ([653, 85], ‘sea otter’), ([487, 191], ‘kelp bass’), ([73, 221], ‘phytoplankton’), ([625, 269], ‘black rockfish’), ([323, 327], ‘zooplankton’), ([149, 363], ‘plainfin’), ([593, 361], ‘kelp’), ([132, 380], ‘midshipman’), ([401, 391], ‘bat star’), ([271, 433], ‘sea cucumber’), ([409, 471], ‘sea urchin’)]]

**Knowledge Retrieval:**

- - The question is about a food web in an ocean ecosystem in Monterey Bay, California.
- - A food web models how matter eaten by organisms moves through an ecosystem.
- - The arrows in a food web represent how matter moves between organisms in an ecosystem.
- - The organisms mentioned in the question are bat star, phytoplankton, and kelp bass.
- - The target organism is the sea cucumber.
- - In a food web, matter moves from one organism to another through consumption, predation, or decomposition.
- - Understanding the relationships between organisms in a food web can help determine the flow of matter between them.

**Solution Generator:**

Analyze the food web to determine which organism’s matter eventually moves to the sea cucumber.

- - The sea cucumber consumes the bat star.
- - **The bat star consumes the sea urchin.**
- - The sea urchin consumes the kelp.
- - The kelp gets its nutrients from the water, **which contains matter from phytoplankton** and other sources.

In this food web, **the matter from the bat star eventually moves to the sea cucumber through consumption. Therefore, the answer is A.**

**Answer Generator (Wrong prediction): (A) bat star**

**Correct Answer (Ground truth): (B) phytoplankton**

Table 21: An example of failure cases in ScienceQA from **Chameleon** (GPT-4) where *the current module inventory falls short in addressing specific abilities*. In this instance, the module inventory does not possess the necessary tools capable of parsing the arrows in the food web illustration.▷ Query

**Table:**

<table border="1"><thead><tr><th>x</th><th>y</th></tr></thead><tbody><tr><td>4</td><td>15</td></tr><tr><td>19</td><td>18</td></tr><tr><td>11</td><td>10</td></tr><tr><td>11</td><td>1</td></tr></tbody></table>

**Table:** (semi-structured format)

x | y  
4 | 15  
19 | 18  
11 | 10  
11 | 1

**Question:** Look at this table. Is this relation a function?

**Options:** (A) yes (B) no

▷ Response

**Generated Program:** ["Knowledge\_Retrieval", "Program\_Generator", "Program\_Verifier", "Program\_Executor", "Answer\_Generator"]

**Knowledge Retrieval:**

- - A relation is a set of ordered pairs  $(x, y)$ .
- - A function is a relation in which each input ( $x$ -value) is paired with exactly one output ( $y$ -value).
- - To determine if a relation is a function, check if each  $x$ -value is paired with only one  $y$ -value.
- - If an  $x$ -value is paired with more than one  $y$ -value, the relation is not a function.

**Program Generator:**

```
relation = {
    4: 15,
    19: 18,
    11: 10,
    11: 1
}

x_values = list(relation.keys())
unique_x_values = set(x_values)

if len(x_values) == len(unique_x_values):
    ans = 'yes'
else:
    ans = 'no'
```

**Program Verifier:** **True**

**Program Executor:**

```
ans = 'yes'
```

**Answer Generator (Wrong prediction):** (A) yes

**Correct Answer (Ground truth):** (B) no

Table 22: An example of failure cases in TabMWP from **Chameleon** (GPT-4) where *some modules perform inaccurately*. In this case, the LLM-based program generator is unsuccessful in creating a program that correctly mirrors the mathematical relationship expressed in the provided query and the retrieved domain knowledge.

