Title: \thetable Performance comparison across different evidence source on LongDocURL.

URL Source: https://arxiv.org/html/2503.13964

Markdown Content:
### \thesubsection Evaluation Benchmarks

*   •MMLongBench[ma2024mmlongbenchdocbenchmarkinglongcontextdocument]: Evaluates models’ ability to understand long documents with rich layouts and multi-modal components, comprising 1091 questions and 135 documents averaging 47.5 pages each. 
*   •LongDocURL[deng2024longdocurl]: Provides a comprehensive multi-modal long document benchmark integrating understanding, reasoning, and locating tasks, covering over 33,000 pages of documents and 2,325 question-answer pairs. 
*   •PaperTab[hui2024uda]: Focuses on evaluating models’ ability to comprehend and extract information from tables within NLP research papers, covering 393 questions among 307 documents. 
*   •PaperText[hui2024uda]: Assesses models’ proficiency in understanding the textual content of NLP research papers, covering 2804 questions among 1087 documents. 
*   •FetaTab[hui2024uda]: a question-answering dataset for tables from Wikipedia pages, challengeing models to generate free-form text answers, comprising 1023 questions and 878 documents. 

### \thesubsection Hyperparameter Settings

*   •Temperature: All models use their default temperature setting. 
*   •Max New Tokens: 256. 
*   •

Max Tokens per Image (Qwen2-VL-7B-Instruct):

    *   –Top-1 retrieval: 16,384 (by default). 
    *   –Top-4 retrieval: 2,048. 

*   •Image Resolution: 144 (for all benchmarks). 

### \thesubsection Prompt Settings

{tcolorbox}

[title=General Agent] You are an advanced agent capable of analyzing both text and images. Your task is to use both the textual and visual information provided to answer the user’s question accurately. Extract Text from Both Sources: If the image contains text, extract it and consider both the text in the image and the provided textual content. Analyze Visual and Textual Information: Combine details from both the image (e.g., objects, scenes, or patterns) and the text to build a comprehensive understanding of the content. Provide a Combined Answer: Use the relevant details from both the image and the text to provide a clear, accurate, and context-aware response to the user’s question. When responding:

*   •If both the image and text contain similar or overlapping information, cross-check and use both to ensure consistency. 
*   •If the image contains information not present in the text, include it in your response if it is relevant to the question. 
*   •If the text and image offer conflicting details, explain the discrepancies and clarify the most reliable source. 

{tcolorbox}

[title=Critical Agent] Provide a Python dictionary of critical information based on all given information—one for text and one for image. Respond exclusively in a valid dictionary format without any additional text. The format should be: {”text”: ”critical information for text”, ”image”: ”critical information for image”} {tcolorbox}[title=Text Agent] You are a text analysis agent. Your job is to extract key information from the text and use it to answer the user’s question accurately. Your tasks:

*   •Extract key details. Focus on the most important facts, data, or ideas related to the question. 
*   •Understand the context and pay attention to the meaning and details. 
*   •Use the extracted information to give a concise and relevant response to the user’s question. Provide a clear answer. 

{tcolorbox}

[title=Image Agent] You are an advanced image processing agent specialized in analyzing and extracting information from images. The images may include document screenshots, illustrations, or photographs. Your tasks:

*   •Extract textual information from images using Optical Character Recognition (OCR). 
*   •Analyze visual content to identify relevant details (e.g., objects, patterns, scenes). 
*   •Combine textual and visual information to provide an accurate and context-aware answer to the user’s question. 

{tcolorbox}

[title=Summarizing Agent] You are tasked with summarizing and evaluating the collective responses provided by multiple agents. You have access to the following information:

*   •Answers: The individual answers from all agents. 

Your tasks:

*   •Analyze: Evaluate the quality, consistency, and relevance of each answer. Identify commonalities, discrepancies, or gaps in reasoning. 
*   •Synthesize: Summarize the most accurate and reliable information based on the evidence provided by the agents and their discussions. 
*   •Conclude: Provide a final, well-reasoned answer to the question or task. Your conclusion should reflect the consensus (if one exists) or the most credible and well-supported answer. 

Return the final answer in the following dictionary format: {”Answer”: Your final answer here} {tcolorbox}[title=Evaluation] Question: {question} Predicted Answer: {answer} Ground Truth Answer: {gt} Please evaluate whether the predicted answer is correct.

*   •If the answer is correct, return 1. 
*   •If the answer is incorrect, return 0. 

Return only a string formatted as a valid JSON dictionary that can be parsed using json.loads, for example: {”correctness”: 1}

### \thesubsection Evaluation Metrics

The metric of all benchmarks is the average binary correctness evaluated by GPT-4o. The evaluation prompt is given in Section •. We use a python script to extract the result provided by GPT-4o. 
1 Additional Results
--------------------

### \thesubsection Fine-grained Performance of LongDocURL

We present the fine-grained performance of LongDocURL, as illustrated in Table•. Similar to MMLongBench, \ours outperforms all LVLM baselines. When using the top 1 retrieval approach, though M3DocRAG performs slightly better on Figure and ColBERTv2+Llama3.1-8B performs slightly better on the type Others, \ours show strong performance in Layout, Text, Table and get the highest average accuracy. With the top 4 retrieval strategy, \ours improves its performance and reach the highest score in the all categories.
### \thesubsection Experiments on different model backbones in \ours

\resizebox

0.85! Table [1](https://arxiv.org/html/2503.13964v1#section1 "1 Additional Results ‣ \thesubsection Evaluation Metrics ‣ \thesubsection Prompt Settings ‣ \thesubsection Hyperparameter Settings ‣ \thesubsection Evaluation Benchmarks") presents an ablation study evaluating the impact of different LVLMs on the performance of our framework. Three LVLMs: Qwen2-VL-7B-Instruct, Qwen2.5-VL-7B-Instruct, and GPT-4o were integrated as the backbone model for all agents except the text agent. Qwen2.5-VL-7B-Instruct performs worse than Qwen2-VL-7B-Instruct on PaperTab, PaperText, and FetaTab, with both top-1 and top-4 retrieval. However, Qwen2.5-VL shows an extremely marked improvement over Qwen2-VL on MMLongBench, resulting higher average scores. MMLongBench’s greater reliance on image-based questions might explain Qwen2.5-VL’s superior performance on this benchmark, possibly indicating that Qwen2.5-VL is better at handling visual question-answering tasks, but worse at handling textual tasks. Importantly, GPT-4o significantly outperforms both Qwen2-VL and Qwen2.5-VL across all benchmarks. Remarkably, GPT-4o’s top-1 performance surpasses even the top-4 results of both Qwen models in almost all cases. This substantial performance increase strongly suggests that our framework effectively leverages more powerful backbone models, showcasing its adaptability and capacity to benefit from improvements in the underlying LVLMs. 
### \thesubsection Additional case studies

In Figure [1](https://arxiv.org/html/2503.13964v1#section1 "1 Additional Results ‣ \thesubsection Evaluation Metrics ‣ \thesubsection Prompt Settings ‣ \thesubsection Hyperparameter Settings ‣ \thesubsection Evaluation Benchmarks"), the question requires identifying a reason from a list that lacks explicit numbering and is accompanied by images. ColBERT fails to retrieve the correct evidence page, resulting ColBERT + Llama’s inability to answer the question. Although ColPali correctly locates the evidence page, M3DocRAG fails to get the correct answer. However, our framework successfully identifies the correct answer (”Most Beautiful Campus”) through the concerted efforts of all agents. The general agent arrives at a preliminary answer and the critical agent identifies critical textual clues (”Most Beautiful Campus”) and corresponding visual elements (images of the NTU campus). Image agent then refines the answer, leveraging the critical information to correctly pinpoint the description lacking people. Though text agent can’t find the related information from the given context, information provided by the critical agent helps it to guess that the answer is ”Most Beautiful Campus”. The summarizing agent combines these insights to arrive at the correct final answer. In Figure [1](https://arxiv.org/html/2503.13964v1#section1 "1 Additional Results ‣ \thesubsection Evaluation Metrics ‣ \thesubsection Prompt Settings ‣ \thesubsection Hyperparameter Settings ‣ \thesubsection Evaluation Benchmarks"), the question asks for Professor Lebour’s degree. ColPali fails to retrieve the relevant page, rendering M3DocRAG ineffective. While ColBERT correctly retrieves the page, ColBERT + Llama still produces an incorrect answer because it incorrectly adds ”F.G.S.” to the answer, which is not a degree. \ours, on the other hand, correctly identifies the ”M.A. degree”. The general agent provides an initial answer, and the critical agent identifies the ”M.A.” designation in both text and image. Based on the clue, the text agent adds a more detailed explanation, and the image agent directly uses the clue as its answer. Finally, the summarizing agent synthesizes the results to provide the verified answer. These two cases highlight \ours’s resilience to imperfect retrieval, demonstrating the effectiveness of collaborative multi-modal information processing and the importance of the general-critical agent’s guidance in achieving high accuracy even with potentially insufficient or ambiguous information.

\includegraphics

[width=0.98height=0.4keepaspectratio]figs/case2.pdf

Figure \thefigure: A Case study of \ours compared with other two baselines. While only ColPali correctly retrieves the evidence page, neither baseline method identifies the correct answer. Our method, through critical information sharing and specialized agent collaboration, correctly pinpoints the ”Most Beautiful Campus” as the only reason without a corresponding image containing people.

\includegraphics

[width=0.98height=0.4keepaspectratio]figs/case3.pdf

Figure \thefigure: A Case study of \ours compared with other two RAG-method baselines. In this case, ColPali fails to retrieve the correct evidence page, hindering M3DocRAG. While ColBERT succeeds in retrieval, the ColBERT + Llama baseline still provides an incorrect answer. Only our multi-agent framework, through precise critical information extraction and agent collaboration, correctly identifies the M.A. degree.

Table \thetable: Performance comparison of using different backbone LVLMs in \ours.

