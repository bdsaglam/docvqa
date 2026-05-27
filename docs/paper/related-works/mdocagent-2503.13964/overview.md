## Introduction

Document Question Answering (DocQA) has emerged as a critical task in natural language processing and computer vision, requiring systems to analyze documents and answer questions based on their content. Documents often contain a rich mixture of text and visual elements, requiring sophisticated AI systems that can understand both modalities and their interrelationships.

![Overview of MDocAgent framework compared to existing approaches](https://paper-assets.alphaxiv.org/figures/2503.13964/img-0.jpeg)

As shown in the figure above, traditional approaches to document understanding face significant challenges when dealing with complex documents. The image illustrates the limitations of Large Vision Language Models (LVLMs) and compares single-modal retrieval approaches with the proposed MDocAgent system, which integrates multiple specialized agents to process both textual and visual information.

## Limitations of Current Approaches

Current DocQA methods typically employ one of two approaches:

1. **Large Language Models (LLMs) and LVLMs**: These powerful models can process text and images but struggle with:
   - Processing long documents due to context length limitations
   - Paying sufficient attention to details
   - Understanding cross-modal relationships

2. **Retrieval Augmented Generation (RAG)**: While RAG systems help overcome context length limitations by retrieving relevant document segments, they face challenges with:
   - Single-modal approaches (either text-only or image-only)
   - Limited ability to integrate information across modalities
   - Reduced accuracy when complex reasoning is required

These limitations lead to suboptimal performance on real-world document understanding tasks, especially those requiring integration of information from text and visual elements.

## The MDocAgent Framework

MDocAgent introduces a novel approach to document understanding by combining multi-modal RAG with a specialized multi-agent system. The framework addresses the limitations of existing methods by leveraging specialized agents for different aspects of document analysis and integrating both textual and visual information.

The key innovation of MDocAgent lies in its multi-agent architecture, which includes five specialized agents:

1. **General Agent**: Provides initial multi-modal understanding of the question and retrieved context
2. **Critical Agent**: Identifies key information within the retrieved context
3. **Text Agent**: Focuses on textual analysis
4. **Image Agent**: Specializes in visual interpretation
5. **Summarizing Agent**: Integrates outputs from all agents to generate the final answer

## System Architecture

The architecture of MDocAgent is designed to effectively process documents through five distinct stages:

![Detailed system architecture showing the five stages of operation](https://paper-assets.alphaxiv.org/figures/2503.13964/img-1.jpeg)

As illustrated in the figure, the system follows a structured workflow:

1. **Document Pre-processing**: Text is extracted via OCR while preserving page images
2. **Multi-modal Context Retrieval**: Both text-based and image-based retrievers select relevant document segments
3. **Initial Analysis**: The General Agent processes retrieved content and produces an initial answer
4. **Critical Information Extraction**: The Critical Agent identifies key information for specialized processing
5. **Specialized Processing and Synthesis**: Text and Image Agents analyze relevant content before the Summarizing Agent produces the final answer

## Specialized Agents

Each agent in the MDocAgent framework has a specific role and contributes uniquely to the document understanding process:

1. **General Agent**
   - Processes the question and retrieved context to generate an initial response
   - Provides a foundation for further specialized analysis
   - Based on Qwen2-VL-7B-Instruct model to leverage multi-modal capabilities

2. **Critical Agent**
   - Reviews the General Agent's response to identify crucial information
   - Extracts key textual and visual elements that require deeper analysis
   - Guides specialized agents by highlighting relevant content
   - Also uses Qwen2-VL-7B-Instruct for multi-modal understanding

3. **Text Agent**
   - Focuses exclusively on textual content identified by the Critical Agent
   - Performs detailed linguistic analysis and reasoning
   - Based on Llama-3.1-8B-Instruct to optimize text processing capabilities

4. **Image Agent**
   - Specializes in visual content analysis
   - Interprets images, charts, figures, and other visual elements
   - Uses Qwen2-VL-7B-Instruct to leverage vision-language capabilities

5. **Summarizing Agent**
   - Integrates outputs from all previous agents
   - Resolves any conflicts between different modalities
   - Produces a coherent, comprehensive final answer
   - Based on Qwen2-VL-7B-Instruct for multi-modal synthesis

## Operational Flow

The operational flow of MDocAgent can be understood through the following steps:

1. **Document Processing**: A document is converted into separate text and image components.

2. **Context Retrieval**:
   - Text-based RAG (using ColBERT) retrieves top-k relevant text segments
   - Image-based RAG (using ColPali) retrieves top-k relevant image segments
   - These segments form the multi-modal context for processing

3. **Multi-Agent Collaboration**:
   ```
   # Pseudocode representing the agent collaboration
   question = get_user_question()
   text_segments = text_retriever.retrieve(question, k=top_k)
   image_segments = image_retriever.retrieve(question, k=top_k)
   
   initial_answer = general_agent.process(question, text_segments, image_segments)
   critical_info = critical_agent.identify_key_info(initial_answer, question)
   
   text_analysis = text_agent.analyze(question, critical_info.text_elements, text_segments)
   image_analysis = image_agent.analyze(question, critical_info.visual_elements, image_segments)
   
   final_answer = summarizing_agent.synthesize(question, text_analysis, image_analysis, initial_answer)
   ```

This collaborative approach enables thorough processing of both textual and visual information, leading to more accurate and comprehensive answers.

## Evaluation Results

MDocAgent was evaluated on five DocQA benchmarks: MMLongBench, LongDocURL, PaperTab, PaperText, and FetaTab. The results demonstrate its superior performance compared to existing methods:

1. **Top-1 Retrieval Performance**:
   - 51.9% improvement over best LVLM baseline
   - 23.7% improvement over text-RAG baseline
   - 12.1% improvement over image-RAG (M3DocRAG) baseline

2. **Top-4 Retrieval Performance**:
   - 73.5% improvement over Qwen2.5-VL-7B
   - Consistent improvements across all benchmarks

3. **Ablation Studies**:
   - Removing either Text Agent or Image Agent results in significant performance drop
   - Removing both General Agent and Critical Agent also reduces performance
   - These results validate the contribution of each agent and the importance of multi-modal integration

The evaluation confirms that MDocAgent's multi-agent approach effectively handles complex document understanding tasks by leveraging specialized processing and multi-modal integration.

## Case Studies

Let's examine two case studies that highlight MDocAgent's capabilities:

![Case study on demographic survey analysis](https://paper-assets.alphaxiv.org/figures/2503.13964/img-2.jpeg)

In this first case study, the system is asked to determine which population is greater in a survey: foreign-born Latinos or Latinos interviewed by cellphone. This requires careful integration of numerical data from the document. While baseline methods struggle with accurate details, MDocAgent correctly identifies that 1,051 Latinos were interviewed by cellphone compared to 795 foreign-born Latinos.

![Case study on NTU campus information](https://paper-assets.alphaxiv.org/figures/2503.13964/img-3.jpeg)

In the second case study, the question requires visual analysis to determine which reason for attending NTU doesn't include people in its corresponding figure. MDocAgent correctly identifies "Most Beautiful Campus" as the answer by analyzing both textual content and visual elements.

![Case study on identifying academic credentials](https://paper-assets.alphaxiv.org/figures/2503.13964/img-4.jpeg)

The third case study demonstrates how MDocAgent correctly identifies academic credentials by integrating information from text. The system accurately determines that Prof. Lebour holds an M.A. (Master of Arts) degree.

These case studies demonstrate MDocAgent's ability to handle questions requiring careful integration of textual and visual information, outperforming baseline methods in accuracy and detail.

## Significance and Impact

MDocAgent makes several significant contributions to document understanding:

1. **Enhanced Multi-modal Integration**: By employing specialized agents for different modalities, the framework achieves superior cross-modal understanding compared to existing approaches.

2. **Improved Handling of Long Documents**: The dual RAG approach and specialized agents effectively process lengthy documents that would exceed the context windows of traditional LLMs and LVLMs.

3. **Superior Attention to Detail**: The Critical Agent ensures that important details are not overlooked during processing, enhancing accuracy on factual questions.

4. **Flexible and Extensible Architecture**: The multi-agent framework can be adapted to different document types and domain-specific requirements.

5. **Practical Applications**: The improved DocQA capabilities have potential applications in:
   - Information retrieval systems
   - Automated document analysis
   - Educational tools
   - Legal and compliance document processing
   - Research assistance

## Conclusion

MDocAgent represents a significant advancement in document understanding by effectively integrating multi-modal retrieval with specialized agent processing. The framework addresses key limitations of existing approaches by enabling detailed cross-modal understanding, effective processing of long documents, and improved attention to details.

The experimental results demonstrate substantial performance improvements across multiple benchmarks, validating the effectiveness of the multi-agent approach. The case studies illustrate how MDocAgent successfully handles complex questions requiring integration of textual and visual information.

As document AI continues to evolve, the multi-agent approach pioneered by MDocAgent provides a promising direction for future research, particularly in developing more sophisticated inter-agent communication protocols and integrating additional knowledge sources to further enhance document understanding capabilities.
## Relevant Citations



Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. MMLongBench-doc: Benchmarking long-context document understanding with visualizations, 2024. 1, 2, 5, 12

  * This paper introduces MMLongBench, a key benchmark dataset used to evaluate MDocAgent's performance in understanding long documents with visualizations.  It provides a standardized way to assess the model's capabilities in a multimodal, long-context setting.

Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song, Bo Zheng, et al. [LongDocURL: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating](https://alphaxiv.org/abs/2412.18424). arXiv preprint arXiv:2412.18424, 2024. 1, 2, 5, 12

  * This citation details LongDocURL, another essential benchmark dataset used for evaluating MDocAgent. It focuses on evaluating understanding, reasoning, and locating information within long documents, which are key tasks addressed by the proposed framework.

Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. [M3DocRAG: Multi-modal retrieval is what you need for multi-page multi-document understanding](https://alphaxiv.org/abs/2411.04952).arXiv preprint arXiv:2411.04952, 2024. 1, 2, 5, 6, 12

  * M3DocRAG, a state-of-the-art multi-modal retrieval augmented generation method, serves as a primary comparison point for MDocAgent. The paper highlights the importance of multi-modal retrieval for understanding complex documents, a challenge that MDocAgent aims to address.

Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, C´eline Hudelot, and Pierre Colombo. [ColPali: Efficient document retrieval with vision language models](https://alphaxiv.org/abs/2407.01449). In The Thirteenth International Conference on Learning Representations, 2024. 2, 4, 6, 12

  * This paper introduces ColPali, the image-based retrieval augmented generation model employed by MDocAgent. It is crucial for retrieving relevant visual content from the document, enabling the framework to effectively integrate visual information.

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Colbertv2: Effective and efficient retrieval via lightweight late interaction.arXiv preprint arXiv:2112.01488, 2021. 5, 6, 12

  * This citation presents Colbertv2, the text-based retrieval model used in MDocAgent. It forms the basis for retrieving relevant text segments, enabling the framework to analyze textual information effectively.