Title: mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding

URL Source: https://arxiv.org/html/2307.02499

Markdown Content:
Jiabo Ye, Anwen Hu 1 1 footnotemark: 1, Haiyang Xu, Qinghao Ye, Ming Yan 2 2 footnotemark: 2, Yuhao Dan, 

Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, Qian Qi, Ji Zhang, Fei Huang 

DAMO Academy, Alibaba Group 

{yejiabo.yjb, huanwen.haw, shuofeng.xhy, yeqinghao.yqh, ym119608}@alibaba-inc.com

###### Abstract

Document understanding refers to automatically extract, analyze and comprehend information from various types of digital documents, such as a web page. Existing Multi-model Large Language Models (MLLMs), including mPLUG-Owl, have demonstrated promising zero-shot capabilities in shallow OCR-free text recognition, indicating their potential for OCR-free document understanding. Nevertheless, without in-domain training, these models tend to ignore fine-grained OCR features, such as sophisticated tables or large blocks of text, which are essential for OCR-free document understanding. In this paper, we propose mPLUG-DocOwl based on mPLUG-Owl for OCR-free document understanding. Specifically, we first construct a instruction tuning dataset featuring a wide range of visual-text understanding tasks. Then, we strengthen the OCR-free document understanding ability by jointly train the model on language-only, general vision-and-language, and document instruction tuning dataset with our unified instruction tuning strategy. We also build an OCR-free document instruction understanding evaluation set LLMDoc to better compare models’ capabilities on instruct compliance and document understanding. Experimental results show that our model outperforms existing multi-modal models, demonstrating its strong ability of document understanding. Besides, without specific fine-tuning, mPLUG-DocOwl generalizes well on various downstream tasks. Our code, models, training data and evaluation set are available at https://github.com/X-PLUG/mPLUG-DocOwl.

1 Introduction
--------------

Large language models (LLMs) like ChatGPT(OpenAI, [2022](https://arxiv.org/html/2307.02499#bib.bib17)), BLOOM(Scao et al., [2022](https://arxiv.org/html/2307.02499#bib.bib19)), and LLaMA(Touvron et al., [2023](https://arxiv.org/html/2307.02499#bib.bib27)) have undergone rapid development to enable the realization of general artificial intelligence, boasting impressive zero-shot capabilities across diverse linguistic applications. With the LLM as the language decoder, Multimodal large language models (MLLMs) such as MiniGPT-4(Zhu et al., [2023](https://arxiv.org/html/2307.02499#bib.bib37)), LLaVA(Liu et al., [2023a](https://arxiv.org/html/2307.02499#bib.bib12)), and mPLUG-Owl(Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36)) have demonstrated remarkable zero-shot performance in various open-ended vision-and-language tasks. These models are trained to align text and images during the pre-training phase, and then to promote diverse abilities during the instruction tuning phase. Interestingly, these MLLMs exhibit superficial OCR-free text recognition abilities without explicit training on visual text understanding datasets(Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36); Liu et al., [2023b](https://arxiv.org/html/2307.02499#bib.bib13)). Nevertheless, due to lacking specific training, these models still face the challenge of comprehending intricate relationships between visual text and objects in diverse types of images, such as charts, documents and webpages.

By performing unified instruction tuning for Document Understanding upon the mPLUG-Owl(Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36)), we further propose a modularized MLLM(Li et al., [2022](https://arxiv.org/html/2307.02499#bib.bib11); Xu et al., [2023b](https://arxiv.org/html/2307.02499#bib.bib32)), namely mPLUG-DocOwl. Our approach utilizes a modularized framework similar to mPLUG-Owl (Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36)), which incorporates a visual abstractor module to link a pre-trained LLM with a visual knowledge module, achieving the alignment of text and images. To enhance diverse document understanding capabilities, we reorganize various downstream document understanding tasks in the same form of instructions. To maintain general uni/multi-modal abilities, we also include language-only and general vision-and-language instruction datasets used by mPLUG-Owl to train the mPLUG-DocOwl. During training, both the visual knowledge module and LLM decoder are frozen, only the visual abstractor and the Low-Rank Adaption (LoRA)(Hu et al., [2022](https://arxiv.org/html/2307.02499#bib.bib7)) in LLM are fine-tuned.

mPLUG-DocOwl achieves ocr-free state-of-the-art performance on multiple commonly used document understanding datasets. Furthermore, our experiments on a carefully-built document instruction understanding evaluation set LLMDoc shows that mPLUG-DocOwl achieves significantly better visual text understanding performance on various domains than existing MLMMs.

Our main contributions can be highlighted as follows:

*   •
We propose a modularized MLLM, mPLUG-DocOwl, which is the first one to balance language-only, general vision-and-language, and document understanding based on unified instruction tuning.

*   •
We carefully construct an instruction understanding test set with human evaluation, dubbed LLMDoc, to assess diverse document understanding capabilities.

*   •
Empirical results demonstrate that our mPLUG-DocOwl surpasses existing methods on ocr-free document understanding, including multiple standard benchmarks and LLMDoc.

2 Related Work
--------------

### 2.1 Visual Text Understanding

There are two types of models for understanding images that contain rich textual information. The first kind of approaches(Xu et al., [2020](https://arxiv.org/html/2307.02499#bib.bib33); Huang et al., [2022](https://arxiv.org/html/2307.02499#bib.bib8); Hu et al., [2021](https://arxiv.org/html/2307.02499#bib.bib6); Tang et al., [2023](https://arxiv.org/html/2307.02499#bib.bib25); Yang et al., [2021](https://arxiv.org/html/2307.02499#bib.bib34)) utilize off-the-shelf OCR models or APIs to recognize text from images, and then design pretraining tasks to facilitate cross-modality alignment between visual and textual inputs. On the other hand, end-to-end approaches(Davis et al., [2022](https://arxiv.org/html/2307.02499#bib.bib5); Kim et al., [2022](https://arxiv.org/html/2307.02499#bib.bib9); Lee et al., [2022](https://arxiv.org/html/2307.02499#bib.bib10)) utilize a high-resolution image encoder to learn text recognition during the pretraining stage. Both two types of models rely on specific finetuning on different downstream datasets and can’t achieve open-domain instruction understanding performance like Multimodal Large Language Models.

### 2.2 Multimodal Large Language Model

Large Language Models (LLMs) have demonstrated impressive zero-shot abilities across various open-ended tasks. Recent research has also explored the application of LLMs for multi-modal generation, utilizing two different paradigms: systematic collaboration and end-to-end trained models. Systematic collaboration approaches, such as Visual ChatGPT (Wu et al., [2023](https://arxiv.org/html/2307.02499#bib.bib30)) and MM-REACT (Yang et al., [2023](https://arxiv.org/html/2307.02499#bib.bib35)), leverage various vision experts or tools to express visual information with text descriptions. Subsequently, LLMs, such as ChatGPT (OpenAI, [2022](https://arxiv.org/html/2307.02499#bib.bib17)), can act as agents and select appropriate experts and tools for visual understanding. Finally, LLMs would summarize the output of these experts to answer user queries. On the other hand, some approaches, such as MiniGPT-4 (Zhu et al., [2023](https://arxiv.org/html/2307.02499#bib.bib37)), LLaVA (Liu et al., [2023a](https://arxiv.org/html/2307.02499#bib.bib12)), and mPLUG-Owl (Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36)), leverage LLMs to build unified models for multi-modality with limited connected parameters. These methods show superficial OCR-free text recognition abilities under the zero-shot setting. However, for complicated document understanding, due to lacking in-domain training, they encounter challenges in handling diverse image types, recognizing rich texts and comprehending relationships between visual semantic and text information. In this work, through unified instruction tuning, mPLUG-DocOwl achieves much better document understanding performance and maintains general uni/multi-modal abilities.

3 mPLUG-DocOwl
--------------

![Image 1: Refer to caption](https://arxiv.org/html/x1.png)

Figure 1: The summary of the instruction tuning paradigm of our mPLUG-DocOwl.

### 3.1 Architecture

The architecture of mPLUG-DocOwl is based on a popular multi-modal language model, mPLUG-Owl (Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36)), which comprises a pre-trained visual foundation model, a visual abstractor, and a language foundation model. The visual foundation model is responsible for extracting visual features from the input images, and the visual abstractor distills these features using a set of learnable tokens. The resulting visual features are then concatenated with the word embeddings of the input sentence and fed into the language model to generate the response. This powerful architecture allows for accurate and efficient multi-modal language processing.

The mPLUG-Owl (Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36)) exhibits superficial OCR ability when presented with images containing salient text. Inspired by this, we propose to further fine-tune the model with document instruction tuning data for better document understanding performance, covering document, table, chart and natural image and webpage. During fine-tuning, we freeze the visual encoder and the language model and train the visual abstractor. We also adopt the low-rank adaptation approach (LoRA)(Hu et al., [2022](https://arxiv.org/html/2307.02499#bib.bib7)) to enhance the language model’s ability.

### 3.2 Instruction Tuning Data

This section introduces the composition of our instruction tuning data in detail. To ensure the versatility of mPLUG-DocOwl, we collect diverse document understanding datasets with different task formats, including Visual Question Answering (VQA) (Antol et al., [2015](https://arxiv.org/html/2307.02499#bib.bib1)), Information Extraction (IE), Natural Language Inference (NLI) (Bowman et al., [2015](https://arxiv.org/html/2307.02499#bib.bib3)), and Image Captioning (IC). mPLUG-Owl(Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36)) performs instruction tuning with a unified format as "<image>Human:{question} AI:{answer}". In this work, we convert different document understanding tasks to the same format as mPLUG-Owl (Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36)) by replacing the {question} and {answer} placeholders as follows.

Visual Question Answering We simply use the raw question and answer as the {question} and {answer} placeholders. We collect VQA datasets on diverse domains, including ChartQA(Masry et al., [2022](https://arxiv.org/html/2307.02499#bib.bib14)), DocVQA(Mathew et al., [2021](https://arxiv.org/html/2307.02499#bib.bib15)), InfographicsVQA (InfoVQA)(Mathew et al., [2022](https://arxiv.org/html/2307.02499#bib.bib16)), WikiTableQuestions (WTQ)(Pasupat and Liang, [2015](https://arxiv.org/html/2307.02499#bib.bib18)), TextVQA (Singh et al., [2019](https://arxiv.org/html/2307.02499#bib.bib21)) and VisualMRC(Tanaka et al., [2021](https://arxiv.org/html/2307.02499#bib.bib24)).

Information Extraction requires the model to extract key-value pairs from the input image. The ‘keys’ (or ‘categories’) are always a stationary set. To convert this task to the instruction tuning format, we treat the value as the {answer} and construct the {question} as ‘What is the value for the {key}?’. When the key does not exist in the image, the {answer} is set to ‘None’. We collect Information Extraction data from DeepForm(Svetlichnaya, [2020](https://arxiv.org/html/2307.02499#bib.bib23)), and Kleister Charity (KLC)(Stanislawek et al., [2021](https://arxiv.org/html/2307.02499#bib.bib22)).

Natural Language Inference is a binary classification task with labels ‘Entailed’ and ‘Refuted’. Given a statement, we construct the {question} as ‘{statement}, Yes or No?’. The {answer} is ‘Yes’ or ‘No’ and refers to ‘Entailed’ or ‘Refuted’, respectively. TabFact(Chen et al., [2020](https://arxiv.org/html/2307.02499#bib.bib4)), a natural language inference dataset about tables, is chosen for instruction tuning.

Image Captioning aims to briefly describe an image with fluent language. We treat the caption as the {answer} and randomly choose a prompt as the {question} like LLaVa (Liu et al., [2023a](https://arxiv.org/html/2307.02499#bib.bib12)). TextCaps(Sidorov et al., [2020](https://arxiv.org/html/2307.02499#bib.bib20)) is an appropriate captioning dataset on natural images with texts.

#### Language-only and General Vision-and-language Instruction Tuning.

To enhance the model’s ability of language comprehension and multi-modal open-ended conversation, we follow mPLUG-Owl (Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36)) to introduce language-only and general vision-and-language instruction tuning data (Taori et al., [2023](https://arxiv.org/html/2307.02499#bib.bib26); Vicuna, [2023](https://arxiv.org/html/2307.02499#bib.bib28); Xu et al., [2023a](https://arxiv.org/html/2307.02499#bib.bib31); Liu et al., [2023a](https://arxiv.org/html/2307.02499#bib.bib12)).

![Image 2: Refer to caption](https://arxiv.org/html/x2.png)

Figure 2: Different types of datasets used to train mPLUG-DocOwl.

[Figure 2](https://arxiv.org/html/2307.02499#S3.F2 "Figure 2 ‣ Language-only and General Vision-and-language Instruction Tuning. ‣ 3.2 Instruction Tuning Data ‣ 3 mPLUG-DocOwl ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") shows the composition of our instruction tuning data grouped by the dataset type. We use training sets of these datasets as instruction tuning data and evaluate models on test sets.

### 3.3 Training Details

We adopt a two-stage training paradigm, where the Vision Transformer and Language model are kept frozen. In the first stage, both the visual abstractor and LoRA (Hu et al., [2022](https://arxiv.org/html/2307.02499#bib.bib7)) in the language model are fine-tuned. The first stage only uses the document understanding data and takes 10 epochs. In the second stage, we further freeze the visual abstractor and only train the LoRA. Besides document understanding data, the language-only and general vision-and-language instruction tuning data are further introduced at this stage and up-sampled 6 times. The second stage takes 3 epochs. Other training hyper-parameters are the same as mPLUG-Owl (Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36)).

4 Experiment
------------

### 4.1 LLMDoc

Existing benchmarks are hard to evaluate the open-ended instruction understanding results given by MLMMs. For better compare the instruction understanding performance in the document domain, we further construct a test set with human evaluation, namely LLMDoc.

#### Data Collection

To comprehensively evaluate the model’s abilities, we consider five scenarios to construct our evaluation dataset, including table (TabFact (Chen et al., [2020](https://arxiv.org/html/2307.02499#bib.bib4))), chart (ChartQA (Masry et al., [2022](https://arxiv.org/html/2307.02499#bib.bib14))), document (DocVQA (Mathew et al., [2021](https://arxiv.org/html/2307.02499#bib.bib15))), natural image (TextVQA (Singh et al., [2019](https://arxiv.org/html/2307.02499#bib.bib21))) and webpage (VisualMRC (Tanaka et al., [2021](https://arxiv.org/html/2307.02499#bib.bib24))). Specifically, for each dataset, we sample 20 images from the test split. For 10 of these images, we adopt a raw question as the instruction. While for the other 10, we ask annotators to write instructions requiring stronger capabilities like summarization, inference, and calculation. In total, we obtain 100 test samples.

#### Human Evaluation

Following the rating criteria proposed in Self-Instruct(Wang et al., [2022](https://arxiv.org/html/2307.02499#bib.bib29)), we perform the human evaluation to score the model’s responses, where A > B > C > D and A represents ‘correct and satisfying response’, B means ‘acceptable response with minor imperfections’, C refers to ‘response to the instruction but has significant errors’ and D means ‘irrelevant or invalid response’.

![Image 3: Refer to caption](https://arxiv.org/html/extracted/2307.02499v1/figs/llm_comp.png)

Figure 3: Human evaluation of mPLUG-DocOwl, mPLUG-Owl and MiniGPT-4 on LLMDoc.

We compare mPLUG-DocOwl with other popular mult-modal large language models, including mPLUG-Owl(Ye et al., [2023](https://arxiv.org/html/2307.02499#bib.bib36)) and Mini-GPT4(Zhu et al., [2023](https://arxiv.org/html/2307.02499#bib.bib37)), on LLMDoc. As shown in [Figure 3](https://arxiv.org/html/2307.02499#S4.F3 "Figure 3 ‣ Human Evaluation ‣ 4.1 LLMDoc ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding"), mPLUG-DocOwl achieves significantly better performance, with 37 responses being scored as “A”, demonstrating the stronger understanding ability of mPLUG-DocOwl in diverse document scenarios. Besides, it’s worth noting that all models have some responses scored as “C” or “D”, showing that instruction understanding performance in the document domain is still far from promising and needs more endeavor.

### 4.2 Benchmark Evaluation

Besides human evaluation, we also compare our mPLUG-DocOwl with ocr-free state-of-the-art document understanding models on public datasets. [Table 1](https://arxiv.org/html/2307.02499#S4.T1 "Table 1 ‣ 4.2 Benchmark Evaluation ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") shows the comparison with Dessurt(Davis et al., [2022](https://arxiv.org/html/2307.02499#bib.bib5)), Donut(Kim et al., [2022](https://arxiv.org/html/2307.02499#bib.bib9)) and Pix2Struct(Lee et al., [2022](https://arxiv.org/html/2307.02499#bib.bib10)) on DUE-Benchmark(Borchmann et al., [2021](https://arxiv.org/html/2307.02499#bib.bib2)), which mainly requires the text recognition and layout understanding abilities on documents and tables. Besides, [Table 2](https://arxiv.org/html/2307.02499#S4.T2 "Table 2 ‣ 4.2 Benchmark Evaluation ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") presents the evaluation on the chart, natural image and webpage datasets, which ask stronger ability to relate visual semantics and text information. Without finetuning on each dataset, our mPLUG-DocOwl achieves comparable or even better performance.

Table 1: Comparison with ocr-free methods on DUE-Benchmark.

Table 2: Comparison with ocr-free methods on chart, natural image and webpage understanding.

### 4.3 Qualitative Analysis

![Image 4: Refer to caption](https://arxiv.org/html/x3.png)

Figure 4: Qualitative results of mPLUG-DocOwl. The crucial regions and corresponding words are annotated with the same colors for clearer visualization. Wrong answers are colored red.

#### Benchmark Results.

Qualitative results on different types of images are shown in [Figure 4](https://arxiv.org/html/2307.02499#S4.F4 "Figure 4 ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding"). Crucial regions and corresponding responses are annotated with the same colors. Case (a) shows that mPLUG-DocOwl can accurately find the answer from a webpage screenshot with complex contents. Case (b) shows that mPLUG-DocOwl is even able to understand hand-drawn tables and correctly recognize handwritten fonts. In case (c), mPLUG-DocOwl can summarize key points from a chart. It successfully understands that the table is about internet usage and infers that “Never” means “Never used internet”. However, it also generates illusory outputs, such as "in the United States". The question in case (d) requires the model to understand the “Result” column, compare the points and return the date with the best results. Case (e) demonstrates that our model is capable of processing scanned documents and distinguishing company and person names. Case (f) shows that mPLUG-DocOwl can not only recognize small and blurry text but also perform simple calculations following the user intent.

![Image 5: Refer to caption](https://arxiv.org/html/x4.png)

Figure 5: Qualitative comparison between mPLUG-DocOwl and Mini-GPT4 on LLMDoc. Part one.

![Image 6: Refer to caption](https://arxiv.org/html/x5.png)

Figure 6: Qualitative comparison between mPLUG-DocOwl and Mini-GPT4 on LLMDoc. Part two.

#### LLMDoc Results

[Figure 5](https://arxiv.org/html/2307.02499#S4.F5 "Figure 5 ‣ Benchmark Results. ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") and [Figure 6](https://arxiv.org/html/2307.02499#S4.F6 "Figure 6 ‣ Benchmark Results. ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") present the comparison between mPLUG-DocOwl and Mini-GPT4 on LLMDoc. [Figure 5](https://arxiv.org/html/2307.02499#S4.F5 "Figure 5 ‣ Benchmark Results. ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") (a) requires models to convert a table into JSON format. Our mPLUG-DocOwl correctly understands the instruction and return a string in JSON format, but misses the last row. Mini-GPT4 fails to comprehend the instruction and doesn’t understand the content within the table. In [Figure 5](https://arxiv.org/html/2307.02499#S4.F5 "Figure 5 ‣ Benchmark Results. ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") (b), both mPLUG-DocOwl and Mini-GPT4 correctly recognize the name of the shop. However, Mini-GPT4 overlooks a smaller sign indicating clothes in this shop are medical uniforms. As for chart understanding in [Figure 6](https://arxiv.org/html/2307.02499#S4.F6 "Figure 6 ‣ Benchmark Results. ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") (c), Mini-GPT4 gives a wrong answer and redundant response, while our mPLUG-DocOwl gives a concise and correct response. In [Figure 6](https://arxiv.org/html/2307.02499#S4.F6 "Figure 6 ‣ Benchmark Results. ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") (d), Bernadette’s actual purpose is to confirm with Suzy if she would like to have the copy sent overnight. This not only requires the model to accurately recognize the text, but also to understand the relationships between involved persons. mPLUG-DocOwl recognizes the phrase "request a copy of chapter," but misunderstands the subject and object. Mini-GPT4 only comprehends that this image is a mail scenario and provides a vague and hallucinatory response. In [Figure 6](https://arxiv.org/html/2307.02499#S4.F6 "Figure 6 ‣ Benchmark Results. ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") (e), mPLUG-DocOwl gives a correct summary of the two latest news but Mini-GPT4 generates news irrelevant to the webpage screenshot.

![Image 7: Refer to caption](https://arxiv.org/html/x6.png)

Figure 7: Failure cases on LLMDoc. Part one.

![Image 8: Refer to caption](https://arxiv.org/html/x7.png)

Figure 8: Failure cases on LLMDoc. Part two.

The LLMDoc contains many challenging instruction understanding cases in the document domain. [Figure 7](https://arxiv.org/html/2307.02499#S4.F7 "Figure 7 ‣ LLMDoc Results ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") and [Figure 8](https://arxiv.org/html/2307.02499#S4.F8 "Figure 8 ‣ LLMDoc Results ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") show some wrong responses given by mPLUG-DocOwl. In [Figure 7](https://arxiv.org/html/2307.02499#S4.F7 "Figure 7 ‣ LLMDoc Results ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") (a), mPLUG-DocOwl only takes note of the three names in the picture, but ignores the fact that the user itself is also a speaker. In [Figure 7](https://arxiv.org/html/2307.02499#S4.F7 "Figure 7 ‣ LLMDoc Results ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") (b), mPLUG-DocOwl fails to perform multi-step calculations on multiple elements in the image. In [Figure 8](https://arxiv.org/html/2307.02499#S4.F8 "Figure 8 ‣ LLMDoc Results ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") (c), the model can understand the scene and the text in it, but fantasizes about non-existent characters. In [Figure 8](https://arxiv.org/html/2307.02499#S4.F8 "Figure 8 ‣ LLMDoc Results ‣ 4.3 Qualitative Analysis ‣ 4 Experiment ‣ mPLUG-DocOwl \scalerel*○: Modularized Multimodal Large Language Model for Document Understanding") (d), mPLUG-DocOwl fails to understand the instruction for writing news and only read the texts in the tablet.

5 Conclusion
------------

In this work, we infuse diverse ocr-free document understanding capabilities into mPLUG-Owl by incorporating document understanding data into instruction finetuning. Experiment results demonstrate that our mPLUG-DocOwl achieves comparable or even better performance than existing OCR-free methods. Besides, benefiting from language-only and general vision-and-language instruction tuning, mPLUG-DocOwl can better comprehend user instructions and intentions, enabling more complex interactions. Moreover, human evaluation on LLMDoc reveals that mPLUG-DocOwl still struggles with document-related commonsense reasoning, mathematical calculations, and creative generation. This provides valuable insights about developing stronger document understanding abilities with the LLM in the future.

References
----------

*   Antol et al. [2015] S.Antol, A.Agrawal, J.Lu, M.Mitchell, D.Batra, C.L. Zitnick, and D.Parikh. Vqa: Visual question answering. In _Proceedings of the IEEE international conference on computer vision_, pages 2425–2433, 2015. 
*   Borchmann et al. [2021] L.Borchmann, M.Pietruszka, T.Stanislawek, D.Jurkiewicz, M.Turski, K.Szyndler, and F.Gralinski. DUE: end-to-end document understanding benchmark. In _NeurIPS Datasets and Benchmarks_, 2021. 
*   Bowman et al. [2015] S.R. Bowman, G.Angeli, C.Potts, and C.D. Manning. A large annotated corpus for learning natural language inference. _arXiv preprint arXiv:1508.05326_, 2015. 
*   Chen et al. [2020] W.Chen, H.Wang, J.Chen, Y.Zhang, H.Wang, S.Li, X.Zhou, and W.Y. Wang. Tabfact : A large-scale dataset for table-based fact verification. In _International Conference on Learning Representations (ICLR)_, Addis Ababa, Ethiopia, April 2020. 
*   Davis et al. [2022] B.L. Davis, B.S. Morse, B.L. Price, C.Tensmeyer, C.Wigington, and V.I. Morariu. End-to-end document recognition and understanding with dessurt. In _ECCV Workshops (4)_, volume 13804 of _Lecture Notes in Computer Science_, pages 280–296. Springer, 2022. 
*   Hu et al. [2021] A.Hu, S.Chen, and Q.Jin. Question-controlled text-aware image captioning. In _ACM Multimedia_, pages 3097–3105. ACM, 2021. 
*   Hu et al. [2022] E.J. Hu, Y.Shen, P.Wallis, Z.Allen-Zhu, Y.Li, S.Wang, L.Wang, and W.Chen. Lora: Low-rank adaptation of large language models. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_. OpenReview.net, 2022. URL [https://openreview.net/forum?id=nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9). 
*   Huang et al. [2022] Y.Huang, T.Lv, L.Cui, Y.Lu, and F.Wei. Layoutlmv3: Pre-training for document AI with unified text and image masking. In _ACM Multimedia_, pages 4083–4091. ACM, 2022. 
*   Kim et al. [2022] G.Kim, T.Hong, M.Yim, J.Nam, J.Park, J.Yim, W.Hwang, S.Yun, D.Han, and S.Park. Ocr-free document understanding transformer. In _ECCV (28)_, volume 13688 of _Lecture Notes in Computer Science_, pages 498–517. Springer, 2022. 
*   Lee et al. [2022] K.Lee, M.Joshi, I.Turc, H.Hu, F.Liu, J.Eisenschlos, U.Khandelwal, P.Shaw, M.Chang, and K.Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding. _CoRR_, abs/2210.03347, 2022. 
*   Li et al. [2022] C.Li, H.Xu, J.Tian, W.Wang, M.Yan, B.Bi, J.Ye, H.Chen, G.Xu, Z.Cao, J.Zhang, S.Huang, F.Huang, J.Zhou, and L.Si. mplug: Effective and efficient vision-language learning by cross-modal skip-connections. In _EMNLP_, pages 7241–7259. Association for Computational Linguistics, 2022. 
*   Liu et al. [2023a] H.Liu, C.Li, Q.Wu, and Y.J. Lee. Visual instruction tuning. _CoRR_, abs/2304.08485, 2023a. 
*   Liu et al. [2023b] Y.Liu, Z.Li, H.Li, W.Yu, M.Huang, D.Peng, M.Liu, M.Chen, C.Li, L.Jin, et al. On the hidden mystery of ocr in large multimodal models. _arXiv preprint arXiv:2305.07895_, 2023b. 
*   Masry et al. [2022] A.Masry, D.X. Long, J.Q. Tan, S.R. Joty, and E.Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In _ACL (Findings)_, pages 2263–2279. Association for Computational Linguistics, 2022. 
*   Mathew et al. [2021] M.Mathew, D.Karatzas, and C.V. Jawahar. Docvqa: A dataset for VQA on document images. In _WACV_, pages 2199–2208. IEEE, 2021. 
*   Mathew et al. [2022] M.Mathew, V.Bagal, R.Tito, D.Karatzas, E.Valveny, and C.V. Jawahar. Infographicvqa. In _WACV_, pages 2582–2591. IEEE, 2022. 
*   OpenAI [2022] OpenAI. Introducing chatgpt. [https://openai.com/blog/chatgpt](https://openai.com/blog/chatgpt), 2022. 
*   Pasupat and Liang [2015] P.Pasupat and P.Liang. Compositional semantic parsing on semi-structured tables. In _ACL (1)_, pages 1470–1480. The Association for Computer Linguistics, 2015. 
*   Scao et al. [2022] T.L. Scao, A.Fan, C.Akiki, E.Pavlick, S.Ilic, D.Hesslow, R.Castagné, A.S. Luccioni, F.Yvon, M.Gallé, J.Tow, A.M. Rush, S.Biderman, A.Webson, P.S. Ammanamanchi, T.Wang, B.Sagot, N.Muennighoff, A.V. del Moral, O.Ruwase, R.Bawden, S.Bekman, A.McMillan-Major, I.Beltagy, H.Nguyen, L.Saulnier, S.Tan, P.O. Suarez, V.Sanh, H.Laurençon, Y.Jernite, J.Launay, M.Mitchell, C.Raffel, A.Gokaslan, A.Simhi, A.Soroa, A.F. Aji, A.Alfassy, A.Rogers, A.K. Nitzav, C.Xu, C.Mou, C.Emezue, C.Klamm, C.Leong, D.van Strien, D.I. Adelani, and et al. BLOOM: A 176b-parameter open-access multilingual language model. _CoRR_, abs/2211.05100, 2022. 
*   Sidorov et al. [2020] O.Sidorov, R.Hu, M.Rohrbach, and A.Singh. Textcaps: A dataset for image captioning with reading comprehension. In _ECCV (2)_, volume 12347 of _Lecture Notes in Computer Science_, pages 742–758. Springer, 2020. 
*   Singh et al. [2019] A.Singh, V.Natarajan, M.Shah, Y.Jiang, X.Chen, D.Batra, D.Parikh, and M.Rohrbach. Towards VQA models that can read. In _CVPR_, pages 8317–8326. Computer Vision Foundation / IEEE, 2019. 
*   Stanislawek et al. [2021] T.Stanislawek, F.Gralinski, A.Wróblewska, D.Lipinski, A.Kaliska, P.Rosalska, B.Topolski, and P.Biecek. Kleister: Key information extraction datasets involving long documents with complex layouts. In _ICDAR (1)_, volume 12821 of _Lecture Notes in Computer Science_, pages 564–579. Springer, 2021. 
*   Svetlichnaya [2020] S.Svetlichnaya. Deepform: Understand structured documents at scale, 2020. 
*   Tanaka et al. [2021] R.Tanaka, K.Nishida, and S.Yoshida. Visualmrc: Machine reading comprehension on document images. In _AAAI_, pages 13878–13888. AAAI Press, 2021. 
*   Tang et al. [2023] Z.Tang, Z.Yang, G.Wang, Y.Fang, Y.Liu, C.Zhu, M.Zeng, C.Zhang, and M.Bansal. Unifying vision, text, and layout for universal document processing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 19254–19264, 2023. 
*   Taori et al. [2023] R.Taori, I.Gulrajani, T.Zhang, Y.Dubois, X.Li, C.Guestrin, P.Liang, and T.B. Hashimoto. Stanford alpaca: An instruction-following llama model. [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), 2023. 
*   Touvron et al. [2023] H.Touvron, T.Lavril, G.Izacard, X.Martinet, M.Lachaux, T.Lacroix, B.Rozière, N.Goyal, E.Hambro, F.Azhar, A.Rodriguez, A.Joulin, E.Grave, and G.Lample. Llama: Open and efficient foundation language models. _CoRR_, abs/2302.13971, 2023. 
*   Vicuna [2023] Vicuna. Vicuna: An open chatbot impressing gpt-4. [https://github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat), 2023. 
*   Wang et al. [2022] Y.Wang, Y.Kordi, S.Mishra, A.Liu, N.A. Smith, D.Khashabi, and H.Hajishirzi. Self-instruct: Aligning language model with self generated instructions. _CoRR_, abs/2212.10560, 2022. doi: [10.48550/arXiv.2212.10560](https://arxiv.org/html/10.48550/arXiv.2212.10560). URL [https://doi.org/10.48550/arXiv.2212.10560](https://doi.org/10.48550/arXiv.2212.10560). 
*   Wu et al. [2023] C.Wu, S.Yin, W.Qi, X.Wang, Z.Tang, and N.Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. _CoRR_, abs/2303.04671, 2023. 
*   Xu et al. [2023a] C.Xu, D.Guo, N.Duan, and J.J. McAuley. Baize: An open-source chat model with parameter-efficient tuning on self-chat data. _CoRR_, abs/2304.01196, 2023a. 
*   Xu et al. [2023b] H.Xu, Q.Ye, M.Yan, Y.Shi, J.Ye, Y.Xu, C.Li, B.Bi, Q.Qian, W.Wang, G.Xu, J.Zhang, S.Huang, F.Huang, and J.Zhou. mplug-2: A modularized multi-modal foundation model across text, image and video. _CoRR_, abs/2302.00402, 2023b. 
*   Xu et al. [2020] Y.Xu, M.Li, L.Cui, S.Huang, F.Wei, and M.Zhou. Layoutlm: Pre-training of text and layout for document image understanding. In R.Gupta, Y.Liu, J.Tang, and B.A. Prakash, editors, _KDD ’20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, CA, USA, August 23-27, 2020_, pages 1192–1200. ACM, 2020. doi: [10.1145/3394486.3403172](https://arxiv.org/html/10.1145/3394486.3403172). URL [https://doi.org/10.1145/3394486.3403172](https://doi.org/10.1145/3394486.3403172). 
*   Yang et al. [2021] Z.Yang, Y.Lu, J.Wang, X.Yin, D.Florêncio, L.Wang, C.Zhang, L.Zhang, and J.Luo. TAP: text-aware pre-training for text-vqa and text-caption. In _CVPR_, pages 8751–8761. Computer Vision Foundation / IEEE, 2021. 
*   Yang et al. [2023] Z.Yang, L.Li, J.Wang, K.Lin, E.Azarnasab, F.Ahmed, Z.Liu, C.Liu, M.Zeng, and L.Wang. MM-REACT: prompting chatgpt for multimodal reasoning and action. _CoRR_, abs/2303.11381, 2023. 
*   Ye et al. [2023] Q.Ye, H.Xu, G.Xu, J.Ye, M.Yan, Y.Zhou, J.Wang, A.Hu, P.Shi, Y.Shi, C.Li, Y.Xu, H.Chen, J.Tian, Q.Qi, J.Zhang, and F.Huang. mplug-owl: Modularization empowers large language models with multimodality. _CoRR_, abs/2304.14178, 2023. 
*   Zhu et al. [2023] D.Zhu, J.Chen, X.Shen, X.Li, and M.Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023.

