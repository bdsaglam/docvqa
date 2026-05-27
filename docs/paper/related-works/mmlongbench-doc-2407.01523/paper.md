Title: MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations

URL Source: https://arxiv.org/html/2407.01523

Published Time: Wed, 17 Sep 2025 18:37:32 GMT

Markdown Content:
Yubo Ma 1, Yuhang Zang 2, Liangyu Chen 1, Meiqi Chen 3, Yizhu Jiao 4

Xinze Li 1, Xinyuan Lu 5, Ziyu Liu 6, Yan Ma 7, Xiaoyi Dong 2, Pan Zhang 2

Liangming Pan 8, Yu-Gang Jiang 9, Jiaqi Wang 2, Yixin Cao 9, Aixin Sun 1

1 S-Lab, Nanyang Technological University, 2 Shanghai AI Laboratory, 3 Peking University 

4 University of Illinois Urbana-Champaign, 5 National University of Singapore, 6 Wuhan University 

7 Singapore Management University, 8 University of Arizona, 9 Fudan University

###### Abstract

Understanding documents with rich layouts and multi-modal components is a long-standing and practical task. Recent Large Vision-Language Models (LVLMs) have made remarkable strides in various tasks, particularly in single-page document understanding (DU). However, their abilities on long-context DU remain an open problem. This work presents MMLongBench-Doc††footnotetext: Project Page: [https://mayubo2333.github.io/MMLongBench-Doc](https://mayubo2333.github.io/MMLongBench-Doc), a long-context, multi-modal benchmark comprising 1,082 expert-annotated questions. Distinct from previous datasets, it is constructed upon 135 lengthy PDF-formatted documents with an average of 47.5 pages and 21,214 textual tokens. Towards comprehensive evaluation, answers to these questions rely on pieces of evidence from (1) different sources (text, image, chart, table, and layout structure) and (2) various locations (_i.e.,_ page number). Moreover, 33.7% of the questions are cross-page questions requiring evidence across multiple pages. 20.6% of the questions are designed to be unanswerable for detecting potential hallucinations. Experiments on 14 LVLMs demonstrate that long-context DU greatly challenges current models. Notably, the best-performing model, GPT-4o, achieves an F1 score of only 44.9%, while the second-best, GPT-4V, scores 30.5%. Furthermore, 12 LVLMs (all except GPT-4o and GPT-4V) even present worse performance than their LLM counterparts which are fed with lossy-parsed OCR documents. These results validate the necessity of future research toward more capable long-context LVLMs.

1 1 footnotetext: Corresponding Authors.![Image 1: Refer to caption](https://arxiv.org/html/2407.01523v3/x1.png)

Figure 1: MMLongBench-Doc evaluates understanding abilities of LVLMs on lengthy documents that span tens of pages and incorporate multi-modal elements. Experiments (bottom-right) indicate that most LVLMs struggle, even falling behind LLMs that are fed with only OCR-parsed documents.

1 Introduction
--------------

Documents are one of the fundamental forms of information preservation and exchange. In each year, tens of millions of documents are created, read, saved, and dispatched[[1](https://arxiv.org/html/2407.01523v3#bib.bib1)]. Beyond unstructured pure-text, documents feature both complicated layout structures and information across distinct modalities such as text, table, chart, image, _etc._ Accordingly, the automatic understanding of documents (Document Understanding; DU) stands as a long-standing task in urgent and practical needs.

Recently, a number of LVLMs, both closed-source ones (GPT-4o[[2](https://arxiv.org/html/2407.01523v3#bib.bib2)], Gemini-1.5[[3](https://arxiv.org/html/2407.01523v3#bib.bib3)], Claude-3[[4](https://arxiv.org/html/2407.01523v3#bib.bib4)], _etc._) and open-source ones (InternLM-XC2-4KHD[[5](https://arxiv.org/html/2407.01523v3#bib.bib5)], InternVL-Chat[[6](https://arxiv.org/html/2407.01523v3#bib.bib6)], Otter[[7](https://arxiv.org/html/2407.01523v3#bib.bib7)], LLaVA-NeXT[[8](https://arxiv.org/html/2407.01523v3#bib.bib8)], CogVLM[[9](https://arxiv.org/html/2407.01523v3#bib.bib9)], mPLUG-DocOwl 1.5[[10](https://arxiv.org/html/2407.01523v3#bib.bib10)], TextMonkey[[11](https://arxiv.org/html/2407.01523v3#bib.bib11)], _etc._) have been developed and presented the great potential to handle documents. Most of them have achieved promising performance on single-page DU datasets like DocVQA[[12](https://arxiv.org/html/2407.01523v3#bib.bib12)], ChartQA[[13](https://arxiv.org/html/2407.01523v3#bib.bib13)], InfoVQA[[14](https://arxiv.org/html/2407.01523v3#bib.bib14)], TAT-DQA[[15](https://arxiv.org/html/2407.01523v3#bib.bib15)], _etc._ However, considerable amounts of documents in the real world are long-context documents with tens or even hundreds of pages. The understanding of these lengthy documents brings new challenges for LVLMs from at least two aspects: (1) Localization: identify and retrieve information from massive, heterogeneous information (similar to the needle in a haystack task); (2) Cross-page comprehension: collect and reason over multi-source information across different pages. These two kinds of abilities are beyond the evaluation scopes of the aforementioned single-page DU datasets. Some recent DU datasets[[16](https://arxiv.org/html/2407.01523v3#bib.bib16); [17](https://arxiv.org/html/2407.01523v3#bib.bib17); [18](https://arxiv.org/html/2407.01523v3#bib.bib18)] feature multiple-page DU, but almost all their documents are either as short of only several pages or of low information density, making the localization-related questions over-simple. Additionally, few (if any) questions in these datasets necessitate cross-page comprehension. See more detailed related work in Section[2](https://arxiv.org/html/2407.01523v3#S2 "2 Related Work ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). In summary, there lacks a unified and high-quality benchmark on lengthy documents, leaving the evaluation of long-context DU largely unexplored.

In this paper, we present MMLongBench-Doc, a benchmark designed to evaluate the M ulti-M odality Long-context Doc ument understanding abilities of LVLMs. Towards a comprehensive benchmark, it incorporates lengthy documents from both four existing datasets[[13](https://arxiv.org/html/2407.01523v3#bib.bib13); [17](https://arxiv.org/html/2407.01523v3#bib.bib17); [18](https://arxiv.org/html/2407.01523v3#bib.bib18); [19](https://arxiv.org/html/2407.01523v3#bib.bib19)] and other various papers, brochures, _etc._ Consequently, our benchmark includes 135 PDF-formatted documents spanning across 7 diverse domains, with each document averaging 47.5 pages and 21,214.1 textual tokens. Regarding the questions, we employ ten expert-level annotators to (1) edit questions associated with documents from existing datasets to meet our benchmark’s standard and (2) create new questions for all collected documents to expand the scale of the benchmark. Then a three-round, semi-automatic reviewing process ensures the benchmark’s annotation quality. As a result, MMLongBench-Doc comprises 1,082 human-annotated questions, with 184 sourced from four existing datasets and 898 newly annotated. Being a multi-modal benchmark, the answer to each question requires evidence from one or more of these five in-document sources: text, layout, chart, table, and image. Questions are categorized into three types based on the number of evidence pages 1 1 1 Given a document D D and a question q q upon D D, We call page P P (in document D D) an evidence page of q q if the answer of q q necessitates one or more pieces of evidence in page P P., with examples illustrated in Figure[1](https://arxiv.org/html/2407.01523v3#S0.F1 "Figure 1 ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations")(a): (1) 494 single-page questions (with one evidence page) mainly to evaluate localization abilities, (2) 365 cross-page questions (with multiple evidence pages) to assess cross-page comprehension, and (3) 223 unanswerable questions (no evidence for answering it, _i.e.,_ no evidence pages) to reduce shortcuts and measure LVLMs’ potential hallucinations. Meta-information including evidence pages, sources, and answer formats, is preserved for fine-grained evaluation and analysis. Detailed descriptions of the annotation pipeline and statistics can be found in Section[3](https://arxiv.org/html/2407.01523v3#S3 "3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations").

We conduct extensive experiments on MMLongBench-Doc to evaluate the long-context DU abilities of 14 LVLMs, including 4 proprietary and 10 open-source ones. Given a document, we screenshot each page and feed all of these PNG-formatted images to LVLMs in an end-to-end approach. For comparison, we also convert the documents to textual format by optical character recognition (OCR) and evaluate another 6 proprietary and 4 open-source 10 LLMs (6 proprietary and 4 open-source ones). The results in Figure[1](https://arxiv.org/html/2407.01523v3#S0.F1 "Figure 1 ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations")(c) highlight the challenges that current LVLMs face with long-context DU. The best-performing LVLM, GPT-4o, achieves an overall F1 score of only 44.9%, while the second-best LVLM, GPT-4V, scores 30.5%. Moreover, all the remaining LVLMs tested with multi-modal documents performed worse than single-modal LLMs handling lossy, OCR-parsed texts. Specifically, the Gemini-1.5-Pro and Claude-3-Opus present 4.2% and 6.4% absolute decrease when the inputs change from document screenshots to OCR-parsed texts. Regarding open-source models, the best-performing LVLM lags behind the best-performing LLM by 11.7%. These results reveal that long-context DU is a far-from-resolved task for current LVLMs.

Table 1: Comparison between our benchmark and previous DU datasets. Unans.: unanswerable question. TXT/L/C/TAB/I: pure text/generalized layout/chart/table/image. Doc. Rel.: document relevance. Whether document information is indispensable for the answer. Avg. Position: the average page index on which the answer evidence is located. *:Statistics from [[20](https://arxiv.org/html/2407.01523v3#bib.bib20)].

Benchmarks Document Question type Answer Evidence
# Pages# Tokens Cross-page (%)Unans. (%)Doc. Rel.Source Avg. Position
DocVQA[[12](https://arxiv.org/html/2407.01523v3#bib.bib12)]1.0 151.5✗✗✔ 
✗TXT/L/C/TAB/I-
ChartQA[[13](https://arxiv.org/html/2407.01523v3#bib.bib13)]2 2 2 We view website screenshots and posters as generalized documents and define equivalent page number (EPN) to measure their context lengths: EPN(D) = ceil​(Pixel(D)P)\texttt{EPN(D) = ceil}(\frac{\texttt{Pixel(D)}}{P}). Here Pixel(D) is the pixel number of generalized document D, and P is the average pixel numbers of each page (converting from .pdf to .png format with resolution 240) in MMLongBench-Doc.1.0 236.9✗✗✓C-
InfoVQA[[14](https://arxiv.org/html/2407.01523v3#bib.bib14)][2](https://arxiv.org/html/2407.01523v3#footnote2 "footnote 2 ‣ Table 1 ‣ 1 Introduction ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations")1.2 288.0✗✗✔ 
✗L/C/TAB/I-
TAT-DQA[[15](https://arxiv.org/html/2407.01523v3#bib.bib15)]1.1 577.0✗✗✔ 
✗TXT/TAB-
VisualWebBench[[21](https://arxiv.org/html/2407.01523v3#bib.bib21)][2](https://arxiv.org/html/2407.01523v3#footnote2 "footnote 2 ‣ Table 1 ‣ 1 Introduction ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations")1.0 452.4✗✗✓LAY/I-
PWC[[22](https://arxiv.org/html/2407.01523v3#bib.bib22)]~12*~7000*✗✗✔ 
✗TAB-
MP-DocVQA[[16](https://arxiv.org/html/2407.01523v3#bib.bib16)]8.3 2026.6✗✗✔ 
✗TXT/L/C/TAB/I 6.0
DUDE[[17](https://arxiv.org/html/2407.01523v3#bib.bib17)]5.7 1831.5✓(2.1%)✓(12.7%)✔ 
✗TXT/L/C/TAB/I 2.5
SlideVQA[[18](https://arxiv.org/html/2407.01523v3#bib.bib18)]20.0 2030.5✓(13.9%)✗✔ 
✗TXT/L/C/TAB/I 9.1
MMLongBench-Doc 47.5 21214.1✓(33.0%)✓(22.5%)✓TXT/L/C/TAB/I 23.6

2 Related Work
--------------

Benchmarks for Document Understanding. A great amount of datasets have emerged to evaluate the DU capabilities of LVLMs. Many datasets focus exclusively on either a single component (_e.g.,_ table, chart)[[13](https://arxiv.org/html/2407.01523v3#bib.bib13); [15](https://arxiv.org/html/2407.01523v3#bib.bib15); [21](https://arxiv.org/html/2407.01523v3#bib.bib21); [22](https://arxiv.org/html/2407.01523v3#bib.bib22)] or a single page[[12](https://arxiv.org/html/2407.01523v3#bib.bib12); [14](https://arxiv.org/html/2407.01523v3#bib.bib14)] from the full documents. Some recent DU datasets[[16](https://arxiv.org/html/2407.01523v3#bib.bib16); [17](https://arxiv.org/html/2407.01523v3#bib.bib17); [18](https://arxiv.org/html/2407.01523v3#bib.bib18); [23](https://arxiv.org/html/2407.01523v3#bib.bib23); [19](https://arxiv.org/html/2407.01523v3#bib.bib19)] attempt to assess multi-page documents, but still exhibit shortcomings in terms of document length (page number), information density (token number) and the construction approaches. Specifically, MP-DocVQA[[16](https://arxiv.org/html/2407.01523v3#bib.bib16)] is an extension of DocVQA[[12](https://arxiv.org/html/2407.01523v3#bib.bib12)] and inherently absent of both cross-page and unanswerable questions. Annotating from scratch, DUDE[[17](https://arxiv.org/html/2407.01523v3#bib.bib17)] includes a small percentage of cross-page questions (2.1%) and unanswerable questions (12.7%). However, due to the relatively short context length (5.3 pages on average) and the use of crowd-sourced annotations, questions in DUDE tend to be less challenging and somewhat less rigorous. SlideVQA features 20-page documents and cross-page questions (12.9%). Nevertheless, the documents in SlideVQA are in slide-deck format and of relatively low information density. Moreover, these cross-page questions are HotpotQA-style[[24](https://arxiv.org/html/2407.01523v3#bib.bib24)] created by instantiating entity graphs and co-referencing in-graph entities across multiple pages. The entity graph from a closed document tends to be sparse and has significant shortcuts (see examples in Appendix[A.4](https://arxiv.org/html/2407.01523v3#A1.SS4 "A.4 Existing Question Editing ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations")). These shortcuts sometimes lead to false cross-page questions that actually do not require answer evidence across different pages. The recent FinanceBench[[19](https://arxiv.org/html/2407.01523v3#bib.bib19)] features both extremely long-context documents and practical, scalable cross-page questions. However, its documents are exclusively financial reports. Additionally, the reference answers are in open-ended formats, making the expert-level manual evaluation indispensable. The above reasons limit the broader applicability of FinanceBench. To our best knowledge, MMLongBench-Doc is the first comprehensive, qualified, and easy-to-use benchmark on the long-context DU task. More detailed descriptions and comparisons are presented in Table[1](https://arxiv.org/html/2407.01523v3#S1.T1 "Table 1 ‣ 1 Introduction ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations").

#### Models for Document Understanding.

There are two main branches of models for automatic DU tasks. The first approach employs two-stream, OCR-dependent architectures to separately encode textual information (parsed via OCR) and visual information (images and/or layout structures)[[25](https://arxiv.org/html/2407.01523v3#bib.bib25); [26](https://arxiv.org/html/2407.01523v3#bib.bib26); [27](https://arxiv.org/html/2407.01523v3#bib.bib27)]. In contrast, the second approach develops OCR-free models that understand documents in an end-to-end manner[[28](https://arxiv.org/html/2407.01523v3#bib.bib28); [29](https://arxiv.org/html/2407.01523v3#bib.bib29)]. With the rapid advancement of LVLMs, the latter approach has dominated the current DU solutions. As mentioned above, a range of LVLMs demonstrate promising performance on single-page DU datasets. However, as shown in Section[4](https://arxiv.org/html/2407.01523v3#S4 "4 Evaluation ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), even the most advanced LVLMs fall significantly short of achieving satisfactory performance on our benchmark. It reveals that understanding lengthy documents still poses great challenges to current LVLMs.

#### Long-context LVLMs and LLMs.

Lengthy documents necessitate the use of LVLMs or LLMs with extended context sizes. Several benchmarks[[30](https://arxiv.org/html/2407.01523v3#bib.bib30); [31](https://arxiv.org/html/2407.01523v3#bib.bib31); [32](https://arxiv.org/html/2407.01523v3#bib.bib32); [33](https://arxiv.org/html/2407.01523v3#bib.bib33)] and solutions[[34](https://arxiv.org/html/2407.01523v3#bib.bib34); [35](https://arxiv.org/html/2407.01523v3#bib.bib35); [36](https://arxiv.org/html/2407.01523v3#bib.bib36); [37](https://arxiv.org/html/2407.01523v3#bib.bib37)] have been proposed to evaluate and develop long-context LLMs. However, there exists limited related work for long-context LVLMs, leaving this area largely unexplored. Until very recently, contemporary studies[[38](https://arxiv.org/html/2407.01523v3#bib.bib38); [39](https://arxiv.org/html/2407.01523v3#bib.bib39); [40](https://arxiv.org/html/2407.01523v3#bib.bib40)] assess and/or improve LVLMs’ multi-image understanding capabilities. Evaluations on both MMLongBench-Doc and these works indicate that current LVLMs are still not fully equipped to handle long-context DU and many other practical tasks that require extensive contextual comprehension.

3 MMLongBench-Doc
-----------------

We design a three-stage annotation pipeline for the construction of our benchmark. The three stages will be introduced in Section[3.1](https://arxiv.org/html/2407.01523v3#S3.SS1 "3.1 Document Collection ‣ 3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), Section[3.2](https://arxiv.org/html/2407.01523v3#S3.SS2 "3.2 Question and Answer Collection ‣ 3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), and Section[3.3](https://arxiv.org/html/2407.01523v3#S3.SS3 "3.3 Quality Control ‣ 3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), respectively. We also provide key statistics of our benchmark in Section[3.4](https://arxiv.org/html/2407.01523v3#S3.SS4 "3.4 Dataset Overview and Analysis ‣ 3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations").

### 3.1 Document Collection

As a long-context DU benchmark, the documents shall be of diverse topics and lengthy enough. To this end, we crawl a great amount of documents from various sources. Then we select the lengthy ones from these documents. Specifically, we encompass a diverse array of documents from two approaches. (1) Existing documents from four previous datasets: DUDE[[17](https://arxiv.org/html/2407.01523v3#bib.bib17)], SlideVQA[[18](https://arxiv.org/html/2407.01523v3#bib.bib18)], ChartQA[[13](https://arxiv.org/html/2407.01523v3#bib.bib13)], and FinanceBench[[19](https://arxiv.org/html/2407.01523v3#bib.bib19)]. (2) Newly-collected documents from Arxiv 3 3 3 https://arxiv.org, ManualsLib 4 4 4 https://www.manualslib.com and Google Search 5 5 5 https://www.google.com.sg. Then we (1) filter out the documents with fewer than 15 pages or license restrictions and (2) down-sample documents from DUDE, SlideVQA, and FinanceBench for a more balanced distribution. Detailed descriptions of our selection and processing procedure can be found in Appendix[A.1](https://arxiv.org/html/2407.01523v3#A1.SS1 "A.1 Existing Document Collection ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations") and Appendix[A.2](https://arxiv.org/html/2407.01523v3#A1.SS2 "A.2 Newly-annotated Document Collection ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations").

In summary, we collect a total of 135 documents. Among them, 76 documents are from existing datasets and incorporate previously annotated questions (represented as triangles). The remaining 59 documents are newly collected and incorporate no existing questions. We manually categorize them into 7 types: Research Report, Financial Report, Academic Paper, Brochure, Guideline, Administration & Industry File, Tutorial / Workshop. We showcase some instances of these documents in Appendix[A.3](https://arxiv.org/html/2407.01523v3#A1.SS3 "A.3 Document Examples ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations").

### 3.2 Question and Answer Collection

To serve as a high-quality and comprehensive benchmark, the question annotation of our benchmark adheres to the following standards: (1) All questions shall be neither over-easy nor over-difficult. (2) Questions are not repetitively derived from the same page or the same pattern. (3) The distribution of evidence numbers, evidence sources, and evidence locations for the questions shall be balanced. (4) No questions shall be answered correctly without accessing the relevant documents.

Ten authors serve as expert-level annotators for the question-and-answer collection. All of them are doctors or Ph.D. students proficient in English reading and writing. Before formal annotation, they undergo a training session and pre-annotate three documents for practice. We iteratively review their annotation results and provide personalized feedback until their annotations meet the standards mentioned above. Regarding the formal annotation, we divide 135 documents into 54 batches (each having 2-4 documents) and dispatch these batches to annotators. We then ask the annotators to submit their results in units of batches and set reasonable time intervals for each batch’s submission. We timely evaluate their annotations after each submission and remind the annotators if their questions in this turn diverge from the standards. It avoids the annotators rushing all assignments in a short time and benefits the annotation quality. We recommend the annotators take 60-90 minutes on each document. Specifically, the annotators shall rapidly read through the whole document in the first 15-30 minutes. For the remaining time, they shall dive deep into specific components to modify existing annotations and/or add new annotations as detailed below.

Modify Existing Questions. Documents collected from existing datasets had been annotated with some questions and answers from previous work. However, their crowd-sourcing annotations inevitably make some questions, answers, and other meta information unqualified. Therefore, we edit their annotations before including them as a component of our benchmark.

Specifically, we classify six potential problems in original annotations: Wrong Answers or Evidence Pages, Repetitive Question, Ambiguous Question, Decontextualization-required Question, Low Document-relevant Question and Potential Shortcut. See detailed explanations and examples about these problems in Appendix[A.4](https://arxiv.org/html/2407.01523v3#A1.SS4 "A.4 Existing Question Editing ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). Given an existing document, the annotators are tasked to evaluate each existing question’s quality according to whether they have one or more above problems and assign a label from {Retain, Revise, Remove} for each question. Then the annotators would revise the Revise questions to meet our quality criteria and remove the Remove questions. Among all 425 original questions from 76 existing documents, 32.2% of them are revised and 46.1% are removed. We finally collect 211 questions in this procedure. The corresponding GUI is shown in Appendix[A.7](https://arxiv.org/html/2407.01523v3#A1.SS7 "A.7 GUI Screenshots ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations").

Add New Questions. We newly annotate questions on both existing and newly collected documents to expand the questions in our benchmark. Specifically, we ask annotators to add about 3 questions on existing documents, and 6 questions on newly-collected documents. Given most existing questions (even after editing) are single-page ones and sourced from texts, we put more focus on (1) cross-page and unanswerable questions and (2) questions sourced from tables, charts, and images for newly added questions to balance the distribution. We detail the quantitative requirements in Appendix[A.5](https://arxiv.org/html/2407.01523v3#A1.SS5 "A.5 New Question Annotation ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). Associated with questions, annotators also provide reference answers and meta-information (_i.e.,_ evidence sources, answer format, evidence locations) for all samples. We finalized a collection of 965 samples in this procedure. The corresponding GUI is shown in Appendix[A.7](https://arxiv.org/html/2407.01523v3#A1.SS7 "A.7 GUI Screenshots ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations").

### 3.3 Quality Control

Combining the merits of humans and LVLMs, we adopt a three-round, semi-automatic quality control procedure to improve the annotation quality of our benchmark. We detail each round in the following components and leave the discussion of potential bias in Appendix[A.6](https://arxiv.org/html/2407.01523v3#A1.SS6 "A.6 Potential Bias for LVLM-based Quality Checking ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations").

Document-relevant Detection. Our benchmark is designed to evaluate LVLMs’ long-context document understanding abilities. All questions are expected to be unanswerable without access to corresponding documents. To remove low document-relevant questions (_i.e.,_ questions not relying on documents), we feed each annotated question WITHOUT documents to GPT-4o. A question will be identified as low document-relevant question if GPT-4o correctly predicts under this case. Ultimately, 94 samples are identified as low document-relevant questions and removed in this round.

Self-reflection. We draw inspirations from MMBench[[41](https://arxiv.org/html/2407.01523v3#bib.bib41)] and leverage LVLMs to reduce the wrongly-annotated samples. Specifically, we feed the remaining questions from the last round WITH their documents to GPT-4o. Samples whose model predictions are inconsistent with the reference answers are sent back to corresponding annotators. The annotators are asked to check each question and identify whether the inconsistency is caused by problematic annotation or not. As a result, 13.8% of the samples are identified as problematic annotations. The annotators revise them accordingly.

Cross-checking. In parallel, annotators cross-check the annotated samples from other annotators and determine the inconsistency reasons the same as described above. We calculate Cohen’s kappa value of their identifications as 0.42 (17.5% inconsistent samples), showing a moderate agreement. Regarding the 17.5% inconsistent samples, two primary authors serve as meta-annotators and make final decisions on them (and if necessary, revise accordingly).

Statistic Number
Documents 135
- Type 7
- Average/Medium pages 47.5 / 28
- Average/Medium length 21,214.1 / 12,179
Total questions 1,082
- Single-page question 494 (45.7%)
- Cross-page questions 365 (33.7%)
- Unanswerable questions 223 (20.6%)
- Derived questions 184 (17.0%)
- Newly-annotated questions 898 (83.0%)
(Evidence source)
- Pure-text 305 (35.5%)
- Layout 119 (13.9%)
- Table 218 (25.4%)
- Chart 178 (20.7%)
- Image 304 (35.4%)
(Answer Format)
- String 250 (29.1%)
- Integer 299 (34.8%)
- Float 159 (18.5%)
- List 151 (17.6%)
Avg./Max. question length 16.4 / 60
Avg./Max. answer length 2.8 / 54

Table 2: Dataset Statistics

![Image 2: Refer to caption](https://arxiv.org/html/2407.01523v3/x2.png)

Figure 2: Detailed distribution of documents. Top: Document type. Middle: Page Number. Bottom: Token Number.

![Image 3: Refer to caption](https://arxiv.org/html/2407.01523v3/x3.png)

Figure 3: Detailed distribution of questions & answers. Left: Absolute position of answer evidences (the page index). Middle: Relative position (the page index/document page number). Right: Evidence page number of each question. (0: unanswerable question; >2: cross-page question).

### 3.4 Dataset Overview and Analysis

The main statistics of MMLongBench-Doc are presented in Table[2](https://arxiv.org/html/2407.01523v3#S3.T2 "Table 2 ‣ Figure 2 ‣ 3.3 Quality Control ‣ 3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). Overall, our benchmark consists of 1,082 questions. These questions are constructed upon 135 lengthy documents across 7 document types, with an average of 47.5 pages and 21,214.1 tokens. Please see detailed distributions of these documents in Figure[2](https://arxiv.org/html/2407.01523v3#S3.F2 "Figure 2 ‣ 3.3 Quality Control ‣ 3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). Regarding the questions, there are 494 single-page questions (1 evidence page), 365 cross-page questions (2+ evidence pages), and 223 unanswerable questions (no evidence page). These three types of questions evaluate the LVLMs’s long-context DU capabilities from complementary aspects: the localization ability, the cross-page comprehension ability, and the hallucination severity, respectively. For single-page and cross-page questions, their answer evidence is scattered among different context sources (_i.e.,_ text, layout, table, chart, image) and evenly distributed across different locations of the documents (see Table[2](https://arxiv.org/html/2407.01523v3#S3.T2 "Table 2 ‣ Figure 2 ‣ 3.3 Quality Control ‣ 3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), Figure[3](https://arxiv.org/html/2407.01523v3#S3.F3 "Figure 3 ‣ 3.3 Quality Control ‣ 3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations") Left and Middle). Also notably, 28.6% of cross-page questions have more than two evidence pages, which further enhances the challenge of our benchmark.

Table 3: Evaluation of various models on MMLongBench-Doc. We report the generalized accuracy of five types of evidence sources including pure text (TXT), layout (LAY), chart (CHA), table (TAB), and image (IMG). We also present the generalized accuracy of questions categorized by the number of evidence pages: single-page (SIN), cross-page (MUL), and unanswerable (UNA) questions. The best and second-best performance in each section are highlighted.

Model#Param Context Evidence Source Evidence Page ACC F1
Window TXT LAY CHA TAB FIG SIN MUL UNA
OCR (Tesseract[[42](https://arxiv.org/html/2407.01523v3#bib.bib42)]) + Large Language Models (LLMs)
Open-source Models
ChatGLM-128k[[37](https://arxiv.org/html/2407.01523v3#bib.bib37)]6B 128k 23.4 12.7 9.7 10.2 12.2 18.8 11.5 18.1 16.3 14.9
Mistral-Instruct-v0.2[[43](https://arxiv.org/html/2407.01523v3#bib.bib43)]7B 32k 19.9 13.4 10.2 10.1 11.0 16.9 11.3 24.1 16.4 13.8
Mixtral-Instruct-v0.1[[44](https://arxiv.org/html/2407.01523v3#bib.bib44)]8x7B 32k 24.2 14.8 12.5 15.0 13.7 21.3 14.1 13.1 17.0 16.9
Mixtral-Instruct-v0.1[[44](https://arxiv.org/html/2407.01523v3#bib.bib44)]8x22B 64k 34.2 21.3 19.5 21.3 19.2 27.7 21.9 32.4 26.9 24.7
Proprietary Models
QWen-Plus[[45](https://arxiv.org/html/2407.01523v3#bib.bib45)]-32k 17.4 15.6 7.4 7.9 8.8 14.2 10.6 42.2 18.9 13.4
DeepSeek-V2[[46](https://arxiv.org/html/2407.01523v3#bib.bib46)]-32k 27.8 19.6 8.8 17.0 9.4 20.2 15.4 48.1 24.9 19.6
Claude-3 Opus[[4](https://arxiv.org/html/2407.01523v3#bib.bib4)]-32k 30.8 30.1 16.4 24.4 16.3 32.0 18.6 30.9 26.9 24.5
Gemini-1.5-Pro[[3](https://arxiv.org/html/2407.01523v3#bib.bib3)]-32k 29.3 15.9 12.5 17.7 11.5 21.2 16.4 73.4 31.2 24.8
GPT-4-turbo[[47](https://arxiv.org/html/2407.01523v3#bib.bib47)]-128k 36.5 21.0 20.7 24.3 17.3 28.7 23.8 31.2 27.6 25.9
GPT-4o[[2](https://arxiv.org/html/2407.01523v3#bib.bib2)]-128k 41.1 23.4 28.5 38.1 22.4 35.4 29.3 18.6 30.1 30.5
Large Visual Language Models (LVLMs)
Open-source, 7-14B Models
DeepSeek-VL-Chat[[48](https://arxiv.org/html/2407.01523v3#bib.bib48)]7.3B 4k 7.2 6.5 1.6 5.2 7.6 5.2 7.0 12.8 7.4 5.4
Idefics2[[49](https://arxiv.org/html/2407.01523v3#bib.bib49)]8B 8k 9.0 10.6 4.8 4.1 8.7 7.7 7.2 5.0 7.0 6.8
MiniCPM-Llama3-V2.5[[50](https://arxiv.org/html/2407.01523v3#bib.bib50); [51](https://arxiv.org/html/2407.01523v3#bib.bib51)]8B 2k 11.9 10.8 5.1 5.9 12.2 9.5 9.5 4.5 8.5 8.6
InternLM-XC2-4KHD[[5](https://arxiv.org/html/2407.01523v3#bib.bib5)]8B 16k 9.9 14.3 7.7 6.3 13.0 12.6 7.6 9.6 10.3 9.8
mPLUG-DocOwl 1.5[[52](https://arxiv.org/html/2407.01523v3#bib.bib52)]8.1B 4k 8.2 8.4 2.0 3.4 9.9 7.4 6.4 6.2 6.9 6.3
Qwen-VL-Chat[[53](https://arxiv.org/html/2407.01523v3#bib.bib53)]9.6B 6k 5.5 9.0 5.4 2.2 6.9 5.2 7.1 6.2 6.1 5.4
Monkey-Chat[[54](https://arxiv.org/html/2407.01523v3#bib.bib54)]9.8B 2k 6.8 7.2 3.6 6.7 9.4 6.6 6.2 6.2 6.2 5.6
Open-source, >14B Models
CogVLM2-LLaMA3-Chat[[9](https://arxiv.org/html/2407.01523v3#bib.bib9)]19B 8k 3.7 2.7 6.0 3.2 6.9 3.9 5.3 3.7 4.4 4.0
InternVL-Chat-v1.5[[6](https://arxiv.org/html/2407.01523v3#bib.bib6)]26B 4k 14.0 16.2 7.1 10.1 16.6 14.9 12.2 17.5 14.6 13.0
EMU2-Chat[[55](https://arxiv.org/html/2407.01523v3#bib.bib55)]37B 2k 6.1 9.7 2.6 3.8 7.7 5.7 6.1 16.5 8.3 5.5
Proprietary Models
Claude-3 Opus[[4](https://arxiv.org/html/2407.01523v3#bib.bib4)]-200k 24.9 24.7 14.8 13.0 17.1 25.6 13.8 7.6 17.4 18.1
Gemini-1.5-Pro[[3](https://arxiv.org/html/2407.01523v3#bib.bib3)]-128k 21.0 17.6 6.9 14.5 15.2 21.1 11.1 69.2 28.2 20.6
GPT-4V(ision)[[47](https://arxiv.org/html/2407.01523v3#bib.bib47)]-128k 34.4 28.3 28.2 32.4 26.8 36.4 27.0 31.2 32.4 31.2
GPT-4o[[2](https://arxiv.org/html/2407.01523v3#bib.bib2)]-128k 46.3 46.0 45.3 50.0 44.1 54.5 41.5 20.2 42.8 44.9
Human Baseline
Human Experts----------65.8 66.0

4 Evaluation
------------

### 4.1 Evaluation Protocol

We follow MATHVISTA[[56](https://arxiv.org/html/2407.01523v3#bib.bib56)] to conduct a three-step evaluation protocol: response generation, answer extraction, and score calculation. We adopt such a protocol out of three considerations: (1) Current LVLMs are instructed to generate long responses, rather than short-form answers, in conventional settings. (2) The evaluation of long responses, however, remains an open and challenging problem. (3) We focus on the document understanding (not instruction following) abilities of LVLMs.

Specifically, we impose no limitations on response generation stage to encourage LVLMs to answer the questions in a freestyle. Then we propose a unified LLM-based answer extractor (GPT-4o under our setting) to convert their long responses to short-form answers. Finally, we use a rule-based score calculator to evaluate the converted short answers. We report both generalized accuracy and generalized F1 score to balance the answerable (positive) and unanswerable (negative) questions. The used prompt, the high correlation between our automatic answer extractor and human evaluation, and the detailed rules of our score calculation are described in Appendix[B](https://arxiv.org/html/2407.01523v3#A2 "Appendix B Experimental Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations").

### 4.2 Experimental Setup

We evaluate 14 LVLMs on MMLongBench-Doc, including 4 proprietary LVLMs and 10 open-source LVLMs. To purely evaluate LVLMs’ long-context DU abilities, we screenshot each page of the PDF-formatted document with 144 DPI and feed all these PNG-formatted images to LVLMs in an end-to-end approach. Notably, all evaluated open-source LVLMs do not support multi-image inputs or present significant performance drops when fed with excessive images (_e.g.,_ more than 10 or 20 images). Therefore, we employ a concatenation strategy that combines all screenshot pages into 1 or 5 images and feeds these concatenated images to open-source LVLMs. Regarding proprietary LVLMs, we adopt the same concatenation strategy and reduce the image number to 20 for Claude-3-Opus to fit its maximum image threshold. For GPT-4o, GPT-4V, and Gemini-1.5-Pro, we directly send all original screenshots to them (_i.e.,_ the image number equals the page number).

For comparison, we also use the Tesseract[[42](https://arxiv.org/html/2407.01523v3#bib.bib42)] OCR model to recognize and extract texts from the documents and feed the parsed documents to 10 LLMs, including 6 proprietary and 4 open-source ones. Texts exceeding their context lengths are truncated. Notably, as a key component of the classical solution for the DU task, the OCR model can handle most flattened texts and some structured tables in the document. However, it cannot perceive the information from the charts or images. Thus the TXT-formatted, OCR-parsed documents are lossy documents in which the information is not fully preserved. More detailed hyperparameters are introduced in Appendix[B.5](https://arxiv.org/html/2407.01523v3#A2.SS5 "B.5 Model Hyperparameters ‣ Appendix B Experimental Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). Additionally, we also conduct manual evaluation on a subset of our datasets (238 questions from 29 documents) to indicate the difficulty of this task for humans.

### 4.3 Main Results

We compare the performance of different LVLMs and LLMs in Table[3](https://arxiv.org/html/2407.01523v3#S3.T3 "Table 3 ‣ 3.4 Dataset Overview and Analysis ‣ 3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), reporting their generalized accuracy and F1 scores (shown in the last two columns). Regarding LVLMs, we draw several conclusions as below: (1) The performance demonstrates that long-context DU is still a challenging and unsolved task for current LVLMs. The best-performing LVLM, GPT-4o, merely achieves a 44.9% F1 score. The second best-performing LVLM, GPT-4V, lags behind by over 10% percent and presents a 31.4% F1 score. All other LVLMs only achieve about 20% or even lower F1 scores. (2) Though far from satisfactory, GPT-4o performs much better than all other models (including GPT-4V). Thus we speculate that the multi-modal pre-training paradigm significantly benefits LVLMs’ cross-modality understanding capabilities. (3) Proprietary LVLMs perform better than open-source LVLMs by a large margin. We attribute it to the difference of acceptable image numbers: open-source LVLMs only support single-image or several-image inputs, while proprietary LVLMs can be fed with at least 20 images or even more. Given that lengthy documents have tens of even hundreds of pages, it is impractical for open-source LVLMs to accurately perceive the information in the documents from the excessively concatenated images. (4) The performances of different models are highly correlated with their acceptable image numbers and maximum image resolutions. Notably, open-source LVLMs that support high-resolution images (_i.e.,_ InternLM-XC2-4KHD and InternVL-Chat-v1.5) exhibit superior performance compared to those with lower resolution limits.

Surprisingly, LVLMs even demonstrate overall worse performance than LLMs, even LLMs are fed with lossy OCR-parsed documents. Specifically, Gemini-1.5-Pro and Claude-3 Opus have 4.2% and 6.4% absolute F1-score degradations on vision versions. And the best-performing LLM (Mixtral) also surpasses the best-performing LVLM (InternVL-v1.5) by 11.7%. The above results clearly reveal that most current LVLMs are still not proficient in cross-modality, long-context document understandings. It is promising that GPT-4o and GPT-4-turbo achieve better performance when seeing multi-modality PDF documents than parsed text by 14.4% and 5.3% F1-score, respectively. Their performances validate the feasibility, benefit, and necessity of understanding documents in an end-to-end, cross-modality approach. We speculate that the scarce related pre-training corpus (_i.e.,_ extremely multi-image or lengthy documents) hinders the long-context DU capabilities of other LVLMs. We will leave related explorations for future work.

Regarding the human evaluation, we observe 66.0% F1-score from our annotators and a significant performance gap (exceeding 20% in absolute) between the current LVLMs and humans. This gap highlights the challenges of document understanding for LVLMs and the necessity of our benchmark.

### 4.4 Fine-grained Results.

![Image 4: Refer to caption](https://arxiv.org/html/2407.01523v3/x4.png)

Figure 4: Fine-grained results on various document types and evidence sources.

Document Type. As illustrated in Figure[4](https://arxiv.org/html/2407.01523v3#S4.F4 "Figure 4 ‣ 4.4 Fine-grained Results. ‣ 4 Evaluation ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), LVLMs and LLMs exhibit distinct performance patterns across various document types. Our findings include: (1) All evaluated models demonstrate decent performance on industrial documents, which tend to have more standardized formats and less non-textual information. (2) The GPT series and Mixtral (_i.e.,_ the SoTA open-source LLM) show relatively balanced performance across different document types. In contrast, other models perform significantly worse in specialized domains such as academic papers and financial reports. (3) When equipped with OCR, LLM-based models like GPT-4 and Mixtral achieve comparable or even superior performance on industrial documents, academic papers, and brochures. Conversely, end-to-end LVLMs outperform OCR+LLMs in areas such as tutorials, research reports, and guidelines. We speculate that comprehending these latter document types requires more extensive multi-modal information, from which LVLMs significantly benefit.

Evidence Source. We categorize questions based on their evidence sources and present fine-grained results in Figure[4](https://arxiv.org/html/2407.01523v3#S4.F4 "Figure 4 ‣ 4.4 Fine-grained Results. ‣ 4 Evaluation ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations") and Table[3](https://arxiv.org/html/2407.01523v3#S3.T3 "Table 3 ‣ 3.4 Dataset Overview and Analysis ‣ 3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). Our observations reveal that only GPT-4o exhibits relatively balanced performance across the different sources. Other LVLMs, however, show inferior performance on questions related to charts and/or images compared to those related to text and/or layout. Additionally, LLMs generally demonstrate better or comparable performance to LVLMs on text- and table-related questions but show worse performance on questions involving other elements. This highlights the limitations of OCR (and other PDF parsers) when dealing with charts and images, as well as the gap in OCR capabilities between LVLMs and pure-text LLMs.

![Image 5: Refer to caption](https://arxiv.org/html/2407.01523v3/x5.png)

Figure 5: Relationships between evidence positions and model performances.

Evidence Position. We also examine how the evidence locations (_i.e.,_ the page indexes where the answer evidence is found) affect model performance. The results shown in Figure[5](https://arxiv.org/html/2407.01523v3#S4.F5 "Figure 5 ‣ 4.4 Fine-grained Results. ‣ 4 Evaluation ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations") reinforce that MMLongBench-Doc poses significant challenges for current models, at least partially due to the extended length of the documents. Almost all models (except InternVL-v1.5) exhibit their best performance on questions derived from the initial pages, while their performance declines progressively as the page index increases. Interestingly, two proprietary models, Gemini-Pro-1.5 and Claude-3-Opus, experience particularly sharp declines in performance.

Number of Evidence Page. We observe a consistent trend that all models achieve higher scores on single-page questions than cross-page questions. It reveals that gathering and reasoning over all necessary information across different pages is not trivial for current LVLMs and LLMs. More interestingly, evaluated LVLMs behave differently on unanswerable questions. GPT-4o and Claude-3 Opus adopt more aggressive strategies and usually tend to provide some answers. It makes their answers more likely helpful, but also increases the risk of hallucination and unfaithfulness (see their scores on unanswerable questions are much lower than answerable questions). On the contrary, Gemini-1.5-Pro, DeepSeek-VL-Chat, and EMU2-Chat are much more cautious and tend to refuse to answer questions about which they are uncertain. It makes their answers safer but less helpful (with large amounts of responses like I don’t know).

5 Analysis & Discussion
-----------------------

### 5.1 Oracle Setting

We conduct additional experiments to explore to what extent the challenges of MMLongBench-Doc are caused by the long-context lengths of documents. Specifically, we feed 820 answerable questions along with their oracle evidence pages (instead of the whole documents) to three representative LVLMs and show results in Figure[6](https://arxiv.org/html/2407.01523v3#S5.F6 "Figure 6 ‣ 5.1 Oracle Setting ‣ 5 Analysis & Discussion ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). On one hand, it indicates that long-context length is a significantly challenging factor for document understanding. Compared with the oracle-page setting, lengthy documents lead to more than 20% absolute performance degradation on Gemini-1.5-Pro and InternLM-XC2-4KHD. Regarding the single-page questions, the performance difference even achieves up to 30%. On the other hand, the overall performance achieves only about 40% and 30% for Gemini-1.5-Pro and InternLM-XC2-4KHD even under oracle-page setting. And the improvement for GPT-4o is much less (about 10%). It demonstrates that the development of long-context LVLMs can largely facilitate, though still can not fully solve, the long-context DU task.

![Image 6: Refer to caption](https://arxiv.org/html/2407.01523v3/x6.png)

Figure 6: Performance comparisons between normal setting (feeding models with the whole documents) and oracle setting (feeding models only with the evidence pages) among three LVLMs.

![Image 7: Refer to caption](https://arxiv.org/html/2407.01523v3/x7.png)

Figure 7: Error distribution

### 5.2 Error Analysis

We further conduct error analysis to understand the bottleneck of current LVLMs in a qualitative approach. Specifically, we randomly select 72 error predictions from GPT-4o’s responses and manually check their error reasons. These errors are categorized into seven types: Perceptual Error, Irrelevant Answer, Incomplete Evidence, Hallucinated Evidence, Extractor Error, Reasoning Error and Knowledge Lacking. The distribution of these errors is illustrated in Figure[7](https://arxiv.org/html/2407.01523v3#S5.F7 "Figure 7 ‣ 5.1 Oracle Setting ‣ 5 Analysis & Discussion ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). It indicates that most errors come from the model’s hallucination (_i.e.,_ wrong explanations and answers to unanswerable questions) and perceptual errors (mainly in visual contexts). Additionally, GPT-4o sometimes misunderstands the intent of questions and provides irrelevant responses. The errors caused by collecting incomplete evidence (for cross-page questions) are also unignorable. The descriptions and examples of these error types are detailed in Appendix[C.1](https://arxiv.org/html/2407.01523v3#A3.SS1 "C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations").

6 Conclusion
------------

In this work, we present MMLongBench-Doc to evaluate the long-context DU capabilities of LVLMs. Extensive experiments on 14 LVLMs (and 10 LLMs for comparison) reveal that the understanding of lengthy documents poses great challenges to current LVLMs. Even though the performance of GPT-4o proves the benefit of end-to-end, multi-modality perception for DU tasks, most LVLMs struggle on long visual contexts (_i.e.,_ extremely multiple images) and show inferior performance compared to OCR+LLM pipelines. We hope that the construction of our benchmark could push forward the development of more powerful LVLMs on lengthy document understanding.

Acknowledgements
----------------

This study is supported under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). This work is also supported by Shanghai Artificial Intelligence Laboratory, the National Key R&D Program of China (2022ZD0160201).

References
----------

*   [1] Lutz Bornmann and Rüdiger Mutz. Growth rates of modern science: A bibliometric analysis based on the number of publications and cited references. Journal of the Association for Information Science and Technology, 66, 2014. 
*   [2] Open AI. Hello gpt-4o, 2024. 
*   [3] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. 
*   [4] Anthropic. Introducing the next generation of claude, 2024. 
*   [5] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. Internlm-Xcomposer2-4KHD: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. ArXiv preprint, abs/2404.06512, 2024. 
*   [6] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. ArXiv preprint, abs/2404.16821, 2024. 
*   [7] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning, 2023. 
*   [8] Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. LLaVA-NeXT: Stronger llms supercharge multimodal capabilities in the wild, 2024. 
*   [9] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. CogVLM: Visual expert for pretrained language models, 2023. 
*   [10] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding, 2024. 
*   [11] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document, 2024. 
*   [12] Minesh Mathew, Dimosthenis Karatzas, R.Manmatha, and C.V. Jawahar. Docvqa: A dataset for vqa on document images. 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 2199–2208, 2020. 
*   [13] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, 2022. Association for Computational Linguistics. 
*   [14] Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. Infographicvqa. 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2582–2591, 2021. 
*   [15] Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng Chua. Towards complex document understanding by discrete reasoning. In Proceedings of the 30th ACM International Conference on Multimedia, pages 4857–4866, 2022. 
*   [16] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multi-page docvqa, 2023. 
*   [17] Jordy Van Landeghem, Rubèn Pérez Tito, Łukasz Borchmann, Michal Pietruszka, Pawel J’oziak, Rafal Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Ackaert, Ernest Valveny, Matthew B. Blaschko, Sien Moens, and Tomasz Stanislawek. Document understanding dataset and evaluation (DUDE). In ICCV, 2023. 
*   [18] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. SlideVQA: A dataset for document visual question answering on multiple images. In AAAI, 2023. 
*   [19] Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen. FinanceBench: A new benchmark for financial question answering, 2023. 
*   [20] Łukasz Borchmann, Michal Pietruszka, Tomasz Stanislawek, Dawid Jurkiewicz, Michał Turski, Karolina Szyndler, and Filip Gralinski. Due: End-to-end document understanding benchmark. In NeurIPS Datasets and Benchmarks, 2021. 
*   [21] Junpeng Liu, Yifan Song, Bill Yuchen Lin, Wai Lam, Graham Neubig, Yuanzhi Li, and Xiang Yue. VisualWebBench: How far have multimodal llms evolved in web page understanding and grounding?, 2024. 
*   [22] Marcin Kardas, Piotr Czapla, Pontus Stenetorp, Sebastian Ruder, Sebastian Riedel, Ross Taylor, and Robert Stojnic. AxCell: Automatic extraction of results from machine learning papers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8580–8594, Online, 2020. Association for Computational Linguistics. 
*   [23] Jon Saad-Falcon, Joe Barrow, Alexa Siu, Ani Nenkova, David Seunghyun Yoon, Ryan A. Rossi, and Franck Dernoncourt. Pdftriage: Question answering over long, structured documents, 2023. 
*   [24] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. 
*   [25] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. Layoutlm: Pre-training of text and layout for document image understanding. In Rajesh Gupta, Yan Liu, Jiliang Tang, and B.Aditya Prakash, editors, KDD ’20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, CA, USA, August 23-27, 2020, pages 1192–1200. ACM, 2020. 
*   [26] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. LayoutLMv2: Multi-modal pre-training for visually-rich document understanding. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2579–2591, Online, 2021. Association for Computational Linguistics. 
*   [27] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In Proceedings of the 30th ACM International Conference on Multimedia, MM ’22, page 4083–4091, New York, NY, USA, 2022. Association for Computing Machinery. 
*   [28] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In European Conference on Computer Vision (ECCV), 2022. 
*   [29] Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: screenshot parsing as pretraining for visual language understanding. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023. 
*   [30] Uri Shaham, Elad Segal, Maor Ivgi, Avia Efrat, Ori Yoran, Adi Haviv, Ankit Gupta, Wenhan Xiong, Mor Geva, Jonathan Berant, and Omer Levy. SCROLLS: Standardized CompaRison over long language sequences. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 12007–12021, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. 
*   [31] Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. L-eval: Instituting standardized evaluation for long context language models, 2023. 
*   [32] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A bilingual, multitask benchmark for long context understanding. ArXiv preprint, abs/2308.14508, 2023. 
*   [33] Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. bench: Extending long context evaluation beyond 100k tokens, 2024. 
*   [34] Szymon Tworkowski, Konrad Staniszewski, Mikolaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Milo’s. Focused transformer: Contrastive training for context scaling. ArXiv preprint, abs/2307.03170, 2023. 
*   [35] Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models. ArXiv preprint, abs/2309.12307, 2023. 
*   [36] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. ArXiv preprint, abs/2309.00071, 2023. 
*   [37] Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. LongAlign: A recipe for long context alignment of large language models. ArXiv preprint, abs/2401.18058, 2024. 
*   [38] Dingjie Song, Shunian Chen, Guiming Hardy Chen, Fei Yu, Xiang Wan, and Benyou Wang. Milebench: Benchmarking mllms in long context. ArXiv preprint, abs/2404.18532, 2024. 
*   [39] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning, 2024. 
*   [40] Yujie Lu, Xiujun Li, Tsu-Jui Fu, Miguel Eckstein, and William Yang Wang. From text to pixel: Advancing long-context understanding in mllms, 2024. 
*   [41] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMbench: Is your multi-modal model an all-around player? ArXiv preprint, abs/2307.06281, 2023. 
*   [42] Ray Smith. An overview of the tesseract ocr engine. In ICDAR, 2007. 
*   [43] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. 
*   [44] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts, 2024. 
*   [45] Qwen Team. Introducing qwen1.5, 2024. 
*   [46] DeepSeek-AI. DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model, 2024. 
*   [47] OpenAI. GPT-4 technical report, 2024. 
*   [48] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. DeepSeek-VL: towards real-world vision-language understanding. ArXiv preprint, abs/2403.05525, 2024. 
*   [49] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?, 2024. 
*   [50] Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui, Taiwen He, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. RLAIF-V: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. ArXiv preprint, abs/2405.17220, 2024. 
*   [51] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. LLaVA-UHD: an lmm perceiving any aspect ratio and high-resolution images. ArXiv preprint, abs/2403.11703, 2024. 
*   [52] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. mPLUG-DocOwl 1.5: Unified structure learning for ocr-free document understanding. ArXiv preprint, abs/2403.12895, 2024. 
*   [53] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A frontier large vision-language model with versatile abilities. ArXiv preprint, abs/2308.12966, 2023. 
*   [54] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. ArXiv preprint, abs/2311.06607, 2023. 
*   [55] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, et al. Generative multimodal models are in-context learners. ArXiv preprint, abs/2312.13286, 2023. 
*   [56] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024. 
*   [57] Tomasz Stanislawek, Filip Grali’nski, Anna Wr’oblewska, Dawid Lipi’nski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, and P.Biecek. Kleister: Key information extraction datasets involving long documents with complex layouts. In IEEE International Conference on Document Analysis and Recognition, 2021. 
*   [58] S.Svetlichnaya. DeepForm: Understand structured documents at scale., 2020. 
*   [59] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. Funsd: A dataset for form understanding in noisy scanned documents. 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW), 2:1–6, 2019. 
*   [60] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C.V. Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520, 2019. 
*   [61] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5376–5384, 2017. 
*   [62] Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In The IEEE Winter Conference on Applications of Computer Vision (WACV), March 2020. 
*   [63] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. ArXiv, abs/2101.11272, 2021. 
*   [64] Xingyu Chen, Zihan Zhao, Lu Chen, JiaBao Ji, Danyang Zhang, Ao Luo, Yuxuan Xiong, and Kai Yu. WebSRC: A dataset for web-based structural reading comprehension. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4173–4185, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. 

Appendix A Benchmark Construction Details
-----------------------------------------

### A.1 Existing Document Collection

Although previous datasets contain a relatively small proportion of lengthy documents, their absolute quantity should not be disregarded. Therefore, we compile lengthy documents from various datasets to include them as part of the documents in this benchmark. Specifically, we review and consider 21 previous document understanding (DU) datasets, and ultimately select 4 of them for further document selection. The selection reasons are shown in Table[4](https://arxiv.org/html/2407.01523v3#A1.T4 "Table 4 ‣ A.1 Existing Document Collection ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). All of these four datasets are licensed under the Creative Commons license (CC-BY) or other open-source licenses. Regarding the 4 selected datasets: DUDE[[17](https://arxiv.org/html/2407.01523v3#bib.bib17)], SlideVQA[[18](https://arxiv.org/html/2407.01523v3#bib.bib18)], ChartQA[[13](https://arxiv.org/html/2407.01523v3#bib.bib13)] and FinanceBench[[19](https://arxiv.org/html/2407.01523v3#bib.bib19)], we collect a total of 76 documents and detail our collection procedures as below.

Table 4: Comparison of selected and considered datasets for our benchmark.

Dataset Selected Comment
DUDE[[17](https://arxiv.org/html/2407.01523v3#bib.bib17)]✓-
SlideVQA[[18](https://arxiv.org/html/2407.01523v3#bib.bib18)]✓-
ChartQA[[13](https://arxiv.org/html/2407.01523v3#bib.bib13)]✓-
FinanceBench[[19](https://arxiv.org/html/2407.01523v3#bib.bib19)]✓-
DocVQA[[12](https://arxiv.org/html/2407.01523v3#bib.bib12)]✗Repetitive with some documents/questions in DUDE; Single-page documents only
MP-DocVQA[[16](https://arxiv.org/html/2407.01523v3#bib.bib16)]✗Repetitive with some documents/questions in DUDE; Single-page questions only
Kleister Charity[[57](https://arxiv.org/html/2407.01523v3#bib.bib57)]✗Repetitive with some documents/questions in DUDE; Over-simple
Kleister NDA[[57](https://arxiv.org/html/2407.01523v3#bib.bib57)]✗Repetitive with some documents/questions in DUDE; Over-simple
DeepForm[[58](https://arxiv.org/html/2407.01523v3#bib.bib58)]✗Repetitive with some documents/questions in DUDE; Over-simple
FUNSD[[59](https://arxiv.org/html/2407.01523v3#bib.bib59)]✗Repetitive with some documents/questions in DUDE; Over-simple
SROIE[[60](https://arxiv.org/html/2407.01523v3#bib.bib60)]✗Repetitive with some documents/questions in DUDE; Over-simple
Infograohics VQA[[14](https://arxiv.org/html/2407.01523v3#bib.bib14)]✗Infographs are not long-context documents
TAT-QA[[15](https://arxiv.org/html/2407.01523v3#bib.bib15)]✗Repetitive with some documents/questions in FinanceBench
PWC[[22](https://arxiv.org/html/2407.01523v3#bib.bib22)]✗Repetitive with our self-annotated questions from academic papers
PaperQA[[56](https://arxiv.org/html/2407.01523v3#bib.bib56)]✗Repetitive with our self-annotated questions from academic papers
TextbookQA[[61](https://arxiv.org/html/2407.01523v3#bib.bib61)]✗Low document-relevance; Over-simple
PlotQA[[62](https://arxiv.org/html/2407.01523v3#bib.bib62)]✗Repetitive with our self-annotated questions from academic papers and research reports
VisualMRC[[63](https://arxiv.org/html/2407.01523v3#bib.bib63)]✗Human performance reached; Website screenshots are not long-context documents
WebSRC[[64](https://arxiv.org/html/2407.01523v3#bib.bib64)]✗Human performance reached; Website screenshots are not long-context documents
VisualWebBench[[21](https://arxiv.org/html/2407.01523v3#bib.bib21)]✗Human performance reached; Website screenshots are not long-context documents
PDFTriage[[23](https://arxiv.org/html/2407.01523v3#bib.bib23)]✗Not publicly available

DUDE: We first filter all documents over 15 pages in the validation set of the original dataset, resulting in 87 documents. From these, we randomly sample 23 to include as a component of our benchmark documents.

SlideVQA: We download slide decks in the test set by following the instructions in the original repository 6 6 6 https://github.com/nttmdlab-nlp/SlideVQA. Pursuing lengthy documents, we slightly modified the code to remove the 20-page truncation procedure. Then we randomly select 27 slide decks for our benchmark documents.

FinanceBench: We randomly sample 5 financial reports from the test set.

ChartQA: Different from the above three datasets, ChartQA only contains chart screenshots cropped from documents. We take the following steps to recover these original documents: (1) We use the Tesseract OCR model[[42](https://arxiv.org/html/2407.01523v3#bib.bib42)] to recognize the text within the charts. (2) We use these texts as keywords to search for related documents on Google Search. (3) We manually identify these documents and remove all those that are less than 15 pages. From the ChartQA test set, we finalize a collection of 53 research reports from the Pew Research Center. We randomly sample 18 of these documents to include as a component of our benchmark documents.

### A.2 Newly-annotated Document Collection

Most documents collected from previous datasets are Industrial Files, Tutorial & Workshop, Finance Report and Research Report. To diversify our benchmark, we additionally collect 59 documents including Academic Paper, Brochure, and Guideline. We detail the collection procedures as below.

Academic Paper We collect 24 academic papers from Arxiv. All selected papers are over 15 pages (including references and appendix). To ensure annotation quality, each paper is either written or thoroughly read by at least one of the annotators.

Guideline and Brochure We collect 21 guidelines and 14 brochures from either ManualsLib or Google Search, covering diverse topics such as school, company, institution, products, service _etc._. Each document is manually reviewed by one corresponding annotator and other primary authors to ensure its availability for academic use 7 7 7 Should any authors request the removal of their documents, we will promptly comply..

### A.3 Document Examples

As stated in Section 2.1, the documents in MMLongBench-Doc can be categorized into seven types. We show the examples of each type as below.

![Image 8: Refer to caption](https://arxiv.org/html/2407.01523v3/figures/industrial_administration_documents.png)

Figure 8: Document example about Administration & Industrial File

![Image 9: Refer to caption](https://arxiv.org/html/2407.01523v3/figures/tutorial.png)

Figure 9: Document example about Tutorial & Workshop (only show first 50 pages)

![Image 10: Refer to caption](https://arxiv.org/html/2407.01523v3/figures/research_report.png)

Figure 10: Document example about Research Report

![Image 11: Refer to caption](https://arxiv.org/html/2407.01523v3/figures/finance_report.png)

Figure 11: Document example about Financial Report (only show first 50 pages)

![Image 12: Refer to caption](https://arxiv.org/html/2407.01523v3/figures/paper.png)

Figure 12: Document example about Academic Paper

![Image 13: Refer to caption](https://arxiv.org/html/2407.01523v3/figures/guideline.png)

Figure 13: Document example about Guidebook

![Image 14: Refer to caption](https://arxiv.org/html/2407.01523v3/figures/brochure.png)

Figure 14: Document example about Brochure

### A.4 Existing Question Editing

Documents collected from existing datasets had been annotated with some questions and answers. However, their crowd-sourcing annotations inevitably make some questions, answers, and other meta information unqualified. So we conduct a systematic and manual pipeline to edit their annotations. Specifically, we classify six potential problems in original annotations. The definitions and examples of these problems are shown below.

1. Wrong Answer or Evidence Pages: The reference answers and/or evidence pages in original datasets are wrongly annotated.

![Image 15: Refer to caption](https://arxiv.org/html/2407.01523v3/x8.png)

Figure 15: Example of the original annotation with Wrong Answer or Evidence Pages.

2. Repetitive Question: Too many questions with the same types (_e.g.,_ key information extraction) occur in a single document (or even on the same page or point).

![Image 16: Refer to caption](https://arxiv.org/html/2407.01523v3/x9.png)

Figure 16: Example of the original annotation with Repetitive Question.

3. Ambiguous Question: The question is ambiguous at the document level (_e.g.,_ the absence of entity, period, exact section or page, _etc._), or too broad to exactly answer.

![Image 17: Refer to caption](https://arxiv.org/html/2407.01523v3/x10.png)

Figure 17: Example of the original annotation with Ambiguous Question.

4. Potential Shortcut: The resolution of the question does not rely on two entities (across different pages) but only one of them, _i.e.,_ there exists a shortcut for this question.

![Image 18: Refer to caption](https://arxiv.org/html/2407.01523v3/x11.png)

Figure 18: Example of the original annotation with Potential Shortcut.

5. Low Document-relevant Question: The resolution of the question does not rely on the information from the document. It can be solved by the parametric knowledge in the LVLMs.

![Image 19: Refer to caption](https://arxiv.org/html/2407.01523v3/x12.png)

Figure 19: Example of the original annotation with Low Document-relevant Question.

6. Decontextulization-required Question: The understanding of the question is conditioned on a single page or even a single component of the document.

![Image 20: Refer to caption](https://arxiv.org/html/2407.01523v3/x13.png)

Figure 20: Example of the original annotation with Decontextulization-required Question.

When dealing with questions categorized under any of these six problem types, annotators are instructed to either revise or remove them. Typically, repetitive questions and those with potential shortcuts are removed. In contrast, wrongly-annotated or decontextualization-required questions are generally revised. For ambiguous and low document-relevant questions, the course of action depends more on the annotators’ discretion.

### A.5 New Question Annotation

We annotate new questions on both existing and newly-collected documents. To ensure a diverse range of questions, we impose limitations on the question distributions categorized by their types (_i.e.,_ single-page, cross-page or unanswerable) and evidence sources (_i.e.,_ table, chart, image). To balance existing questions which are mostly single-page and text-based, we place greater emphasis on cross-page, unanswerable, table-related, chart-related, and image-related questions. The detailed standards are as follows:

Document Type Evidence Page Evidence Source All
Cross-page Unanswerable Table Chart Image
Industrial File 2\geq 2--3\geq 3
Workshop & Tutorial 2\geq 2 1\geq 1—— 3\geq 3 ——6\geq 6
Research Report 3\geq 3 1\geq 1 2\geq 2 2\geq 2-5\geq 5
Financial Report 5\geq 5 2\geq 2 7\geq 7--10\geq 10
Academic Paper 3\geq 3 1\geq 1 2\geq 2—- 3\geq 3 —-6\geq 6
Guidebook 3\geq 3 1\geq 1--4\geq 4 7\geq 7
Brochure 2\geq 2 1\geq 1--3\geq 3 7\geq 7

Table 5: The minimum requirements for the number and distribution of questions, categorized by the evidence page numbers and evidence sources. We have set varying requirements for different document types based on their specific characteristics.

### A.6 Potential Bias for LVLM-based Quality Checking

As described in Section[3.3](https://arxiv.org/html/2407.01523v3#S3.SS3 "3.3 Quality Control ‣ 3 MMLongBench-Doc ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), we employ GPT-4o to remove document-agnostic (_i.e.,_ can be correctly answer without documents) samples and review potential wrongly-labeled samples. A reasonable speculation raises that our final benchmark can be biased toward GPT-4o’s answers, especially when GPT-4o outperforms others by a large margin. We discuss this potential bias as follows.

We check the effect of GPT-4o’s involvement in the quality control step-by-step. Specifically, we compare the performance of samples remained after each step across GPT-4o and two other competitive models (GPT-4V and Gemini-1.5-Pro). We show their results in the table below.

GPT-4o GPT-4V Gemini-1.5-Pro
No quality control 43.1%35.2%23.3%
+ document-relevance detection 41.2%31.0%20.5%
+ document-relevance detection + self-reflection / cross-checking 42.7%31.4%20.9%

Table 6: Step-wise performance comparison with and without LVLM-based quality checking

The results illustrate that the potential bias in step 1 (document-relevance detection) actually reduce, rather than increase, the performance gap between GPT4o and other models. It is because that we filter out all samples correctly answered by GPT4o without the access to documents. Under this case, the more significant performance drop of GPT-4V and Gemini-1.5-Pro can only be attributed to their limited document understanding and over-reliance on their internal knowledge. Regarding the step 2 and 3 (self-reflection and cross-checking), we provide inconsistent answers between human annotations and GPT4o’s predictions to annotators and ask them to check and revise accordingly. The potential bias of this step does lead to a slight performance bias (1.1% absolute difference at maximum). We believe that such bias is NOT the main cause of GPT4o’s significantly best performance. Without the involvement of GPT-4o in the quality control process, GPT-4o still significantly outperforms GPT-4V by 7.9% (43.1% - 35.2%) and Gemini-1.5-Pro by 19.8% (43.1% - 23.3%). Accordingly, all primary conclusions in our paper still hold.

### A.7 GUI Screenshots

We present the screenshots for editing existing questions and annotating new questions (along with their reference answers and meta-data) in Figure[21](https://arxiv.org/html/2407.01523v3#A1.F21 "Figure 21 ‣ A.7 GUI Screenshots ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations") and Figure[22](https://arxiv.org/html/2407.01523v3#A1.F22 "Figure 22 ‣ A.7 GUI Screenshots ‣ Appendix A Benchmark Construction Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations") respectively.

![Image 21: Refer to caption](https://arxiv.org/html/2407.01523v3/figures/GUI_existing_question.png)

Figure 21: GUI screenshot for editing existing questions (along with reference answers and meta-data)

![Image 22: Refer to caption](https://arxiv.org/html/2407.01523v3/figures/GUI_new_question.png)

Figure 22: GUI screenshot for annotating new questions (along with reference answers and meta-data)

### A.8 Annotation Cost

This benchmark is annotated by the authors of this paper. Therefore, the data collection does not need compensation. And we count the time cost of our benchmark as below.

Pre-annotation (about 45h): the development of annotation interface (10h), the writing of annotation guideline (5h), training session (10h), preliminary annotation and personalized feedback (20h).

Annotation (about 150h): It takes about 60-90 minutes for the annotation of each document. And all of the 130 documents take about 150 hours.

Post-annotation (about 45h): quality checking (30h), data processing and release preparation (15h).

In summary, our benchmark annotation approximately takes a total of 45+150+45=240 hours (1.36 man months).

Appendix B Experimental Details
-------------------------------

### B.1 Prompt for Response Generation

Listing 1: Prompt used for response generation. The [Document] is in PNG format (page screenshots) for LVLMs, and TXT format for LLMs.

[Document]

Read the above documents and answer this question:

[question]

Please make your answer as concise as possible.

### B.2 Prompt for Answer Extraction

Listing 2: Prompt used for answer extraction.

Given the question and analysis,you are tasked to extract answers with required formats from the free-form analysis.

-Your extracted answers should be one of the following formats:(1)Integer,(2)Float,(3)String and(4)List.If you find the analysis the question can not be answered from the given documents,type"Not answerable".Exception:If the analysis only tells you that it can not read/understand the images or documents,type"Fail to answer".

-Please make your response as concise as possible.Also note that your response should be formatted as below:

‘‘‘

Extracted answer:[answer]

Answer format:[answer format]

‘‘‘

Please read the following example,then extract the answer from the model response and type it at the end of the prompt.

---

Question:List the primary questions asked about the services in this report.

Analysis:The primary questions asked about the services in the report for The Limes Residential Home are:

1.Is the service safe?

2.Is the service effective?

3.Is the service caring?

4.Is the service responsive?

5.Is the service well-led?

Extracted answer:[’Is the servife safe?’,’Is the service effective’,’Is the serve caring?’,’Is the service responsive?’,’Is the service well-led?’]

Answer format:List

---

Question:How many regulations of the HSCA 2008 are breached in all according to this report?

Analysis:According to the report,the provider breached 10 Health and Social Care Act 2008(Regulated Activities)Regulations in total.Here are the specifics:

1.Regulation 13:Safeguarding service users from abuse and improper treatment

2.Regulation 12:Safe care and treatment

3.Regulation 18:Staffing

4.Regulation 11:Need for consent

5.Regulation 10:Dignity and respect

6.Regulation 9:Person-centred care

7.Regulation 17:Good governance

8.Regulation 18(CQC Registration Regulations 2009):Notification of other incidents

9.Regulation 18:Failure to maintain an accurate and up-to-date care plan

10.Regulation 11:Failure to implement the Mental Capacity Act 2005 code of practice effectively

These breaches involve issues concerning staffing,safeguarding,medicines management,dignity and respect,consent,care planning,governance,and failure to notify the CQC of incidents.

Extracted answer:10

Answer format:Integer

---

Question:According to the survey that is the percentage of Chinese who are paying more or about the same attention to politics after Trump’s election?

Analysis:The survey provided does not specify the percentage of Chinese individuals specifically who are paying more or about the same attention to politics after Trump’s election.The report focuses primarily on American demographics and does not include specific details about the Chinese population in relation to this question.If you need information about a different demographic or a summary of the findings from the American demographic,I can certainly help with that!

Extracted answer:Not answerable

Answer format:None

---

Question:How many quotations from male respondent over 50 years old are included in this report?

Analysis:The image you’ve provided appears to be a screenshot of a document with multiple charts.However,the text is too small and blurry to read accurately.If you can provide a clearer image or more context,I might be able to help you with your question.

Extracted answer:Fail to answer

Answer format:None

---

### B.3 Rules for Score Calculation

We evaluate the model’s responses by scoring the extracted answers against the reference answers. The scorer is rule-based and employs different strategies according to the format of the reference answer. We detail its rules as below:

String: We firstly use a series of regular expressions to determine whether the answers require exact matching (_e.g.,_ telephone numbers, email addresses, website addresses, file names, times, dates, _etc._). If an exact match is needed, we perform a straightforward string comparison and score the answer either 0 or 1. Otherwise, we follow previous work[[17](https://arxiv.org/html/2407.01523v3#bib.bib17)] and calculate the ANLS (Average Normalized Levenshtein Similarity) with a pre-defined threshold (τ=0.5\tau=0.5).

Integer: We perform an exact match comparison and score the answer either 0 or 1.

Float: We view the prediction and reference answers as equal if they fall within a 1% relative tolerance.

List: We adopt a relatively strict rule for scoring answers in list format: predictions that do not have the same number of elements as the reference receive a score of 0. For the remaining predictions, as Eq.[1](https://arxiv.org/html/2407.01523v3#A2.E1 "In B.3 Rules for Score Calculation ‣ Appendix B Experimental Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations") indicates, we score each element in order and use the minimum element-wise score as the score for the entire list. The element-wise scoring strategies is determined by the formats of elements (_i.e.,_ string, integer or float).

pred_list,ref_list=sorted(pred_list),sorted(ref_list)\displaystyle\texttt{pred\_list},\texttt{ref\_list}=\texttt{sorted(pred\_list)},\texttt{sorted(ref\_list)}(1)
Score(pred_list, ref_list)=min(\displaystyle\texttt{Score(pred\_list, ref\_list)}=\texttt{min}(
(Score(pred, ref) for pred, ref in zip(pred_list, ref_list)⌋\displaystyle\quad\quad[\texttt{Score(pred, ref) for pred, ref in zip(pred\_list, ref\_list)}]
)\displaystyle)

Evaluation detailed in the Appendix[B.4](https://arxiv.org/html/2407.01523v3#A2.SS4 "B.4 Human Evaluation on the Automatic Evaluation Pipeline ‣ Appendix B Experimental Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations") shows that while this scorer is not perfect, it aligns well with human judgment. We will continue refining these rules to cover more corner cases and enhance their accuracy.

### B.4 Human Evaluation on the Automatic Evaluation Pipeline

We conduct human evaluations to assess the performance of our automatic evaluation pipeline, which includes the answer extractor and the score calculator. Specifically, we randomly select 100 questions and review their responses from two representative LVLMs: GPT-4o and Gemini-1.5-Pro. We manually evaluate the correctness of each response and compare the results between human evaluation and automatic evaluation. The performance, as shown in Table[7](https://arxiv.org/html/2407.01523v3#A2.T7 "Table 7 ‣ B.4 Human Evaluation on the Automatic Evaluation Pipeline ‣ Appendix B Experimental Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), indicates a high correlation between human judgment and our automatic pipeline.

Model Inconsistent Evaluation
Ans. Extractor Scorer Overall
GPT-4o 4 2 6
Gemini-1.5-Pro 2 2 4

Table 7: We manually check 100 responses from GPT-4o and Gemni-1.5-Pro, and compare the evaluation results between humans and our automatic pipeline.

### B.5 Model Hyperparameters

The hyperparameters of used LVLMs and LLMs in Section 3.3 are detailed in Table[8](https://arxiv.org/html/2407.01523v3#A2.T8 "Table 8 ‣ B.5 Model Hyperparameters ‣ Appendix B Experimental Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). The temperature is set as 0.0 0.0, and the max_new_tokens is set as 1024 1024 for all the models. The ‘concatenated_images’ parameter determines the maximum number of images that can be combined into a single input for LVLMs. By concatenating multiple images, we can meet the minimum context window requirements. The ‘max_pages’ parameter specifies the maximum number of images that can be directly input into the LVLMS without concatenation.

Model Hyperparameters
LLM
ChatGLM-128k max_input_words=60000
Mistral-Instruct-v0.2-7B max_input_words=20000
Mixtral-Instruct-v0.1-8x7B max_input_words=20000
Mixtral-Instruct-v0.1-8x22B max_input_words=40000
QWen-Plus max_input_words=16000
DeepSeek-V2 max_input_words=20000
LVLM
DeepSeek-VL-Chat concatenated_images=5
Qwen-VL-Chat concatenated_images=5
Idefics2 concatenated_images=5
MiniCPM-Llama3-V2.5 concatenated_images=2
InternLM-XC2-4KHD concatenated_images=2
Monkey-Chat concatenated_images=1
CogVLM2-Llama3-Chat concatenated_images=1
InternVL-Chat-v1.5 concatenated_images=5
EMU2-Chat concatenated_images=5
LLM & LVLM
Claude-3 Opus version=claude-3-opus-20240229, concatenated_images=20
Gemini-1.5-Pro max_pages=120, version=gemini-1.5-pro-latest
GPT-4-turbo max_pages=120, version=gpt-4-turbo-2024-04-09
GPT-4o max_pages=120, version=gpt-4o-2024-05-13

Table 8: Model Hyperparameters

Appendix C Qualitative Study
----------------------------

### C.1 Error Analysis

We delve into the analysis of error by GPT-4o to further understand its bottlenecks and potentials on long-context document understanding. We manually check 72 incorrect responses and categorized their error reasons into 7 types. Except for the Extraction Error caused by our automatic evaluation pipeline (see Appendix[B.4](https://arxiv.org/html/2407.01523v3#A2.SS4 "B.4 Human Evaluation on the Automatic Evaluation Pipeline ‣ Appendix B Experimental Details ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations")), we detail and showcase another six reasons as below:

Perceptual Error: GPT-4o sometimes struggles to extract or understand visual information from document screenshots. For instance, it misinterprets the axes and colored circles in the charts shown in Figure[23](https://arxiv.org/html/2407.01523v3#A3.F23 "Figure 23 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). Additionally, it inaccurately counts the number of green bars in Figure[24](https://arxiv.org/html/2407.01523v3#A3.F24 "Figure 24 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). They demonstrate that even the cutting-edge LVLMs still fall short in fundamental perceptual capabilities.

Incomplete Evidence: Though GPT-4o has achieved significantly better global searching abilities compared to other models when dealing with lengthy, multi-modal documents, it sometimes still omits certain information. For example, GPT-4o misses one chapter author from Columbia University in the full list (Figure[25](https://arxiv.org/html/2407.01523v3#A3.F25 "Figure 25 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations")). Additionally, it overlooks an app that appears across two pages (Figure[26](https://arxiv.org/html/2407.01523v3#A3.F26 "Figure 26 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations")).

Hallucinated Evidence: As stated in Section 3.4, GPT-4o adopts more aggressive strategies and tends to provide more false-positive answers. It sometimes even fabricates non-existent evidence in documents to support its incorrect responses. For example, it references a non-existent page in Figure[27](https://arxiv.org/html/2407.01523v3#A3.F27 "Figure 27 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), and fabricates the content of a page in Figure[28](https://arxiv.org/html/2407.01523v3#A3.F28 "Figure 28 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). The above examples clearly reveal the importance of further research on LVLMs’ hallucination and safety.

Knowledge Lacking: Resolving certain questions requires both information from the documents and the parametric knowledge within LVLMs. We have observed error cases stemming from the absence of specific knowledge. For example, GPT-4o overlooks details about the fixed asset turnover ratio and uses the single-point value instead of the average value to calculate this metric (Figure[29](https://arxiv.org/html/2407.01523v3#A3.F29 "Figure 29 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations")). Additionally, it misidentifies buildings at Tsinghua University in Figure[30](https://arxiv.org/html/2407.01523v3#A3.F30 "Figure 30 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations").

Reasoning Error: Though not a primary cause, flawed reasoning based on correctly collected evidence and information from documents can sometimes lead to wrong answers. For example, GPT-4o correctly gathers all data but calculates a relative percentage instead of an absolute percentage in Figure[31](https://arxiv.org/html/2407.01523v3#A3.F31 "Figure 31 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"). Additionally, as shown in Figure[32](https://arxiv.org/html/2407.01523v3#A3.F32 "Figure 32 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), it correctly lists all quizzes but inaccurately counts them in the final step.

Irrelevant Answer: GPT-4o sometimes misunderstands the intent of questions and provides irrelevant responses. For instance, in Figure[33](https://arxiv.org/html/2407.01523v3#A3.F33 "Figure 33 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), GPT-4o answers about button operations when the question asks about button functions. Similarly, in Figure[34](https://arxiv.org/html/2407.01523v3#A3.F34 "Figure 34 ‣ C.1 Error Analysis ‣ Appendix C Qualitative Study ‣ MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations"), where the question asks for the MOST discrimination type, GPT-4o summarizes all types instead.

Perceptual Error: Case 1

![Image 23: Refer to caption](https://arxiv.org/html/2407.01523v3/x14.png)

Figure 23: Error example about Perceptual Error

Perceptual Error: Case 2

![Image 24: Refer to caption](https://arxiv.org/html/2407.01523v3/x15.png)

Figure 24: Error example about Perceptual Error

Incomplete Evidence: Case 1

![Image 25: Refer to caption](https://arxiv.org/html/2407.01523v3/x16.png)

Figure 25: Error example about Incomplete Evidence

Incomplete Evidence: Case 2

![Image 26: Refer to caption](https://arxiv.org/html/2407.01523v3/x17.png)

Figure 26: Error example about Incomplete Evidence

Hallucinated Evidence: Case 1

![Image 27: Refer to caption](https://arxiv.org/html/2407.01523v3/x18.png)

Figure 27: Error example about Hallucinated Evidence

Hallucinated Evidence: Case 2

![Image 28: Refer to caption](https://arxiv.org/html/2407.01523v3/x19.png)

Figure 28: Error example about Hallucinated Evidence

Knowledge Lacking: Case 1

![Image 29: Refer to caption](https://arxiv.org/html/2407.01523v3/x20.png)

Figure 29: Error example about Knowledge Lacking

Knowledge Lacking: Case 2

![Image 30: Refer to caption](https://arxiv.org/html/2407.01523v3/x21.png)

Figure 30: Error example about Knowledge Lacking

Reasoning Error: Case 1

![Image 31: Refer to caption](https://arxiv.org/html/2407.01523v3/x22.png)

Figure 31: Error example about Reasoning Error

Reasoning Error: Case 2

![Image 32: Refer to caption](https://arxiv.org/html/2407.01523v3/x23.png)

Figure 32: Error example about Reasoning Error

Irrelevant Answer: Case 1

![Image 33: Refer to caption](https://arxiv.org/html/2407.01523v3/x24.png)

Figure 33: Error example about Irrelevant Answer

Irrelevant Answer: Case 2

![Image 34: Refer to caption](https://arxiv.org/html/2407.01523v3/x25.png)

Figure 34: Error example about Irrelevant Answer

### C.2 Case Study

![Image 35: Refer to caption](https://arxiv.org/html/2407.01523v3/x26.png)

Figure 35: Case Study. Evidence source: table. The evidence pages are zoomed in. The correct extracted information and reasoning are colored in green, and the wrong ones are colored in red.

![Image 36: Refer to caption](https://arxiv.org/html/2407.01523v3/x27.png)

Figure 36: Case Study. Evidence source: text. The evidence pages are zoomed in. The correct extracted information and reasoning are colored in green, and the wrong ones are colored in red.

![Image 37: Refer to caption](https://arxiv.org/html/2407.01523v3/x28.png)

Figure 37: Case Study. Evidence source: layout. The evidence page is zoomed in. The correct extracted information and reasoning are colored in green, and the wrong ones are colored in red.

![Image 38: Refer to caption](https://arxiv.org/html/2407.01523v3/x29.png)

Figure 38: Case Study. Evidence source: image. The evidence page is zoomed in. The correct extracted information and reasoning are colored in green, and the wrong ones are colored in red.

Appendix D Limitations
----------------------

MMLongBench-Doc is the first comprehensive benchmark designed to evaluate the long-context document understanding capabilities of LVLMs. While our benchmark addresses significant gaps in the previous datasets, we acknowledge several limitations.

One primary limitation is the scale of the benchmark. Currently, our benchmark includes a test set comprising 135 documents and 1,082 questions. It is much smaller compared to previous datasets. The complexity and difficulty of annotations limit the scale of our benchmark. As a long-context benchmark, our documents average about 50 pages and 20,000 tokens. And most questions require either complicated reasoning or cross-page comprehension. It takes more than one hour for an expert-level annotator to read through a single document, and then edit existing instances and create new instances on this document. Given the purpose of MMLongBench-Doc as an evaluation benchmark, we prioritize annotation quality over quantity. Moreover, the results presented in Sections 3.3 and 3.4 confirm that the scale of our benchmark is sufficient for fine-grained evaluations across different document types, evidence sources, evidence pages, _etc._. Additionally, we plan to expand our benchmark by adding more documents and questions in future iterations.

We roughly categorize these questions into three types, _i.e.,_ single-page, cross-page, and unanswerable questions, based on whether evidence can be found in the documents and the number of evidence pages. However, unlike MMBench[[41](https://arxiv.org/html/2407.01523v3#bib.bib41)] or MathVista[[56](https://arxiv.org/html/2407.01523v3#bib.bib56)], we provide no further taxonomy to classify some (_e.g.,_ 7 or 20) fine-grained, evaluated reasoning or perception capabilities out of two main reasons: (1) Prior (_i.e.,_ pre-annotation) taxonomy limits the diversity of the questions. Therefore we provide no predefined classifications in our guideline and encourage the expert-level annotators to freely write questions without constraints. (2) The intrinsic complexity of document understanding presents significant challenges for establishing a posterior (_i.e.,_ post-annotation) taxonomy.

While there exist limitations in our benchmark, MMLongBench-Doc surely represents a significant step forward in this field. We would iteratively maintain and refine this benchmark and hope it could push forward the development of long-context document understanding.

Appendix E Social Impacts
-------------------------

The development and use of MMLongBench-Doc may have potential societal implications. For instance, biased or inaccurate outputs from benchmarked models could perpetuate harmful stereotypes or reinforce existing social inequalities. Additionally, the ability to process and analyze long documents could potentially be used to surveil or monitor individuals’ personal information. Developers and users of MMLongBench-Doc benchmark must be aware of these potential consequences and take steps to ensure responsible development and deployment of AI models.

Appendix F Author Statement
---------------------------

The authors state that all of the previous datasets that we collected are licensed under the Creative Commons license (CC-BY) or other open-source licenses. Using this dataset should abide by the [policy of OpenAI](https://openai.com/policies/terms-of-use). Regarding the newly collected documents, we manually check them to ensure their availability for academic use. Should any authors request the removal of their documents, we will promptly comply.

