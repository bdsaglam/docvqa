# Document Understanding Dataset and Evaluation (DUDE 😎)

Jordy Van Landeghem<sup>1,2</sup>    Rubèn Tito<sup>5</sup>    Łukasz Borchmann<sup>3</sup>    Michał Pietruszka<sup>3,6</sup>    Paweł Józiaik<sup>3,4</sup>  
 Rafał Powalski<sup>8</sup>    Dawid Jurkiewicz<sup>3,7</sup>    Mickaël Coustaty<sup>9</sup>    Bertrand Ackaert<sup>2</sup>    Ernest Valveny<sup>5</sup>  
 Matthew Blaschko<sup>1</sup>    Sien Moens<sup>1</sup>    Tomasz Stanisławek<sup>3</sup>

<sup>1</sup>KU Leuven   <sup>2</sup>Contract.fit   <sup>3</sup>Snowflake   <sup>4</sup>Warsaw University of Technology   <sup>5</sup>Computer Vision Center, Universitat Autònoma de Barcelona  
<sup>6</sup>Jagiellonian University   <sup>7</sup>Adam Mickiewicz University   <sup>8</sup>Instabase   <sup>9</sup>University of La Rochelle

**#non-answerable**

Q: In which year does the Net Requirement exceed 25,000?

A: None

**#abstractive #counting**

Q: How many attorneys are listed for the plaintiffs?

A: Two

**#layout-navigating #graphic-intensive**

Q: Are the margins of the page uniform on all pages?

A: Yes

**#extractive #list**

Q: What are the Years mentioned in Chart 1?

A: [2020, 2021, 2022]

**#multi-hop #layout-navigating**

Q: From the list of Top 10 Key Recovery Components, which is the last component listed on the second page?

A: Hope

**#abstractive #graphic-intensive**

Q: Does this document contain any checkboxes?

A: No

## Abstract

We call on the Document AI (DocAI) community to re-evaluate current methodologies and embrace the challenge of creating more practically-oriented benchmarks. Document Understanding Dataset and Evaluation (DUDE) seeks to remediate the halted research progress in understanding visually-rich documents (VRDs). We present a new dataset<sup>1</sup> with novelties related to types of questions, answers, and document layouts based on **multi-industry, multi-domain, and multi-page** VRDs of various origins, and dates. Moreover, we are pushing the boundaries of current methods by creating multi-task and multi-domain evaluation setups that more accurately simulate real-world situations where powerful generalization and adaptation under low-resource settings are desired. DUDE aims to set a new standard as a more practical, long-standing benchmark for the community, and we hope that it will lead to future extensions and contributions that address real-world challenges. Finally, our work illustrates the importance of finding more efficient ways to model language, images, and layout in DocAI.

<sup>1</sup>[huggingface.co/datasets/jordyvl/DUDE\\_loader](https://huggingface.co/datasets/jordyvl/DUDE_loader)

## 1. Introduction

Early stages of research and growth in any field are characterized by enacting proof-of-concept and demonstrating the feasibility of the proposed solution. In the Deep Learning era, this is often echoed by building narrow and simplified datasets that do not reflect real-world complexity, leading to models that may not be suitable for practical use.

The field of Document Understanding (DU) is not an exception to the recent proliferation of deep architectures, which in this case are predominantly used for classification and information extraction from documents. However, the wide and complex nature of documents presents many challenges that remain unsolved or not yet addressed. One such challenge is domain generalization, where a model trained on medical documents may not be directly applicable to financial or tabular content. Another challenge concerns task-agnostic architectures, where a model must be able to adapt to various DU subtasks such as document classification, key information extraction (KIE), and question answering (QA). Lastly, the high variability of document contents and layouts often leads to highly imbalanced sampleswithin document types, resulting in a long-tailed distribution with few or almost no samples to train a model.

Despite the importance of these challenges, there is currently no DU benchmark dataset that simultaneously addresses all of these issues. This paper proposes a novel dataset formulated as an instance of Document Visual Question Answering (DocVQA) to evaluate how well current DU solutions deal with multi-page documents, if they can navigate and reason over visual layouts, and if they can generalize their skills to different document types and domains.

The data collection and evaluation design of **DUDE** naturally motivates targeting models that can answer natural yet highly diverse questions (e.g., regarding document elements, their properties, and compositions) for any VRD (e.g., drawn from potentially unseen distributions of layouts, domains, and types). The presented problem setting relates to Multi-Domain Long-Tailed Recognition (MDLT) [96], which concerns learning from multi-domain imbalanced data whilst addressing label imbalance, divergent label distributions across domains, and possible train-test domain shift. Put plainly, since we cannot provide ground truth QA pairs for, e.g., stamps, on every document type (domain), we expect a solution to transfer the subtask ‘stamp detection’ learned on document types where stamps naturally occur (and thus training QA pairs were created organically) to other domains. The DocVQA and MDLT formulations of **DUDE** allow us to create a longstanding, challenging benchmark that in the future can be easily extended with more subtasks formulated as QA pairs, and domains relating to document types (see Limitations).

The contribution of this work is twofold. First, we have created **DUDE**, a novel large-scale, multi-paged, multi-domain, multi-industry DocVQA benchmark for evaluating DU progress. Second, we show that the zero-shot and fine-tuned performance of current state-of-the-art models applied to DU lags far behind human baselines, explained in part by the need for more holistic and efficient modeling of language, vision, and richly structured layouts.

## 2. Related Work

Document Understanding encompasses datasets related to various subtasks like document layout analysis [109, 49], classification [30], key information extraction [85, 35], table extraction [83, 108, 107], and visual question answering [57, 59, 91]. These benchmarks lead to end-to-end DU architectures that have transformed common DocAI practices [72, 5, 33, 23, 25, 50, 71]. These task-specific benchmarks, however, are often tailored to a single domain, limiting the ability to create and assess how well DU models generalize to other document types and domains. To fill this gap, we adopt a visual question answering (VQA) approach, which has been crucial in the growth of the DU field.

The VQA paradigm provides a natural language inter-

face for various tasks from both computer vision and natural language processing. In the latter, the question-answering approach has been successfully used in several domains, including medicine [67, 39, 64, 36, 48, 76, 61], open-domain knowledge [97, 54, 58, 53], emotions [26, 9], code [2, 51], logical reasoning [52, 100, 106, 95], claim verification [88, 32, 103], and math [104, 31, 16, 60, 4]. As a result of its ability to function as a natural language interface for various forms of data, this paradigm has been applied to other domains. For example, the question-answering approach is combined with modalities such as videos [44, 13, 14, 28, 17], images [98, 3, 29, 68, 7, 8], speech [99, 43], knowledge graphs [93, 84, 80, 22, 37], and maps [70, 15].

Overall, the convergence of computer vision and NLP through the emergence of VQA tasks has also opened up new avenues for research in the DU field, with many DU datasets now including rich visual content alongside questions. Yet, prior study on document VQA has mainly focused on single-page documents [57, 89, 56] with rare exceptions such as MP-DocVQA [90]. However, [57, 89] pose only extractive questions where the answer follows the context on which the question is defined as in other question answering benchmarks [78, 92, 42]. Moreover, these datasets do not contain *non-answerable* questions as in established (natural language) QA datasets like [77, 42]. To the best of our knowledge, there are no VQA datasets containing questions requiring lists as an answer. There are however few text-only QA datasets that contain such answer types [69, 46, 18]. Other datasets mainly related to our work are rather domain-specific like [111, 87, 56, 86, 73]. We give a detailed comparison of most related document VQA datasets in Table 1 highlighting the major contributions.

## 3. DUDE Dataset

While **DUDE** shares some similarities with existing VQA datasets, a closer comparison (see Table 1) highlights its unique features. We are confident that the model’s proficiency in the areas introduced in this work will showcase its capability to handle the intricacy and diversity of document understanding tasks in real-world scenarios.

**Documents.** The dataset covers a wide range of document types, sources and dates, as shown in Table 1 and Figure 1 where its diverse nature is confirmed by the spread of document content representations.<sup>2</sup> Moreover, it covers a broad range of domains, including medical, legal, technical, and financial, among others, to evaluate models’ ability to handle diverse topics and the specific knowledge each requires. Furthermore, the dataset contains documents with varying layouts: diverse text arrangements, font sizes, and styles, to

<sup>2</sup>This holds not only when textual content is considered but also for document images (Figure 9 in the Appendix).Figure 1: Visualization of inter-document similarities between samples from different datasets (t-SNE over TF-IDF representations of 1k passages from each source).

ensure that models can handle visually diverse documents.

In contrast to our proposal, current VQA datasets often focus on homogeneous documents, such as invoices in VQA-CD [55] or financial reports in TAT-DQA [111]. Even when not restricted to a single domain or layout, these datasets share essential characteristics. For example, InfographicsVQA [56] demonstrates significant diversity in topics and designs, but still embodies a preference for visual aids over complex tables or long text passages. Moreover, VQA datasets are commonly restricted to either born-digital or scanned documents, which limits their ability to measure the robustness to mixed-origin files that one usually finds in real-world applications. In particular, this restriction makes it uncertain whether state-of-the-art performers on website fragments from VisualMRC [87] can be efficient on multi-column layouts and documents with OCR errors or incorrectly-detected reading orders. Finally, a typical dataset for document visual question answering contains documents from a limited period, i.e., a few years (Table 1).

Considering the properties mentioned above, the most diverse dataset to date is Single Page DocVQA (SP-DocVQA) [57], which contains mixed-origin documents of different types created over several decades. However, it is built exclusively on single-page document excerpts and is limited to several domains represented in the Industry Documents Library. As a result, it complements rather than serves as a touchstone for general-purpose DU systems. MP-DocVQA [90] extends this including previous and posterior pages of the documents. However, the questions are kept the same which makes the extra pages mere distractors.

**Questions.** We use VQA as a natural language interface to VRDs, challenging the DU model with diverse questions, advanced operations, and multi-step reasoning to achieve real-world success.

Firstly, we assert that various layouts and visual elements must be comprehended semantically. As such, we introduce

complex questions targeting these document elements, requiring comprehension beyond the document content, such as ‘*how many text columns are there?*’, ‘*does the document contain words with diacritics?*’ or ‘*which page contains the largest table in the document?*’. These Layout-navigating questions bridge the gap between Document Layout Analysis and Question Answering paradigms.

Our unique and detailed compositional questions demand a model that comprehends semantics and generalizes to new questions in a zero-shot setting. For example, >90% of our questions are unique, while we target questions whose answer scope is much more diverse than in previous works.<sup>3</sup> Since neural networks are known to perform poorly at mathematical reasoning and symbolical processing, we provide training and evaluation questions demanding arithmetic and comparison operations on numbers and dates.

Moreover, we feature multi-hop questions that indicate a model’s robustness to sequential reasoning and mimic how humans ask questions. They may be useful in real-world tasks such as ‘*If the checkbox on page 1 section 3a indicates that the company is incorporated, how much yearly revenue did it generate in 2022 (given the table on page 5)?*’

**Answers.** Even though some VQA datasets are deliberately limited to questions of exclusively extractive (SP-DocVQA) or abstractive (VisualMRC) nature, others do not obey such restrictions and include both question types (see Table 1). The dataset we provide includes both abstractive and extractive answers, covering various types such as *textual, numerical, dates, yes/no, lists, or no answer*.

This allows us to cover all possible business use cases and reveal major deficiencies of existing DU systems beyond typical textual answers. For instance, no existing VQA dataset includes not answerable questions and questions answered with a list. In turn, the models considered to date supposedly tend to make unreliable guesses on questions with an answer not entailed by the content [77]. Our dataset is designed to cover answers beyond plain extractive text such as a list of items or even ‘None’.

The ‘None’ answer type demands that the model correctly identifies that the answer cannot be provided, as the question needs to be better formed, e.g., it asks about the value of an empty cell in the table. In addition, list generation problems pose challenges to the model, as (1) more tokens need to be generated, (2) they may be sourced from different places in the document, and (3) OCR reading order may influence the element ordering.

### 3.1. Gathering Documents

A fundamental difficulty in gathering raw source files was ensuring dataset diversity while fulfilling strict licens-

<sup>3</sup>Answer type comparison is included in supplementary materials.ing requirements. Therefore, rather than depending on initial sources of files, e.g., libraries that originally published digitized materials, we resorted to aggregate websites.

The document collection process was manual and assumed formulating queries to [archive.org](https://archive.org) (containing 36M books and texts), [commons.wikimedia.org](https://commons.wikimedia.org) (with 86M media types of various types), and [documentcloud.org](https://documentcloud.org) (with around 5M public documents). The queries consisted of keywords relevant to some category of interest, e.g., the *resume* category of our proposal consists of ‘resume’, ‘cv’, ‘curriculum’, and ‘biography’ keywords). Where necessary, a separate query parameter ensured that the resulting files belonged to the public domain or were released under a permissive license. Information on keywords and the search procedure is distributed as a part of the DUDE dataset.

From the resulting documents, we selected those representing the requested category and visually distinctive from the ones already gathered. Special care was put into removing examples that visibly expose controversial content or may be subject to privacy or legal concerns, despite the declared license. We collected five thousand, typically multi-page, English documents using this methodology.

### 3.2. Annotation Process

The annotation process involved in-house annotators and Amazon Mechanical Turk freelancers. For the latter, there is limited control over the expertise, and where justified, we resorted to limiting task availability depending on the number of completed tasks and historical acceptance rate.<sup>4</sup> The former are five highly qualified people with a Ph.D. in Linguistics. These three annotation scenarios will be referred to as *All MTurkers*, *Best MTurkers*, and *Qualified Linguists*.

We estimate the total cost of annotation involving both *Linguists* and *MTurkers* as \$20,000.

**Phase 1.** We started by providing *All MTurkers* documents described in Section 3.1 in separate batches aimed at collecting abstractive, extractive, and list QA pairs. Each freelancer was asked to propose up to five questions of a particular type, and in the case of extractive ones to provide an evidence bounding box. The exception to this process is the annotation of non-answerable questions previously shown to be particularly challenging [77]. These are predominantly annotated by *Qualified Linguists* and because of their quality promoted without passing through Phases 2-3.

Candidate QA pairs are semi-automatically filtered to exclude annotations that cannot be valid due to the length, use of non-typical character combinations, or type-specific criteria, such as non-list answers for list batches. Additionally, we cluster duplicate and near-duplicate question-answer pairs to ensure dataset diversity and

promote them directly to Phase 3 after a manual review (the same QA pairs provided independently by several annotators indicate their validity).

**Phase 2.** The rest of the annotations promoted from Phase 1 were directed to *All MTurkers*, but this time instead of providing complete QA pairs, they were asked to answer the question from the previous round. Obtained triples of questions and two answer variants (one from each phase) were evaluated using inter-answer ANLS (defined in Section 3.5) promoted to the final dataset if the agreement was  $>0.8$ . Otherwise, QA triples were directed to Phase 3.

**Phase 3.** *Best MTurkers* were provided with document, question, and answer variants to decide the correctness of each answer and optionally overrule both variants if they are not correct. Outliers from decisions in this phase, such as repealing without a judgment on previous answers, were reviewed by *Qualified Linguists* and corrected if needed.

**Optional Phase 4.** Annotations of the test set were reviewed by *Qualified Linguists*. Given data from Phase 3, they corrected questions, answers and created metadata related to diagnostic categories described in Section 3.4.

### 3.3. Dataset Statistics

We conducted a statistical analysis of our dataset and found that the distribution of document length, question length, and answer type was much more diverse than in other datasets in the same domain. We also used the Simpson diversity coefficient [81] for analysis and summarized the results in Table 1. The following are the statistics for the data split:

<table border="1"><thead><tr><th></th><th>train</th><th>val</th><th>test (diagnostic)</th></tr></thead><tbody><tr><td>documents</td><td>3,010</td><td>749</td><td>1,215 (530)</td></tr><tr><td>questions</td><td>23,728</td><td>6,315</td><td>11,448 (2,462)</td></tr></tbody></table>

Table 2: Data split counts.

The number of tokens in the document distribution is much more diverse compared to other datasets, a consequence of the more diverse distribution of pages (see Figure 3). Note some of the documents are more visual than textual (or even visual-only), making the left whisker essentially reach 0 ( $\log_2$ -scaling of  $x$ -axis).

The distribution of the number of tokens in answers is heavy-tailed, to some extent this is also the property of the distribution of number of tokens in questions. Furthermore, 90.9% of questions are unique, and so are 70.7% of answers (taking answer variants into account).

<sup>4</sup>Approval above 97% over at least 5k HITS.<table border="1">
<thead>
<tr>
<th>Dataset</th>
<th>Ours</th>
<th>SP-DocVQA</th>
<th>VisualMRC</th>
<th>InfographicsVQA</th>
<th>TAT-DQA</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6" style="text-align: center;"><i>Dataset-level properties</i></td>
</tr>
<tr>
<td>Sources</td>
<td>Multi</td>
<td>Industry docs</td>
<td>Web pages</td>
<td>Infographics</td>
<td>Finance reports</td>
</tr>
<tr>
<td>Origin</td>
<td>BD, Scan</td>
<td>Mostly scans</td>
<td>BD</td>
<td>BD</td>
<td>BD</td>
</tr>
<tr>
<td>Period</td>
<td>1860-2022</td>
<td>1960-2000</td>
<td>Jan-Mar 2020</td>
<td>not specified</td>
<td>2018-2020</td>
</tr>
<tr>
<td>Documents</td>
<td>5,019</td>
<td>12,767</td>
<td>10,234</td>
<td>5,485</td>
<td>2,758</td>
</tr>
<tr>
<td>Pages (<math>avg \pm std</math>)</td>
<td>5.72<math>\pm</math>6.4</td>
<td>1.0<math>\pm</math>0.0</td>
<td>1.0<math>\pm</math>0.0</td>
<td>1.0<math>\pm</math>0.0</td>
<td>1.11<math>\pm</math>0.32</td>
</tr>
<tr>
<td>Tokens (<math>avg \pm std</math>)</td>
<td>1,831.53<math>\pm</math>2,545.06</td>
<td>183<math>\pm</math>149.96</td>
<td>154.19<math>\pm</math>79.34</td>
<td>287.98<math>\pm</math>214.57</td>
<td>576.99<math>\pm</math>290.12</td>
</tr>
<tr>
<td>Simpson coeff. (ResNet)</td>
<td>0.82</td>
<td>0.76</td>
<td>0.83</td>
<td>0.86</td>
<td>0.73</td>
</tr>
<tr>
<td>Simpson coeff. (Tf-Idf)</td>
<td>0.95</td>
<td>0.93</td>
<td>0.99</td>
<td>0.94</td>
<td>0.15</td>
</tr>
<tr>
<td colspan="6" style="text-align: center;"><i>Question-level properties</i></td>
</tr>
<tr>
<td>Questions</td>
<td>41,541</td>
<td>50,000</td>
<td>30,562</td>
<td>30,035</td>
<td>16,558</td>
</tr>
<tr>
<td>Unique (%)</td>
<td>90.9</td>
<td>72.34</td>
<td>96.26</td>
<td>99.11</td>
<td>95.65</td>
</tr>
<tr>
<td>Length (<math>avg \pm std</math>)</td>
<td>8.65<math>\pm</math>3.35</td>
<td>8.34<math>\pm</math>3.04</td>
<td>9.38<math>\pm</math>4.01</td>
<td>11.57<math>\pm</math>3.71</td>
<td>12.51<math>\pm</math>4.18</td>
</tr>
<tr>
<td>Semantics</td>
<td>All</td>
<td>T, L, F, Ch</td>
<td>T, L, F, Ch</td>
<td>T, L, F, Ch, M</td>
<td>T, L</td>
</tr>
<tr>
<td colspan="6" style="text-align: center;"><i>Answer-level properties</i></td>
</tr>
<tr>
<td>Unique (%)</td>
<td>70.7</td>
<td>64.29</td>
<td>91.82</td>
<td>48.84</td>
<td>77.54</td>
</tr>
<tr>
<td>Length (<math>avg \pm std</math>)</td>
<td>3.35<math>\pm</math>6.1</td>
<td>2.11<math>\pm</math>1.67</td>
<td>8.38<math>\pm</math>6.36</td>
<td>1.66<math>\pm</math>1.43</td>
<td>3.44<math>\pm</math>7.20</td>
</tr>
<tr>
<td>Extractive (%)</td>
<td>42.39</td>
<td>100.0</td>
<td>0.0</td>
<td>71.96</td>
<td>55.72</td>
</tr>
<tr>
<td>Abstractive (%)</td>
<td>38.25</td>
<td>0.0</td>
<td>100.0</td>
<td>24.91</td>
<td>44.28</td>
</tr>
<tr>
<td>List (%)</td>
<td>6.62</td>
<td>0.0</td>
<td>0.0</td>
<td>5.69</td>
<td>0.0</td>
</tr>
<tr>
<td>None</td>
<td>12.74</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
</tbody>
</table>

Table 1: Summary of the existing English document datasets and our challenge. BD stands for born-digital. Layout semantics are abbreviated as (T)able, (L)ist, (F)igure, (Ch)art, and M(ap). Comparison based on Azure Cognitive Services (3.2) OCR.

Figure 2: Distribution of the number of tokens in documents, answers, and questions.

We scrutinized the answer types by aggregating possible answers into classes representing the information they conveyed. The study used heuristics to determine if the answers fit into NER labeling scheme [1] or categories we anticipated, such as *yes/no* and *none*, or did not anticipate, such as *color*. This resulted in 25 different groups of answers, with the *other* answer type being the fourth largest group. Cramer’s V coefficient was used to check for correlations between question types and answer types, and the results indicated that there were few correlations (see Appendix D.1). The expected correlations, such as *none* answers with *not-answerable* questions or *yes/no* answers with *abstractive* questions, were present, but barely any correlation was significant. This suggests it is hard to guess the answer based

on the question solely.

We study relative diversity measure, called Simpson coefficient [110, 81]. To define it, consider a fixed distance function  $d(a_1, a_2)$  defined for pair of documents  $a_1, a_2 \in A$ : the dataset. In our applications, it is the cosine similarity of a document embedding. Further, for an arbitrary number of datasets  $A_1, \dots, A_N$  the diversity of  $A_1$  with respect to  $A_2, \dots, A_N$  is defined as

$$\text{Div}_{A_2, \dots, A_N}(A_1) = 1 - p\left(d(a_{i1}, a_{i2}) < \min_{i=2:N} d(a_{i1}, a_{i2})\right)$$

where  $a_{i1}, a_{i2} \in A_i$ , are randomly selected,  $i = 2 : Ni = 2 : N$ . We report relative diversities of each of the datasets, relative to other datasets in the study, based on two embeddings: visual (ResNet-101 embeddings-based) and se-Figure 3: While other datasets are predominantly single-page only, the number of pages featuring in **DUDE** is more diverse, yet still biased towards shorter documents.

Figure 4: Count of particular diagnostic categories in a subset of 2.5k test set QA pairs annotated in detail to help analyze models’ performance.

semantic (Tf-Idf embeddings-based), in Table 1. The results show that the probability that two random documents from **DUDE** are more similar than each random pair of documents from other datasets is small, meaning that documents in our dataset are well-distributed and diverse.

### 3.4. Diagnostic Subsets

Following previous DU datasets, we gather diagnostic metadata for close to half of the documents and QA pairs in the test set (see Figure 4). These are intended to enable a fine-grained analysis of the models’ performance. The taxonomy used is an extension of the one from earlier works [57, 56, 10], covering **DUDE**-specific questions and enables a more detailed examination of visual artifacts under consideration.

**Question type and perceived complexity.** We distinguish questions perceived as *simple*, i.e., those based on spotting value near a phrase mentioned explicitly as a part

of the question. For example, "Who is the Secretary of the U.S. Department of Commerce?" when the document contains "Penny Pritzker, Secretary, U.S. Department of Commerce." Such could be guessed given an approximate string matching algorithm and does not require much comprehension beyond that. The remaining questions are marked as *hard* with distinguished categories of *hard multi-hop questions*, and *hard meta/layout-navigating questions*.

**Answer evidence.** We provide information on what types of elements have to be comprehended to provide an answer, including *free text*, *handwriting*, *table or list*, and *layout*, i.e., non-tabular spatial understanding of text placement. These follow the ontology established by previous works [57, 56, 10]. In addition, we supply hints on graphical artifacts one needs to consider for particular questions, such as *image/photo*, *plot/chart*, *checkbox*, and *annotation*.

**Required operation.** We distinguish *arithmetic*, *comparison*, *counting*, and *normalization* operations to provide information on the need for performing, respectively, arithmetic operations on extractable data, comparing numerical values or sizes, counting elements or converting data present in the document to another format (e.g., rounding or date format conversion).

**Answer form/shape.** Finally, we provide information on the shallow form of the returned answer, including *date*, *numeric*, and *proper name*.

### 3.5. Evaluation

The evaluation process follows the typical paradigm of separate training, validation, and test splits. We provide both a standalone evaluator and a website<sup>5</sup> [?] to submit test set predictions.

To assess models’ performance, we rely on the ANLS metric introduced by authors of the ST-VQA dataset [8]. Roughly speaking, it is a generalization of accuracy that does not penalize the system for an answer whose similarity to the gold standard measured with normalized Levenshtein similarity is above a specified threshold. Moreover, the metric assumes the presence of multiple, equally valid reference answers. The mentioned properties account for possible OCR errors or different phrasings, such as the same numerical answer represented as *two* and 2 by different annotators.

In practice, production DU systems provide an estimation of confidence in order to triage documents that do not need to be manually reviewed by a human. While the reliability of the automation ability of a DU solution is deemed quintessential for generating business value in practice [11], DU research rarely reports any confidence evalu-

<sup>5</sup>[rcc.cvc.uab.es/?ch=23](http://rcc.cvc.uab.es/?ch=23)ation. Some exceptions are in closely related task domains like scene text recognition [82] and QA [38, 105].

With DUDE, we want to establish calibration evaluation and confidence ranking as a default evaluation methodology in DU, especially since the field is so close to applications.

To this end, we report (next to ANLS) two additional metrics, Expected Calibration Error (ECE) [65, 63, 27], and Area-Under-Risk-Coverage-Curve (AURC) [24, 34].

Calibration requires that the probability a model assigns to its predictions equals their true likelihood of being correct [19, 20, 101].

ECE approximates top-1 calibration error by a weighted average over the accuracy/confidence difference of histogram bins. Particularly in our evaluation setting, we consider a predicted answer correct if its ANLS to the ground truth answer is above a pre-defined threshold ( $\tau=0.5$ ). For consistency, not-answerable and list-answers both have confidence estimated for the answer as a whole (regardless of the number of answers). Following [66], we apply equal-size binning (with 100 bins,  $\mathcal{L}_{pnorm} = 1$ ), avoiding some pathologies of equal-range binning [41, 94].

AURC is a selective classification metric that evaluates how well an estimator prevents silent failures on an *i.i.d* test set. As an aggregate measure of estimator performance (ANLS) and confidence ranking, it provides a more practically useful estimate of overall performance when the estimator can abstain from (low-confidence) decisions and defer to a human for feedback.

By reporting the above metrics, we hope that in future work there will be contributions (e.g., calibration methods for improved forecasting or metrics for better predictive uncertainty evaluation) that concretely target the empirical observations of overconfidence/miscalibration in DU models.

### 3.6. Baselines

**Human performance.** To establish the human baseline, we assign test set questions to *Qualified Linguists*, ensuring none of them will face the same documents as reviewed in Phase 4. The procedure results in an estimation of 74.76 ANLS points (Table 3). At first glance, this result seems low. Still, when analyzing results case by case, it turns out that it’s hard to score much better since the answer format can influence the overall results a lot: *Eagle* vs. *an eagle* (0.625 ANLS), 62% vs. 62 (0.67 ANLS), 1958-04-29 vs. 4-29-58 (0 ANLS), *Clemson University*, *Clemson South Carolina* vs. *Clemson University* (0 ANLS). We achieved the lowest performance (67.58) on the extractive question type, which confirms our hypothesis since the abstractive answers are shorter (mostly numbers, yes/no, or colors).

We analyzed the maximum score achieved by the best-performing model for each diagnostic test category and plotted that against the human performance in Figure 5.

**Reference models.** We assessed a group of models to determine how their performance is influenced by different factors such as (1) their ability to handle textual, layout, and visual elements, (2) whether they were fine-tuned for the task, (3) their size in (trainable parameters), and (4) the maximum input length they can handle.

To analyze factors (1) and (2), we conducted a zero-shot evaluation of several baseline text-only models. We used three encoder-based models (BERT [21], Longformer [6], and BigBird [102]) that cannot generate text and three that feature a decoder (T5 [75], GPT-3-Davinci [12], and ChatGPT) and have this capability. Next, we extended the T5 architecture with 2D layout embeddings [10, 72] and fine-tuned models with increasing maximum sequence lengths (512  $\rightarrow$  8192) on DUDE. Finally, we evaluated our replication of the hierarchical Hi-VT5 model [90], as this model has the ability to decode text, understand multi-page layouts, and comprehend visual page features using DiT [47].

Regarding factors (2) and (3), we evaluated models of various sizes ranging from 131M (BigBird) to 175B (GPT-3-Davinci) and varied the input context from 512 (BERT) to 20480 (Hi-VT5) tokens. Overall, we thoroughly evaluated multiple models in the different testing setups to determine their performance under various conditions, as seen in Table 3.

### 3.7. Analysis & Discussion

To summarize, our study reveals that existing advanced language models such as BERT, Longformer, and BigBird struggle with comprehending visual elements and document layouts. To address this issue, we introduced T5, T5-2D, and Hi-VT5 models that incorporate layout and visual information. Still, their performance remains unsatisfactory, as evidenced by the comparison with the human baseline, similar to what has been reported for InfographicsVQA. This indicates that there is still scope for enhancing the visual understanding of DUDE models. Moreover, our findings indicate that a large LLM capable of processing long inputs alone is insufficient for achieving strong performance in DUDE, especially for the extractive type of answer. Finally, the dataset’s length significantly affects the models’ scores, as seen by the increase in scores by 4.4 – 5.0 points when the T5 and T5+2D context length is extended from 512 to 8192. Similarly, the model size has a positive correlation with the final score, but it holds only within a particular model-type and is not the main factor influencing the results. State-of-the-art performance of 46.04 ANLS<sub>all</sub> was achieved on  $T5_{large}$  with a 2D layout understanding that consumed 8192 tokens, confirming the observation above.

## 4. Conclusion

In conclusion, this paper introduces a new large-scale multi-paged, multi-domain, multi-industry Document Vi-Figure 5: We report the average ANLS for the human expert vs. the best-performing model per diagnostic category as a ceiling analysis.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Init.</th>
<th>Params</th>
<th>Max Seq. Length</th>
<th>Test Setup</th>
<th>ANLS<sub>all</sub> ↑</th>
<th>ECE<sub>all</sub> ↓</th>
<th>AURC<sub>all</sub> ↓</th>
<th>ANLS<sub>do</sub></th>
<th>ANLS<sub>do</sub> Abs</th>
<th>ANLS<sub>do</sub> Ex</th>
<th>ANLS<sub>do</sub> NA</th>
<th>ANLS<sub>do</sub> Li</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="13"><i>text-only</i> Encoder-based models</td>
</tr>
<tr>
<td>Big Bird</td>
<td>MPDocVQA</td>
<td>131M</td>
<td>4096</td>
<td>Concat*</td>
<td>26.27</td>
<td>30.14</td>
<td>44.22</td>
<td>30.67</td>
<td>7.11</td>
<td>40.26</td>
<td>12.75</td>
<td>8.46</td>
</tr>
<tr>
<td>BERT-Large</td>
<td>MPDocVQA</td>
<td>334M</td>
<td>512</td>
<td>Max Conf.*</td>
<td>25.48</td>
<td>34.06</td>
<td>48.60</td>
<td>32.18</td>
<td>7.28</td>
<td>42.23</td>
<td>5.88</td>
<td>11.13</td>
</tr>
<tr>
<td>Longformer</td>
<td>MPDocVQA</td>
<td>148M</td>
<td>4096</td>
<td>Concat*</td>
<td>27.14</td>
<td>27.59</td>
<td>44.59</td>
<td>33.45</td>
<td>8.55</td>
<td>43.58</td>
<td>10.78</td>
<td>10.62</td>
</tr>
<tr>
<td colspan="13"><i>text-only</i> Encoder-Decoder based models</td>
</tr>
<tr>
<td>T5</td>
<td>base</td>
<td>223M</td>
<td>512</td>
<td>Concat-0*</td>
<td>19.65</td>
<td>19.14</td>
<td>48.83</td>
<td>25.62</td>
<td>5.24</td>
<td>33.91</td>
<td>0</td>
<td>7.31</td>
</tr>
<tr>
<td>T5</td>
<td>MPDocVQA</td>
<td>223M</td>
<td>512</td>
<td>Max Conf.*</td>
<td>29.48</td>
<td>27.18</td>
<td>43.06</td>
<td>37.56</td>
<td>21.19</td>
<td>44.22</td>
<td>0</td>
<td>10.56</td>
</tr>
<tr>
<td>T5</td>
<td>base</td>
<td>223M</td>
<td>512</td>
<td>Concat+FT</td>
<td>37.41</td>
<td><b>10.82</b></td>
<td>41.09</td>
<td>40.61</td>
<td>42.61</td>
<td>48.20</td>
<td>53.92</td>
<td>16.87</td>
</tr>
<tr>
<td>T5</td>
<td>base</td>
<td>223M</td>
<td>8192</td>
<td>Concat+FT</td>
<td>41.80</td>
<td>17.33</td>
<td>49.53</td>
<td>44.95</td>
<td>47.62</td>
<td>50.49</td>
<td>63.72</td>
<td>7.56</td>
</tr>
<tr>
<td colspan="13"><i>text-only</i> Large Language models (LLM)</td>
</tr>
<tr>
<td>ChatGPT</td>
<td>gpt-3.5-turbo</td>
<td>20B</td>
<td>4096</td>
<td>Concat-0</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>35.07</td>
<td>16.73</td>
<td>42.52</td>
<td>70.59</td>
<td>15.97</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Concat-4</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>41.89</td>
<td>22.19</td>
<td>49.90</td>
<td><b>77.45</b></td>
<td>17.74</td>
</tr>
<tr>
<td>GPT3</td>
<td>davinci3</td>
<td>175B</td>
<td>4000</td>
<td>Concat-0</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>43.95</td>
<td>18.16</td>
<td>54.44</td>
<td>73.53</td>
<td>36.32</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Concat-4</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>47.04</td>
<td>22.37</td>
<td><b>57.09</b></td>
<td>63.73</td>
<td><b>40.01</b></td>
</tr>
<tr>
<td colspan="13"><i>text+layout</i> Encoder-Decoder based models</td>
</tr>
<tr>
<td>T5-2D</td>
<td>base</td>
<td>223M</td>
<td>512</td>
<td>Concat+FT</td>
<td>37.10</td>
<td>10.85</td>
<td>41.46</td>
<td>40.50</td>
<td>42.48</td>
<td>48.62</td>
<td>52.94</td>
<td>3.49</td>
</tr>
<tr>
<td>T5-2D</td>
<td>base</td>
<td>223M</td>
<td>8192</td>
<td>Concat+FT</td>
<td>42.10</td>
<td>17.00</td>
<td>48.83</td>
<td>45.73</td>
<td>48.37</td>
<td>52.29</td>
<td>63.72</td>
<td>8.02</td>
</tr>
<tr>
<td>T5-2D</td>
<td>large</td>
<td>770M</td>
<td>8192</td>
<td>Concat+FT</td>
<td><b>46.06</b></td>
<td>14.40</td>
<td><b>35.70</b></td>
<td><b>48.14</b></td>
<td><b>50.81</b></td>
<td>55.65</td>
<td>68.62</td>
<td>5.43</td>
</tr>
<tr>
<td colspan="13"><i>text+layout+vision</i> models</td>
</tr>
<tr>
<td>HiVT5</td>
<td></td>
<td>316M</td>
<td>20480</td>
<td>Hierarchical+FT</td>
<td>23.06</td>
<td>11.91</td>
<td>54.35</td>
<td>22.33</td>
<td>33.94</td>
<td>17.60</td>
<td>61.76</td>
<td>6.83</td>
</tr>
<tr>
<td>LayoutLMv3</td>
<td>MPDocVQA</td>
<td>125M</td>
<td>512</td>
<td>Max Conf.*</td>
<td>20.31</td>
<td>34.97</td>
<td>47.51</td>
<td>25.27</td>
<td>8.10</td>
<td>32.60</td>
<td>8.82</td>
<td>7.82</td>
</tr>
<tr>
<td colspan="8"><i>Human baseline</i></td>
<td>74.76</td>
<td>81.95</td>
<td>67.58</td>
<td>83.33</td>
<td>67.74</td>
</tr>
</tbody>
</table>

Table 3: Summary of Baseline performance on the **DUDE** test set (*all*) and diagnostic subset (*do*). Test setups are defined as *Max Conf.*: predict one answer per page and return an answer with the highest probability over all pages, *Concat*: predict on tokens truncated to maximum sequence length, *FT* stands for fine-tuning on **DUDE** training data, and *-0* refers to zero-shot and *-4* few-shot inference. Average ANLS results per question type are abbreviated as (Abs)tractive, (Ex)tractive, (N)ot-(A)nswerable, (Li)st. (\*) We report only results for best performing test setup (either *Max Conf.* or *Concat*). All scalars are scaled between 0 and 100 for readability.

sual Question Answering Benchmark for document understanding. Our dataset is adjusted to the real-world environment where we need to process long documents and understand different types of documents. The benchmark includes visual semantics such as *tables*, *charts*, *figures*, *lists*, *checkboxes*, *stamps*, and more, which are essential for real-world document understanding. The performance of state-of-the-art textual and multi-modal models still lags behind human performance, indicating the need for further improvement in visual understanding for DU models. Nevertheless, we believe evaluating systems on **DUDE** could inspire new architectures and methods.

**Limitations.** As our approach is closer to real-world industrial applications, and enables models to recognize and understand new unseen data without the need for re-training, it does come with some limitations and constraining factors, including the use of only English language documents. Future work could address these limitations and expand the benchmark to include other languages. Moreover, although our dataset can be considered large-scale, it still represents a relatively small sample size of the plethora of documents that exist in the real world.## References

- [1] SpaCy en\_core\_web\_lg label scheme. <https://spacy.io/models/en>. Accessed: 2023-03-08. 5
- [2] Rajas Agashe, Srinivasan Iyer, and Luke Zettlemoyer. JulCe: A large scale distantly supervised dataset for open domain context-based code generation. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 5436–5446, Hong Kong, China, Nov. 2019. Association for Computational Linguistics. 2
- [3] Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol, Margaret Mitchell, C. Lawrence Zitnick, Dhruv Batra, and Devi Parikh. Vqa: Visual question answering, 2015. 2
- [4] Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms, 2019. 2
- [5] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R Manmatha. Docformer: End-to-end transformer for document understanding. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 993–1003, 2021. 2
- [6] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*, 2020. 7, 1
- [7] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluís Gómez, Marçal Rusinol, Minesh Mathew, CV Jawahar, Ernest Valveny, and Dimosthenis Karatzas. Icdar 2019 competition on scene text visual question answering. In *2019 International Conference on Document Analysis and Recognition (ICDAR)*, pages 1563–1570. IEEE, 2019. 2
- [8] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluís Gómez, Marçal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In *Proceedings of the IEEE/CVF international conference on computer vision*, 2019. 2, 6, 3
- [9] Johannes Bjerva, Nikita Bhutani, Behzad Golshan, Wang-Chiew Tan, and Isabelle Augenstein. SubjQA: A Dataset for Subjectivity and Review Comprehension. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 5480–5494, Online, Nov. 2020. Association for Computational Linguistics. 2
- [10] Łukasz Borchmann, Michał Pietruszka, Tomasz Stanisławek, Dawid Jurkiewicz, Michał Turski, Karolina Szyndler, and Filip Galiński. Due: End-to-end document understanding benchmark. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021. 6, 7
- [11] Pascal Bornet, Ian Barkin, and Jochen Wirtz. *Intelligent automation: Welcome to the world of hyperautomation: learn how to harness artificial intelligence to boost business & make our world more human*. World Scientific, 2021. 6
- [12] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020. 7
- [13] Santiago Castro, Mahmoud Azab, Jonathan Stroud, Cristina Noujaim, Ruoyao Wang, Jia Deng, and Rada Mihalcea. LifeQA: A real-life dataset for video question answering. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 4352–4358, Marseille, France, May 2020. European Language Resources Association. 2
- [14] Santiago Castro, Naihao Deng, Pingxuan Huang, Mihai Burzo, and Rada Mihalcea. In-the-wild video question answering. In *Proceedings of the 29th International Conference on Computational Linguistics*, pages 5613–5635, Gyeongju, Republic of Korea, Oct. 2022. International Committee on Computational Linguistics. 2
- [15] Shuaichen Chang, David Palzer, Jialin Li, Eric Fosler-Lussier, and Ningchuan Xiao. Mapqa: A dataset for question answering on choropleth maps, 2022. 2
- [16] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric Xing, and Liang Lin. GeoQA: A geometric question answering benchmark towards multimodal numerical reasoning. In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 513–523, Online, Aug. 2021. Association for Computational Linguistics. 2
- [17] Anthony Colas, Seokhwan Kim, Franck Dernoncourt, Siddhesh Gupte, Zhe Wang, and Doo Soon Kim. TutorialVQA: Question answering dataset for tutorial videos. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 5450–5455, Marseille, France, May 2020. European Language Resources Association. 2
- [18] Pradeep Dasigi, Nelson F Liu, Ana Marasović, Noah A Smith, and Matt Gardner. Quoref: A reading comprehension dataset with questions requiring coreferential reasoning. *arXiv preprint arXiv:1908.05803*, 2019. 2
- [19] A Philip Dawid. The well-calibrated bayesian. *Journal of the American Statistical Association*, 77(379):605–610, 1982. 7
- [20] Morris H DeGroot and Stephen E Fienberg. The comparison and evaluation of forecasters. *Journal of the Royal Statistical Society: Series D (The Statistician)*, 32(1-2):12–22, 1983. 7
- [21] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, 2019. 7
- [22] Ritam Dutt, Kasturi Bhattacharjee, Rashmi Gangadharaiah, Dan Roth, and Carolyn Rose. PerKGQA: Question answering over personalized knowledge graphs. In *Findings of the Association for Computational Linguistics: NAACL 2022*, pages 253–268, Seattle, United States, July 2022. Association for Computational Linguistics. 2
- [23] Łukasz Garncarek, Rafał Powalski, Tomasz Stanisławek, Bartosz Topolski, Piotr Halama, Michał Turski, and FilipGraliński. Lambert: Layout-aware language modeling using bert for information extraction. In *ICDAR*, 2021. 2

[24] Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. *Advances in neural information processing systems*, 30, 2017. 7, 4

[25] Jiuxiang Gu, Jason Kuen, Vlad I Morariu, Handong Zhao, Rajiv Jain, Nikolaos Barmpalios, Ani Nenkova, and Tong Sun. Unidoc: Unified pretraining framework for document understanding. *Advances in Neural Information Processing Systems*, 34:39–50, 2021. 2

[26] Lin Gui, Jiannan Hu, Yulan He, Ruifeng Xu, Qin Lu, and Jiachen Du. A question answering approach for emotion cause extraction. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pages 1593–1602, Copenhagen, Denmark, Sept. 2017. Association for Computational Linguistics. 2

[27] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning - Volume 70*, ICML’17, page 1321–1330, 2017. 7

[28] Deepak Gupta and Dina Demner-Fushman. Overview of the MedVidQA 2022 shared task on medical video question-answering. In *Proceedings of the 21st Workshop on Biomedical Language Processing*, pages 264–274, Dublin, Ireland, May 2022. Association for Computational Linguistics. 2

[29] Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people, 2018. 2

[30] Adam W Harley, Alex Ufkes, and Konstantinos G Derpanis. Evaluation of deep convolutional nets for document image classification and retrieval. In *2015 13th International Conference on Document Analysis and Recognition (ICDAR)*, pages 991–995. IEEE, 2015. 2

[31] Mark Hopkins, Ronan Le Bras, Cristian Petrescu-Prahova, Gabriel Stanovsky, Hannaneh Hajishirzi, and Rik Koncel-Kedziorski. SemEval-2019 task 10: Math question answering. In *Proceedings of the 13th International Workshop on Semantic Evaluation*, pages 893–899, Minneapolis, Minnesota, USA, June 2019. Association for Computational Linguistics. 2

[32] Xuming Hu, Zhijiang Guo, GuanYu Wu, Aiwei Liu, Lijie Wen, and Philip Yu. CHEF: A pilot Chinese dataset for evidence-based fact-checking. In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 3362–3376, Seattle, United States, July 2022. Association for Computational Linguistics. 2

[33] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. *arXiv preprint arXiv:2204.08387*, 2022. 2

[34] Paul F Jaeger, Carsten Tim Lüth, Lukas Klein, and Till J. Bungert. A call to reflect on evaluation practices for failure detection in image classification. In *International Conference on Learning Representations*, 2023. 7, 1, 4

[35] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. Funsd: A dataset for form understanding in noisy scanned documents. In *2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)*, volume 2, pages 1–6. IEEE, 2019. 2

[36] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. PubMedQA: A dataset for biomedical research question answering. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 2567–2577, Hong Kong, China, Nov. 2019. Association for Computational Linguistics. 2

[37] Endri Kacupaj, Joan Plepi, Kuldeep Singh, Harsh Thakkar, Jens Lehmann, and Maria Maleshkova. Conversational question answering over knowledge graphs with transformer and graph attention networks. In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, pages 850–862, Online, Apr. 2021. Association for Computational Linguistics. 2

[38] Amita Kamath, Robin Jia, and Percy Liang. Selective question answering under domain shift. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 5684–5696, 2020. 7

[39] Sanjay Kamath, Brigitte Grau, and Yue Ma. Verification of the Expected Answer Type for Biomedical Question Answering. In *First International Workshop on Hybrid Question Answering with Structured and Unstructured Knowledge (HQA’18)*, WWW ’18 Companion Proceedings of the The Web Conference 2018, pages 1093–1097, Lyon, France, Apr. 2018. ACM Press. 2

[40] Andreas Kirsch. Player of jeopardy: Chatgpt evaluation, 2023. 3

[41] Ananya Kumar, Percy Liang, and Tengyu Ma. Verified uncertainty calibration. In *Advances in Neural Information Processing Systems*, 2019. 7, 4

[42] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 2019. 2

[43] Egor Lakomkin, Sven Magg, Cornelius Weber, and Stefan Wermter. KT-speech-crawler: Automatic dataset construction for speech recognition from YouTube videos. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 90–95, Brussels, Belgium, Nov. 2018. Association for Computational Linguistics. 2

[44] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara Berg. TVQA: Localized, compositional video question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 1369–1379, Brussels, Belgium, Oct.-Nov. 2018. Association for Computational Linguistics. 2- [45] Vladimir I Levenshtein et al. Binary codes capable of correcting deletions, insertions, and reversals. In *Soviet physics doklady*, volume 10, pages 707–710. Soviet Union, 1966. 3
- [46] Haonan Li, Martin Tomko, Maria Vasardani, and Timothy Baldwin. Multispanqa: A dataset for multi-span question answering. In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 1250–1260, 2022. 2
- [47] Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, and Furu Wei. Dit: Self-supervised pre-training for document image transformer. In *Proceedings of the 30th ACM International Conference on Multimedia*, pages 3530–3539, 2022. 7
- [48] Jing Li, Shangping Zhong, and Kaizhi Chen. MLEC-QA: A Chinese Multi-Choice Biomedical Question Answering Dataset. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 8862–8874, Online and Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. 2
- [49] Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. Docbank: A benchmark dataset for document layout analysis, 2020. 2
- [50] Peizhao Li, Jiuxiang Gu, Jason Kuen, Vlad I Morariu, Handong Zhao, Rajiv Jain, Varun Manjunatha, and Hongfu Liu. Selfdoc: Self-supervised document representation learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5652–5660, 2021. 2
- [51] Chexiao Liu and Xiaojun Wan. CodeQA: A question answering dataset for source code comprehension. In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pages 2618–2632, Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. 2
- [52] Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. In Christian Bessiere, editor, *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20*, pages 3622–3628. International Joint Conferences on Artificial Intelligence Organization, 7 2020. Main track. 2
- [53] Jiahua Liu, Yankai Lin, Zhiyuan Liu, and Maosong Sun. XQA: A cross-lingual open-domain question answering dataset. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 2358–2368, Florence, Italy, July 2019. Association for Computational Linguistics. 2
- [54] Shayne Longpre, Yi Lu, and Joachim Daiber. Mkqa: A linguistically diverse benchmark for multilingual open domain question answering, 2020. 2
- [55] Ibrahim Souleiman Mahamoud, Mickaël Coustaty, Aurélie Joseph, Vincent Poulain d’Andecy, and Jean-Marc Ogier. Qalayout: Question answering layout based on multimodal attention for visual question answering on corporate document. In Seiichi Uchida, Elisa Barney, and Véronique Eglin, editors, *Document Analysis Systems*, pages 659–673, Cham, 2022. Springer International Publishing. 3
- [56] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 1697–1706, 2022. 2, 3, 6
- [57] Minesh Mathew, Ruben Tito, Dimosthenis Karatzas, R Manmatha, and CV Jawahar. Document visual question answering challenge 2020. *arXiv preprint arXiv:2008.08899*, 2020. 2, 3, 6
- [58] Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. Ambigqa: Answering ambiguous open-domain questions, 2020. 2
- [59] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In *2019 international conference on document analysis and recognition (ICDAR)*, pages 947–952. IEEE, 2019. 2
- [60] Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. NumGLUE: A suite of fundamental yet challenging mathematical reasoning tasks. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3505–3523, Dublin, Ireland, May 2022. Association for Computational Linguistics. 2
- [61] Timo Möller, Anthony Reina, Raghavan Jayakumar, and Malte Pietsch. COVID-QA: A question answering dataset for COVID-19. In *Proceedings of the 1st Workshop on NLP for COVID-19 at ACL 2020*, Online, July 2020. Association for Computational Linguistics. 2
- [62] Muhammad Akhtar Munir, Muhammad Haris Khan, M Saquib Sarfraz, and Mohsen Ali. Towards improving calibration in object detection under domain shift. In *Advances in Neural Information Processing Systems*, 2022. 3
- [63] Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using Bayesian binning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 29, 2015. 7
- [64] Anastasios Nentidis, Georgios Katsimpras, Eirini Vandorou, Anastasia Krithara, Antonio Miranda-Escalada, Luis Gasco, Martin Krallinger, and Georgios Paliouras. Overview of BioASQ 2022: The tenth BioASQ challenge on large-scale biomedical semantic indexing and question answering. In *Lecture Notes in Computer Science*, pages 337–361. Springer International Publishing, 2022. 2
- [65] Alexandru Niculescu-Mizil and Rich Caruana. Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine learning*, pages 625–632, 2005. 7
- [66] Jeremy Nixon, Michael W Dusenberry, Linchuan Zhang, Ghassen Jerfel, and Dustin Tran. Measuring calibration in deep learning. In *CVPR Workshops*, volume 2, 2019. 7, 4
- [67] Dimitris Pappas, Petros Stavropoulos, Ion Androutsopoulos, and Ryan McDonald. BioMRC: A dataset for biomedical machine reading comprehension. In *Proceedings of the 19th SIGBioMed Workshop on Biomedical Language Processing*, pages 140–149, Online, July 2020. Association for Computational Linguistics. 2[68] Jae Sung Park, Chandra Bhagavatula, Roozbeh Mottaghi, Ali Farhadi, and Yejin Choi. Visualcomet: Reasoning about the dynamic context of a still image, 2020. [2](#)

[69] Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. *arXiv preprint arXiv:1508.00305*, 2015. [2](#)

[70] Tzuf Paz-Argaman and Reut Tsarfaty. RUN through the streets: A new dataset and baseline models for realistic urban navigation. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 6449–6455, Hong Kong, China, Nov. 2019. Association for Computational Linguistics. [2](#)

[71] Michał Pietruszka, Michał Turski, Łukasz Borchmann, Tomasz Dwojak, Gabriela Palka, Karolina Szyndler, Dawid Jurkiewicz, and Łukasz Garncarek. Stable: Table generation framework for encoder-decoder models, 2022. [2](#)

[72] Rafał Powalski, Łukasz Borchmann, Dawid Jurkiewicz, Tomasz Dwojak, Michał Pietruszka, and Gabriela Palka. Going full-tilt boogie on document understanding with text-image-layout transformer. In *ICDAR*, 2021. [2](#), [7](#)

[73] Le Qi, Shangwen Lv, Hongyu Li, Jing Liu, Yu Zhang, Qiaojiao She, Hua Wu, Haifeng Wang, and Ting Liu. DuReader<sub>vis</sub>: A Chinese dataset for open-domain document visual question answering. In *Findings of the Association for Computational Linguistics: ACL 2022*, pages 1338–1351, Dublin, Ireland, May 2022. Association for Computational Linguistics. [2](#)

[74] Yiwei Qin, Weizhe Yuan, Graham Neubig, and Pengfei Liu. T5score: Discriminative fine-tuning of generative evaluation metrics. *arXiv preprint arXiv:2212.05726*, 2022. [3](#)

[75] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. *J. Mach. Learn. Res.*, 21(140):1–67, 2020. [7](#)

[76] Preethi Raghavan, Jennifer J Liang, Diwakar Mahajan, Rachita Chandra, and Peter Szolovits. emrKBQA: A clinical knowledge-base question answering dataset. In *Proceedings of the 20th Workshop on Biomedical Language Processing*, pages 64–73, Online, June 2021. Association for Computational Linguistics. [2](#)

[77] Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions for squad. *arXiv preprint arXiv:1806.03822*, 2018. [2](#), [3](#), [4](#)

[78] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. *arXiv preprint arXiv:1606.05250*, 2016. [2](#)

[79] Rebecca Roelofs, Nicholas Cain, Jonathon Shlens, and Michael C Mozer. Mitigating bias in calibration error estimation. In *International Conference on Artificial Intelligence and Statistics*, pages 4036–4054. PMLR, 2022. [4](#)

[80] Apoorv Saxena, Soumen Chakrabarti, and Partha Talukdar. Question answering over temporal knowledge graphs. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 6663–6676, Online, Aug. 2021. Association for Computational Linguistics. [2](#)

[81] E. H. SIMPSON. Measurement of diversity. *Nature*, 163(4148):688–688, apr 1949. [4](#), [5](#)

[82] Ron Slossberg, Oron Anschel, Amir Markovitz, Ron Litman, Aviad Aberdam, Shahar Tsiper, Shai Mazor, Jon Wu, and R Manmatha. On calibration of scene-text recognition models. *arXiv preprint arXiv:2012.12643*, 2020. [7](#)

[83] Brandon Smock, Rohith Pesala, and Robin Abraham. Pubtables-1m: Towards comprehensive table extraction from unstructured documents. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 4634–4642, 2022. [2](#)

[84] Tarcísio Souza Costa, Simon Gottschalk, and Elena Demidova. Event-qa: A dataset for event-centric question answering over knowledge graphs. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management, CIKM ’20*, page 3157–3164, New York, NY, USA, 2020. Association for Computing Machinery. [2](#)

[85] Tomasz Stanislawek, Filip Gralinski, Anna Wróblewska, Dawid Lipinski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, and Przemysław Biecek. Kleister: Key information extraction datasets involving long documents with complex layouts. In *ICDAR*, volume 12821 of *Lecture Notes in Computer Science*, pages 564–579. Springer, 2021. [2](#)

[86] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: A dataset for document visual question answering on multiple images, 2023. [2](#)

[87] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. In *AAAI*, 2021. [2](#), [3](#)

[88] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a large-scale dataset for fact extraction and VERification. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pages 809–819, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. [2](#)

[89] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. Document collection visual question answering. In *Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part II 16*, pages 778–792. Springer, 2021. [2](#), [3](#)

[90] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multi-page docvqa. *arXiv preprint arXiv:2212.05935*, 2022. [2](#), [3](#), [7](#), [1](#)

[91] Rubèn Tito, Minesh Mathew, CV Jawahar, Ernest Valveny, and Dimosthenis Karatzas. Icdar 2021 competition on document visual question answering. In *International Conference on Document Analysis and Recognition*, pages 635–649. Springer, 2021. [2](#)[92] Adam Trischler, Tong Wang, Xingdi Yuan, Justin Harris, Alessandro Sordoni, Philip Bachman, and Kaheer Suleman. Newsqa: A machine comprehension dataset. *arXiv preprint arXiv:1611.09830*, 2016. 2

[93] Priyansh Trivedi, Gaurav Maheshwari, Mohnish Dubey, and Jens Lehmann. Lc-quad: A corpus for complex question answering over knowledge graphs. In *International Semantic Web Conference*, pages 210–218. Springer, 2017. 2

[94] Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, and Thomas Schön. Evaluating model calibration in classification. In *The 22nd International Conference on Artificial Intelligence and Statistics*, pages 3459–3467. PMLR, 2019. 7, 4

[95] Linyi Yang, Zhen Wang, Yuxiang Wu, Jie Yang, and Yue Zhang. Towards fine-grained causal reasoning and qa, 2022. 2

[96] Yuzhe Yang, Hao Wang, and Dina Katabi. On multi-domain long-tailed recognition, generalization and beyond. *arXiv preprint arXiv:2203.09513*, 2022. 2

[97] Yi Yang, Wen-tau Yih, and Christopher Meek. WikiQA: A challenge dataset for open-domain question answering. In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pages 2013–2018, Lisbon, Portugal, Sept. 2015. Association for Computational Linguistics. 2

[98] Yuya Yoshikawa, Yutaro Shigeto, and Akikazu Takeuchi. STAIR captions: Constructing a large-scale Japanese image caption dataset. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 417–421, Vancouver, Canada, July 2017. Association for Computational Linguistics. 2

[99] Chenyu You, Nuo Chen, Fenglin Liu, Shen Ge, Xian Wu, and Yuexian Zou. End-to-end spoken conversational question answering: Task, dataset and model. In *Findings of the Association for Computational Linguistics: NAACL 2022*, pages 1219–1232, Seattle, United States, July 2022. Association for Computational Linguistics. 2

[100] Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. Reclor: A reading comprehension dataset requiring logical reasoning. In *International Conference on Learning Representations (ICLR)*, April 2020. 2

[101] Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In *Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pages 694–699, 2002. 7

[102] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. *Advances in Neural Information Processing Systems*, 33:17283–17297, 2020. 7, 1

[103] Majid Zarharan, Mahsa Ghaderan, Amin Pourtabiri, Zahra Sayedi, Behrouz Minaei-Bidgoli, Sauleh Eetemadi, and Mohammad Taher Pilehvar. ParsFEVER: a dataset for Farsi fact extraction and verification. In *Proceedings of \*SEM 2021: The Tenth Joint Conference on Lexical and Computational Semantics*, pages 99–104, Online, Aug. 2021. Association for Computational Linguistics. 2

[104] Qiyuan Zhang, Lei Wang, Sicheng Yu, Shuohang Wang, Yang Wang, Jing Jiang, and Ee-Peng Lim. NOAHQA: Numerical reasoning with interpretable graph question answering dataset. In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pages 4147–4161, Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. 2

[105] Shujian Zhang, Chengyue Gong, and Eunsol Choi. Knowing more about questions can help: Improving calibration in question answering. *arXiv preprint arXiv:2106.01494*, 2021. 7

[106] Xinbo Zhang, Changzhi Sun, Yue Zhang, Lei Li, and Hao Zhou. NAIL: A challenging benchmark for naïve logical reasoning, 2022. 2

[107] Xinyi Zheng, Douglas Burdick, Lucian Popa, Xu Zhong, and Nancy Xin Ru Wang. Global table extractor (gte): A framework for joint table identification and cell structure recognition using visual context. In *Proceedings of the IEEE/CVF winter conference on applications of computer vision*, pages 697–706, 2021. 2

[108] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: data, model, and evaluation. In *Computer Vision—ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXI 16*, pages 564–580. Springer, 2020. 2

[109] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: largest dataset ever for document layout analysis. In *2019 International Conference on Document Analysis and Recognition (ICDAR)*, pages 1015–1022. IEEE, 2019. 2

[110] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(6):1452–1464, June 2018. 5

[111] Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng Chua. Towards complex document understanding by discrete reasoning. In *Proceedings of the 30th ACM International Conference on Multimedia*, pages 4857–4866, 2022. 2, 3# Supplementary Materials

<table><tr><td><b>A Detailed Results Analysis</b></td><td><b>1</b></td></tr><tr><td>    A.1. Within Model Class Analysis</td><td>1</td></tr><tr><td>        A.1.1 Encoder vs. Decoder</td><td>1</td></tr><tr><td>        A.1.2 Incorporating Layout &amp; Vision</td><td>1</td></tr><tr><td>        A.1.3 Toward Long Document Processing</td><td>1</td></tr><tr><td>        A.1.4 Diagnosis of LLM Results</td><td>1</td></tr><tr><td>    A.2 Assessing Confidence</td><td>1</td></tr><tr><td><b>B Baseline Experiments Setup</b></td><td><b>2</b></td></tr><tr><td>    B.1. Hyperparameter Defaults</td><td>2</td></tr><tr><td>    B.2. Generative LLM Prompt Fine-tuning</td><td>2</td></tr><tr><td>    B.3. Confidence Estimation</td><td>3</td></tr><tr><td>    B.4. Evaluation metrics</td><td>3</td></tr><tr><td>        B.4.1 ANLS</td><td>3</td></tr><tr><td>        B.4.2 ECE</td><td>3</td></tr><tr><td>        B.4.3 AURC</td><td>4</td></tr><tr><td><b>C Qualitative Examples</b></td><td><b>4</b></td></tr><tr><td><b>D Additional Dataset Statistics</b></td><td><b>7</b></td></tr><tr><td>    D.1. Answer Types</td><td>7</td></tr><tr><td>    D.2 Dataset Diversity</td><td>7</td></tr></table>## A. Detailed Results Analysis

### A.1. Within Model Class Analysis

#### A.1.1 Encoder vs. Decoder

A key difference between encoder-only and (encoder-) decoder-based models is the ability to generate answers beyond the explicit document textual content. This is clearly reflected in the results for BigBird, Longformer, BERT, and LayoutLMv3, which score  $< 10$  ANLS% on abstractive questions, whereas they have just average scores for extractive questions. On **DUDE**, we can claim that a generative model is necessary given all considered question types.

Quite remarkably, while the human baseline demonstrates that humans find abstractive questions (ANLS  $\pm 82\%$ ) easier than extractive questions (ANLS  $\pm 68\%$ ), the reverse is true for all machine baselines. A potential confounder for these results could be the difference in output formatting for extractive vs. abstractive answers, which is hard to take into account with ANLS evaluation.

#### A.1.2 Incorporating Layout & Vision

When comparing T5 with and without 2D position embeddings on the diagnostic categories, we find the highest improvements on ‘evidence table or list’, ‘complexity simple’, and ‘evidence plain’.

Our study with the proposed baselines shows that questions requiring visual evidence to be answered are an important future challenge for the vision community. To get further insights into models’ performance on these questions, we calculate a weighted average of ANLS over visual categories. This reveals that GPT3 (4-shot) and T5-2d-large-8K obtain a tied score of (ANLS=37%), even though they only have access to the text. The human performance, on the other hand, is close to double (ANLS=72%), thus showing the need for better integration of the visual modality in DU models.

#### A.1.3 Toward Long Document Processing

**DUDE** clearly requires methods that can process long sequences, as evidenced by its average document length of  $1832 \pm 2545$  tokens. This is particularly evident when comparing standard NLP QA methods like BERT-concat, which underperforms Longformer [6] and BigBird [102], despite being the *large* version. Experiments with T5 and T5-2D further support this claim, as extending the sequence length from 512 to 8192 leads to a  $\sim 5\%$  ANLS improvement.

The exception is HiVT5 [90], which performs worse than the rest of the methods. This is due to the authors of HiVT5 performing a pre-training task of text denoising that helped to better model the [PAGE] tokens. This resulted in a better, compressed representation of the relevant

information within a document conditioned by a question. Moreover, the authors also did extensive experimentation and found that 10 [PAGE] tokens per page were the best fit for the MP-DocVQA [90] dataset. We used similar hyperparameters, but **DUDE** might require better fine-tuning of [PAGE] tokens since the images are more visually rich with colored graphics and layouts. The hierarchical processing of documents with a meaningful visual component is a promising avenue for future research.

#### A.1.4 Diagnosis of LLM Results

The reasoning for including these LLMs as baselines stems from our question: “Does advanced text understanding suffice for solving **DUDE**?”. Our results for diagnostic categories reveal some strengths and weaknesses of LLMs in the DocVQA task setting.

**Strengths** GPT3 trumps all other tested models for list-type questions (ANLS=36-40%), which can be explained by the extractive nature of these questions. After 4-shot fine-tuning, ChatGPT (4-shot) is better than all other tested baselines in answering not-answerable questions (ANLS=77.45%). This can partly explain the appeal of this particular GPT checkpoint in recent times. GPT3 (4-shot) outperforms (ANLS=52.51%) other tested baselines on questions from the ‘complexity multi-hop’ category such as *What city name appears the most often in the timetables?*.

**Weaknesses** Compared to another (more simple text-only generative baseline, T5-base-512 (ANLS=47%), LLMs perform two times worse on abstractive questions (ANLS=22%). Closer analysis reveals that LLMs (even after 4-shot fine-tuning) predict abstractive questions to be *Not-answerable* in 55% of cases (in reality: 10%). Operations such as arithmetic, counting, and comparisons remain generally elusive skills ( $< 25\%$  ANLS).

Both LLMs we tested scored significantly lower than the human baseline in questions that require visual understanding, with an average ANLS score of 21%. This is understandable because these are text-only models.

While LLMs’ zero-shot performance is relatively high, we note that **DUDE** consists of public-license documents from the web, which potentially might have been included in the LLMs’ pre-training corpus.

### A.2. Assessing Confidence

ECE measures calibration of confidence, whereas AURC assesses both performance and confidence ranking [34] (more detail Appendix B.4). The latter results in an appropriate metric to select the best model in real-world applications, where wrong predictions can yield undesired scenarios, which could be prevented by manually revising low-confidence answers.

Interestingly, T5-base-512 scores better on calibration<table border="1">
<thead>
<tr>
<th>Model</th>
<th>ANLS</th>
<th>ECE</th>
<th>AURC</th>
</tr>
</thead>
<tbody>
<tr>
<td>BertQA MPDocVQA Concat</td>
<td>29.8</td>
<td><b>13.83</b></td>
<td><b>43.28</b></td>
</tr>
<tr>
<td>BertQA MPDocVQA MaxConf</td>
<td><b>32.18</b></td>
<td>28.93</td>
<td>48.73</td>
</tr>
<tr>
<td>BigBird MPDocVQA Concat</td>
<td><b>30.67</b></td>
<td><b>25.07</b></td>
<td><b>47.2</b></td>
</tr>
<tr>
<td>BigBird MPDocVQA MaxConf</td>
<td>29.38</td>
<td>50.79</td>
<td>56.81</td>
</tr>
<tr>
<td>LayoutLMv3 MPDocVQA Concat</td>
<td>22.61</td>
<td><b>13.19</b></td>
<td>57.11</td>
</tr>
<tr>
<td>LayoutLMv3 MPDocVQA MaxConf</td>
<td><b>25.27</b></td>
<td>31.31</td>
<td>58.54</td>
</tr>
<tr>
<td>Longformer MPDocVQA Concat</td>
<td><b>33.45</b></td>
<td><b>22.21</b></td>
<td>45.83</td>
</tr>
<tr>
<td>Longformer MPDocVQA MaxConf</td>
<td>28.67</td>
<td>48.6</td>
<td>58.11</td>
</tr>
<tr>
<td>T5 MPDocVQA Concat</td>
<td>34.37</td>
<td><b>18.97</b></td>
<td>47.31</td>
</tr>
<tr>
<td>T5 MPDocVQA MaxConf</td>
<td><b>37.56</b></td>
<td>23.73</td>
<td><b>46.69</b></td>
</tr>
<tr>
<td>T5-base Concat-0</td>
<td><b>25.62</b></td>
<td><b>20.05</b></td>
<td>62.25</td>
</tr>
<tr>
<td>T5-base MaxConf-0</td>
<td>22.21</td>
<td>39.47</td>
<td><b>58.89</b></td>
</tr>
</tbody>
</table>

Table 4: Comparison of baselines using Concat or Max Conf strategies.

(ECE=10.82) than T5-2D-large-8K, the baseline with the highest ANLS, yet worse calibration (ECE=14.4). In general, it seems calibration worsens when extending the maximum sequence length, whereas adding 2D position embeddings only positively affects ANLS. From the baselines tested, T5-2D-large-8K achieves the highest AURC.

Another interesting result comes from analyzing the calibration of models evaluated using the *Concat* strategy vs. *Max Conf.* strategy. In the main paper, we reported results for the model with the relative best ANLS. Thanks to our varied set of evaluation metrics, we discover that *Max Conf.* overall results in poor calibration (see Table 4), whereas considering ANLS, there is not always a clear winning strategy. This shows that predicting each page separately and necessarily assuming conditional independence across pages is not a reliable strategy for multipage DocVQA.

## B. Baseline Experiments Setup

In this Section, we describe the implementation details for the architectures and inference methods used in our benchmark.

### B.1. Hyperparameter Defaults

Refer to Table 5.

### B.2. Generative LLM Prompt Fine-tuning

The performance of GPT3.5 models was assessed in two settings: 0-shot and 4-shot. In the 0-shot setting, the prompt included instructions similar to those provided to annotators to teach them how to annotate. In the 4-shot setting, the prompt was enhanced with the content of a single document from the training set along with four questions of different types (extractive, abstractive, list, and not answerable) to better gauge the models’ abilities.

<table border="1">
<thead>
<tr>
<th>Hyper-Parameter</th>
<th>T5</th>
<th>T5+2D</th>
<th>HiVT5</th>
</tr>
</thead>
<tbody>
<tr>
<td>Epochs</td>
<td>10</td>
<td>10</td>
<td>10</td>
</tr>
<tr>
<td>Warm-up (iterations)</td>
<td>1000</td>
<td>250</td>
<td>1000</td>
</tr>
<tr>
<td>Optimizer</td>
<td>Adam, AdamW</td>
<td>Adafactor</td>
<td>Adam</td>
</tr>
<tr>
<td>Gradient acc.</td>
<td>False</td>
<td>8</td>
<td>False</td>
</tr>
<tr>
<td>Lower case</td>
<td>True</td>
<td>True</td>
<td>True</td>
</tr>
<tr>
<td>Max. Seq. Length</td>
<td>512, 8192</td>
<td>512, 8192</td>
<td>20480</td>
</tr>
<tr>
<td>Generation (Max. Tokens)</td>
<td>100</td>
<td>100</td>
<td>50</td>
</tr>
<tr>
<td>Batch size</td>
<td>3</td>
<td>8</td>
<td>1</td>
</tr>
<tr>
<td>Learning rate</td>
<td>1E-04, 2E-04</td>
<td>2E-04</td>
<td>2E-04</td>
</tr>
<tr>
<td>Training time (per epoch)</td>
<td>1h, 10h</td>
<td>1.5h, 5h</td>
<td>10h</td>
</tr>
<tr>
<td>GPU Hardware</td>
<td>TITAN RTX, A100</td>
<td>A100 (80GB)</td>
<td>TITAN RTX (24GB)</td>
</tr>
</tbody>
</table>

Table 5: Hyperparameters used for fine-tuning T5, T5-2D and HiVT5 on DUDE. When two values are placed in a single column, they refer to the model’s versions with 512 and 8192 input sequence length, respectively.

### Few-shot Prompts

#### Document:

<content of single training set document>

**Question:** <extractive question>

**Answer:** <extractive answer>

**Question:** <abstractive question>

**Answer:** <abstractive answer>

**Question:** <list question>

**Answer:** <list item> | <list item>

**Question:** <not-answerable question>

**Answer:** none

#### Document:

<content of evaluated document>

Questions and answers pairs to above document:

Answers contains either:

- - a span inside of document

- - a list of spans inside of document (each span should be separated by "|")

- - not exist explicitly as span of document (the answer should be freely generated text)

- - question couldn’t be answered (the answer should be "none")

**Question:** <question>

**Answer:** \_\_\_\_\_

The 0-shot prompt is analogous to the 4-shot prompt, but the key distinction is that it lacks the first document and the example question-and-answer pairs.

For the inference process, we utilized the prompt completion default settings outlined in the OpenAI documentation, with the exception of the temperature parameter, whichwas lowered to a value of 0.0. This adjustment was made to ensure that the output would be more deterministic and focused, with less emphasis on generating creative variations.

Only after our prompting experiments had been completed, we realized the opportunity to assess confidence estimation using chained prompts (*Please give a confidence between 0 and 1 about how certain you are this is the answer.*) as in [40]. Since we did not save our dialogue states and considered the expenses, we leave this for future work.

### B.3. Confidence Estimation

This Subsection details confidence scoring functions for the baselines, as this is not reported in standard practice.

We define *confidence* as the predicted probability of the top-1 prediction, often arising as the largest value from softmax normalization of logits from a final model layer (head).

**Encoder-based** models will output logits for all possible start and end positions of the answer within the provided context. While the predicted answer of such a span prediction architecture will come from the highest valid (no negative span) combination of the sum of a start and end logit, the predicted answer confidence can be obtained by the following procedure (*BS*: batch size and *S*: sequence length): (to be added in next version)

**Decoder-based** models are not restricted to spans and can output an arbitrary, though often controllable, amount of text tokens, indicated as  $S'$ . The logits at the final layer take the shape of  $BS \times S' \times V$ , where  $V$  is the tokenizer’s vocabulary size (32.1K for T5-base). The following confidence estimation procedure is applied for decoder outputs: (to be added in next version)

### B.4. Evaluation metrics

All metric implementations (ANLS, ECE, AURC) are made available as a standalone repository. Additionally, we provide an online service where researchers can evaluate their methods against a blind (questions-only) test dataset. Below, we expound on the implementation details of the metrics and motivate design choices.

#### B.4.1 ANLS

Average Normalized Levenshtein Similarity (ANLS) is a metric introduced in [8], which was then extended [89] to support *lists* and be invariant to the order of provided answers. We adapt the underlying Levenshtein Distance metric [45] to support *not-answerable* questions,  $NA(G) = \mathbb{I}[\text{type}(G) = \text{not-answerable}]$  (see Equation (1)).

Consider for simplicity, the evaluation of a single non-list ground truth answer  $G$  and prediction  $\hat{P}$ , each with

string lengths  $|G|$  and  $|\hat{P}|$ , respectively.

$$LD(G, \hat{P}) = \begin{cases} 1 & \text{if } NA(G) \wedge |\hat{P}| > 0, \\ 0 & \text{if } NA(G) \wedge |\hat{P}| = 0, \\ |G| & \text{if } |\hat{P}| = 0, \\ LD(\text{tail}(G), \text{tail}(\hat{P})) & \text{if } G[0] = \hat{P}[0], \\ 1 + \min \begin{cases} LD(\text{tail}(G), \hat{P}) \\ LD(G, \text{tail}(\hat{P})), \text{ otherwise} \\ LD(\text{tail}(G), \text{tail}(\hat{P})) \end{cases} & \end{cases} \quad (1)$$

The normalized similarity metric is then defined as

$$NLS(G, \hat{P}) = \frac{1 - LD(G, \hat{P})}{\max(1, |G|, |\hat{P}|)}.$$

Given multiple ground truth answer variants  $G = \{g_1, g_2, \dots\}$  and a predicted answer for  $\hat{P}_{Q_i}$  for each question  $Q$  in the test set of size  $N$ , we define the complete metric as follows:

$$ANLS = \frac{1}{N} \sum_{i=0}^N \left( \max_{g \in G_i} s(g, \hat{P}_{Q_i}) \right) \quad (2)$$

$$s(g, \hat{P}_{Q_i}) = \begin{cases} NLS(g, \hat{P}_{Q_i}) & \text{if } NLS(g, \hat{P}_{Q_i}) \geq \tau \\ 0 & \text{if } NLS(g, \hat{P}_{Q_i}) < \tau \end{cases}, \quad (3)$$

where we follow prior literature [8, 89] in setting the threshold  $\tau = 0.5$ .

In the case of a *list-type* question, Hungarian matching is performed following [89] according to NLS between each ground truth answer part and each prediction answer part.

While ANLS can account for shortcomings of OCR and formatting issues, evaluation of generated text is notoriously complex [74] and requires more research.

#### B.4.2 ECE

Expected Calibration Error (ECE) is a default metric to evaluate top-1 prediction miscalibration. It measures the  $\mathcal{L}_p$  norm difference between a model’s posterior and the true likelihood of being correct, as formally defined below:

$$ECE_p(f)^p = \mathbb{E}_{(X,Y)} [\|\mathbb{E}[Y = \hat{y} \mid f(X) = \hat{p}] - f(X)\|_p^p],$$

where  $\hat{y} = \arg \max_{y'} [f(X)]_{y'}$  is a class prediction with associated posterior probability  $\hat{p} = \max_{y'} [f(X)]_{y'}$ .

In our setting, the exact accuracy condition  $\mathbb{I}[Y = \hat{y}]$  is replaced by  $\mathbb{I}[ANLS(y, \hat{y}) > \tau]$ . Prior work [62] already introduced the strategy of thresholding continuous quality scores (in the case of IOU larger than  $\tau$ ) in order to be able to estimate ECE.In practice, ECE is implemented as a histogram binning estimator that discretizes predicted probabilities into ranges of possible values (bins) for which conditional expectation can be estimated. In order to minimize the drawbacks inherited from histogram binning, as suggested by the literature [66, 94, 41, 79], we apply an equal-mass binning scheme with 100 bins (close to  $\sqrt{N}$ ).

**Low complexity.** *Where the document has been printed?*

Simple, extractive question, plain-text evidence.

### B.4.3 AURC

Area-Under-Risk-Coverage-Curve (AURC) [24, 34] measures the possible trade-offs between coverage (proportion of test set%) and risk (error % under given coverage). It assumes predictions to come with a confidence estimate. The curve can be obtained by sorting all confidence estimates and evaluating risk from high to low, together with their respective correctness (typically based on exact match).

Similar to ECE as defined above, we apply ANLS thresholding instead. Formulated this way, the best possible AURC is constrained by the model’s test error (1-ANLS) and the number of test instances. We have extended the very detailed implementation of [34], to which we refer for further information. On a final note, AURC might be more sensible for evaluating highly-accurate settings (e.g., 90% accuracy), where risk can be better controlled (as it is typically a business decision to decide tolerance to mistakes).

<table border="1">
<thead>
<tr>
<th>Source</th>
<th>Answer</th>
<th>ANLS</th>
<th>Conf.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ground truth</td>
<td><i>New Delhi, India</i></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Human</td>
<td><i>India</i></td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>T5</td>
<td><i>IS : 9304 - 1979</i></td>
<td>0.0</td>
<td>0.56</td>
</tr>
<tr>
<td>ChatGPT</td>
<td><i>The document does not mention where it has been printed.</i></td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>GPT3</td>
<td><i>Bela Pack n Print.<br/>New Delhi, India</i></td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>T5-2D</td>
<td><i>New Delhi, India</i></td>
<td>1.0</td>
<td>0.09</td>
</tr>
<tr>
<td>HiVT5</td>
<td><i>Page 1</i></td>
<td>0.0</td>
<td>0.18</td>
</tr>
<tr>
<td>Longformer</td>
<td><i>new delhi, india</i></td>
<td>1.0</td>
<td>0.72</td>
</tr>
</tbody>
</table>

## C. Qualitative Examples

As is customary, we provide some interesting, hand-picked test set examples with predictions from some of the baselines in our study.**High complexity.** Is there any redacted section on the document? Abstractive question that requires knowledge about possible document elements.

<table border="1">
<thead>
<tr>
<th>Source</th>
<th>Answer</th>
<th>ANLS</th>
<th>Conf.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ground truth</td>
<td>No</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Human</td>
<td>No</td>
<td>1.0</td>
<td>—</td>
</tr>
<tr>
<td>T5</td>
<td>yes</td>
<td>0.0</td>
<td>0.17</td>
</tr>
<tr>
<td>ChatGPT</td>
<td>[Not-answerable]</td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>GPT3</td>
<td>[Not-answerable]</td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>T5-2D</td>
<td>No</td>
<td>1.0</td>
<td>0.43</td>
</tr>
<tr>
<td>HiVT5</td>
<td>Yes</td>
<td>0.0</td>
<td>0.55</td>
</tr>
<tr>
<td>LayoutLMv3</td>
<td>approved for release</td>
<td>0.0</td>
<td>0.01</td>
</tr>
</tbody>
</table>

**Requires arithmetic.** What is the difference between how much Operator II and Operator III makes per hour? The question requires table comprehension, determining relevant values, dividing extracted integers, and correcting the subject-verb agreement.

<table border="1">
<thead>
<tr>
<th>Source</th>
<th>Answer</th>
<th>ANLS</th>
<th>Conf.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ground truth</td>
<td>$5</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Human</td>
<td>$5</td>
<td>1.0</td>
<td>—</td>
</tr>
<tr>
<td>T5</td>
<td>200</td>
<td>0.0</td>
<td>0.28</td>
</tr>
<tr>
<td>ChatGPT</td>
<td>$5 per hour.</td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>GPT3</td>
<td>Operator II ($17/hr) |<br/>Operator III ($22/hr)</td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>T5-2D</td>
<td>[Not-answerable]</td>
<td>0.0</td>
<td>0.31</td>
</tr>
<tr>
<td>HiVT5</td>
<td>[Not-answerable]</td>
<td>0.0</td>
<td>0.15</td>
</tr>
</tbody>
</table>

**Visual evidence (chart).** What is the maximum percentage of the blue graph line on page 8? A highly demanding question that requires simultaneous competency of visual comprehension (locating chart and line color), navigating through layout (determining adequate page), and numerical comparison (deciding on the highest value).

<table border="1">
<thead>
<tr>
<th>Source</th>
<th>Answer</th>
<th>ANLS</th>
<th>Conf.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ground truth</td>
<td>75%</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Human</td>
<td>75</td>
<td>0.7</td>
<td>—</td>
</tr>
<tr>
<td>T5</td>
<td>76</td>
<td>0.0</td>
<td>0.25</td>
</tr>
<tr>
<td>ChatGPT</td>
<td>[Not-answerable]</td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>GPT3</td>
<td>76%</td>
<td>0.7</td>
<td>—</td>
</tr>
<tr>
<td>T5-2D</td>
<td>32.0</td>
<td>0.0</td>
<td>0.00</td>
</tr>
<tr>
<td>HiVT5</td>
<td>45%</td>
<td>0.7</td>
<td>0.05</td>
</tr>
<tr>
<td>BigBird</td>
<td>32</td>
<td>0.0</td>
<td>0.47</td>
</tr>
<tr>
<td>LayoutLMv3</td>
<td>80%</td>
<td>0.0</td>
<td>0.15</td>
</tr>
</tbody>
</table>**Visual evidence (handwriting).** What is the handwritten date on page 1? The question requires visual comprehension (recognition of handwriting) and layout navigation (determining the adequate page).

<table border="1">
<thead>
<tr>
<th>Source</th>
<th>Answer</th>
<th>ANLS</th>
<th>Conf.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ground truth</td>
<td>13-XII-50</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Human</td>
<td>13-XII-50</td>
<td>1.0</td>
<td>—</td>
</tr>
<tr>
<td>T5</td>
<td>1977-01-01</td>
<td>0.0</td>
<td>0.24</td>
</tr>
<tr>
<td>ChatGPT</td>
<td>[Not-answerable]</td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>GPT3</td>
<td>15 December 1950</td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>T5-2D</td>
<td>1950-12-15</td>
<td>0.0</td>
<td>0.24</td>
</tr>
<tr>
<td>HiVT5</td>
<td>1977-07-01</td>
<td>0.0</td>
<td>0.11</td>
</tr>
<tr>
<td>BERTQA</td>
<td>2006 / 1</td>
<td>0.0</td>
<td>0.5</td>
</tr>
</tbody>
</table>

**Visual evidence (map), multi-hop.** Which states don't have any marijuana laws? The multi-hop question requires visually comprehending the map and linking knowledge from its legend with depicted regions.

<table border="1">
<thead>
<tr>
<th>Source</th>
<th>Answer</th>
<th>ANLS</th>
<th>Conf.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ground truth</td>
<td>ID | SD | KS</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Human</td>
<td>ID | SD | KS</td>
<td>1.0</td>
<td>—</td>
</tr>
<tr>
<td>T5</td>
<td>WA ME MT ND MN<br/>OR VT ID NH SD WI<br/>NY MA MI</td>
<td>0.0</td>
<td>0.28</td>
</tr>
<tr>
<td>ChatGPT</td>
<td>[Not-answerable]</td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>GPT3</td>
<td>American Samoa</td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>T5-2D</td>
<td>i</td>
<td>0.0</td>
<td>0.03</td>
</tr>
<tr>
<td>HiVT5</td>
<td>-</td>
<td>0.0</td>
<td>0.02</td>
</tr>
</tbody>
</table>

**Requires counting.** How many pages have a signature?

The question requires visual comprehension (recognition of signature), knowledge about layout, and counting.

<table border="1">
<thead>
<tr>
<th>Source</th>
<th>Answer</th>
<th>ANLS</th>
<th>Conf.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ground truth</td>
<td>2</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Human</td>
<td>2</td>
<td>1.0</td>
<td>—</td>
</tr>
<tr>
<td>T5</td>
<td>1</td>
<td>0.0</td>
<td>0.01</td>
</tr>
<tr>
<td>ChatGPT</td>
<td>4</td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>GPT3</td>
<td>[Not-answerable]</td>
<td>0.0</td>
<td>—</td>
</tr>
<tr>
<td>T5-2D</td>
<td>4</td>
<td>0.0</td>
<td>0.69</td>
</tr>
<tr>
<td>HiVT5</td>
<td>4</td>
<td>0.0</td>
<td>0.41</td>
</tr>
</tbody>
</table>## D. Additional Dataset Statistics

### D.1. Answer Types

Figure 6 shows that there are barely any correlations between question type and answer type, except for the most expected ones (e.g. ‘None’ answers and ‘Not answerable’ questions), by means of Cramer’s V coefficient. For instance, date and duration types of answers are equally likely for both extractive and abstractive questions.

Figure 7 shows the answer type distribution per question type in **DUDE**, followed by a comparison to answer type distributions in related DocVQA datasets (Figure 8).

Figure 6: Answer types correlation heatmap. Results obtained with Cramer’s V coefficient. Note that values on the scale are below 0.1, suggesting a lack of correlation.

### D.2. Dataset Diversity

Similar to the text-based comparison, Figure 9 visualizes the diversity of the visual embeddings of all documents’ first pages in **DUDE**, relative to those from other DocVQA datasets.

Figure 7: Answer type distribution per question type in **DUDE**.

Figure 8: Answer type distribution per dataset, sorted in descending order of total answer type occurrences. We have found: 13 answer types in TAT-DQA; 20 answer types in InfographicsVQA and SP-DocVQA, 23 answer types in VisualMRC, and 24 answer types in **DUDE**Figure 9: Visualization of document image similarities between samples from different datasets (t-SNE over ResNet101 features of 1k documents, first pages only).

