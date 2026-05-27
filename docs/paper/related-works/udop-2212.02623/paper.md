# Unifying Vision, Text, and Layout for Universal Document Processing

Zineng Tang<sup>1,2</sup>    Ziyi Yang<sup>2\*</sup>    Guoxin Wang<sup>3</sup>    Yuwei Fang<sup>2</sup>    Yang Liu<sup>2</sup>  
 Chenguang Zhu<sup>2</sup>    Michael Zeng<sup>2</sup>    Cha Zhang<sup>3</sup>    Mohit Bansal<sup>1\*</sup>

<sup>1</sup>University of North Carolina at Chapel Hill

<sup>2</sup>Microsoft Azure Cognitive Services Research

<sup>3</sup>Microsoft Azure Visual Document Intelligence

## Abstract

*We propose Universal Document Processing (UDOP), a foundation Document AI model which unifies text, image, and layout modalities together with varied task formats, including document understanding and generation. UDOP leverages the spatial correlation between textual content and document image to model image, text, and layout modalities with one uniform representation. With a novel Vision-Text-Layout Transformer, UDOP unifies pretraining and multi-domain downstream tasks into a prompt-based sequence generation scheme. UDOP is pretrained on both large-scale unlabeled document corpora using innovative self-supervised objectives and diverse labeled data. UDOP also learns to generate document images from text and layout modalities via masked image reconstruction. To the best of our knowledge, this is the first time in the field of document AI that one model simultaneously achieves high-quality neural document editing and content customization. Our method sets the state-of-the-art on 8 Document AI tasks, e.g., document understanding and QA, across diverse data domains like finance reports, academic papers, and websites. UDOP ranks first on the leaderboard of the Document Understanding Benchmark.<sup>1</sup>*

## 1. Introduction

Document Artificial Intelligence studies information extraction, understanding, and analysis of digital documents, e.g., business invoices, tax forms, academic papers, etc. It is a multimodal task where text is structurally embedded in documents, together with other vision information like symbols, figures, and style. Different from classic vision-language research, document data have a 2D spatial layout: text content is structurally spread around in different locations based on diverse document types and formats (e.g., invoices vs.

tax forms); formatted data such as figures, tables and plots are laid out across the document. Hence, effectively and efficiently modeling and understanding the layout is vital for document information extraction and content understanding, for example, title/signature extraction, fraudulent check detection, table processing, document classification, and automatic data entry from documents.

Document AI has unique challenges that set it apart from other vision-language domains. For instance, the cross-modal interactions between text and visual modalities are much stronger here than in regular vision-language data, because the text modality is visually-situated in an image. Moreover, downstream tasks are diverse in domains and paradigms, e.g., document question answering [45], layout detection [57], classification [13], information extraction [28], etc. This gives rise to two challenges: (1) how to utilize the strong correlation between image, text and layout modalities and unify them to model the document as a whole? (2) how can the model efficiently and effectively learn diverse vision, text, and layout tasks across different domains?

There has been remarkable progress in Document AI in recent years [1, 10–12, 15, 16, 24, 26, 29, 30, 36, 37, 48, 52–55]. Most of these model paradigms are similar to traditional vision-language frameworks: one line of work [1, 11, 29, 30, 36, 37, 52–55] inherits vision-language models that encode images with a vision network (e.g., vision transformer) and feed the encodings to the multimodal encoder along with text [17, 27, 44, 47]; another line of work uses one joint encoder [22, 46] for both text and image [16]. Some models regard documents as text-only inputs [10, 12, 15, 26, 48]. In these works, the layout modality is represented as shallow positional embeddings, e.g., adding a 2D positional embedding to text embeddings. The strong correlation between modalities inherent in document data are not fully exploited. Also to perform different tasks, many models have to use task-specific heads, which is inefficient and requires manual design for each task.

To address these challenges, we propose Universal Docu-

\*Corresp. authors: ziyiyang@microsoft.com, mbansal@cs.unc.edu

<sup>1</sup>Code and models: <https://github.com/microsoft/i-Code/tree/main/i-Code-Doc>The diagram illustrates the UDOP architecture, which unifies vision, text, and layout through a Vision-Text-Layout Transformer. The architecture consists of a Unified Encoder and a Decoder (Text-Layout Decoder and Vision Decoder). The tasks are categorized into Vision Task (red), Text Task (green), Layout Task (blue), and Mixed Task (yellow).

**Input Document:** A sample document from RJ REYNOLDS with OCR text and Bounding Boxes (x0, y0, x1, y2).

**Tasks and Prompts:**

- **Mixed Task:** Text reconstruction with layout. <text\_layout\_0> Retail: Week of March 14, 1994
- **Text Task:** Visual text recognition. <text\_0> <100><350><118><372> </text\_0> Week of March 14, 1994
- **Text Task:** Question answering. What is the date?
- **Layout Task:** Layout modeling. <layout\_0> Ship Date </layout\_0> to Retail: Week of March 14, 1994
- **Layout Task:** Layout analysis. Title
- **Vision Task:** Masked image reconstruction. Ship Date to Retail: Week of March 14, 1994

**Outputs:**

- **Vision Outputs:** Ship Date ...
- **Text Outputs:** <100><350><118><372>...
- **Layout Outputs:** <text\_layout\_0> Ship Date <0><10><2><20> <text\_0> Ship Date Week of March 14, 1994 <layout\_0> <100><350><118><372> Title <20><50><40><80>

Figure 1. UDOP unifies vision, text, and layout through vision-text-layout Transformer and unified generative pretraining tasks including vision task, text task, layout task, and mixed task. We show the task prompts (left) and task targets (right) for all self-supervised objectives (joint text-layout reconstruction, visual text recognition, layout modeling, and masked autoencoding) and two example supervised objectives (question answering and layout analysis).

ment Processing (UDOP), a foundation Document AI model that unifies vision, text, and layout and different document tasks. Different from regarding image and document text as two separate inputs in previous works, in UDOP we propose to model them with the uniform layout-induced representation (Sec. 3.1): in the input stage, we add embeddings of text tokens with the features of the image patch where the tokens are located. This simple and novel layout-induced representation greatly enhances the interaction between the text and vision modalities.

Besides the layout-induced representation, to form a uniform paradigm for different vision, text, layout tasks, UDOP first builds a homogeneous vocabulary for texts and document layout that converts layout, i.e. bounding boxes, to discretized tokens. Second, we propose Vision-Text-Layout (VTL) Transformer, consisting of a modality-agnostic encoder, text-layout decoder and vision decoder. VTL Transformer allows UDOP to jointly encode and decode vision, text, and layout. UDOP unites all downstream tasks with a sequence-to-sequence generation framework.

Besides the challenges of modalities unification and task paradigms discussed above, another issue is previous works utilized self-supervised learning objectives that were originally designed for single-modality learning, e.g., masked language modeling, or classical vision-language pretraining, e.g., contrastive learning. We, on the other hand, propose novel self-supervised learning objectives designed to allow holistic document learning, including layout modeling, text and layout reconstruction, and vision recognition that account for text, vision and layout modeling together (Sec. 4). Besides sequential generation, UDOP can also generate vision documents by leveraging masked autoencoders (MAE) [14] by reconstructing the document image from text and layout modalities. With such generation capacity, UDOP is the first document AI model to achieve high-quality

customizable, joint document editing and generation.

Finally, our uniform sequence-to-sequence generation framework enables us to conveniently incorporate all major document supervised learning tasks to pretraining, i.e., document layout analysis, information extraction, document classification, document Q&A, and Table QA/NLI, despite their significant differences in task and data format. In contrast, pretraining in previous document AI works is constrained to unlabeled data only (or using one single auxiliary supervised dataset such as FUNSD [55]), while abundant labeled datasets with high quality supervision signals are ignored due to the lack of modeling flexibility. Overall, UDOP is pretrained on 11M public unlabeled documents, together with 11 supervised datasets of 1.8M examples. Ablation study in Table 4 shows that UDOP only pretrained with the proposed self-supervised objectives exhibits great improvements over previous models, and adding the supervised data to pretraining further improves the performance.

We evaluate UDOP on FUNSD [18], CORD [34], RVL-CDIP [13], DocVQA [33], and DUE-Benchmark [2]. UDOP ranks the 1st place on the DUE-Benchmark leaderboard with 7 tasks, and also achieves SOTA on CORD, hence making UDOP a powerful and unified foundation Document AI model for diverse document understanding tasks,

To summarize, our major contributions include:

1. 1. Unified representations and modeling for vision, text and layout modalities in document AI.
2. 2. Unified all document tasks to the sequence-to-sequence generation framework.
3. 3. Combined novel self-supervised objectives with supervised datasets in pretraining for unified document pretraining.
4. 4. UDOP can process and generate text, vision, and layout modalities together, which to the best of our knowledge is first one in the field of document AI.5. UDOP is a foundation model for Document AI, achieving SOTA on 8 tasks with significant margins.

## 2. Related Work

**Unifying Model Architectures in Multimodal Learning.** Unifying model architectures for different modalities, such as vision, language, and speech, is an emergent direction. Inspired by the immense success in natural language processing, computer vision and speech processing, model architectures in multimodal learning is converging to Transformers. One type of works concatenates text token embeddings and projected image patches as the input [6, 42] to a multimodal Transformer. Other models uses two-tower or three-tower architecture where each modality is encoded respectively. Projection heads or fusion networks on top of the two-tower architecture generate multimodal representations [38, 56].

**Unifying Tasks with the Generative Framework.** Research on unifying training processes across different tasks and domains recently has made significant progress. [8] fine-tunes language models with instructions on 1.8k tasks. [7] unifies several vision-language tasks by converting training objectives to sequence generation. [31, 49, 50] further combines more tasks, e.g., image generation, by converting images and bounding boxes to discrete tokens.

**Document Artificial Intelligence.** LayoutLM [53] pre-trains BERT models on document data with masked language modeling and document classification task, with 2D positional information and image embeddings integrated. Subsequent works [15, 16, 55] also adopt VL-BERT alike architecture and includes additional pretraining tasks, e.g., masked image/region modeling proposed, and leverages the reading order in layout information [12]. [11, 29] use a multimodal encoder to model region features extracted by CNN with sentence-level text representations and train with self-supervised objectives. [20] proposes an OCR-free model to directly generate textual output from document images. [36] trains generative language models on both unlabeled and labeled document data using generative training objectives. [10] proposed to model documents as collections of tokens bounding boxes.

## 3. Universal Document Processing

We introduce UDOP, a novel document AI framework with unified learning objectives and model architecture for text, vision, and layout as shown in Figure 1. In this section, we will concretely discuss the proposed Vision-Text-Layout Transformer in UDOP, and will introduce the unified generative pretraining method in the next section. In document processing, given a document image  $v$ , typically optical character recognition (OCR) is used on  $v$  to extract text tokens  $\{s_i\}$  in the document and their bounding boxes  $\{(x_i^1, y_i^1, x_i^2, y_i^2)\}$ , i.e., the layout information for each token.

Text token  $s_1$  is in image patch  $v_0$ ,  $s_2$  is in  $v_2$ ,  $s_3$  is in  $v_3$ .  
 $v_1$   $v_4$  do not contain any text.  $s_0$  is usually task prompt.

Figure 2. Layout-induced vision-text embedding.

$(x_i^1, y_i^1)$  and  $(x_i^2, y_i^2)$  respectively represent the coordinates of the left-upper and right-bottom corner of the bounding box. Thus, suppose we have  $M$  word tokens, the input is the triple,  $(v, \{s_i\}_{i=1}^M, \{(x_i^1, y_i^1, x_i^2, y_i^2)\}_{i=1}^M)$ . Figure 1 shows an example document (left) and downstream tasks (right).

### 3.1. A Unified Vision, Text, and Layout Encoder

We fuse the vision, text, and layout modalities in the input stage using one unified transformer encoder. For traditional vision-text data, the text modality is usually the high-level description of the corresponding image or task prompt (e.g., question). While in document images, text is embedded inside the image, i.e., text and image pixels have one-to-one correspondence. To leverage this correspondence, we propose a new Vision-Text-Layout (VTL) Transformer architecture to dynamically fuse and unite the image pixels and text tokens based on the layout information.

Concretely, given the document image  $v \in \mathbb{R}^{H \times W \times C}$ ,  $M$  word tokens  $\{s_i\}_{i=1}^M$  inside the image and the extracted layout structure  $\{(x_i^1, y_i^1, x_i^2, y_i^2)\}_{i=1}^M$ , we first partition  $v$  into  $\frac{H}{P} \times \frac{W}{P}$  image patches, where each patch is of size  $P \times P \times C$ . We then encode each patch with a  $D$ -dim vector and group all patch embeddings into a sequence of vectors  $\{v_i \in \mathbb{R}^D\}_{i=1}^N$  where  $N = \frac{H}{P} \times \frac{W}{P}$ . Text tokens are also converted to numerical  $D$ -dim embeddings  $\{s_i\}_{i=1}^M$  by vocabulary look-up.

**Layout-Induced Vision-Text Embedding.** Next, we build a unified representation for vision, text, and layout as shown in Figure 2. We define the layout indicator function  $\phi$  of image patch and token embeddings as follows:

$$\phi(s_i, v_j) = \begin{cases} 1, & \text{if the center of } s_i \text{'s bounding box} \\ & \text{is within the image patch } v_j. \\ 0, & \text{otherwise.} \end{cases} \quad (1)$$Then for each text token embedding  $s_i$ , the joint representation is the sum of its image patch feature<sup>2</sup> and the text feature:

$$s'_i = s_i + v_j, \text{ where } \phi(s_i, v_j) = 1.$$

For image patches  $v_j$  without any text tokens, i.e.  $\forall i, \phi(s_i, v_j) = 0$ , the joint representation,  $v'_j$  is itself:

$$v'_j = v_j.$$

Note we do not have a designated joint representation for image patch containing tokens, since features of these image patches are already integrated with the text embeddings. Then  $\{s'_i\}$  and  $\{v'_j\}$  are fed into the VTL transformer encoder. These joint representations greatly enhance the interaction between vision, text and layout in the model input stage by explicitly leveraging their spatial correlations.

To further unify layout and text representation, inspired by the recent progress in generative object detection [4, 49], we discretize the layout modality, i.e., continuous coordinates text bounding box, to layout tokens. Suppose we have bounding box  $(x_i^1, y_i^1, x_i^2, y_i^2)$  normalized in  $[0, 1]$ . The resulting layout token will be each coordinate multiplied by vocabulary size and then rounded to nearest integer. For example, if we have bounding box  $(0.1, 0.2, 0.5, 0.6)$  with layout vocabulary size 500, the layout tokens will then be  $\langle 50 \rangle \langle 100 \rangle \langle 250 \rangle \langle 300 \rangle$ . Layout tokens can be conveniently inserted into text context, and elegantly used for layout generation tasks (e.g., location detection). More details are discussed in Section 4.

**Position Bias.** We follow TILT [36] to encode 2D text token position as 2D relative attention bias, similar to the relative attention bias used in T5. However, unlike T5, TILT, or transformer models in previous Document AI works [16, 36], we do not use 1D position embeddings in VTL transformer encoder, since the joint embedding and the 2D position bias already incorporate the layout structure of the input document.

### 3.2. Vision-Text-Layout Decoder

As introduced in the previous section, the VTL encoder is able to compactly and jointly encode vision, text, and their layout. To perform various document generative tasks (will be discussed in Section 4), the VTL decoder is designed to jointly generate all vision, text, and layout modalities.

The VTL decoder consists of a text-layout decoder and a vision decoder, as shown in Figure 1 (middle). The text-layout decoder is a uni-directional Transformer decoder to generate text and layout tokens in a sequence-to-sequence

<sup>2</sup>Some text token like manually crafted prompts have no locations. So, we set their layout bounding boxes to be  $(0, 0, 0, 0)$ , i.e., they fall into a pseudo image patch.

manner. For the vision decoder, we adopt the decoder of MAE [14] and directly generate the image pixels with text and layout information. Details of the image decoding process will be discussed in the segment “**Masked Image Reconstruction with Text and Layout**” of Section 4.1. Both text-layout decoder and vision decoder will cross-attend to the VTL encoder.

Information such as model configurations are presented in Section 5.1.

## 4. Unified Generative Pretraining

To unify across different training objectives and datasets, we create a universal generative task format with task prompt. We pretrain UDOP on large-scale documents with and without human labels. We summarize the tasks prompts and targets in Table 1 which includes all self-supervised and supervised tasks respectively in upper and lower blocks.

### 4.1. Self-Supervised Pretraining Tasks

We propose various innovative self-supervised learning objectives for unlabeled documents. The unlabeled document contains OCR text inputs with token-level bounding boxes and the document image. In the rest of this subsection, we use the following input text as example: “Ship Date to Retail: Week of March 14, 1994”

**(1) Joint Text-Layout Reconstruction** requires the model to reconstruct the missing texts and locate them in the document image. Concretely, we mask a percentage of text tokens and ask the model to both the tokens and their bounding boxes (i.e. layout tokens). E.g., assume masking “Ship Date” and “of”, the input sequence and target sequence is given below:

---

**Input Sequence:**

“*Joint Text-Layout Reconstruction.*  $\langle \text{text\_layout\_0} \rangle$  to Retail: Week  $\langle \text{text\_layout\_1} \rangle$  March 14, 1994”

---

**Target Sequence:**

“ $\langle \text{text\_layout\_0} \rangle$  Ship Date  $\langle 100 \rangle \langle 350 \rangle \langle 118 \rangle \langle 372 \rangle \langle \text{text\_layout\_1} \rangle$  of  $\langle 100 \rangle \langle 370 \rangle \langle 118 \rangle \langle 382 \rangle$ ”

---

Here  $\langle \text{text\_layout\_0} \rangle$  and  $\langle \text{text\_layout\_1} \rangle$  denote the text-layout sentinel tokens,  $\langle 100 \rangle \langle 350 \rangle \langle 118 \rangle \langle 372 \rangle$  and  $\langle 100 \rangle \langle 370 \rangle \langle 118 \rangle \langle 382 \rangle$  represent the layout tokens of “Date to” and “of” respectively. We use masking ratio 15% similar to Masked Language Modeling (MLM) [9] as this task can be interpreted as masked text-layout modeling.

**(2) Layout Modeling** asks the model to predict positions of (group of) text tokens, given the document image and context text. E.g., to predict positions of “Ship Date” and “of”, the input sequence and target sequence is given below:Table 1. A summary of all generative pretraining objectives with task names, task prompts, and task targets.

<table border="1">
<thead>
<tr>
<th>Self-Supervised Tasks</th>
<th>Task Prompts</th>
<th>Task Targets</th>
</tr>
</thead>
<tbody>
<tr>
<td>Layout Modeling</td>
<td><i>Layout Modeling.</i> &lt;layout_0&gt; Ship Date to Retail &lt;/layout_0&gt; Week of March 14, 1994</td>
<td>&lt;layout_0&gt;<br/>&lt;100&gt;&lt;350&gt;&lt;118&gt;&lt;372&gt;</td>
</tr>
<tr>
<td>Visual Text Recognition</td>
<td><i>Visual Text Recognition.</i> &lt;text_0&gt; &lt;100&gt;&lt;350&gt;&lt;118&gt;&lt;372&gt; &lt;/text_0&gt; to Retail: Week of March 14, 1994</td>
<td>&lt;text_0&gt; Ship Date</td>
</tr>
<tr>
<td>Joint Text-Layout Reconstruction</td>
<td><i>Joint Text-Layout Reconstruction.</i> &lt;text_layout_0&gt; to Retail: Week of March 14, 1994</td>
<td>&lt;text_layout_0&gt; Ship Date &lt;100&gt;&lt;350&gt;&lt;118&gt;&lt;372&gt;</td>
</tr>
<tr>
<td>Masked Image Reconstruction</td>
<td><i>Masked Image Reconstruction.</i> Ship Date to Retail: Week of March 14, 1994</td>
<td>[Pixels of the original image]</td>
</tr>
<tr>
<th colspan="3">Supervised Tasks</th>
</tr>
<tr>
<td>Classification</td>
<td><i>Document Classification.</i> Ship Date to Retail: Week of March 14, 1994</td>
<td>Memo.</td>
</tr>
<tr>
<td>Layout Analysis</td>
<td><i>Layout Analysis.</i> Paragraph.</td>
<td>Paragraph &lt;82&gt;&lt;35&gt;&lt;150&gt;&lt;439&gt;</td>
</tr>
<tr>
<td>Information Extraction</td>
<td><i>Information Extraction.</i> Ship Date to Retail</td>
<td>Week of March 14, 1994</td>
</tr>
<tr>
<td>Question Answering</td>
<td><i>Question Answering.</i> What is the ship year?</td>
<td>1994</td>
</tr>
<tr>
<td>Document NLI</td>
<td><i>Document Natural Language Inference.</i> Ship Date to Retail: Week of March 14, 1994</td>
<td>Entailment.</td>
</tr>
</tbody>
</table>

**Input Sequence:**

“*Layout Modeling.* <layout\_0> Ship Date </layout\_0> to Retail: Week <layout\_1> of </layout\_1> March 14, 1994”

**Target Sequence:**

“<layout\_0> <100><350><118><372> <layout\_1> <100><370><118><382>”

Note this pretraining task has a different sentinel token, <layout\_sent\_0>, from the previous task “Joint Text-Layout Reconstruction” because the generation content is different (layout vs. text + layout). We use large masking ratio 75% since masking with small ratio results in an easy task.

**(3) Visual Text Recognition** identifies text at given location in the image. E.g., to recognize the text tokens at <100><350><118><372> and <100><370><118><382>, the input and target is:

**Input Sequence:**

“*Visual Text Recognition.* <text\_0> <100><350><118><372> </text\_0> to Retail: Week <text\_1> <100><370><118><382> </text\_1> March 14, 1994”

**Target Sequence:**

“<text\_0> Ship Date <text\_1> of”

Note this pretraining task also has a different sentinel token, <text\_0>. We use masking ratio 50% to distinguish this task from “Joint Text-Layout Reconstruction” and set the layout (bounding box) of sentinel token, e.g., <text\_0>, and layout token, e.g., <0><10><2><20>, to (0,0,0,0). This objective helps model learn joint vision-text embedding by understanding vision-text correspondence.

**(4) Masked Image Reconstruction with Text and Layout**

Figure 3. Masked autoencoding with text and layout.

aims to reconstruct image with text and layout as shown in Figure 3. We adopt the MAE objective [14] for vision self-supervised learning. Originally, MAE masks a percentage of the image patches and feed non-masked patches into a vision encoder. It then feeds encoder outputs to a vision decoder to reconstruct masked patches. MAE uses mean squared error and apply loss only on masked patches. We make the following modifications to the MAE decoding process to customize it for document image generation and our task unification framework:

**(4.a) Cross-Attention with Character Embeddings.** In document, the textual content mostly consists of alphabetic characters, numbers and punctuation. The character-level composition of text tokens should be helpful for the vision generation. We add cross-attention in the vision decoder that it attends to both the text token encoder features and embeddings of characters in the token (Figure 3 left upper). These characters embeddings are trainable parameters and not en-coded by the encoder. This cross-attention with characters only adds linear computation complexity but considerably improves the image generation quality.

**(4.b) Image Decoding.** Next, we describe the MAE decoding process. For UDOP, we cannot directly feed the unified encoder output to the vision decoder, since the joint vision-text embedding only contains non-masked image patches to the unified encoder (Section 3.1), and image patches are fused with text tokens. Therefore, we propose that the vision decoder takes in a sequence of trainable placeholder embeddings. The length and order of the placeholder sequence is same as the patches of target image. We use two types of placeholder embeddings to indicate whether the image patch is masked in the input document image. The vision decoder attends to encoder vision-text output AND character embeddings via cross-attention. The above process is illustrated in Figure 3. We show the high quality generation visualization in Section 6.1.

## 4.2. Supervised Pretraining Tasks

Self-supervised tasks leverage large-scale unlabeled data to learn robust representations. On the other hand, supervised tasks use labeled data for fine-grained model supervision. We include the following supervised tasks in pretraining: document classification, layout analysis, information extraction, question answering, and document natural language inference. Details of the following supervised dataset are in Appendix D. Note that we do not conduct self-supervised tasks on the supervised datasets since we already have large-scale and diverse unlabeled data. Note that the validation or test set of downstream tasks is not used in supervised pretraining.

**Classification.** The task is to predict the document type. The task prompt is “*Document Classification on (Dataset Name)*” like “*Document Classification on RVLCDIP*”, then followed by text tokens. The target is the document class. We use RVL-CDIP [13] with 16 document categories.

**Layout Analysis.** This task is to predict locations of an entity in the document like title, paragraph, etc. The task prompt is “*Layout Analysis on (Dataset Name)*”, then followed by the entity name. The target are all bounding boxes that cover the given entity. We use PubLayNet [57].

**Information Extraction.** This task predict the entity type and location of a text query (e.g., the abstract paragraph). The task prompt is “*Information Extraction on (Dataset Name) (Text Query)*”. The target is the entity label and the bounding box of each token of the query. We use DocBank [28], Kleister Charity (KLC) [41], PWC [19], and DeepForm [43].

**Question Answering.** The task is to answer a given question associated with the document image. The task prompt is “*Question Answering on (Dataset Name)*”, then followed by the question and all document tokens. The tar-

get is the answer. We use WebSRC [3], VisualMRC [45], DocVQA [33], InfographicsVQA [32], and WTQ (WikiTableQuestions) [35].

**Document NLI.** Document Natural Language Inference predicts the entailment relationship between two sentences in a document. The prompt is “*Document Natural Language Inference on (Dataset Name)*”, then followed by the sentence pair. The target is the “Entailment” or “Not Entailment”. We use TabFact [5] for this task.

## 5. Experimental Setup

### 5.1. Model Pretraining

**Model Configuration.** In UDOP, the unified encoder and text-layout decoder follows the encoder-decoder architecture of T5-large [39]. The vision decoder is MAE-large decoder [14]. Overall UDOP has 794M trainable parameters. For tokenizer, we use T5 tokenizer and embedding from Hugging Face Transformers [51]. We also extend the vocabulary to accommodate special tokens (e.g., new sentinel and layout tokens).

**Data.** For self-supervised learning, we use IIT-CDIP Test Collection 1.0 [25], a large-scale document collections commonly-used in previous works [16, 53, 55]. It contain 11 million scanned document with contains text and token-level bounding boxes extracted by OCR. Supervised datasets are as introduced in Section 4.2.

**Curriculum Learning.** We use large image resolution, 1024, in our final settings since low resolution makes document text unidentifiable for both detection and generation. It will result in  $(1024/16)^2 = 4096$  image patch sequence length which takes longer training time than small image resolution, e.g., 224. Therefore, we use curriculum learning to start from a relatively small resolution and gradually scale up to 1024 resolution. In practice, we use scale with 3 resolutions during the pretraining  $224 \rightarrow 512 \rightarrow 1024$ . We show the performance of the 3 stages in Appendix E.

**Training.** We use Adam [23] optimizer with learning rate  $5e-5$ , 1000 warmup steps, batch size 512, weight decay of  $1e-2$ ,  $\beta_1 = 0.9$ , and  $\beta_2 = 0.98$ . For each curriculum learning stage, we train for 1 epoch.

### 5.2. Downstream Evaluations

We report the results on FUNSD [18], CORD [34], RVL-CDIP [13], and DocVQA [33] in Table 3 and describe their respective settings in below. We also report the results on 7 datasets of DUE-Benchmark [2] in Table 2. Finetuning training details are available in Appendix D.6 and performance variance is available in Table 9 and Table 10. Note that for all downstream tasks, we use the original OCR annotations provided in the datasets.

**FUNSD** (Form Understanding in Noisy Scanned Documents [18]) has 149 and 50 samples for train and test. WeTable 2. Comparison with existing published models on the DUE-Benchmark. Modality T, L, V denote text, layout, or vision.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Modality</th>
<th colspan="2">Question Answering</th>
<th colspan="3">Information Extraction</th>
<th colspan="2">Table QA/NLI</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>DocVQA</th>
<th>InfoVQA</th>
<th>KLC</th>
<th>PWC</th>
<th>DeepForm</th>
<th>WTQ</th>
<th>TabFact</th>
</tr>
</thead>
<tbody>
<tr>
<td>Donut [21]</td>
<td>V</td>
<td>72.1</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>BERT<sub>large</sub> [9]</td>
<td>T</td>
<td>67.5</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>T5<sub>large</sub> [39]</td>
<td>T</td>
<td>70.4</td>
<td>36.7</td>
<td>74.3</td>
<td>25.3</td>
<td>74.4</td>
<td>33.3</td>
<td>58.9</td>
<td>50.7</td>
</tr>
<tr>
<td>T5<sub>large</sub>+U [36]</td>
<td>T</td>
<td>76.3</td>
<td>37.1</td>
<td>76.0</td>
<td>27.6</td>
<td>82.9</td>
<td>38.1</td>
<td>76.0</td>
<td>56.5</td>
</tr>
<tr>
<td>T5<sub>large</sub>+2D [36]</td>
<td>T+L</td>
<td>69.8</td>
<td>39.2</td>
<td>72.6</td>
<td>25.7</td>
<td>74.0</td>
<td>30.8</td>
<td>58.0</td>
<td>50.4</td>
</tr>
<tr>
<td>T5<sub>large</sub>+2D+U [36]</td>
<td>T+L</td>
<td>81.0</td>
<td>46.1</td>
<td>75.9</td>
<td>26.8</td>
<td>83.3</td>
<td>43.3</td>
<td>78.6</td>
<td>59.8</td>
</tr>
<tr>
<td>LAMBERT [10]</td>
<td>T+L</td>
<td>-</td>
<td>-</td>
<td>81.3</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>StructuralLM<sub>large</sub> [26]</td>
<td>T+L</td>
<td>83.9</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>LayoutLMv2<sub>large</sub> [55]</td>
<td>V+T+L</td>
<td>78.8</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>LayoutLMv3<sub>large</sub> [16]</td>
<td>V+T+L</td>
<td>83.4</td>
<td>45.1</td>
<td>77.1</td>
<td>26.9</td>
<td>84.0</td>
<td>45.7</td>
<td>78.1</td>
<td>62.9</td>
</tr>
<tr>
<td><b>UDOP</b></td>
<td>V+T+L</td>
<td><b>84.7</b></td>
<td><b>47.4</b></td>
<td><b>82.8</b></td>
<td><b>28.0</b></td>
<td><b>85.5</b></td>
<td><b>47.2</b></td>
<td><b>78.9</b></td>
<td><b>64.8</b></td>
</tr>
</tbody>
</table>

Table 3. Performance on FUNSD, CORD, and RVL-CDIP datasets. Modality V, T, L denote vision, text and layout.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Modality</th>
<th colspan="2">Info Ext.</th>
<th>Classification</th>
</tr>
<tr>
<th>FUNSD</th>
<th>CORD</th>
<th>RVL-CDIP</th>
</tr>
</thead>
<tbody>
<tr>
<td>Donut [21]</td>
<td>V</td>
<td>-</td>
<td>91.6</td>
<td>95.3</td>
</tr>
<tr>
<td>BERT<sub>large</sub> [9]</td>
<td>T</td>
<td>65.63</td>
<td>90.25</td>
<td>89.92</td>
</tr>
<tr>
<td>BROS<sub>large</sub> [15]</td>
<td>T+L</td>
<td>84.52</td>
<td>97.40</td>
<td>-</td>
</tr>
<tr>
<td>StructuralLM<sub>large</sub> [26]</td>
<td>T+L</td>
<td>85.14</td>
<td>-</td>
<td><b>96.08</b></td>
</tr>
<tr>
<td>LiLT [48]</td>
<td>T+L</td>
<td>88.41</td>
<td>96.07</td>
<td>95.68</td>
</tr>
<tr>
<td>FormNet [24]</td>
<td>T+L</td>
<td>84.69</td>
<td>97.28</td>
<td>-</td>
</tr>
<tr>
<td>LayoutLM<sub>large</sub> [53]</td>
<td>T+L</td>
<td>77.89</td>
<td>-</td>
<td>91.90</td>
</tr>
<tr>
<td>SelfDoc [29]</td>
<td>V+T+L</td>
<td>83.36</td>
<td>-</td>
<td>92.81</td>
</tr>
<tr>
<td>UniDoc [11]</td>
<td>V+T+L</td>
<td>87.93</td>
<td>96.86</td>
<td>95.05</td>
</tr>
<tr>
<td>DocFormer<sub>large</sub> [1]</td>
<td>V+T+L</td>
<td>84.55</td>
<td>96.99</td>
<td>95.50</td>
</tr>
<tr>
<td>TILT<sub>large</sub> [36]</td>
<td>V+T+L</td>
<td>-</td>
<td>96.33</td>
<td>95.52</td>
</tr>
<tr>
<td>LayoutLMv2<sub>large</sub> [55]</td>
<td>V+T+L</td>
<td>84.20</td>
<td>96.01</td>
<td>95.64</td>
</tr>
<tr>
<td>LayoutLMv3<sub>large</sub> [16]</td>
<td>V+T+L</td>
<td><b>92.08</b></td>
<td>97.46</td>
<td>95.93</td>
</tr>
<tr>
<td><b>UDOP</b></td>
<td>V+T+L</td>
<td>91.62</td>
<td><b>97.58</b></td>
<td>96.00</td>
</tr>
</tbody>
</table>

evaluate on the entity recognition task: predicting the entity, "question", "answer", "header", or "other", for the text token. The task format is, suppose we have the title, "The Title", and its entity "[I-Header]", then the encoder input is "The Title" and the generation target is "The Title [I-Header]". The metric is F1 scores.

**CORD** (Consolidated Receipt Dataset for Post-OCR Parsing) [34] is a key information extraction dataset with 30 labels under 4 categories such as "total" or "subtotal". It has 1,000 receipt samples. The train, validation, and test splits contain 800, 100, and 100 samples respectively. The metric is F1 and the task format is the same as FUNSD.

**RVL-CDIP** is the document classification dataset that we have discussed previously. It has 320k/40k/40k images for training/validation/test. The metric is classification accuracy.

**DUE-Benchmark** contains 7 datasets and 3 domains, including document question answering (DocVQA [33], In-

fographicsVQA [32]), key information extraction (KLC [41], PWC [19], DeepForm [43]), and Table QA/NLI (WTQ [35], TabFact [5]). Task prompt formats can be found in Section 4.2 and details of datasets can be found in the appendix.

**Results.** Pretrained models are finetuned on each evaluation dataset. As shown in Table 2, our models UDOP achieve SOTA performance on all 7 tasks of DUE-Benchmark, ranking the 1st place on the leaderboard as of November 11, 2022. It also sets SOTA on CORD and (Table 3). It is worth noting that UDOP is an **open-vocabulary generative model** and uses **one single model for all tasks**. In comparison, most baselines leverage task-specific network for each dataset and are classification-based models. Nonetheless, UDOP still exhibits better results than those models.

Curriculum learning on image resolution (appendix Table 8) shows that with larger resolution, UDOP steadily gains stronger performance. E.g., UDOP average performance on DUE-Benchmark with 224, 512 and 1024 resolution is 63.9, 64.3 and 65.1 respectively. Note our model with 224 resolution already outperform previous best models (e.g., average 62.9 on DUE-Benchmark). We then train UDOP only with self-supervised objectives (224 resolution). Its performance (Table 4) also surpasses baselines, which shows the effectiveness of the unified representations, TVL transformer and the proposed self-supervised objectives.

## 6. Analysis

### 6.1. Visualization Analysis

**Masked Image Reconstruction.** Figure 6 presents masked image reconstruction. Even with high masking ratio, the model can reconstruct the document image from text and layout signals with high quality: reconstructed contents are clear, consistent, and almost identical with the original image (all demonstrations are conducted on unseen documents.).Figure 4. Document generation with customized content (right). Left is the original document. We show four document edits within the same figure including title replacement, text addition, text replacement, and tilted text replacement. All edits are done with one model run.

**Document Generation & Editing.** For the first time in Document AI, UDOP achieves controllable high-quality document generation and editing. As shown in Fig. 4), one can edit and add to the document image content with customized contents. The generated content is of high resolution and is consistent with the context in font, size, style and orientation (e.g., vertical numbers in Fig. 4). More generation examples are available in Appendix B. This is done by masking the regions to edit in the document image, and specifying the customized content in the text input, and their positions through layout embeddings. This novel functionality can generate augmentation document data for future research.

**Layout Customization.** UDOP can perform controllable high-quality document layout edits. We show examples in Figure 5, where our model can edit the layout of the document by regenerating the document from scratch. This is done by keeping only a few image patch as prompt, change the bounding boxes of the content, and then regenerate the document image with the new layout.

## 6.2. Ablation Analysis

**Pretraining Objectives.** Table 4 presents the ablation study of pretraining objectives on DocVQA and RVL-CDIP

validation sets. We first develop a MLM (Masked Language Modeling) baseline that is a UDOP model pre-trained only on the BERT’s MLM [9] that masks 15% of the input tokens. UDOP models (224 image resolution) pretrained with layout/text self-supervised objectives (“Layout Modeling”, “Visual Text Dataition”, and “Joint Text-Layout Reconstruction”) outperforms the one trained with masked language modeling (MLM), confirming their effectiveness. Table 4 also shows relative effectiveness of each pretraining task. Layout modeling improves upon Joint Text-Layout Modeling; Masked Image Reconstruction improves on text-based pretraining tasks. Adding vision self-supervised learning (masked image reconstruction) and supervised learning further improves the performance.

Table 4. Ablation study on pre-training objectives.

<table border="1">
<thead>
<tr>
<th>Pretrain Objectives</th>
<th>#Pretrain Data</th>
<th>DocVQA</th>
<th>RVL-CDIP</th>
</tr>
</thead>
<tbody>
<tr>
<td>MLM</td>
<td>11.0M</td>
<td>79.7 <math>\pm</math> 0.4</td>
<td>95.3 <math>\pm</math> 0.3</td>
</tr>
<tr>
<td>Joint Text-Layout</td>
<td>11.0M</td>
<td>82.8 <math>\pm</math> 0.1</td>
<td>95.4 <math>\pm</math> 0.3</td>
</tr>
<tr>
<td>+ Visual Text Recognition</td>
<td>11.0M</td>
<td>83.3 <math>\pm</math> 0.2</td>
<td>95.4 <math>\pm</math> 0.2</td>
</tr>
<tr>
<td>+ Layout Modeling</td>
<td>11.0M</td>
<td>84.0 <math>\pm</math> 0.3</td>
<td>95.6 <math>\pm</math> 0.2</td>
</tr>
<tr>
<td>+ Image Reconstruction</td>
<td>11.0M</td>
<td>84.4 <math>\pm</math> 0.2</td>
<td>96.2 <math>\pm</math> 0.2</td>
</tr>
<tr>
<td>+ Supervised</td>
<td>12.8M</td>
<td><b>85.0</b> <math>\pm</math> 0.2</td>
<td><b>96.3</b> <math>\pm</math> 0.1</td>
</tr>
</tbody>
</table>Figure 5. Document generation with customized layout (right). Left is the original document. We change the layout of the document text including line breaks change and text rearrangement. All edits are done with one model run.

Table 5. Ablations on model architecture.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Question Answering</th>
<th colspan="2">Information Extraction</th>
<th colspan="3">Table QA/NLI</th>
<th colspan="2">Avg.</th>
</tr>
<tr>
<th>DocVQA</th>
<th>InfoVQA</th>
<th>KLC</th>
<th>PWC</th>
<th>DeepForm</th>
<th>WTQ</th>
<th>TabFact</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>UDOP-Dual</b></td>
<td><b>84.4</b></td>
<td><b>47.1</b></td>
<td><b>81.9</b></td>
<td><b>28.0</b></td>
<td><b>85.2</b></td>
<td><b>46.7</b></td>
<td><b>79.5</b></td>
<td><b>64.6</b></td>
</tr>
<tr>
<td><b>UDOP</b></td>
<td><b>84.7</b></td>
<td><b>47.4</b></td>
<td><b>82.8</b></td>
<td><b>28.0</b></td>
<td><b>85.5</b></td>
<td><b>47.2</b></td>
<td>78.9</td>
<td><b>64.8</b></td>
</tr>
</tbody>
</table>

Figure 6. MAE demonstrations with 75% masking. Middle: reconstruction, Right: original.

**Modality-Specific Model Variant.** In the field of multi-modal learning, a common model architecture is the two-tower model, where vision and text are encoded by two modality-specific encoders respectively [38, 56]. Therefore,

we explore a variant of UDOP such that instead of having one unified encoder, we separately use a text encoder (to encode both text and layout tokens) and a vision encoder. Position bias are used in both encoders to represent layout information following previous works. We name this variant UDOP-Dual. For UDOP-Dual, the text-layout encoder-decoder follows T5-large, and the vision encoder-decoder has the same configuration as MAE-large. It has in total 1098M trainable parameters. As shown in Table 5 and Table 11, using one unified encoder is better than having separated encoders in most datasets. The exceptions are WTQ and RVL-CDIP on which UDOP-Dual achieves SOTA.

**Additional Supervised Training Stage** TILT [36] performs additional training on a wide range of QA datasets, such as reading comprehension dataset SQuAD [40], before the finetuning on DocVQA. This results in considerable performance improvement of the TILT model on DocVQA and InfographicsVQA. To have a fair comparison, we also finetune UDOP on the same set of datasets before testing on DocVQA or InfographicsVQA. As shown in Table 6,UDOP is further improved with this auxiliary training and outperforms TILT.

Table 6. Training UDOP on auxiliary QA datasets as in TILT. The performance of UDOP on DocVQA and InfographicsVQA is further improved (performance without the auxiliary training was not reported in the TILT paper).

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>DocVQA</th>
<th>InfoVQA</th>
</tr>
</thead>
<tbody>
<tr>
<td>TILT<sub>large</sub>(w/ auxiliary training)</td>
<td>87.1</td>
<td>61.2</td>
</tr>
<tr>
<td>UDOP (w/o auxiliary training)</td>
<td>84.7</td>
<td>47.4</td>
</tr>
<tr>
<td>UDOP (w/ auxiliary training)</td>
<td><b>87.8</b></td>
<td><b>63.0</b></td>
</tr>
</tbody>
</table>

### 6.3. Effectiveness of the Vision Modality

In the field of Document AI, the effectiveness of the vision modality, i.e., document images, is unclear. We explore this by removing the visual embedding from the model input, with results shown in Table 7. It shows that the vision modality is more prominent on visually-rich tasks, e.g., InfographicsVQA, compared with text-dominant data such as DocVQA.

Table 7. Effectiveness of the vision modality.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>DocVQA</th>
<th>InfoVQA</th>
</tr>
</thead>
<tbody>
<tr>
<td>UDOP</td>
<td><b>84.7</b></td>
<td><b>47.4</b></td>
</tr>
<tr>
<td>UDOP w/o image input embeddings</td>
<td>84.4</td>
<td>45.0</td>
</tr>
</tbody>
</table>

## 7. Conclusion

In this work, we propose UDOP, a foundation model for document AI. UDOP unifies the vision, text and layout modalities of documents by utilizing their strong spatial correlations through layout-induced vision-text representations and Vision-Text-Layout transformer. It also unites all self-supervised and supervised document tasks with a generative framework. UDOP achieves SOTA on 8 tasks and currently ranks the 1st place on the Document Understanding Benchmark Leaderboard. For the first time in document AI, UDOP achieves customizable realistic document generation and editing. We discuss the limitations and societal impact of our work in the appendix.

## References

1. [1] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R Manmatha. Docformer: End-to-end transformer for document understanding. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 993–1003, 2021. [1](#), [7](#), [14](#)
2. [2] Łukasz Borchmann, Michał Pietruszka, Tomasz Stanisławek, Dawid Jurkiewicz, Michał Turski, Karolina Szyndler, and Filip Graliński. Due: End-to-end document understanding benchmark. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021. [2](#), [6](#)
3. [3] Lu Chen, Xingyu Chen, Zihan Zhao, Danyang Zhang, Jiabao Ji, Ao Luo, Yuxuan Xiong, and Kai Yu. Websrc: A dataset for web-based structural reading comprehension. *arXiv preprint arXiv:2101.09465*, 2021. [6](#), [14](#)
4. [4] Ting Chen, Saurabh Saxena, Lala Li, David J. Fleet, and Geoffrey Hinton. Pix2seq: A language modeling framework for object detection. In *International Conference on Learning Representations*, 2022. [4](#)
5. [5] Wenhui Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. Tabfact: A large-scale dataset for table-based fact verification. *arXiv preprint arXiv:1909.02164*, 2019. [6](#), [7](#), [14](#)
6. [6] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In *European conference on computer vision*, pages 104–120. Springer, 2020. [3](#)
7. [7] Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying vision-and-language tasks via text generation. In *International Conference on Machine Learning*, pages 1931–1942. PMLR, 2021. [3](#)
8. [8] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. *arXiv preprint arXiv:2210.11416*, 2022. [3](#)
9. [9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In *NAACL*, 2018. [4](#), [7](#), [8](#), [13](#), [14](#)
10. [10] Łukasz Garncarek, Rafał Powalski, Tomasz Stanisławek, Bartosz Topolski, Piotr Halama, Michał Turski, and Filip Graliński. Lambert: layout-aware language modeling for information extraction. In *International Conference on Document Analysis and Recognition*, pages 532–547. Springer, 2021. [1](#), [3](#), [7](#), [13](#)
11. [11] Jiuxiang Gu, Jason Kuen, Vlad I Morariu, Handong Zhao, Rajiv Jain, Nikolaos Barmpalios, Ani Nenkova, and Tong Sun. Unidoc: Unified pretraining framework for document understanding. *Advances in Neural Information Processing Systems*, 34:39–50, 2021. [1](#), [3](#), [7](#), [14](#)
12. [12] Zhangxuan Gu, Changhua Meng, Ke Wang, Jun Lan, Weiqiang Wang, Ming Gu, and Liqing Zhang. Xylayoutlm: Towards layout-aware multimodal networks for visually-rich document understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 4583–4592, 2022. [1](#), [3](#)
13. [13] Adam W Harley, Alex Ufkes, and Konstantinos G Derpanis. Evaluation of deep convolutional nets for document image classification and retrieval. In *2015 13th International Conference on Document Analysis and Recognition (ICDAR)*, pages 991–995. IEEE, 2015. [1](#), [2](#), [6](#), [13](#)
14. [14] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalablevision learners. *arXiv preprint arXiv:2111.06377*, 2021. [2](#), [4](#), [5](#), [6](#)

[15] Teakgyu Hong, DongHyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, and Sungrae Park. Bros: A pre-trained language model focusing on text and layout for better key information extraction from documents. *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(10):10767–10775, Jun. 2022. [1](#), [3](#), [7](#), [14](#)

[16] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. *arXiv preprint arXiv:2204.08387*, 2022. [1](#), [3](#), [4](#), [6](#), [7](#), [13](#), [14](#)

[17] Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu, and Jianlong Fu. Pixel-bert: Aligning image pixels with text by deep multi-modal transformers. *arXiv preprint arXiv:2004.00849*, 2020. [1](#)

[18] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. Funsd: A dataset for form understanding in noisy scanned documents. In *2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)*, volume 2, pages 1–6. IEEE, 2019. [2](#), [6](#)

[19] Marcin Kardas, Piotr Czapla, Pontus Stenetorp, Sebastian Ruder, Sebastian Riedel, Ross Taylor, and Robert Stojnic. Axcell: Automatic extraction of results from machine learning papers. *arXiv preprint arXiv:2004.14356*, 2020. [6](#), [7](#), [13](#)

[20] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In *European Conference on Computer Vision*, pages 498–517. Springer, 2022. [3](#)

[21] Geewook Kim, Teakgyu Hong, Moonbin Yim, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Donut: Document understanding transformer without ocr. *arXiv preprint arXiv:2111.15664*, 2021. [7](#), [14](#)

[22] Wonjae Kim, Bokyung Son, and Ildoo Kim. ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision. In *ICML*, 2021. [1](#)

[23] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *ICLR*, 2014. [6](#), [14](#)

[24] Chen-Yu Lee, Chun-Liang Li, Timothy Dozat, Vincent Perot, Guolong Su, Nan Hua, Joshua Ainslie, Renshen Wang, Yasuhisa Fujii, and Tomas Pfister. Formnet: Structural encoding beyond sequential modeling in form document information extraction. *arXiv preprint arXiv:2203.08411*, 2022. [1](#), [7](#), [14](#)

[25] David Lewis, Gady Agam, Shlomo Argamon, Ophir Frieder, David Grossman, and Jefferson Heard. Building a test collection for complex document information processing. In *Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval*, pages 665–666, 2006. [6](#)

[26] Chenliang Li, Bin Bi, Ming Yan, Wei Wang, Songfang Huang, Fei Huang, and Luo Si. Structurallm: Structural pre-training for form understanding. *arXiv preprint arXiv:2105.11210*, 2021. [1](#), [7](#), [13](#), [14](#)

[27] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. *arXiv preprint arXiv:1908.03557*, 2019. [1](#)

[28] Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. Docbank: A benchmark dataset for document layout analysis. *arXiv preprint arXiv:2006.01038*, 2020. [1](#), [6](#), [13](#)

[29] Peizhao Li, Jiuxiang Gu, Jason Kuen, Vlad I Morariu, Handong Zhao, Rajiv Jain, Varun Manjunatha, and Hongfu Liu. Selfdoc: Self-supervised document representation learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5652–5660, 2021. [1](#), [3](#), [7](#), [14](#)

[30] Yulin Li, Yuxi Qian, Yuechen Yu, Xiameng Qin, Chengquan Zhang, Yan Liu, Kun Yao, Junyu Han, Jingtuo Liu, and Errui Ding. Structext: Structured text understanding with multi-modal transformers. In *Proceedings of the 29th ACM International Conference on Multimedia*, pages 1912–1920, 2021. [1](#)

[31] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Motaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. *arXiv preprint arXiv:2206.08916*, 2022. [3](#)

[32] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 1697–1706, 2022. [6](#), [7](#), [14](#)

[33] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In *Proceedings of the IEEE/CVF winter conference on applications of computer vision*, pages 2200–2209, 2021. [2](#), [6](#), [7](#), [14](#)

[34] Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. Cord: a consolidated receipt dataset for post-ocr parsing. In *Workshop on Document Intelligence at NeurIPS 2019*, 2019. [2](#), [6](#), [7](#)

[35] Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. *arXiv preprint arXiv:1508.00305*, 2015. [6](#), [7](#), [14](#)

[36] Rafał Powalski, Łukasz Borchmann, Dawid Jurkiewicz, Tomasz Dwojak, Michał Pietruszka, and Gabriela Pałka. Going full-tilt boogie on document understanding with text-image-layout transformer. In *International Conference on Document Analysis and Recognition*, pages 732–747. Springer, 2021. [1](#), [3](#), [4](#), [7](#), [9](#), [13](#), [14](#)

[37] Subhojeet Pramanik, Shashank Mujumdar, and Hima Patel. Towards a multi-modal, multi-task learning based pre-training framework for document representation learning. *arXiv preprint arXiv:2009.14457*, 2020. [1](#)

[38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning*, pages 8748–8763. PMLR, 2021. [3](#), [9](#)

[39] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, andPeter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *JMLR*, 2020. [6](#), [7](#), [13](#)

[40] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. In *EMNLP*, 2016. [9](#)

[41] Tomasz Stanisławek, Filip Graliński, Anna Wróblewska, Dawid Lipiński, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, and Przemysław Biecek. Kleister: key information extraction datasets involving long documents with complex layouts. In *International Conference on Document Analysis and Recognition*, pages 564–579. Springer, 2021. [6](#), [7](#), [13](#)

[42] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visual-linguistic representations. In *International Conference on Learning Representations*, 2019. [3](#)

[43] S Svetlichnaya. Deepform: Understand structured documents at scale, 2020. [6](#), [7](#), [13](#)

[44] Hao Tan and Mohit Bansal. Lxmert: Learning cross-modality encoder representations from transformers. In *EMNLP*, 2019. [1](#)

[45] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Visualmrc: Machine reading comprehension on document images. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(15):13878–13888, May 2021. [1](#), [6](#), [14](#)

[46] Zineng Tang, Jaemin Cho, Jie Lei, and Mohit Bansal. Perceiver-vl: Efficient vision-and-language modeling with iterative latent attention. *arXiv preprint arXiv:2211.11701*, 2022. [1](#)

[47] Zineng Tang, Jie Lei, and Mohit Bansal. Decembert: Learning from noisy instructional videos via dense captions and entropy minimization. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2415–2426, 2021. [1](#)

[48] Jiapeng Wang, Lianwen Jin, and Kai Ding. Lilt: A simple yet effective language-independent layout transformer for structured document understanding. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 7747–7757, 2022. [1](#), [7](#), [14](#)

[49] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In *International Conference on Machine Learning*, pages 23318–23340. PMLR, 2022. [3](#), [4](#)

[50] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. *arXiv preprint arXiv:2208.10442*, 2022. [3](#)

[51] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45, Online, Oct. 2020. Association for Computational Linguistics. [6](#)

[52] Te-Lin Wu, Cheng Li, Mingyang Zhang, Tao Chen, Spurthi Amba Hombaiah, and Michael Bendersky. Lampret: Layout-aware multimodal pretraining for document understanding. *arXiv preprint arXiv:2104.08405*, 2021. [1](#)

[53] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. Layoutlm: Pre-training of text and layout for document image understanding. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pages 1192–1200, 2020. [1](#), [3](#), [6](#), [7](#), [14](#)

[54] Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, and Furu Wei. Layoutxlm: Multimodal pre-training for multilingual visually-rich document understanding. *arXiv preprint arXiv:2104.08836*, 2021. [1](#)

[55] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, et al. Layoutlmv2: Multi-modal pre-training for visually-rich document understanding. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 2579–2591, 2021. [1](#), [2](#), [3](#), [6](#), [7](#), [13](#), [14](#)

[56] Ziyi Yang, Yuwei Fang, Chenguang Zhu, Reid Pryzant, Dongdong Chen, Yu Shi, Yichong Xu, Yao Qian, Mei Gao, Yi-Ling Chen, et al. i-code: An integrative and composable multimodal learning framework. *arXiv preprint arXiv:2205.01818*, 2022. [3](#), [9](#)

[57] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: largest dataset ever for document layout analysis. In *2019 International Conference on Document Analysis and Recognition (ICDAR)*, pages 1015–1022. IEEE, 2019. [1](#), [6](#), [13](#)

## A. Appendix Overview

The appendix has the following contents:

- • Vision demonstrations of UDOP localizing answers in documents, the effectiveness of the cross attention with character embeddings in vision generation, and more neural editing examples Appendix [B](#).
- • UDOP-Dual performance in Appendix [C](#).
- • More details for pretraining and evaluation datasets, and finetuning experiment set up in Appendix [D](#).
- • Experiment results of curriculum learning in Appendix [E](#).
- • Performance variance of UDOP in Appendix [F](#).
- • Discussion of limitations and societal impacts in Appendix [G](#).Table 8. Comparison of different image size in curriculum learning on the DUE-Benchmark. Modality T, L, V denote text, layout, or vision.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Modality</th>
<th colspan="2">Question Answering</th>
<th colspan="3">Information Extraction</th>
<th colspan="2">Table QA/NLI</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>DocVQA</th>
<th>InfoVQA</th>
<th>KLC</th>
<th>PWC</th>
<th>DeepForm</th>
<th>WTQ</th>
<th>TabFact</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>UDOP (224)</b></td>
<td>V+T+L</td>
<td>84.4</td>
<td>46.1</td>
<td>82.1</td>
<td>26.7</td>
<td>83.6</td>
<td>46.1</td>
<td>78.2</td>
<td>63.9</td>
</tr>
<tr>
<td><b>UDOP (512)</b></td>
<td>V+T+L</td>
<td>84.5</td>
<td>47.3</td>
<td>82.0</td>
<td>27.1</td>
<td>84.7</td>
<td>46.2</td>
<td>78.3</td>
<td>64.3</td>
</tr>
<tr>
<td><b>UDOP (1024)</b></td>
<td>V+T+L</td>
<td><b>84.7</b></td>
<td><b>47.4</b></td>
<td><b>82.8</b></td>
<td><b>28.9</b></td>
<td><b>85.5</b></td>
<td><b>47.2</b></td>
<td>78.9</td>
<td><b>65.1</b></td>
</tr>
</tbody>
</table>

Table 9. Performance with standard deviations on on the DUE-Benchmark. Modality T, L, V denote text, layout, or vision.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Modality</th>
<th colspan="2">Question Answering</th>
<th colspan="3">Information Extraction</th>
<th colspan="2">Table QA/NLI</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>DocVQA</th>
<th>InfoVQA</th>
<th>KLC</th>
<th>PWC</th>
<th>DeepForm</th>
<th>WTQ</th>
<th>TabFact</th>
</tr>
</thead>
<tbody>
<tr>
<td>Donut</td>
<td>V</td>
<td>72.1</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>BERT<sub>large</sub> [9]</td>
<td>T</td>
<td>67.5</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>T5<sub>large</sub> [39]</td>
<td>T</td>
<td>70.4</td>
<td>36.7</td>
<td>74.3</td>
<td>25.3</td>
<td>74.4</td>
<td>33.3</td>
<td>58.9</td>
<td>50.7</td>
</tr>
<tr>
<td>T5<sub>large</sub>+U [36]</td>
<td>T</td>
<td>76.3</td>
<td>37.1</td>
<td>76.0</td>
<td>27.6</td>
<td>82.9</td>
<td>38.1</td>
<td>76.0</td>
<td>56.5</td>
</tr>
<tr>
<td>T5<sub>large</sub>+2D [36]</td>
<td>T+L</td>
<td>69.8</td>
<td>39.2</td>
<td>72.6</td>
<td>25.7</td>
<td>74.0</td>
<td>30.8</td>
<td>58.0</td>
<td>50.4</td>
</tr>
<tr>
<td>T5<sub>large</sub>+2D+U [36]</td>
<td>T+L</td>
<td>81.0</td>
<td>46.1</td>
<td>75.9</td>
<td>26.8</td>
<td>83.3</td>
<td>43.3</td>
<td>78.6</td>
<td>59.8</td>
</tr>
<tr>
<td>LAMBERT [10]</td>
<td>T+L</td>
<td>-</td>
<td>-</td>
<td>81.3</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>StructuralLM<sub>large</sub> [26]</td>
<td>T+L</td>
<td>83.9</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>LayoutLMv2<sub>large</sub> [55]</td>
<td>V+T+L</td>
<td>78.8</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>LayoutLMv3<sub>large</sub> [16]</td>
<td>V+T+L</td>
<td>83.4</td>
<td>45.1</td>
<td>77.1</td>
<td>26.9</td>
<td>84.0</td>
<td>45.7</td>
<td>78.1</td>
<td>62.9</td>
</tr>
<tr>
<td><b>UDOP-Dual</b></td>
<td>V+T+L</td>
<td>84.4±0.1</td>
<td>47.1±0.2</td>
<td>81.9±0.4</td>
<td>28.7±0.5</td>
<td>85.2±0.2</td>
<td>46.7±0.4</td>
<td><b>79.5</b>±0.3</td>
<td>64.7±0.3</td>
</tr>
<tr>
<td><b>UDOP</b></td>
<td>V+T+L</td>
<td><b>84.7</b>±0.2</td>
<td><b>47.4</b>±0.2</td>
<td><b>82.8</b>±0.3</td>
<td><b>28.9</b>±0.4</td>
<td><b>85.5</b>±0.2</td>
<td><b>47.2</b>±0.2</td>
<td>78.9±0.1</td>
<td><b>65.1</b>±0.2</td>
</tr>
</tbody>
</table>

## B. Visualization Analysis

**Creative Image Generation.** UDOP achieves controllable high-quality document generation and editing as described in Section 6.1. We show additional examples here in Fig. 7. Our model can edit and add to the document image content with customized contents. Note that even if the document content is vertical (the first subfigure of Fig. 7), UDOP can still achieve high generation quality.

**Answer Localization for Document QA.** UDOP can perform question answering while predicting the location of the answer. We show examples on VisualMRC in Figure 8 and our model can answer the questions regarding the document correctly while locating the area of interest.

## C. UDOP-Dual Performance

We list the performance of UDOP-Dual on FUNSD, CORD, and RVL-CDIP in Table 11.

## D. Supervised Pretraining Tasks

In this section, we list more details about the supervised datasets in pretraining and evaluations.

### D.1. Classification

RVL-CDIP [13] contains 16 document categories, such as “invoice”, “scientific publication” and “form”. The dataset has 320k training, 40k validation and 40k test images.

### D.2. Layout Analysis

PubLayNet [57] is a layout analysis dataset created from medical publications. It contains over 360k document images and labeled with typical document layout elements such as titles, paragraphs, etc.

### D.3. Information Extraction

DocBank [28] is a richly-annotated large-scale IE dataset. It consists of 500K document pages, where 400K for training, 50K for validation and 50K for testing. It has 12 semantic structure labels like abstract, title, and author. Each token has corresponding bounding box and semantic structure label.

Kleister Charity [41] is an IE dataset with complex invoice page layout and has 21.6k entities and 2.7k document images from UK Charity Commission. Its entities for extraction include invoice date, invoice number, net amount, vendor name, etc.

PWC [19] is an IE dataset which has 2,291 leaderboards, where the data is collected from the Papers with Code labelling interface. It asks information like task, dataset, metric, etc. Different from original implementation, DUE-Benchmark provides complete papers as input instead of tables.

DeepForm [43] is an IE dataset collected from political television ads in US elections and has 20k receipts and over 100k document images. This task is to extract entities like advertiser name, contract number, amount paid, etc.Table 10. Performance with standard deviations on FUNSD, CORD, and RVL-CDIP datasets.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Modality</th>
<th colspan="2">Info Ext.</th>
<th>Classification</th>
</tr>
<tr>
<th>FUNSD</th>
<th>CORD</th>
<th>RVL-CDIP</th>
</tr>
</thead>
<tbody>
<tr>
<td>Donut</td>
<td>V</td>
<td>-</td>
<td>91.6</td>
<td>95.3</td>
</tr>
<tr>
<td>BERT<sub>large</sub></td>
<td>T</td>
<td>65.63</td>
<td>90.25</td>
<td>89.92</td>
</tr>
<tr>
<td>BROS<sub>large</sub> [15]</td>
<td>T+L</td>
<td>84.52</td>
<td>97.40</td>
<td>-</td>
</tr>
<tr>
<td>StructuralLM<sub>large</sub></td>
<td>T+L</td>
<td>85.14</td>
<td>-</td>
<td>96.08</td>
</tr>
<tr>
<td>LiLT [48]</td>
<td>T+L</td>
<td>88.41</td>
<td>96.07</td>
<td>95.68</td>
</tr>
<tr>
<td>FormNet [24]</td>
<td>T+L</td>
<td>84.69</td>
<td>97.28</td>
<td>-</td>
</tr>
<tr>
<td>LayoutLM<sub>large</sub></td>
<td>T+L</td>
<td>77.89</td>
<td>-</td>
<td>91.90</td>
</tr>
<tr>
<td>SelfDoc</td>
<td>V+T+L</td>
<td>83.36</td>
<td>-</td>
<td>92.81</td>
</tr>
<tr>
<td>UDoc</td>
<td>V+T+L</td>
<td>87.93</td>
<td>98.94</td>
<td>95.05</td>
</tr>
<tr>
<td>DocFormer<sub>large</sub> [1]</td>
<td>V+T+L</td>
<td>84.55</td>
<td>96.99</td>
<td>95.50</td>
</tr>
<tr>
<td>TILT<sub>large</sub></td>
<td>V+T+L</td>
<td>-</td>
<td>96.33</td>
<td>95.52</td>
</tr>
<tr>
<td>LayoutLMv2<sub>large</sub></td>
<td>V+T+L</td>
<td>84.20</td>
<td>96.01</td>
<td>95.64</td>
</tr>
<tr>
<td>LayoutLMv3<sub>large</sub></td>
<td>V+T+L</td>
<td>92.08</td>
<td>97.46</td>
<td>95.93</td>
</tr>
<tr>
<td><b>UDOP-Dual</b></td>
<td>V+T+L</td>
<td>91.20±0.21</td>
<td>97.64±0.12</td>
<td>96.22±0.27</td>
</tr>
<tr>
<td><b>UDOP</b></td>
<td>V+T+L</td>
<td>91.62±0.34</td>
<td>97.58±0.15</td>
<td>96.00±0.26</td>
</tr>
</tbody>
</table>

Table 11. Performance of UDOP-Dual on FUNSD, CORD, and RVL-CDIP.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Modality</th>
<th colspan="2">Info Ext.</th>
<th>Classification</th>
</tr>
<tr>
<th>FUNSD</th>
<th>CORD</th>
<th>RVL-CDIP</th>
</tr>
</thead>
<tbody>
<tr>
<td>Donut [21]</td>
<td>V</td>
<td>-</td>
<td>91.6</td>
<td>95.3</td>
</tr>
<tr>
<td>BERT<sub>large</sub> [9]</td>
<td>T</td>
<td>65.63</td>
<td>90.25</td>
<td>89.92</td>
</tr>
<tr>
<td>BROS<sub>large</sub> [15]</td>
<td>T+L</td>
<td>84.52</td>
<td>97.40</td>
<td>-</td>
</tr>
<tr>
<td>StructuralLM<sub>large</sub> [26]</td>
<td>T+L</td>
<td>85.14</td>
<td>-</td>
<td>96.08</td>
</tr>
<tr>
<td>LiLT [48]</td>
<td>T+L</td>
<td>88.41</td>
<td>96.07</td>
<td>95.68</td>
</tr>
<tr>
<td>FormNet [24]</td>
<td>T+L</td>
<td>84.69</td>
<td>97.28</td>
<td>-</td>
</tr>
<tr>
<td>LayoutLM<sub>large</sub> [53]</td>
<td>T+L</td>
<td>77.89</td>
<td>-</td>
<td>91.90</td>
</tr>
<tr>
<td>SelfDoc [29]</td>
<td>V+T+L</td>
<td>83.36</td>
<td>-</td>
<td>92.81</td>
</tr>
<tr>
<td>UniDoc [11]</td>
<td>V+T+L</td>
<td>87.93</td>
<td>96.86</td>
<td>95.05</td>
</tr>
<tr>
<td>DocFormer<sub>large</sub> [1]</td>
<td>V+T+L</td>
<td>84.55</td>
<td>96.99</td>
<td>95.50</td>
</tr>
<tr>
<td>TILT<sub>large</sub> [36]</td>
<td>V+T+L</td>
<td>-</td>
<td>96.33</td>
<td>95.52</td>
</tr>
<tr>
<td>LayoutLMv2<sub>large</sub> [55]</td>
<td>V+T+L</td>
<td>84.20</td>
<td>96.01</td>
<td>95.64</td>
</tr>
<tr>
<td>LayoutLMv3<sub>large</sub> [16]</td>
<td>V+T+L</td>
<td><b>92.08</b></td>
<td>97.46</td>
<td>95.93</td>
</tr>
<tr>
<td><b>UDOP-Dual</b></td>
<td>V+T+L</td>
<td>91.20</td>
<td>97.64</td>
<td><b>96.22</b></td>
</tr>
<tr>
<td><b>UDOP</b></td>
<td>V+T+L</td>
<td>91.62</td>
<td><b>97.58</b></td>
<td>96.00</td>
</tr>
</tbody>
</table>

#### D.4. Question Answering

WebSRC [3] stands for Web-based Structural Reading Comprehension. It consists of 0.44M questions collected from 6.5K web pages with corresponding HTML, screenshots and metadata. The answer is either the text span of context or yes/no.

VisualMRC [45] stands for visual machine reading comprehension. It consists of 10,197 images 30,562 abstractive questions-answers.

DocVQA [33] is a QA dataset for excerpts from industry documents and has 50k questions on 12k document images. It asks questions on topics like text content, non-textual

elements like marks or diagrams, layout, style, etc.

InfographicsVQA [32] is a QA dataset with a focus on infographic images and has 30K questions on 5.3k document images. It requires reasoning on text content, images, data visualizations, layout, etc.

WTQ [35] is a table-based QA dataset on HTML tables collected from Wikipedia. It has 2.1k tables and 22k questions hand crafted by humans and cover a wide range of topics like table lookup, superlatives, arithmetic operations, etc.

#### D.5. Document NLI

TabFact [5] is an open-domain table-based NLI task and has 16k Wikipedia tables for 118k statements by human annotations.

#### D.6. Finetuning Experiment Setting

For all DUE-Benchmark finetuning experiments, we use Adam [23] optimizer with learning rate  $5e-5$ , 1000 warmup steps, batch size 16, weight decay of  $1e-2$ ,  $\beta_1 = 0.9$ , and  $\beta_2 = 0.98$ . For FUNSD and CORD, we use learning rate  $3e-4$  and for RVL-CDIP, we use learning rate  $1e-3$  both with 1000 warmup steps, batch size 16, weight decay of  $1e-2$ ,  $\beta_1 = 0.9$ , and  $\beta_2 = 0.98$ .

#### E. Curriculum Learning

In this section, we present the results of curriculum learning of input image resolution (224, 512, 1024) on the validations sets of evaluation benchmarks. As shown in Table 8, while the model already performs competitively well on224 resolution, its performance further increases on 512 and 1024.

## F. Performance Variance

For results in Table 2 and Table 3, we report their standard deviations as shown in Table 9 and Table 10. The deviations are computed from 5 runs with different seeds for parameter initialization.

## G. Limitations and Societal Impact

UDOP can assist users with document analysis, understanding and information extraction. This automatic processing technology will make the document processing workflow more efficient and potential more accurate. It is also worth noting that, similar to all AI generation technology, the document generation capacity of UDOP can be potentially abused for malicious document counterfeit, e.g., signature forgery, tampering monetary amount in checks, fake medical/financial records generation, etc. To avoid abuse, for model release we plan to open source the vision generation model only with limited access, e.g., through an API. Documents submitted by users that are classified as sensitive (the classifier can be a finetuned UDOP model), such as checks and personal ID, will be denied.

Applying UDOP on non-English data, especially those with non-Latin writing systems, may require further modifications to the model. For example, in Sec. 4.1, the vision decoder cross-attends with character embeddings. Then for non-English data, we need to include more character embeddings to attend with.30565 8897

SALEM PROMOTION EFFECTIVENESS REVIEW

<table border="1">
<thead>
<tr>
<th colspan="4">11. PROGRAM REVIEW - 1995</th>
</tr>
<tr>
<th colspan="4">B. PROGRAM PERFORMANCE - 1995</th>
</tr>
<tr>
<th rowspan="2">VOLUME GENERATION</th>
<th rowspan="2">TIMING</th>
<th colspan="2">REDEMPTION</th>
</tr>
<tr>
<th>EST.</th>
<th>ACTUAL</th>
</tr>
<tr>
<th></th>
<th></th>
<th>COST PER CARTON</th>
<th>COST PER COMP. CARTON</th>
</tr>
</thead>
<tbody>
<tr>
<td>• $1.00/Ctn. BOUNCEBACK</td>
<td>11/95</td>
<td>70.0</td>
<td>55.3</td>
</tr>
<tr>
<td>• $1.50 &amp; 3-4.75/Ctn. BFD INSERT</td>
<td>11/95</td>
<td>8.0/5.5</td>
<td>3.3/1.7</td>
</tr>
<tr>
<td>• 4-41.50/Ctn. Solo FSI</td>
<td>11/95</td>
<td>5.5</td>
<td>3.6</td>
</tr>
<tr>
<td>• $1.50-3-4.75/Ctn. Co-op FSI</td>
<td>11/95</td>
<td>8.0/5.5</td>
<td>3.3/2.2</td>
</tr>
<tr>
<td>• 4-42.00/Ctn. Solo FSI</td>
<td>11/95</td>
<td>8.5</td>
<td>4.9</td>
</tr>
<tr>
<td>• $1.00 &amp; 3-4.50/Ctn. Co-op FSI</td>
<td>11/95</td>
<td>5.5/3.5</td>
<td>2.7/1.4</td>
</tr>
<tr>
<td>• 42.00 &amp; 2-41.00/Ctn. Solo FSI</td>
<td>12/95</td>
<td>6.0/4.0</td>
<td>2.7/1.6</td>
</tr>
<tr>
<td><b>TARGETED BROWTH</b></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>• FREE PACK MAGAZINE Pop-Up</td>
<td>11/95</td>
<td>7.0</td>
<td>6.4</td>
</tr>
<tr>
<td>• FREE-IN-THE-MALL PREMIUM OFFER</td>
<td></td>
<td>20.0</td>
<td>15.0</td>
</tr>
<tr>
<td>• FREE CARTON BOUNCEBACK (SALEM BOX)</td>
<td></td>
<td>22.0</td>
<td>18.5</td>
</tr>
</tbody>
</table>

30565 8897

Replace Title

SALEM PROMOTION EFFECTIVENESS

<table border="1">
<thead>
<tr>
<th colspan="4">11. PROGRAM REVIEW - 1995</th>
</tr>
<tr>
<th colspan="4">B. PROGRAM PERFORMANCE - 1995</th>
</tr>
<tr>
<th rowspan="2">VOLUME GENERATION</th>
<th rowspan="2">TIMING</th>
<th colspan="2">REDEMPTION</th>
</tr>
<tr>
<th>EST.</th>
<th>ACTUAL</th>
</tr>
<tr>
<th></th>
<th></th>
<th>COST PER CARTON</th>
<th>COST PER COMP. CARTON</th>
</tr>
</thead>
<tbody>
<tr>
<td>• $1.00/Ctn. BOUNCEBACK</td>
<td>11/95</td>
<td>2.96</td>
<td>55.3</td>
</tr>
<tr>
<td>• $1.50 &amp; 3-4.75/Ctn. BFD INSERT</td>
<td>11/95</td>
<td>8.0/5.5</td>
<td>3.3/1.7</td>
</tr>
<tr>
<td>• 4-41.50/Ctn. Solo FSI</td>
<td>11/95</td>
<td>5.5</td>
<td>3.6</td>
</tr>
<tr>
<td>• $1.50-3-4.75/Ctn. Co-op FSI</td>
<td>11/95</td>
<td>8.0/5.5</td>
<td>3.3/2.2</td>
</tr>
<tr>
<td>• 4-42.00/Ctn. Solo FSI</td>
<td>11/95</td>
<td>8.5</td>
<td>4.9</td>
</tr>
<tr>
<td>• $1.00 &amp; 3-4.50/Ctn. Co-op FSI</td>
<td>11/95</td>
<td>5.5/3.5</td>
<td>2.7/1.4</td>
</tr>
<tr>
<td>• 42.00 &amp; 2-41.00/Ctn. Solo FSI</td>
<td>12/95</td>
<td>6.0/4.0</td>
<td>2.7/1.6</td>
</tr>
<tr>
<td><b>TARGETED BROWTH</b></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>• FREE PACK MAGAZINE Pop-Up</td>
<td>11/95</td>
<td>7.0</td>
<td>6.4</td>
</tr>
<tr>
<td>• FREE-IN-THE-MALL PREMIUM OFFER</td>
<td></td>
<td>20.0</td>
<td>15.0</td>
</tr>
<tr>
<td>• FREE CARTON BOUNCEBACK (SALEM BOX)</td>
<td></td>
<td>22.0</td>
<td>18.5</td>
</tr>
</tbody>
</table>

**SIDESTREAM VISIBILITY ID** (H\_1420\_)

Requestor: John Paine Date: 9/24/95 Paper Code Number: KHVV-4A  
 Paper Filler: 305 nominal 3040-136, hand ground sample of Northupite  
 Na, Mg(CO<sub>3</sub>), Cl. Coarse surface on paper due to filler. Opacity B1  
 Basis Weight (g/m<sup>2</sup>): 45.2 Porosity (COMESTA): 6.3  
 Sizing Agents (type and level): 30 nominal % Succinate. (7.85% by analysis)  
 Cross-reference Similar Models  
 Comments: By analysis: Na 3.92, Mg 1.01, K 1.161 (possible leaching of Mg)  
 Tobacco Filler: BXT Filter: DAV Cigarette Weight: 1031  
 Method (check one):  Other  Risla Super  Risla Luxury  
 Date Cigarettes Prepared: 4/16/91 Number prepared: 1 By: 3.8

**SINGLE-PORT SIDESTREAM VISIBILITY DATA:**  
 n = 1 % Attenuation: 25 Static Burn Time (min.): 10.0 S.D.: 1.8  
 Extinction Coefficient: 0.23 S.D.: 0.05 KC x BWT: 2.90  
 Ash: Adhesion: 1 Color: 4 Fall Off: 2 Solidity: 3  
 Analyze: \_\_\_\_\_ Dates Analyzed: \_\_\_\_\_  
 % Reduction: 61 relative to KAG with KC: 0.75 S.D.: \_\_\_\_\_ n = 6

**SUBJECTIVE SCREENING:**  
 Smoker: B. Floyd Relative Rating: 5.0  
 Comments hot: sl. harsh, sl. green, astringent  
 Smoker: D. Boxelman Relative Rating: 2.0  
 Comments mid: impact, spicy, astringent, tongue bite, thick black charline, ataining, dark grey ash  
 Smoker: T. Sanders Relative Rating: 2.5  
 Comments low: impact, harsh, unpleasant, bitter, affects taste, no tobacco, poor ash appearance

2024072722

**SIDESTREAM VISIBILITY ID** (H\_1420\_)

Requestor: John Paine Date: 9/24/95 Paper Code Number: KHVV-4A  
 Paper Filler: 305 nominal of the substance  
 Na, Mg(CO<sub>3</sub>), Cl. Coarse surface on paper due to filler. Opacity B1  
 Basis Weight (g/m<sup>2</sup>): 45.2 Porosity (COMESTA): 6.3  
 Sizing Agents (type and level): 30 nominal % Succinate. (7.85% by analysis)  
 Cross-reference Similar Models  
 Comments: By analysis: Na 3.92, Mg 1.01, K 1.161 (possible leaching of Mg)  
 Tobacco Filler: BXT Filter: DAV Cigarette Weight: 1031  
 Method (check one):  Other  Risla Super  Risla Luxury  
 Date Cigarettes Prepared: 4/16/91 Number prepared: 1 By: 3.8

**SINGLE-PORT UDOP SIDESTREAM DATA** Modify Subtitle  
 n = 1 % Attenuation: 25 Static Burn Time (min.): 10.0 S.D.: 1.8  
 Extinction Coefficient: 0.23 S.D.: 0.05 KC x BWT: 2.90  
 Ash: Adhesion: 1 Color: 4 Fall Off: 2 Solidity: 3  
 Analyze: \_\_\_\_\_ Dates Analyzed: \_\_\_\_\_  
 % Reduction: 61 relative to KAG with KC: 0.75 S.D.: \_\_\_\_\_ n = 6

**SUBJECTIVE SCREENING:**  
 Smoker: B. Floyd Relative Rating: 5.0  
 Comments hot: sl. harsh, sl. green, astringent  
 Smoker: D. Boxelman Relative Rating: 2.0  
 Comments mid: impact, spicy, astringent, tongue bite, thick black charline, ataining, dark grey ash  
 Smoker: T. Sanders Relative Rating: 2.5  
 Comments low: impact, harsh, unpleasant, bitter, affects taste, no tobacco, poor ash appearance

2024072722

Comments A new comment added at the end of document by UDOP Add Text

**FINISHED FILTER ROD DESCRIPTIVE** Effective Date: 02/03/1997

LOUISVILLE NO FRILLS LIGHTS 100 REGULAR SP BRAND CODE: 0H4J1 (STD) STANDARD PRODUCTION

<table border="1">
<tbody>
<tr><td>MACHINE TYPE</td><td>KDF-2</td></tr>
<tr><td>FILTER ROD LENGTH</td><td>126.0</td></tr>
<tr><td>FILTER CIRCUMFERENCE</td><td>24.45</td></tr>
<tr><td>HUBBER UP</td><td>0</td></tr>
<tr><td>PLUG LENGTH</td><td>31.5</td></tr>
<tr><td>ROD R.T.D.</td><td>419</td></tr>
<tr><td>PLUG R.T.D.</td><td>108</td></tr>
<tr><td>FILTER ROD</td><td>126MM/24.45/FT-777/4SH1/8XPF 07-033-A</td></tr>
<tr><td>PLASTICIZER</td><td>TRIACETIN PF 05-880-A</td></tr>
<tr><td>APPLICATION PERCENT</td><td>0.08</td></tr>
<tr><td>APPLICATION WEIGHT</td><td>6.08 GRAMS/100 RODS 1.50 GRAMS/25 RODS</td></tr>
<tr><td>FILTER TON</td><td>FT-777 #10 (EASTMAN) 05-777-E<br/>FT-777 #10 (CELANESE) 05-777-C</td></tr>
<tr><td>DENIER PER FILAMENT</td><td>2.7</td></tr>
<tr><td>TOTAL DENIER</td><td>35,000</td></tr>
<tr><td>CROSS SECTION</td><td>4.10</td></tr>
<tr><td>PLUG WRAP</td><td>26.25MM X 600MM KC 45-MI 86-362-A</td></tr>
<tr><td>ANCHOR ADHESIVE</td><td>NATIONAL 32-2095 PVA 41-943-A</td></tr>
<tr><td>GLUE APP. FOR ANCH. ADH.</td><td>TWO GLUE LINES</td></tr>
<tr><td>LAP ADHESIVE</td><td>NATIONAL 34-2743 HH 61-666-A</td></tr>
<tr><td>DRY WEIGHT</td><td>79.99 GRAMS/100 RODS 18.75 GRAMS/25 RODS</td></tr>
<tr><td>DRY WEIGHT W/ GLUE</td><td>79.94 GRAMS/100 RODS 18.89 GRAMS/25 RODS</td></tr>
<tr><td>WET WEIGHT W/ GLUE</td><td>81.84 GRAMS/100 RODS 20.39 GRAMS/25 RODS</td></tr>
</tbody>
</table>

SAP CROSS REFERENCE #:          FILTER: 07033A

REASON FOR CHANGE: FILTER ROD CODE CHANGED FROM 807833 TO 07-033-A FOR ADMINISTRATIVE REASONS. FILTER ROD CODE ADDED TO BODY OF SPECIFICATION. MATERIALS UPDATED TO CURRENT IF APPLICABLE. NO OTHER CHANGES.

PREPARED BY: PAM LOMELIN PRINTING DATE: 1/19/92  
 PREPARATION DATE: 1/23/97 PRINTING TIME: 8:51:49

APPROVED BY:          SUPERCEDES SPECIFICATION DATED: 2/19/96

206071592

**FINISHED DESCRIPTIVE UDOP** Effective Date: 02/03/1997

LOUISVILLE NO FRILLS LIGHTS 100 REGULAR SP BRAND CODE: 0H4J1 (STD) STANDARD PRODUCTION

<table border="1">
<tbody>
<tr><td>MACHINE TYPE</td><td>KDF-2</td></tr>
<tr><td>FILTER ROD LENGTH</td><td>126.0</td></tr>
<tr><td>FILTER CIRCUMFERENCE</td><td>24.45</td></tr>
<tr><td>HUBBER UP</td><td>0</td></tr>
<tr><td>PLUG LENGTH</td><td>31.5</td></tr>
<tr><td>ROD R.T.D.</td><td>419</td></tr>
<tr><td>PLUG R.T.D.</td><td>108</td></tr>
<tr><td>FILTER ROD</td><td>126MM/24.45/FT-777/4SH1/8XPF 07-033-A</td></tr>
<tr><td>PLASTICIZER</td><td>TRIACETIN PF 05-880-A</td></tr>
<tr><td>APPLICATION PERCENT</td><td>0.08</td></tr>
<tr><td>APPLICATION WEIGHT</td><td>6.08 GRAMS/100 RODS 1.50 GRAMS/25 RODS</td></tr>
<tr><td>FILTER TON</td><td>FT-777 #10 (EASTMAN) 05-777-E<br/>FT-777 #10 (CELANESE) 05-777-C</td></tr>
<tr><td>DENIER PER FILAMENT</td><td>2.7</td></tr>
<tr><td>TOTAL DENIER</td><td>35,000</td></tr>
<tr><td>CROSS SECTION</td><td>4.10</td></tr>
<tr><td>PLUG WRAP</td><td>26.25MM X 600MM KC 45-MI 86-362-A</td></tr>
<tr><td>ANCHOR ADHESIVE</td><td>NATIONAL 32-2095 PVA 41-943-A</td></tr>
<tr><td>GLUE APP. FOR ANCH. ADH.</td><td>TWO GLUE LINES</td></tr>
<tr><td>LAP ADHESIVE</td><td>NATIONAL 34-2743 HH 61-666-A</td></tr>
<tr><td>DRY WEIGHT</td><td>79.99 GRAMS/100 RODS 18.75 GRAMS/25 RODS</td></tr>
<tr><td>DRY WEIGHT W/ GLUE</td><td>79.94 GRAMS/100 RODS 18.89 GRAMS/25 RODS</td></tr>
<tr><td>WET WEIGHT W/ GLUE</td><td>81.84 GRAMS/100 RODS 20.39 GRAMS/25 RODS</td></tr>
</tbody>
</table>

UDOP Comments: 1449 P&S and 360.1 equipments Add Comment

SAP CROSS REFERENCE #:          FILTER: 07033A

REASON FOR CHANGE: FILTER ROD CODE CHANGED FROM 807833 TO 07-033-A FOR ADMINISTRATIVE REASONS. FILTER ROD CODE ADDED TO BODY OF SPECIFICATION. MATERIALS UPDATED TO CURRENT IF APPLICABLE. NO OTHER CHANGES.

PREPARED BY: PAM LOMELIN PRINTING DATE: 1/19/92  
 PREPARATION DATE: 1/23/97 PRINTING TIME: 8:51:49

APPROVED BY: UDOP Author Add Signature SUPERCEDES SPECIFICATION DATED: 2/19/96

206071592

Figure 7. Document generation with customized content (right). Left is the original document. We show different document edits within the same figure including title replacement, text addition, text replacement, and tilted text replacement. All edits are done with one model run.Hindawi / Blog / **Blog Post**

SciencePod 15 Nov 2019

**Latest from our journals**

**Understanding the most powerful magnets in the universe**

Science

**New study examines bizarre workings of rare type of magnetic star.**

Neutron stars – ‘dead’ stars left over when a giant star collapses – are some of the densest objects in the universe. Young, spinning neutron stars, known as magnetars, can have magnetic fields 1,000 trillion times stronger than Earth’s. These rare stars, of which 29 are currently known, include a group that is rarer.

A new study, “Observations of Radio Magnetars with the Deep Space Network”, published in Hindawi’s open access journal *Advances in Astronomy*, has used a network of space telescopes to look in detail at three of the four known radio magnetars and one magnetar candidate, a star showing some magnetar-like behaviour.

The Deep Space Network (DSN), an array of radio telescopes located in California, Spain and Australia, is mostly used by NASA to track spacecraft – but the telescopes are sometimes used to study other objects in the sky too.

Study authors, Aaron B. Pearlman, Walid A. Majid and Thomas A. Prince from the California Institute of Technology in Pasadena used the DSN to monitor the emission from three radio magnetars and a magnetar candidate over more than a year. They found that the pulsations from these magnetars varied greatly during the observation time.

**Share Post**

Twitter LinkedIn Facebook Email

**Question 1:**  
**Where is the DSN located?**

**Answer 1:**  
**California, Spain and Australia.**

**Region of Interest 1**

**Question 2:**  
**How many magnetars are known to people?**

**Answer 2:**  
**29**

**Region of Interest 2**

Figure 8. Document QA and answer localization with UDOP on VisualMRC dataset. As shown, besides generating the answer, UDOP can predict the region of interest (RoI) that answer is located in by generating the layout tokens. Note that the the labeled RoI VisualMRC dataset is at paragraph level.

