I participated in the DocVQA 2026 competition organized by the Computer Vision Center (CVC) as part of ICDAR 2026. The challenge involves answering questions about visually rich documents — engineering drawings, comics, maps, scientific papers, infographics, etc. — where answers require understanding both the visual layout and the textual content.

- Competition page: https://rrc.cvc.uab.es/?ch=34&com=introduction
- Dataset: https://huggingface.co/datasets/VLR-CVC/DocVQA-2026
- Our code: https://github.com/bdsaglam/docvqa

Our method uses a code-generation agent built on DSPy's Recursive Language Model (RLM) framework. A single Qwen3.5-27B model serves as both the reasoning LLM and the vision-language model. The agent operates in a Python REPL where it writes code to explore documents, call the VLM for visual perception on page crops, search OCR text via BM25, and compute answers programmatically. We preprocess documents with Docling for OCR and include per-category prompt guidance (e.g., coarse-to-fine scanning for maps, panel-by-panel enumeration for comics).

We scored 36% on the test set and 45% on the validation set. For reference, the strongest baseline (Gemini 3 Pro with direct prompting) scores 37.5%.

The organizers require a method summary for inclusion in the ICDAR competition report. I've attached the summary — could you review it before I submit?

---

ICDAR 2026 bünyesinde Computer Vision Center (CVC) tarafından düzenlenen DocVQA 2026 yarışmasına katıldım. Yarışma, görsel açıdan zengin belgeler (mühendislik çizimleri, çizgi romanlar, haritalar, bilimsel makaleler, infografikler vb.) üzerinde soru yanıtlamayı içeriyor; cevaplar hem görsel düzeni hem de metin içeriğini anlamayı gerektiriyor.

- Yarışma sayfası: https://rrc.cvc.uab.es/?ch=34&com=introduction
- Veri seti: https://huggingface.co/datasets/VLR-CVC/DocVQA-2026
- Kodumuz: https://github.com/bdsaglam/docvqa

Yöntemimiz, DSPy'nin Recursive Language Model (RLM) çerçevesi üzerine kurulu bir kod üretim ajanı kullanıyor. Tek bir Qwen3.5-27B modeli hem akıl yürütme LLM'i hem de görsel dil modeli (VLM) olarak görev yapıyor. Ajan, bir Python REPL ortamında çalışarak belgeleri keşfetmek, sayfa kırpmaları üzerinde VLM ile görsel algılama yapmak, BM25 ile OCR metni aramak ve cevapları programatik olarak hesaplamak için kod yazıyor. Belgeleri Docling ile OCR'den geçiriyoruz ve kategoriye özel yönlendirmeler ekliyoruz (ör. haritalar için kaba-ince tarama, çizgi romanlar için panel panel numaralandırma).

Test setinde ~%36, validation setinde %45 skor elde ettik. Karşılaştırma olarak, en güçlü referans yöntem (doğrudan promptlama ile Gemini 3 Pro) %37.5 alıyor.

Organizatörler, ICDAR yarışma raporu için bir yöntem özeti istiyor. Ekte gönderiyorum. 
