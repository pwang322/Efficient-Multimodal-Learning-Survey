# From Models to Systems: A Comprehensive Survey of Efficient Multimodal Learning
The official GitHub page for the survey paper "From Models to Systems: A Comprehensive Survey of Efficient Multimodal Learning". And this paper is under review.


## Model

### Modality-specific Encoders
#### Vision Encoder
1. 2017_arXiv_Mobilenets: Efficient convolutional neural networks for mobile vision applications [arXiv](https://arxiv.org/abs/1704.04861)
2. 2018_CVPR_Shufflenet: An extremely efficient convolutional neural network for mobile devices [arXiv](https://arxiv.org/abs/1707.01083)
3. 2019_ICML_Efficientnet: Rethinking model scaling for convolutional neural networks [arXiv](https://arxiv.org/abs/1905.11946)
4. 2021_arXiv_Mobilevit: light-weight, general-purpose, and mobile-friendly vision transformer [arXiv](https://arxiv.org/abs/2110.02178)
5. 2021_NeurIPS_Coatnet: Marrying convolution and attention for all data sizes [arXiv](https://arxiv.org/abs/2106.04803)
6. 2022_CVPR_Metaformer is actually what you need for vision [arXiv](https://arxiv.org/abs/2111.11418)
7. 2020_arXiv_An image is worth 16x16 words: Transformers for image recognition at scale [arXiv](https://arxiv.org/abs/2010.11929)
8. 2021_ICCV_Swin transformer: Hierarchical vision transformer using shifted windows [arXiv](https://arxiv.org/abs/2103.14030)[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)
9. 2021_CVPR_Masked autoencoders are scalable vision learners [arXiv](https://arxiv.org/abs/2111.06377)
10. 2022_arXiv_Beit v2: Masked image modeling with vector-quantized visual tokenizers [arXiv](https://arxiv.org/abs/2208.06366)
11. 2023_COLM_Mamba: Linear-Time Sequence Modeling with Selective State Spaces [arXiv](https://arxiv.org/abs/2312.00752)
12. 2024_arXiv_Vision mamba: Efficient visual representation learning with bidirectional state space model [arXiv](https://arxiv.org/abs/2401.09417)
13. 2024_arXiv_Kan: Kolmogorov-arnold networks [arXiv](https://arxiv.org/abs/2404.19756)
14. 2021_ICLR_Learning transferable visual models from natural language supervision [arXiv](https://arxiv.org/abs/2103.00020)
15. 2021_ICLR_Scaling up visual and vision-language representation learning with noisy text supervision [arXiv](https://arxiv.org/abs/2102.05918)
16. 2023_ICCV_Sigmoid loss for language image pre-training [arXiv](https://arxiv.org/abs/2303.15343)
17. 2021_ICCV_Emerging properties in self-supervised vision transformers [arXiv](https://arxiv.org/abs/2104.14294)
18. 2023_arXiv_Dinov2: Learning robust visual features without supervision [arXiv](https://arxiv.org/abs/2304.07193)
19. 2025_arXiv_DINOv3 [arXiv](https://arxiv.org/abs/2508.10104)
#### Text Encoders
1. 1997_NeuralComputation_Hochreiter, Sepp and Schmidhuber, Jürgen [IEEE](https://ieeexplore.ieee.org/abstract/document/6795963)
2. 2014_EMNLP_Learning phrase representations using RNN encoder-decoder for statistical machine translation [arXiv](https://arxiv.org/abs/1406.1078)
3. 2018_NAACL_Deep contextualized word representations [arXiv](https://arxiv.org/abs/1802.05365)
4. 2018_CVPR_Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN [arXiv](https://arxiv.org/abs/1803.04831)
5. 2016_NeurIPS_LightRNN: Memory and computation-efficient recurrent neural networks [arXiv](https://arxiv.org/abs/1610.09893)
6. 2017_NeurIPS_Attention is all you need [arXiv](https://arxiv.org/abs/1706.03762)
7. 2019_NAACL_Bert: Pre-training of deep bidirectional transformers for language understanding [arXiv](https://arxiv.org/abs/1810.04805)
8. 2020_arXiv_Longformer: The long-document transformer [arXiv](https://arxiv.org/abs/2004.05150)
9. 2020_NeurIPS_Big Bird: Transformers for Longer Sequences [arXiv](https://arxiv.org/abs/2007.14062)
10. 2020_arXiv_Reformer: The efficient transformer [arXiv](https://arxiv.org/abs/2001.04451)
11. 2020_arXiv_Linformer: Self-Attention with Linear Complexity [arXiv](https://arxiv.org/abs/2006.04768)
12. 2023_arXiv_Llama: Open and efficient foundation language models [arXiv](https://arxiv.org/abs/2302.13971)
13. 2020_ICLR_Transformers are rnns: Fast autoregressive transformers with linear attention [arXiv](https://arxiv.org/abs/2006.16236)
14. 2025_arXiv_TextMamba: Scene Text Detector with Mamba [arXiv](https://arxiv.org/abs/2512.06657)
#### Audio Encoders
1. 2017_ICASSP_CNN ARCHITECTURES FOR LARGE-SCALE AUDIO CLASSIFICATION [arXiv](https://arxiv.org/abs/1609.09430)
2. 2020_ACM_Panns: Large-scale pretrained audio neural networks for audio pattern recognition [arXiv](https://arxiv.org/abs/1912.10211)
3. 2020_NeurIPS_wav2vec 2.0: a framework for self-supervised learning of speech representations [arXiv](https://arxiv.org/abs/2006.11477)
4. 2021_ACM_Hubert: Self-supervised speech representation learning by masked prediction of hidden units [arXiv](https://arxiv.org/abs/2106.07447)
5. 2021_Interspeech_AST: Audio Spectrogram Transformer [arXiv](https://arxiv.org/abs/2104.01778)
6. 2022_AAAI_SSAST: Self-Supervised Audio Spectrogram Transformer [arXiv](https://arxiv.org/abs/2110.09784)
7. 2024_Interspeech_Audio Mamba: Selective State Spaces for Self-Supervised Audio Representations [arXiv](https://arxiv.org/abs/2406.02178)
#### Thermal/Depth/Time-series
1. 2022_CVPR_Target‑aware Dual Adversarial Learning and a Multi‑scenario Multi‑Modality Benchmark to Fuse Infrared and Visible for Object Detection [arXiv](https://arxiv.org/abs/2203.16220)
2. 2024_CVPR_Flexible Window‑based Self‑attention Transformer in Thermal Image Super‑Resolution [CVPR](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/papers/Jiang_Flexible_Window-based_Self-attention_Transformer_in_Thermal_Image_Super-Resolution_CVPRW_2024_paper.pdf)
3. 2023_CVPR_Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation [arXiv](https://arxiv.org/abs/2211.13202)
4. 2023_ICCV_MonoDETR: Depth‑guided Transformer for Monocular 3D Object Detection [arXiv](https://arxiv.org/abs/2203.13310)
5. 2021_AAAI_Informer: Beyond efficient transformer for long sequence time-series forecasting [arXiv](https://arxiv.org/abs/2012.07436)
6. 2023_COLM_Mamba: Linear-Time Sequence Modeling with Selective State Spaces [arXiv](https://arxiv.org/abs/2312.00752)
7. 2023_ICLR_Liquid Structural State-Space Models [arXiv](https://arxiv.org/abs/2209.12951)
### Unified Encoders
1. 2020_ECCV_Uniter: Universal image-text representation learning [arXiv](https://arxiv.org/abs/1909.11740)
2. 2021_ICLR_ViLT: Vision‑and‑Language Transformer Without Convolution or Region Supervision [arXiv](https://arxiv.org/abs/2102.03334)
3. 2023_AAAI_BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning [arXiv](https://arxiv.org/abs/2206.08657)
4. 2022_arXiv_Learning audio-visual speech representation by masked multimodal cluster prediction [arXiv](https://arxiv.org/abs/2201.02184)
5. 2022_CVPR_Flava: A foundational language and vision alignment model [arXiv](https://arxiv.org/abs/2112.04482)
6. 2022_ICLR_Perceiver IO: A General Architecture for Structured Inputs \& Outputs [arXiv](https://arxiv.org/abs/2107.14795)
7. 2023_WACV_Perceiver-vl: Efficient vision-and-language modeling with iterative latent attention [arXiv](https://arxiv.org/abs/2211.11701)
8. 2024_NeurIPS_UNIT: Unifying Image and Text Recognition in One Vision Encoder [arXiv](https://arxiv.org/abs/2409.04095)
9. 2024_ICLR_Emu: Generative Pretraining in Multimodality [arXiv](https://arxiv.org/abs/2307.05222)
10. 2024_CVPR_Unified‑IO 2: Scaling Autoregressive Multimodal Models with Vision Language Audio and Action [arXiv](https://arxiv.org/abs/2312.17172)
11. 2024_NAACL_i‑Code V2: An Autoregressive Generation Framework over Vision, Language, and Speech Data [arXiv](https://arxiv.org/abs/2305.12311)
12. 2025_arXiv_UGen: Unified Autoregressive Multimodal Model with Progressive Vocabulary Learning [arXiv](https://arxiv.org/abs/2503.21193)
13. 2024_Grok-1.5 Vision: a preview of xAI’s multimodal model [XAI](https://x.ai/news/grok-1.5v)
### Structural Sparsity
1. 2022_NeurIPS_Vlmo: Unified vision-language pre-training with mixture-of-modality-experts [arXiv](https://arxiv.org/abs/2111.02358)
2. 2022_NeurIPS_Multimodal contrastive learning with limoe: the language-image mixture of experts [arXiv](https://arxiv.org/abs/2206.02770)
3. 2025_TPAMI_Uni-moe: Scaling unified multimodal llms with mixture of experts [arXiv](https://arxiv.org/abs/2405.11273)
4. 2025_arXiv_LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token Reduction and Mixture of Multi-Modal Experts [arXiv](https://arxiv.org/abs/2504.04653)
5. 2024_arXiv_Flex-moe: Modeling arbitrary modality combination via the flexible mixture-of-experts [arXiv](https://arxiv.org/abs/2410.08245)
6. 2025_CVPR_Nvila: Efficient frontier visual language models [arXiv](https://arxiv.org/abs/2412.04468)
7. 2025_arXiv_Smolvlm: Redefining small and efficient multimodal models [arXiv](https://arxiv.org/abs/2412.04468)
### Structural Decoding
1. 2022_neurIPS_Flamingo: a visual language model for few-shot learning [arXiv](https://arxiv.org/abs/2204.14198)
2. 2023_ICLR_Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models [arXiv](https://arxiv.org/abs/2301.12597)
### Modular Adaptation
1. 2019_ICML_Parameter-efficient transfer learning for NLP [arXiv](https://arxiv.org/abs/1902.00751)
2. 2022_ICLR_Lora: Low-rank adaptation of large language models [arXiv](https://arxiv.org/abs/2106.09685)
3. 2022_CVPR_Vl-adapter: Parameter-efficient transfer learning for vision-and-language tasks [arXiv](https://arxiv.org/abs/2112.06825)
4. 2022_NeurIPS_Lst: Ladder side-tuning for parameter and memory efficient transfer learning [arXiv](https://arxiv.org/abs/2206.06522)
5. 2023_NeurIPS_Visual instruction tuning [arXiv](https://arxiv.org/abs/2304.08485)
6. 2023_arXiv_Minigpt-4: Enhancing vision-language understanding with advanced large language models [arXiv](https://arxiv.org/abs/2304.10592)
7. 2025_AAAI_A wander through the multimodal landscape: Efficient transfer learning via low-rank sequence multimodal adapter [arXiv](https://arxiv.org/abs/2412.08979)
8. 2024_CVPR_Mma: Multi-modal adapter for vision-language models [arXiv](https://arxiv.org/abs/2409.02958)
9. 2024_arXiv_PaLM2-VAdapter: progressively aligned language model makes a strong vision-language adapter [arXiv](https://arxiv.org/abs/2402.10896)
## Algorithm
### Token Compression & Selective Computing
### Pruning
### Quantization
### Knowledge Distillation
### Prompting & Speculative Decoding
### Caching & Reuse
### Runtime Sparsity

## System
### KV Cache Management & Serving
### Edge–cloud Collaboration
### Latency-Aware Scheduling & Pipelining
### Hardware-software Co-design
### Federated Learning

## Application
### Affective Computing and Human Behavior Analysis
### Embodied AI & Robotics
### Media Understanding and Generation
### Healthcare and Biomedical Intelligence
### Spatial and Physical Scene Understanding
### Multimodal Reasoning
