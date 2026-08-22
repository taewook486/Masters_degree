# Master's Thesis (English Edition)

<!-- pdf:strip-meta -->
> Source: [THESIS_FINAL_v2.0.md](THESIS_FINAL_v2.0.md) (Korean edition, v2.0-draft)
> This English edition is a faithful translation of the Korean thesis. All numeric
> results, statistics, and citations are carried over verbatim from the source.
> Remaining work: citation style alignment with the department's required format,
> supervisor review.
<!-- /pdf:strip-meta -->

## Thesis Information

- **University**: Graduate School of Information & Telecommunications, Konkuk University, Department of Convergence Information Technology, Major in Artificial Intelligence
- **Title (Korean)**: 경량 멀티모달 모델의 의료 영상 VQA 도메인 적응: QLoRA 파인튜닝과 자율 하이퍼파라미터 최적화
- **Title (English)**: Domain Adaptation of Lightweight Vision-Language Models for Medical Visual Question Answering: QLoRA Fine-Tuning with Autonomous Hyperparameter Optimization

---

## ABSTRACT

This study examines whether lightweight Vision-Language Models (VLMs) can be adapted to the medical Visual Question Answering (VQA) domain on consumer-grade GPUs, through a three-stage experiment conducted under a single evaluation protocol. Four open models in the 2-3B parameter range (Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B) were evaluated on three public medical VQA datasets (PathVQA, SLAKE, VQA-RAD) across zero-shot baselines (12 conditions, Phase 1), QLoRA fine-tuning (75 conditions, Phase 2), and a comparison of hyperparameter search strategies together with a configuration-consistency re-experiment (810 trials, Phase 3).

Zero-shot accuracy differed significantly across models (Cochran's Q = 1904.28, df = 3, p < .001), with Qwen3-VL-2B achieving the highest pooled accuracy (0.3843). This ranking persisted after removing samples flagged as likely pretraining exposure by Min-K% Probability, indicating robustness to data contamination. The effect of QLoRA fine-tuning was heterogeneous in direction across models: it significantly improved Qwen2.5-VL-3B (Cohen's d = +2.646) and Qwen3-VL-2B (d = +1.620), yet significantly degraded SmolVLM2-2.2B (d = -2.284). A mixed-effects model pooling all four models reported no significant effect (p = .3629) because these opposing effects cancelled in the aggregate. In autonomous hyperparameter search, the large language model agent (0.4184) performed significantly worse than Bayesian optimization (0.4490; Mann-Whitney U = 16.00, p = .0112, r = -0.68) and was statistically indistinguishable from random search (0.4186), although all three automated strategies exceeded the manual configuration (0.3776). The agent's disadvantage arose not from proposal quality but from a reduced effective search budget caused by duplicate proposals (12.5 of 20 unique configurations per repeat). This diagnosis was confirmed by a 200-trial re-experiment that removed the configuration inconsistencies between the prompt and the implementation: unique configurations recovered to 18.90 of 20 and the significant disadvantage against Optuna disappeared (p = .1726), yet the run-level mean still fell short of Optuna (0.4292 vs. 0.4490) and no improvement over the original condition was demonstrated (p = .2387), so no performance advantage was established.

The study makes three contributions. First, it provides three-stage empirical evidence on medical-domain adaptation of lightweight VLMs under a single protocol, showing that performance and resource consumption are not monotonically related and thereby offering a concrete basis for model selection under resource constraints. Second, it quantifies the model-level heterogeneity of fine-tuning effects and demonstrates how pooled analysis conceals opposing effects. Third, it reports a negative result for LLM-based autonomous optimization together with a verification of its failure mechanism. Notably, the variation observed when repeating an identical configuration (0.056) exceeded the mean difference between strategies (0.031), constituting a quantitative counterexample to the practice of comparing methods from single runs.

Keywords : Medical Visual Question Answering, Vision-Language Model, QLoRA, Parameter-Efficient Fine-Tuning, Hyperparameter Optimization, Large Language Model Agent

---

## Chapter I. Introduction

### 1.1 Background

Driven by the success of large language models (LLMs), Vision-Language Models (VLMs) that jointly understand images and text have advanced rapidly, led by systems such as GPT-4V and Gemini. These general-purpose VLMs perform well on question answering over natural images, but their performance tends to be limited in domains requiring specialist knowledge, such as medical imaging, because they lack an adequate grasp of medical terminology and of pathological and radiological findings. At the same time, retraining large models from scratch is not realistic in research and clinical environments with constrained GPU resources. Parameter-Efficient Fine-Tuning (PEFT) methods such as QLoRA (Quantized Low-Rank Adaptation) have therefore attracted attention as an alternative that makes domain-specific fine-tuning feasible even on consumer-grade GPUs with 16-24GB of VRAM.

Applying a PEFT method such as QLoRA in practice, however, requires selecting a number of hyperparameters — LoRA rank, target modules, learning rate, and others — and the influence of these choices on final performance often remains unexamined in a systematic way. Automating hyperparameter optimization (HPO) was identified as an open problem for VLM PEFT in a 2024 NeurIPS survey. Beyond traditional grid and random search or Bayesian optimization (Optuna TPE), an autonomous search paradigm has been proposed in which an LLM agent interprets prior experimental results on its own and proposes the next configuration.

### 1.2 Research Objectives

This study aims to empirically verify the full process of adapting lightweight Vision-Language Models to the medical imaging VQA (Visual Question Answering) domain on consumer GPUs. Specifically, it pursues the following three objectives.

1. Demonstrate the feasibility of medical VQA domain adaptation for lightweight VLMs on consumer-class GPUs (executed here on 24GB cards; applicable to the 16GB class on the basis of measured VRAM).
2. Systematically analyse how the principal hyperparameters of QLoRA fine-tuning — data scale, LoRA rank, and the scope of target modules — affect performance.
3. Verify whether autoresearch-style autonomous hyperparameter search driven by an LLM agent delivers performance competitive with Bayesian optimization while providing interpretable search rationales.

These three objectives are formalized as the following research questions (RQs).

| # | Research Question | Null Hypothesis |
|---|----------|----------|
| RQ1 | Do lightweight VLMs (2-3B) differ significantly across models in zero-shot medical VQA performance? | H0: No difference in VQA accuracy across models |
| RQ2 | Does QLoRA fine-tuning significantly improve medical VQA performance? | H0: Base = Fine-tuned performance |
| RQ3 | Does LLM-agent-based autonomous hyperparameter search achieve performance competitive with Bayesian optimization (Optuna TPE) while providing interpretable search rationales? | H0: Autoresearch = Optuna (TPE) |

The null hypothesis for RQ3 was set against Optuna (TPE) rather than plain random search because the justification for adopting LLM-based HPO — natural-language explanation of search rationale, use of prior knowledge to understand the structure of the search space, and traceability of the reasons for configuration changes — holds only if such a method can compete with Bayesian optimization, which is already established as a practical standard, rather than merely being "better than random." Random search is retained solely as a lower-bound reference.

RQ1, RQ2, and RQ3 are verified against measured data in Sections 4.1, 4.2, and 4.3 of Chapter IV respectively, and the discussion cutting across all three results is consolidated in Section 4.4.

### 1.3 Scope and Delimitations

The experimental scope of this study is limited to four lightweight VLMs (Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B) and three public medical VQA datasets (PathVQA, SLAKE, VQA-RAD). Models were selected on the criteria that QLoRA fine-tuning was expected to be feasible within 16GB-class VRAM (confirmed post hoc at a measured maximum of 14.4GB, Section 3.2.1) and that the license permits free redistribution (Apache 2.0 / MIT). Datasets were restricted to publicly accessible benchmarks in which clinical question types are labelled.

The principal limitations of the study are as follows; detailed grounds and mitigations are discussed in Section 5.3 of Chapter V. (1) Because the target datasets were released before the pretraining cut-off of the models, contamination of pretraining data is possible; this study therefore actively measures such contamination using the Min-K% Probability technique and separately verifies the robustness of its conclusions (see the contamination-robustness verification in Section 4.1.1). (2) Direct experimental comparison with existing medical-specialized VLMs such as LLaVA-Med and Med-Flamingo lies outside the scope of this study and is replaced by indirect comparison with figures reported in prior work. (3) Owing to GPU time and cost constraints, a `max_steps` ceiling was applied to QLoRA training, so that the effective training volume (in epoch terms) varies with dataset size.

### 1.4 Organization of the Thesis

This thesis is organized as follows. Chapter II reviews the theoretical background and prior work on Vision-Language Models, Parameter-Efficient Fine-Tuning, medical VQA, and autonomous hyperparameter optimization. Chapter III describes the experimental environment, the target models and datasets, the concrete design of the three-stage experiment (Phase 1 zero-shot baseline, Phase 2 QLoRA fine-tuning, Phase 3 autonomous HPO), and the evaluation and statistical analysis methods. Chapter IV presents the experimental results of each phase in accordance with RQ1-RQ3 and discusses them collectively. Chapter V summarizes the findings and presents the contributions, limitations, and directions for future research.

---

## Chapter II. Theoretical Background

### 2.1 Overview of Vision-Language Models

#### 2.1.1 Development of Multimodal Learning

A Vision-Language Model (VLM) combines an image encoder with a large language model (LLM) so as to understand and generate visual and linguistic information jointly. Early multimodal learning centred on contrastive learning over image-text pairs to learn a shared embedding space. As the instruction-following capability of LLMs advanced, the dominant approach shifted to projecting image features into the input token space of an LLM and using instruction tuning to acquire visual question answering and description generation abilities. LLaVA by Liu et al. [1] is a representative case in which an open-source LLM was fine-tuned on visual instruction data generated by GPT-4, achieving visual dialogue ability approaching that of commercial models; many subsequent VLM studies inherited this instruction-tuning paradigm. Commercial large-scale VLMs such as GPT-4V and Gemini have secured general-purpose visual understanding through vast parameter counts and training data, but their inference cost and deployment constraints are correspondingly large, which limits their use on consumer-grade hardware.

#### 2.1.2 Lightweight VLM Architectures

The four models targeted in this study are all lightweight VLMs in the 2-3B class released in 2025-2026, each pursuing parameter efficiency in a different way. Qwen2.5-VL [2] is characterized by dynamic-resolution processing, which adjusts the number of visual tokens according to image size, and by an MRoPE extension that aligns temporal information to absolute time; it is strong at document parsing and multilingual OCR. Its successor, Qwen3-VL, expanded into dense (2B/4B/8B/32B) and Mixture-of-Experts (30B-A3B/235B-A22B) families and introduced a "thinking mode" together with DeepStack-style multi-level fusion of visual features. SmolVLM2 [3] is an on-device-oriented architecture that combines a SigLIP image encoder with the SmolLM2 language model, focusing on extremely low memory usage (5.2GB for video inference in the 2.2B model). Gemma4-E2B (Google) is a Mixture-of-Experts-family structure that activates only 2.3B parameters at inference through Per-Layer Embeddings (PLE) while exploiting the expressive capacity of a 5.1B-class total parameter count; it is lightweight in terms of active parameters, but its stored parameter scale is larger than that of the other three models, an architectural difference whose implications for interpreting the results are discussed in Section 5.3 of Chapter V.

### 2.2 Parameter-Efficient Fine-Tuning

#### 2.2.1 LoRA (Low-Rank Adaptation)

LoRA [4] freezes a pretrained weight matrix $W_0$ and learns an approximation of its update $\Delta W$ as the product $BA$ of two low-rank matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$, with $r \ll \min(d,k)$. The forward pass is computed as $h = W_0 x + BAx$, and since only $A$ and $B$ are trained, the ratio of trainable to total parameters can be reduced dramatically (in Ablation B of this study, the trainable-parameter ratio at rank = 64 is approximately 0.2-1.6% of the total; see Section 4.2.3). The rank $r$ is the key hyperparameter governing the trade-off between expressive capacity and parameter efficiency, while alpha is a coefficient that scales $\Delta W$ (multiplying $BAx$ by $\alpha/r$).

#### 2.2.2 QLoRA (Quantized LoRA)

QLoRA [5] combines LoRA with 4-bit quantization to reduce VRAM usage further. Its core components are (1) 4-bit NormalFloat (NF4) quantization optimized for normally distributed weights, (2) double quantization, which quantizes the quantization constants themselves, and (3) a paged optimizer that offloads GPU memory spikes to the CPU. The base model is held fixed in 4-bit quantized form and only the LoRA adapter above it is trained at 16-bit precision; the original paper showed that this allows fine-tuning of a 65B-class model on a single 48GB GPU while maintaining performance close to full 16-bit fine-tuning. This study applies the same QLoRA scheme (NF4 quantization + LoRA + paged AdamW 8-bit) to all four lightweight VLMs in order to verify the feasibility of medical-domain fine-tuning on consumer-class GPUs (Section 3.6). Execution took place on 24GB cards (RTX 4090 and 3090), and feasibility in the 16GB class is judged from measured peak VRAM (Section 3.2.1).

#### 2.2.3 Comparison with Other PEFT Methods

Besides LoRA and QLoRA, various PEFT methods have been proposed, including partial fine-tuning that trains only some layers, prefix tuning that prepends trainable prefix vectors to the input, and adapter approaches that insert small adapter modules between transformer layers. Compared with these, the advantages of the LoRA family are that (1) $BA$ can be merged into $W_0$ at inference time so that no additional latency is incurred, and (2) adapters can be stored and swapped independently, allowing a single base model to be reused across multiple tasks. This study adopts LoRA/QLoRA because these two advantages are particularly favourable for the repeated hyperparameter search conducted on consumer GPUs (Section 3.7, Phase 3).

### 2.3 Medical Visual Question Answering

#### 2.3.1 Definition of the Medical VQA Task

Medical VQA is the task of generating a correct answer given a medical image (a pathology tissue slide, a radiological image, and so on) together with a natural-language question. Unlike general-domain VQA, it requires both an understanding of specialist medical terminology and fine-grained visual grounding of imaging findings. Question types divide broadly into (1) closed-ended questions answered yes/no or by selecting among options, and (2) open-ended questions requiring free-form answers; the two types differ greatly in scoring method and difficulty, as confirmed empirically in Section 4.1.2.

#### 2.3.2 Principal Benchmark Datasets

The three datasets used in this study each represent a different sub-domain of medical imaging. PathVQA [6] provides 32,799 question-answer pairs over 4,998 pathology tissue images extracted from pathology textbooks and the PEIR digital library, organized into seven question types modelled on the American Board of Pathology (ABP) certification examination format. SLAKE [7] combines 14,028 bilingual English-Chinese question-answer pairs over 642 radiology/CT images with a medical knowledge graph of 5,232 knowledge triplets. VQA-RAD [8] was the first dataset to collect questions that clinicians naturally raised while viewing radiological images (CT/MRI/X-ray), containing roughly 3,500 question-answer pairs over 315 images.

#### 2.3.3 Prior Work

Attempts to specialize general-purpose VLMs for the medical domain divide broadly into large-scale retraining and adaptation approaches. LLaVA-Med [9] generated GPT-4-based instruction data from large-scale biomedical figure-caption data in PubMed Central and adapted general-purpose LLaVA to the biomedical domain through curriculum learning. Med-Flamingo [10] performed continued pretraining of OpenFlamingo-9B on image-text data from medical papers and textbooks, acquiring the ability to adapt to medical VQA from only a few examples. CheXagent [11] is a foundation model composed of a clinical LLM, a visual encoder, and a cross-modal bridge network specialized for chest X-ray interpretation, representing an approach deeply specialized to one imaging sub-domain.

These prior studies commonly target models on the scale of billions to hundreds of billions of parameters and presuppose either continued pretraining on large biomedical corpora or the generation of large-scale instruction data. By contrast, this study fine-tunes 2-3B-class lightweight models with QLoRA using only small domain datasets (thousands to tens of thousands of samples), focusing on practical adaptability under constrained computational resources; the approach therefore differs in kind. Direct experimental comparison with these prior studies lies outside the present scope. However, since LLaVA-Med reports figures on the standard test splits of the same three datasets used here, an indirect comparison restricted to closed-ended metrics — where the scoring criteria coincide — is presented in Section 4.4.6 (Table 4.4a) of Chapter IV. Open-ended metrics are excluded from comparison because the two scoring schemes differ fundamentally; the grounds and the remaining constraints are discussed in Section 5.3(2) of Chapter V.

### 2.4 Autonomous Hyperparameter Optimization

#### 2.4.1 Traditional HPO (Grid, Random, Bayesian)

The simplest method of hyperparameter optimization (HPO) is grid search, which evaluates every combination on a predefined grid; the number of required evaluations, however, grows exponentially with the dimensionality of the search space. Bergstra and Bengio [12] showed theoretically and empirically that random sampling of combinations can yield substantially better — or equivalent — performance than grid search within the same computational budget, because the effective dimensionality of the hyperparameters that actually influence performance is often low. Bayesian optimization constructs a probabilistic surrogate model of the objective function from previous evaluations and selects the next evaluation point accordingly, aiming to find a better solution with fewer evaluations than grid or random search. Optuna [13], used as the control in this study, is a Bayesian optimization framework adopting the Tree-structured Parzen Estimator (TPE) algorithm, providing a define-by-run API that allows the search space to be defined dynamically at runtime together with efficient pruning strategies.

Among early-stopping-based methods, Hyperband [14] formulates HPO as a pure-exploration bandit problem by repeatedly applying successive halving, allocating little resource (training steps and the like) to unpromising configurations and progressively more to promising ones. For this method to hold, there must be a significant rank correlation between the early learning curve (performance over the first few steps) and final converged performance; Ablation A of Phase 2 in this study (learning curves by data ratio, Section 4.2.2) provides grounds for observing a similar early-signal-to-final-performance relationship within its own experimental context.

#### 2.4.2 LLM-Agent-Based Optimization (autoresearch)

In contrast to Bayesian optimization, which relies on a numerical surrogate model, a scheme in which an LLM agent directly interprets previous experimental logs and proposes the next configuration through natural-language reasoning has recently been discussed as a new alternative for hyperparameter search. The theoretical distinctions of this approach can be summarized in three points: (1) the LLM can transfer knowledge across different tasks and architectures (cross-domain transfer) by drawing on domain knowledge acquired during pretraining; (2) unlike Bayesian optimization, which treats individual hyperparameters as independent variables, it can understand interactions among hyperparameters structurally on the basis of prior knowledge; and (3) it records the reason for changing a configuration explicitly in natural language, providing interpretability and traceability of the search process. The autoresearch-style loop adopted in this study (read previous results → propose the next configuration with a natural-language rationale → execute → record the result, Section 3.7) is the experimental apparatus by which RQ3 tests whether these three distinctions can in fact be combined with performance competitive against Bayesian optimization (Optuna/TPE). This is, however, less an established standard methodology than one instance of an emerging research current on the use of LLM agents for automating scientific experimentation; rather than claiming general superiority for the approach, this study focuses on empirically verifying its competitiveness on the concrete task of medical VQA QLoRA tuning.

### 2.5 Summary of Prior Work and the Distinctiveness of This Study

Taken together, the prior work reviewed above shows that research on medical-specialized VLMs (2.3.3) generally presupposes large models and large domain datasets; research on PEFT methodology (2.2) concentrates on verifying parameter efficiency in general domains; and research on HPO automation (2.4) has rarely compared LLM-agent-based methods directly against established Bayesian optimization on the same task. This study is distinguished from prior work in that, at the intersection of these three currents — **consumer GPU environment, lightweight VLMs, QLoRA domain adaptation, and LLM-based autonomous HPO** — it (1) verifies the zero-shot and fine-tuned medical VQA performance of lightweight VLMs with statistical rigour (RQ1, RQ2), and (2) validates the practical value of autoresearch-style autonomous search by comparing it directly against the industry-standard Optuna (TPE) (RQ3).

---

## Chapter III. Research Method

### 3.1 Overview of the Research Design

This study is designed as empirical research that verifies medical VQA domain adaptation of lightweight VLMs sequentially through a three-phase experiment. Phase 1 measures the zero-shot performance of the four models before fine-tuning in order to answer RQ1 (differences across models). Phase 2 applies QLoRA fine-tuning to verify RQ2 (the effect of fine-tuning) while simultaneously exploring the optimal QLoRA configuration (data scale, LoRA rank, target modules) through ablation studies. Phase 3 takes the optimal configuration identified in Phase 2 as its starting point and compares four HPO strategies — Manual, Random Search, Optuna (TPE), and Autoresearch — to verify RQ3 (the competitiveness of autonomous search). Each phase has a sequential structure in which the results of the preceding phase (best model, optimal QLoRA configuration) become fixed conditions for the next.

### 3.2 Experimental Environment and Tools

#### 3.2.1 Hardware Specification

All experiments reported in this study were conducted on a cloud GPU service (RunPod). Phases 1 and 2 used an RTX 4090 (24GB VRAM) instance and Phase 3 an RTX 3090 (24GB VRAM) instance. The change of GPU in Phase 3 was not a design choice but the result of securing whatever instance was available under tightened budget constraints after institutional cost support proved unavailable. Both are consumer-class cards with 24GB of VRAM, and every Phase 3 trial ran on the same 3090 instance, so the hardware conditions of the strategy comparison were controlled.

A local workstation (RTX 5060 Ti 16GB + RTX 4060 8GB, Ryzen 5 5600X, 32GB RAM) was used only for a preliminary smoke test to size the Phase 3 execution (Section 3.7); **none of the figures reported in this thesis were produced in the local environment.**

Feasibility in the 16GB-class environment targeted by this study was confirmed **from measured VRAM usage rather than by execution on such a card**. Peak VRAM during Phase 2 QLoRA training was at most 14,373MB across the four models (Gemma4-E2B), within the 16GB limit, while the other three models required only 4,015-7,943MB. Because this was not verified by actually running on a 16GB card, the limits of this inference are discussed in Section 5.3(14) of Chapter V.

#### 3.2.2 Software Stack

Model loading and QLoRA fine-tuning used the HuggingFace `transformers` library together with the `unsloth` backend for accelerated 4-bit quantized training, with LoRA adapters configured through the `peft` library. The Bayesian optimization control in Phase 3 used `Optuna` (TPE sampler), and experiment management and logging used `wandb`. BERTScore computation for scoring open-ended responses used the `bert-score` library (roberta-large / BioBERT backbones), and statistical analysis used `scipy` and `statsmodels` (Mixed-Effects Model). The LLM agent of the Autoresearch strategy calls the Anthropic Claude API.

### 3.3 Target Models and Selection Criteria

The four models examined in this study and the grounds for their selection are given in Table 3.1.

**Table 3.1. Target models**

| Model | Parameters | Architectural features | Expected QLoRA VRAM |
|------|---------|-------------|:---:|
| Qwen3-VL-2B | 2B | Thinking mode, DeepStack | ~8-10 GB |
| Qwen2.5-VL-3B | 3B | Dynamic resolution, OCR in 19 languages | ~8-10 GB |
| SmolVLM2-2.2B | 2.2B | HuggingFace lightweight VLM | ~8-10 GB |
| Gemma4-E2B | 2.3B (active) / 5.1B (total) | PLE (Per-Layer Embeddings), Apache 2.0 | ~12-14 GB |

The selection criteria were threefold: (1) QLoRA fine-tuning must be feasible within 16GB of VRAM (an a priori expectation, confirmed post hoc from measured peak VRAM - Section 3.2.1); (2) the license must be Apache 2.0 or MIT so that research use is unrestricted; and (3) the model must have sufficient community and framework support. Gemma4-E2B was included because its Mixture-of-Experts-family PLE technique activates only 2.3B parameters at inference while providing 5.1B-class expressive capacity; the influence of this architectural characteristic on the interpretation of results is discussed separately in Section 5.3 of Chapter V.

### 3.4 Datasets and Preprocessing

The three public medical VQA datasets used in the experiments are summarized in Table 3.2.

**Table 3.2. Target datasets**

| Dataset | Images | QA pairs | Language | Domain | Question types |
|----------|:---:|:---:|:---:|:---:|:---:|
| PathVQA | 4,998 | 32,799 | English | Pathology | Open + Closed (7 types) |
| SLAKE | 642 | 14,028 | English + Chinese | Radiology / CT | Open + Closed |
| VQA-RAD | 315 | 2,248 | English | Radiology | Open + Closed |

Each dataset was used with its official train/val/test split unchanged.

**Contamination control**: PathVQA (2018), SLAKE (2021), and VQA-RAD (2018) were all released before the pretraining cut-off of the target models (2025-2026), so the possibility of pretraining data contamination cannot be excluded. This study measures such contamination actively using the Min-K% Probability Attack [15]. The mean token-level log-probability of the lowest K% (K = 20) of tokens in each sample's gold answer text is used as the contamination indicator — on the theory that samples exposed during pretraining exhibit higher mean probability — and samples in the top 5% within a dataset are classified as suspected contamination, after which the principal conclusion (RQ1) is re-verified on the reduced sample set with those samples removed (see the contamination-robustness verification in Section 4.1.1). The procedure and interpretation criteria (a difference of less than 1%p between original and reduced results is deemed robust, 1-5%p is stated as a limitation, and more than 5%p requires reconsideration of the conclusion) are implemented in `scripts/measure_contamination.py`.

### 3.5 Experiment 1: Zero-Shot Baseline Evaluation (Phase 1)

Zero-shot performance before fine-tuning was measured for 4 models × 3 datasets = 12 conditions. Because evaluation uses greedy decoding, it is deterministic: changing the seed does not change the result, so repeated trials are meaningless. Evaluation therefore used a single seed (42), and uncertainty is reported as a bootstrap 95% confidence interval over the per-sample correct/incorrect decisions of each condition. Since the four models are evaluated on an identical test set, comparison across models has a paired structure; accordingly, Cochran's Q test (binary correctness on a shared test set, H0: equal accuracy) and pairwise McNemar post-hoc tests with Bonferroni correction were used instead of ANOVA, which assumes independent samples.

The measured metrics are closed-ended accuracy (multiple choice), open-ended accuracy (gold-token matching) together with BERTScore F1 (roberta-large), bootstrap 95% CIs for each accuracy, response time (ms per item), and peak VRAM (MB).

### 3.6 Experiment 2: QLoRA Fine-Tuning (Phase 2)

The base QLoRA configuration applied to every Phase 2 condition is given in Table 3.3.

**Table 3.3. Base QLoRA configuration**

| Parameter | Value |
|----------|-----|
| Quantization | NF4 (4-bit NormalFloat) |
| LoRA rank | 64 (determined by Ablation B) |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | all-linear (determined by Ablation C) |
| Learning rate | 2e-4 |
| Batch size | 1 (gradient accumulation 8, effective batch = 8) |
| Optimizer | paged_adamw_8bit |

The training budget targeted 3 epochs, but a ceiling of `max_steps=500` (samples_seen fixed at 4,000 per condition) was imposed because of cloud GPU time and cost constraints. Since the training volume is fixed regardless of dataset size, the small VQA-RAD receives roughly 2 epochs or more while the medium SLAKE and large PathVQA receive less than 1 epoch; the effect of this asymmetry on the interpretation of results is discussed in Section 5.3 of Chapter V.

**Experimental conditions**: 4 models × 3 datasets = 12 conditions, each repeated 3 times (seeds 42/123/456). The values LoRA rank = 64 and target = all-linear were fixed on the basis of the three ablation studies below; detailed results appear in Sections 4.2.2-4.2.4 of Chapter IV.

- **Ablation A (effect of data size)**: PathVQA and Qwen3-VL-2B fixed; training-data ratio 5/10/25/50/100%
- **Ablation B (effect of LoRA rank)**: PathVQA and Qwen3-VL-2B fixed; rank ∈ {4, 8, 16, 32, 64}
- **Ablation C (effect of target modules)**: PathVQA and Qwen3-VL-2B fixed; {q/v_proj} vs {q/k/v/o_proj} vs {all-linear}

**Catastrophic forgetting** is measured in two ways. (A) Change in general capability: the rate of accuracy decrease before and after fine-tuning is measured on a VQAv2 validation subset (2,000 samples) across all 12 conditions. (B) Cross-dataset generalization within the medical domain: evaluation on a dataset other than the training dataset (for example, train on PathVQA → evaluate on SLAKE/VQA-RAD), giving 12 conditions × 2 cross-datasets = 24 additional evaluations. Because PathVQA (pathology) and SLAKE/VQA-RAD (radiology) differ in image domain itself, the results of (B) are interpreted as a domain-generalization gap rather than as catastrophic forgetting in the strict sense (Section 4.2.5).

### 3.7 Experiment 3: Autonomous Hyperparameter Optimization (Phase 3)

The hyperparameter search space shared by the four strategies in Phase 3 is given in Table 3.4.

**Table 3.4. Phase 3 search space**

| Parameter | Search range | Type |
|----------|----------|------|
| lora_rank | {4, 8, 16, 32, 64} | Discrete |
| lora_alpha | rank × {1, 2, 4} | Discrete |
| learning_rate | [1e-5, 5e-4] | Continuous (log scale) |
| batch_size | {1, 2, 4} | Discrete |
| grad_accum_steps | {4, 8, 16} | Discrete |
| warmup_ratio | [0.0, 0.1] | Continuous |
| weight_decay | [0.0, 0.1] | Continuous |
| lora_targets | {minimal, medium, full} | Categorical |

**Four strategies compared**: Manual (the researcher's default values, one run) / Random Search (random sampling) / Optuna TPE (Bayesian optimization) / Autoresearch (autonomous search by an LLM agent). Autoresearch repeats a loop that (1) reads the previous experimental results (results.tsv), (2) proposes the next configuration together with a natural-language rationale (config.yaml + rationale.md), (3) performs a git commit, (4) runs fixed training, (5) evaluates on the validation set, and (6) retains the configuration if performance improves and discards it otherwise.

All trials share the same model (the best model from Phase 2), the same dataset (PathVQA), and a fixed `max_steps=200` in order to control training volume; the wall-clock ceiling `time_budget_min` serves as a safety device to prevent pathological combinations rather than as an experimental control variable. The original design called for Manual 10 + Random Search 400 + Optuna 400 + Autoresearch 400 = 1,210 trials in total (40 trials per strategy × 10 independent repeats). A **preliminary smoke test conducted solely to size the execution** (on the local dual-GPU workstation, for timing measurement only and not reported in this thesis), however, indicated on a wall-clock basis — counting not only training but also validation and final test evaluation time — that the original scale would require roughly 24-25 days even with two GPUs in parallel. The number of repeats (10), which is the unit of statistical testing and the basis of run-level statistical power, was therefore preserved, while the number of search trials per strategy was reduced from 40 to 20, halving the total time to approximately 12.8 days. **The execution scale of the four-strategy comparison is Manual 10 + Random Search 200 + Optuna 200 + Autoresearch 200 = 610 trials in total (20 trials per strategy × 10 independent repeats).** This reduction carries the trade-off of halving the diversity of hyperparameter combinations explored per strategy, but it does not affect the validity of the run-level statistical tests based on 10 independent repeats.

In addition, a **200-trial configuration-consistency re-experiment (Autoresearch-v2)** was conducted separately (Section 4.3.5). This is not part of the four-strategy same-condition comparison but a follow-up intervention designed to test whether the duplicate-proposal phenomenon of Section 4.3.4 dissolves once the prompt-implementation inconsistencies identified in Section 5.3(8) are removed. The search space, model, dataset, `max_steps`, and repetition count were held identical to the original, and only the five configuration inconsistencies were corrected; the original condition was preserved intact for reproducibility. **The total execution scale of Phase 3 is 810 trials.**

**Statistical verification is performed only at the run level, not at the trial level.** Because Autoresearch and Optuna are sequential optimizers, trials within the same run are dependent (the result of trial t influences the proposal at t+1), violating the assumption of independent observations. The unit of testing is the 10 final performance values obtained from the 10 independent repeats of each strategy, analysed with the Kruskal-Wallis test (four-group comparison), the Mann-Whitney U test (pairwise, Autoresearch vs Optuna), and BCa bootstrap 95% CIs. Trial-level data are used only for visualization such as anytime performance curves.

### 3.8 Evaluation Metrics and Statistical Analysis

#### 3.8.1 Dual Reporting of BERTScore

Open-ended responses are reported with both exact match and BERTScore F1. The general-purpose criterion (roberta-large, threshold ≥ 0.7) is the sole decision metric (primary) for accuracy and statistical testing, while the medical-specialized criterion (BioBERT, dmis-lab/biobert-v1.1) is reported alongside as a secondary metric only; no dual gating (requiring both metrics to pass before an answer is judged correct) is applied.

#### 3.8.2 Dual Measurement of Catastrophic Forgetting

Both (A) the change in general capability measured on VQAv2 and (B) the cross-dataset generalization gap described in Section 3.6 are reported, so that side effects of fine-tuning that a single metric cannot capture are measured from multiple angles.

#### 3.8.3 Clinical Significance Analysis (WCA + ECE)

Motivated by the concern that plain accuracy does not fully capture the clinical value of medical AI, Weighted Clinical Accuracy (WCA) is computed using the labels of the seven PathVQA question types (diagnosis, location, measurement, description, temporal, yes_no, unknown).

`WCA = Σ(accuracy per type × weight) / Σweights`

Weights were assigned by clinical importance in the order diagnosis = 1.0 (a diagnostic error directly affects the direction of treatment) > location = 0.8 > measurement = 0.7 > description = 0.6 > temporal = yes_no = 0.5 (binary judgements with limited information content). These weights, however, were set arbitrarily by the researcher without external clinical literature or Delphi consensus; they cannot be interpreted as an absolute scale of clinical importance and are used only in a limited way as a supplementary reference metric complementing the primary metrics (accuracy and BERTScore).

Expected Calibration Error (ECE) [16], a metric measuring how well a model's predicted confidence matches its actual accuracy, was planned for inclusion but could not be computed because the current evaluation pipeline does not store per-sample confidence (see the limitations in Section 5.3 of Chapter V).

#### 3.8.4 Robust Statistics (Bootstrap + Mixed-Effects)

Given the sample-size limitation of n = 9 (3 datasets × 3 seeds), the test of the fine-tuning effect in Phase 2 uses three methods in parallel: (1) a paired t-test with Cohen's d (the conventional comparison), (2) a 95% CI for Cohen's d obtained by BCa bootstrap with 10,000 resamples (robust estimation), and (3) the Wilcoxon signed-rank test (non-parametric). A Mixed-Effects Model pooling the four models (`accuracy ~ condition + dataset`, group = seed) is additionally applied as a supplement; however, because heterogeneous effects across models may cancel in a pooled estimate, the triple verification performed per model is treated as the primary evidence and the MEM result is used only as supporting explanation — a phenomenon actually observed in Section 4.2.1.

#### 3.8.5 Contamination Control (Min-K% Probability)

The methodology is as described in Section 3.4. The robustness verification results are presented in Section 4.1.1 of Chapter IV.

---

## Chapter IV. Experimental Results and Analysis

### 4.1 Phase 1: Zero-Shot Baseline Results

Phase 1 measured the zero-shot performance of four lightweight VLMs (Gemma4-E2B, Qwen2.5-VL-3B, Qwen3-VL-2B, SmolVLM2-2.2B) before fine-tuning on three medical VQA datasets (PathVQA, SLAKE, VQA-RAD), answering RQ1 ("do lightweight VLMs show practical performance on medical VQA without fine-tuning, and are the differences across models significant?"). Evaluation was performed with seed 42 on the full test split of each dataset (PathVQA 6,719 / SLAKE 1,061 / VQA-RAD 451 items).

#### 4.1.1 Performance Comparison across Models

**Table 4.1. Zero-shot baseline results (model × dataset)**

| Model | Dataset | Closed Acc | Open Acc | Overall Acc | BERTScore F1 | Response time (ms) | Peak VRAM (MB) |
|------|:--------:|:----------:|:--------:|:-----------:|:-------------:|:------------:|:-------------:|
| Gemma4-E2B | PathVQA | 0.1633 | 0.0477 | **0.1055** | 0.8069 | 892.6 | 13,932.7 |
| Gemma4-E2B | SLAKE | 0.6394 | 0.4178 | **0.4920** | 0.8679 | 689.8 | 13,927.3 |
| Gemma4-E2B | VQA-RAD | 0.4502 | 0.3100 | **0.3880** | 0.8373 | 721.5 | 13,929.1 |
| Qwen2.5-VL-3B | PathVQA | 0.6130 | 0.0354 | **0.3245** | 0.8613 | 483.4 | 7,581.3 |
| Qwen2.5-VL-3B | SLAKE | 0.7465 | 0.4632 | **0.5580** | 0.9359 | 254.8 | 7,561.9 |
| Qwen2.5-VL-3B | VQA-RAD | 0.6614 | 0.2800 | **0.4922** | 0.8886 | 310.6 | 7,580.5 |
| Qwen3-VL-2B | PathVQA | 0.6336 | 0.0605 | **0.3472** | 0.8487 | 419.1 | 4,527.3 |
| Qwen3-VL-2B | SLAKE | 0.7915 | 0.4575 | **0.5693** | 0.9081 | 344.9 | 4,412.9 |
| Qwen3-VL-2B | VQA-RAD | 0.7211 | 0.2250 | **0.5011** | 0.8894 | 245.7 | 4,428.5 |
| SmolVLM2-2.2B | PathVQA | 0.5892 | 0.0274 | **0.3085** | 0.8557 | 666.0 | 6,021.3 |
| SmolVLM2-2.2B | SLAKE | 0.6648 | 0.3598 | **0.4618** | 0.9130 | 774.7 | 5,991.2 |
| SmolVLM2-2.2B | VQA-RAD | 0.6574 | 0.3150 | **0.5055** | 0.9029 | 756.7 | 5,996.2 |

> Overall Acc denotes accuracy computed over closed-ended and open-ended items combined. Open items are scored by BERTScore (roberta-large backbone, threshold method) and closed items by exact string match against the gold answer.

On pooled accuracy across all datasets, the performance of the four models is given in Table 4.1a (n = 8,231, the total number of items across the three datasets).

**Table 4.1a. Pooled accuracy and statistical tests**

| Rank | Model | Pooled Overall Acc | 95% CI |
|:---:|------|:-------------------:|:------:|
| 1 | **Qwen3-VL-2B** | **0.3843** | [0.3740, 0.3947] |
| 2 | Qwen2.5-VL-3B | 0.3637 | [0.3533, 0.3740] |
| 3 | SmolVLM2-2.2B | 0.3391 | [0.3289, 0.3495] |
| 4 | Gemma4-E2B | 0.1708 | [0.1627, 0.1790] |

Cochran's Q test showed that the correctness patterns of the four models differ significantly on a pooled basis (Q = 1904.28, df = 3, p < .001). Testing each dataset individually, all three were also significant (PathVQA: Q = 2067.08, p < .001 / SLAKE: Q = 71.34, p < .001 / VQA-RAD: Q = 27.18, p < .001).

In pairwise McNemar post-hoc tests with Bonferroni correction, **Gemma4-E2B performed significantly worse than all three other models on the pooled basis and on the PathVQA and VQA-RAD datasets individually** (all such comparisons p(adj) < .005). SLAKE was an exception, where the difference between Gemma4-E2B and SmolVLM2-2.2B was not significant (0.492 vs 0.462, p(adj) = 0.326) — Gemma4-E2B performs relatively better on SLAKE than on the other two datasets, so this single pair is not statistically separable. Significance also varied by dataset among the three top models (Qwen2.5-VL-3B, Qwen3-VL-2B, SmolVLM2-2.2B): all three differed significantly from one another on the pooled basis (p(adj) < .001), yet on SLAKE and VQA-RAD individually the difference between Qwen2.5-VL-3B and Qwen3-VL-2B was not significant (p(adj) = 1 in both cases). In short, **Qwen3-VL-2B is the best model on pooled accuracy, but it is close enough to Qwen2.5-VL-3B that the two are not statistically separable on some datasets.**

**Contamination robustness verification**: Samples suspected of pretraining exposure — identified by the Phase 1.5 Min-K% Probability analysis (K = 20%, outliers in the top 5% within each dataset; 1,020 for PathVQA, 233 for SLAKE, and 73 for VQA-RAD, taken as the union across the four models) — were removed and the same tests were re-run. Cochran's Q remained significant on all three datasets, and **the pooled model ranking and the best model (Qwen3-VL-2B) were preserved** (original 0.3849 → reduced set 0.3041: absolute accuracy falls but the ranking is unchanged). The conclusions of this section are therefore robust to potential data contamination (details: `results/phase1_baseline/phase1_robustness.md`).

#### 4.1.2 Analysis of Dataset Difficulty

The difficulty of the three datasets follows a clear order independent of the model. Comparing the mean Overall Acc of the four models by dataset gives **PathVQA (mean 0.271) < VQA-RAD (mean 0.472) < SLAKE (mean 0.520)**, so PathVQA is markedly harder than the other two.

This gap is especially pronounced on **open-ended items**. On all three datasets, open-item accuracy (0.03-0.46) is far below closed-item accuracy (0.45-0.79), but the drop is most severe on PathVQA: its open accuracy is 0.027-0.061 across all models, more than an order of magnitude below SLAKE (0.360-0.463) and VQA-RAD (0.225-0.315). PathVQA contains a high proportion of items requiring free-form description of detailed findings in pathology tissue images, so generative open-ended responses are interpreted as being harder than on SLAKE and VQA-RAD, which contain relatively more multiple-choice-type items.

Response time and VRAM scale with model size, and differences across datasets are negligible (VRAM variation across datasets within the same model is < 1%). Response time does, however, vary somewhat by model according to item characteristics such as the length of description required — for example, Gemma4-E2B takes 892.6 ms on PathVQA, longer than on SLAKE or VQA-RAD.

#### 4.1.3 Error Type Analysis

For the WCA (Weighted Clinical Accuracy) analysis, PathVQA items (seed 42) were classified into seven clinical question types (diagnosis, location, measurement, description, temporal, yes_no, unknown) and accuracy was computed per type. As discussed in Section 5.3, the weights are a provisional scale without clinical-literature validation and are used here only to identify reference error patterns.

**Table 4.1b. PathVQA accuracy by question type**

| Type | Samples | Gemma4-E2B | Qwen2.5-VL-3B | Qwen3-VL-2B | SmolVLM2-2.2B |
|------|:-------:|:----------:|:--------------:|:-----------:|:--------------:|
| diagnosis | 23 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| location | 433 | 0.0647 | 0.1062 | 0.1478 | 0.0762 |
| measurement | 33 | 0.2121 | 0.0909 | 0.1515 | 0.0909 |
| description | 2,729 | 0.0425 | 0.0224 | 0.0447 | 0.0198 |
| temporal | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| yes_no | 3,362 | 0.1633 | 0.6130 | 0.6336 | 0.5892 |
| unknown | 130 | 0.0692 | 0.0692 | 0.0923 | 0.0154 |

The most striking pattern is that **all four models score exactly 0.0000 on the diagnosis and temporal question types**. Although diagnosis carries the highest WCA weight (1.0) and is clinically the most important type, the models in their zero-shot state were entirely unable to derive a diagnostic name from pathological findings. The description type (free-form narration) also remains at 0.02-0.04 accuracy despite having the largest sample count (2,729 items, 41% of the total), and is confirmed as the principal factor depressing overall PathVQA accuracy.

Conversely, **the yes_no type shows relatively high accuracy** (for example, 0.6336 for Qwen3-VL-2B), and the ranking across models largely matches the overall ranking in Section 4.1.1 (Qwen3-VL-2B > Qwen2.5-VL-3B > SmolVLM2-2.2B ≫ Gemma4-E2B). Gemma4-E2B, however, lags far behind the other three even on yes_no (0.1633 versus 0.59-0.63), confirming that its overall inferiority in Section 4.1.1 is not confined to particular types but extends across all of them.

In summary, zero-shot lightweight VLMs **can cope to some degree with binary discrimination (yes_no) tasks, but are effectively unable to handle clinically important open-ended tasks such as deriving a diagnosis or generating free-form descriptions of findings.** This supports the need for RQ2 (the effect of fine-tuning), and is discussed in connection with how these type-level gaps change after fine-tuning in Section 4.2 (clinical significance analysis in Section 4.2.5).

---

### 4.2 Phase 2: QLoRA Fine-Tuning Results

Phase 2 fine-tuned each of the four models on each of the three datasets with QLoRA (rank = 64, alpha = 128, target = all-linear, ceiling `max_steps=500`) and verified RQ2 ("does domain-specific fine-tuning significantly improve performance over zero-shot?"). In addition to the 36 main conditions (4 models × 3 datasets × 3 seeds), 39 further conditions were run under fixed Qwen3-VL-2B and PathVQA settings for Ablation A (data ratio), B (LoRA rank), and C (target modules), giving 75 conditions in total.

#### 4.2.1 Base vs Fine-Tuned Performance

Per-model performance before and after fine-tuning, together with the paired tests, is given in Table 4.2.

**Table 4.2. Fine-tuning effect by model (paired, n = 9 = 3 datasets × 3 seeds)**

| Model | Base Acc | FT Acc | Cohen's d | d 95% CI (BCa) | paired t-test p | Wilcoxon p |
|------|:--------:|:------:|:---------:|:---------------:|:----------------:|:----------:|
| Qwen2.5-VL-3B | 0.4582 | 0.5749 | **+2.646** | [1.953, 4.723] | < .001 | .0039 |
| Qwen3-VL-2B | 0.4725 | 0.5845 | **+1.620** | [0.932, 3.153] | .0013 | .0039 |
| SmolVLM2-2.2B | 0.4253 | 0.4036 | **-2.284** | [-3.160, -1.552] | < .001 | .0039 |
| Gemma4-E2B | 0.3285 | 0.2288 | -0.652 | [-1.599, 0.032] | .0864 (n.s.) | .1289 (n.s.) |

The effect of fine-tuning is heterogeneous across models. **Qwen2.5-VL-3B and Qwen3-VL-2B improved significantly** (p < .01, large effect size), whereas **SmolVLM2-2.2B significantly deteriorated**, and Gemma4-E2B moved in a negative direction without reaching statistical significance. All three tests (paired t-test, BCa bootstrap Cohen's d, and Wilcoxon) reached consistent conclusions for each model, supporting the robustness of the result.

A Mixed-Effects Model estimated by pooling the four models without distinguishing them (`accuracy ~ condition + dataset`, group = seed) found no significant fixed effect (coefficient = 0.0268, p = .3629, ICC(seed) = 0.0, n = 72). This is not a computational error but a consequence of **heterogeneous effects across models cancelling in the pooled mean**. The per-model triple verification above is therefore taken as the primary evidence for RQ2, and the pooled MEM result is cited only as supporting explanation that "the overall effect without distinguishing models is not significant because of heterogeneity."

#### 4.2.2 Effect of Data Size (Ablation A)

Performance across training-data ratios is given in Table 4.2a.

**Table 4.2a. Performance by training-data ratio (Qwen3-VL-2B, PathVQA, mean of 3 seeds)**

| subset_ratio | Training samples | Overall Acc (mean) |
|:---:|:---:|:---:|
| 0.05 | 982 | 0.4150 |
| 0.10 | 1,965 | 0.4309 |
| 0.25 | 4,913 | 0.4357 |
| 0.50 | 9,827 | 0.4628 |
| 1.00 | 19,654 | **0.5019** |

Accuracy increases monotonically with the training-data ratio, with no sign of reaching a performance ceiling over the interval from 0.05 to 1.0; within the experimental range, **using the full dataset (ratio = 1.0) is optimal**.

> **Bug in the `train_time_min` field (cause identified)**: Although not included in the table above, the `train_time_min` column of `phase2_summary.csv` is unusually large for seed 42 alone under identical conditions (for example, at ratio = 1.0, seed 42 = 369.6 minutes versus about 28 minutes for seeds 123 and 456). Direct comparison against each condition's `train_result.json` showed that `train_runtime_sec`, measured internally by the Trainer, was **consistently normal at 27-29 minutes across all seeds**; the problem lay only in `train_time_min`, which the wrapping script measures by wall clock. The bug appears only in the condition that **first creates** the preprocessing cache for a (model, dataset) combination, where the one-off cost of cache creation is added to wall-clock time; the same pattern was confirmed in an exhaustive check of the 36 main conditions (for example, `qwen25-vl-3b/pathvqa/seed42` at 395.4 minutes versus 44.9 minutes actual). **Training-outcome metrics such as accuracy and loss are unaffected by this bug, and the accuracy-based conclusions of this section are unchanged.** Future analyses requiring time comparisons (such as Phase 3 cost estimation) should use `train_runtime_sec` rather than `train_time_min`.

#### 4.2.3 Effect of LoRA Rank (Ablation B)

Performance across LoRA ranks is given in Table 4.2b.

**Table 4.2b. Performance by LoRA rank (Qwen3-VL-2B, PathVQA, mean of 3 seeds)**

| LoRA rank | Peak VRAM (MB) | Overall Acc (mean) |
|:---:|:---:|:---:|
| 4 | 3,870.6 | 0.4733 |
| 8 | 3,875.2 | 0.4907 |
| 16 | 3,884.4 | 0.5020 |
| 32 | 3,902.8 | 0.5172 |
| 64 | 3,918.8 | **0.5210** |

Performance increases monotonically with rank, though the gain flattens between 32 and 64 (+0.0038 from 32 to 64 versus +0.0152 from 16 to 32), while the VRAM increase across the whole range from rank 4 to 64 is a negligible 1.3% (3,870.6 → 3,918.8 MB). Since the performance gain relative to VRAM cost remains positive, **rank = 64 was adopted** (the setting used in the Phase 2 main experiment).

#### 4.2.4 Effect of Target Modules (Ablation C)

Performance across target-module scopes is given in Table 4.2c.

**Table 4.2c. Performance by target-module scope (Qwen3-VL-2B, PathVQA, mean of 3 seeds)**

| Setting | Target modules | Trainable-parameter ratio | Overall Acc (mean) |
|------|----------------|:---:|:---:|
| minimal | q_proj, v_proj | 0.21% | 0.5015 |
| medium | q/k/v/o_proj | 0.43% | 0.5155 |
| **full** | all-linear | 1.55% | **0.5400** |

Performance increases monotonically as the target-module scope widens (LoRA applied to more linear layers). **full (all-linear) gives the best performance of the three settings** and was adopted as the default configuration of the Phase 2 main experiment (rank = 64, alpha = 128, **target = all-linear**). It should be noted, however, that these three axes (ratio, rank, target) were each verified independently with the other two held fixed; the combination applying all three optima simultaneously was not itself separately verified (see the limitations in Section 5.3).

#### 4.2.5 Catastrophic Forgetting Analysis (VQAv2 + cross-dataset)

**(A) Loss of general capability — measured on a VQAv2 validation subset (2,000 samples)**

The VQAv2 performance degradation rate by model is given in Table 4.2d.

**Table 4.2d. VQAv2 performance degradation rate by model (n = 9 = mean over 3 datasets × 3 seeds)**

| Model | Mean degradation (%) | SD | Range |
|------|:---:|:---:|:---:|
| **Gemma4-E2B** | **51.50** | 4.74 | 43.50 - 57.49 |
| Qwen3-VL-2B | 7.23 | 3.11 | 3.29 - 10.47 |
| Qwen2.5-VL-3B | 4.42 | 3.60 | -0.34 - 8.30 |
| SmolVLM2-2.2B | 0.49 | 0.29 | 0.15 - 0.96 |

The degree of general-capability loss measured on VQAv2 diverges sharply across models. **Gemma4-E2B loses on average 51.5% of its VQAv2 performance after fine-tuning**, exhibiting clear catastrophic forgetting, whereas **SmolVLM2-2.2B shows essentially no degradation (mean 0.49%)**. Interestingly, this ordering runs opposite to the ranking of domain-performance improvement in Section 4.2.1: the Qwen family, which gained most in the domain, gives up some general capability (4-7%), while Gemma4-E2B suffers a double loss — neither significant domain improvement (Section 4.2.1) nor retention of general capability (51.5% loss) — and SmolVLM2-2.2B preserves general capability but significantly deteriorates in the domain. This correlation is an observed pattern; the present study did not separately verify a causal relationship.

**(B) Cross-dataset generalization gap — rate of change when evaluated on a dataset other than the training domain**

The rate of performance change observed on datasets other than the training domain is given in Table 4.2e.

**Table 4.2e. Cross-dataset performance change rate by model (n = 18 = 2 eval sets × 3 training sets × 3 seeds)**

| Model | Mean change (%) | SD | Share positive |
|------|:---:|:---:|:---:|
| Gemma4-E2B | +73.94 | 79.97 | 15/18 |
| Qwen2.5-VL-3B | +0.38 | 4.98 | 9/18 |
| Qwen3-VL-2B | -9.46 | 9.23 | 3/18 |
| SmolVLM2-2.2B | **-31.53** | 17.49 | 0/18 |

As already defined in the research design, PathVQA (pathology tissue) and SLAKE/VQA-RAD (radiological images) differ in image domain itself, so (B) is interpreted as a **domain-generalization gap** rather than catastrophic forgetting in the strict sense. SmolVLM2-2.2B shows a clear fall in performance on other datasets the more it specializes on one (mean -31.5%, negative in all 18 conditions), whereas Gemma4-E2B rises substantially on average (+73.9%), albeit with extreme variability (SD 79.97, ranging from -8.06% to +205.56%). This large positive mean for Gemma4-E2B may be a floor effect arising from the low zero-shot baseline identified in Section 4.1 (about 0.10 on PathVQA in particular): when base accuracy is very low, relative change rates are easily exaggerated in either direction. For all 72 conditions in detail, see `results/phase2_finetune/cross_dataset_cf_summary.md`.

---

### 4.3 Phase 3: Autonomous Hyperparameter Optimization Results

Phase 3 fixed Qwen3-VL-2B — the best performer in Phase 2 — on PathVQA and compared four hyperparameter search strategies (Manual, Random Search, Optuna (TPE), and Autoresearch (LLM agent)) under identical conditions, answering RQ3 ("does an LLM agent's autonomous search reach performance competitive with existing HPO techniques?"). Training volume was controlled at `max_steps=200` for all trials, and the execution scale of the four-strategy comparison was Manual 10 + Random 200 + Optuna 200 + Autoresearch 200 = 610 trials in total (20 trials per strategy × 10 independent repeats; Manual has one trial per repeat). To this is added the 200-trial configuration-consistency re-experiment (Autoresearch-v2) reported in Section 4.3.5, bringing Phase 3 to **810 trials** overall.

As described in Section 3.7, **statistical testing is performed only at the run level, not at the trial level**, because Optuna and Autoresearch are sequential optimizers whose trials within a run are dependent, so individual trials cannot be treated as independent observations. The unit of testing is the **10 per-repeat best val_accuracy values** obtained from the 10 independent repeats of each strategy (for Manual, which has one trial per repeat, that value is itself the run-level value); trial-level data are used only for describing the search process in Sections 4.3.3-4.3.4.

#### 4.3.1 Final Performance Comparison by Strategy (run level)

**Table 4.3. Run-level performance by HPO strategy (n = 10 = per-repeat best val_accuracy over 10 independent repeats)**

| Strategy | n | Mean val_accuracy | 95% CI (bootstrap) |
|------|:-:|:-----------------:|:------------------:|
| **Optuna (TPE)** | 10 | **0.4490** | [0.4368, 0.4594] |
| Random Search | 10 | 0.4186 | [0.4106, 0.4274] |
| Autoresearch (LLM) | 10 | 0.4184 | [0.4064, 0.4328] |
| Manual | 10 | 0.3776 | [0.3760, 0.3794] |

> A run-level value is the highest val_accuracy among the completed trials of each independent repeat. Failed trials (status ≠ completed) are excluded from aggregation. Under the Manual strategy there are records of 10 failures followed by retries in repeats 6-9, but every repeat ultimately completed one trial successfully, so the run-level sample size of 10 is preserved.

The difference in run-level performance among the four strategies was **statistically significant under the Kruskal-Wallis test** (H = 27.92, df = 3, p < .001).

The Mann-Whitney U test for the pairwise comparison central to RQ3, **Autoresearch vs Optuna**, was significant at U = 16.00, p = .0112, rank-biserial r = **-0.68**. Under the sign convention of this study (`run_mann_whitney` in `src/evaluate/statistics.py`), r > 0 means the first sample (Autoresearch) is stochastically superior to the second (Optuna); **a negative r therefore means Optuna is significantly superior to Autoresearch**.

This direction is confirmed independently of the test by the confidence intervals: the lower bound of Optuna's 95% CI (0.4368) exceeds the upper bound of Autoresearch's (0.4328), so **the two intervals do not overlap**. By contrast, Autoresearch (0.4184) and Random Search (0.4186) have practically identical means with largely overlapping intervals, so the two strategies are not statistically distinguishable.

In sum, the ordering among strategies under these experimental conditions is **Optuna > (Random ≈ Autoresearch) > Manual**. That is, **the LLM agent's autonomous search fell significantly short of established Bayesian optimization (TPE) and showed no advantage even over random search.** All three automated strategies including Autoresearch nevertheless exceeded the researcher's manual configuration (Manual, 0.3776), confirming the validity of automated hyperparameter search as such.

#### 4.3.2 Hyperparameters Found by Each Strategy

The hyperparameter configuration of the best-performing trial reached by each strategy is given in Table 4.3a.

**Table 4.3a. Hyperparameter configuration of the best-performing trial per strategy**

| Strategy | rank | alpha | learning_rate | batch | grad_accum | warmup | weight_decay | targets | val_acc | closed | open |
|------|:----:|:-----:|:-------------:|:-----:|:----------:|:------:|:------------:|:-------:|:-------:|:------:|:----:|
| Manual | 16 | 32 | 2.00e-4 | 1 | 8 | 0.030 | 0.010 | minimal | 0.3840 | 0.7213 | 0.0625 |
| Random | 64 | 256 | 2.08e-4 | 2 | 16 | 0.072 | 0.013 | full | 0.4440 | 0.7951 | 0.1094 |
| **Optuna** | 32 | 128 | 4.91e-4 | 4 | 16 | 0.089 | 0.054 | full | **0.4700** | 0.8115 | 0.1445 |
| Autoresearch | 64 | 256 | 2.00e-4 | 4 | 16 | 0.050 | 0.010 | full | 0.4640 | 0.7992 | 0.1445 |

> This table describes the single best configuration reached by each strategy; the sample size is one, and it is not used as evidence for the relative merit of the strategies (which is judged by the run-level tests in Section 4.3.1). The closed and open columns are the `val_closed_acc` and `val_open_acc` of that trial.

In all four strategies, **the gap between closed accuracy (0.72-0.81) and open accuracy (0.06-0.14) persists**. The overall accuracy gain achieved by search (Manual 0.3840 → Optuna 0.4700) arose mainly on closed items, while open accuracy remains at 0.1445 even under the best configuration. This gap shows that the weakness in open-ended responses observed in Phase 1 (Sections 4.1.2, 4.1.3) was not resolved by hyperparameter optimization; the three phases are consolidated on this point in Section 4.4.4.

The three strategies that explored the search space freely (Random, Optuna, Autoresearch) all **converged on `lora_targets=full` (all-linear) with rank in the 32-64 region**. This coincides independently with the finding of Ablations B and C in Phase 2 that rank = 64 and target = all-linear are optimal (Sections 4.2.3, 4.2.4); the fact that two different experimental designs — fixed-axis ablation and free search — point to the same region reinforces the Phase 2 conclusion. The best-performing Optuna configuration, meanwhile, combined rank = 32 with a relatively high learning rate (4.91e-4) and weight decay (0.054), a region the other strategies did not reach.

#### 4.3.3 Search Trajectory Analysis (trial level, descriptive only)

The trial at which each strategy reached its best performance is given in Table 4.3b.

**Table 4.3b. Trial at which the best performance was reached, by strategy (over 10 repeats)**

| Strategy | Final best (median) | Trial of best (median) | IQR of that trial |
|------|:----------------:|:---------------------:|:-------------:|
| Manual | 0.3770 | 1.0 | — |
| Random Search | 0.4150 | 15.0 | [12.2, 17.0] |
| Optuna (TPE) | 0.4550 | 15.0 | [13.5, 18.0] |
| Autoresearch | 0.4060 | 17.5 | [11.5, 20.0] |

Autoresearch reaches its best performance latest, with a median trial of 17.5, and **the upper bound of its IQR touches 20.0, the ceiling of the search budget**. This means that nearly half of the runs were still improving at the point the budget was exhausted, suggesting that a budget of 20 trials may have been insufficient for this strategy — a point directly connected to the reduction from 40 to 20 described in Section 3.7 and discussed as a limitation in Section 5.3. This is, however, observed circumstantial evidence; whether the strategy would actually catch up with Optuna under a larger budget was not verified in this study.

The trial-level performance distribution by strategy is given in Table 4.3c.

**Table 4.3c. Trial-level performance distribution by strategy (all 200 trials, descriptive statistics)**

| Strategy | Trial-level mean | SD | Total training time (min) |
|------|:---------------:|:-------:|:--------------:|
| Manual | 0.3776 | 0.0031 | 228.3 |
| Random Search | 0.3672 | 0.0321 | 5,078.6 |
| Optuna (TPE) | 0.3905 | 0.0374 | 5,967.3 |
| Autoresearch | 0.3980 | **0.0145** | 5,450.0 |

Notably, **Autoresearch has the highest trial-level mean (0.3980) while also having the lowest standard deviation (0.0145)**. That is, the average quality of its individual proposals is superior, but its low variance meant it reached exceptionally good configurations less often. Optuna, conversely, has the largest standard deviation (0.0374). As long as the run-level metric is defined as "the best of 20 trials," a high-variance search is structurally advantaged; the reversal observed in Section 4.3.1 — Autoresearch higher on trial mean but Optuna higher on run-level best — is explained by this difference in variance.

The anytime performance curve (median and IQR of cumulative best performance as trials progress) is in `results/phase3_autoresearch/phase3_anytime.png`, with the underlying figures in `phase3_anytime_curve.csv`.

#### 4.3.4 Search Behaviour of the Autoresearch Agent

To identify the cause of the low variance observed in Section 4.3.3, the number of **unique hyperparameter combinations** proposed by the agent in each repeat was counted; the results are given in Table 4.3d.

**Table 4.3d. Unique hyperparameter combinations per repeat (out of 20 trials)**

| Strategy | Unique combinations per repeat (mean) | Range |
|------|:------------------------:|:----:|
| Random Search | 20.0 / 20 | 20 - 20 |
| Optuna (TPE) | 20.0 / 20 | 20 - 20 |
| **Autoresearch** | **12.5 / 20** | 10 - 17 |

Random and Optuna produced 20 mutually distinct configurations in all 10 repeats, whereas **Autoresearch attempted only 12.5 unique configurations on average.** Roughly 37% of the search budget was thus consumed re-running configurations already tried, and this phenomenon was not confined to particular repeats but was observed consistently across all 10 (minimum 10, maximum 17).

The pattern emerges more clearly in the proposal trajectory of individual repeats. In repeat 8, for example, the agent explored by varying rank 16 → 64 → 32 → 8 → 32 → 64 over the first six trials, then became fixed after the seventh trial on the combination `rank=64, alpha=256, lr=2.0e-4, batch=2, targets=full`, proposing it 11 times consecutively. Notably, **val_accuracy fluctuated between 0.388 and 0.444 while the identical configuration was being re-run** — this variation reflects the stochastic behaviour of training and evaluation rather than any difference in configuration, suggesting that the agent may have interpreted this noise as a performance signal. In that repeat, the agent broke out of the fixation by changing batch_size to 4 only on the final trial, and that configuration recorded the repeat's best performance (0.4640).

In summary, the low performance of Autoresearch appears to stem not from poor quality of the configurations it proposes — its trial-level mean is in fact the highest — but from **a reduced effective search budget caused by repeatedly proposing identical configurations**. This section is, however, a post-hoc observation of result logs and does not directly verify the agent's internal reasoning. The original rationale texts are reproduced in Appendix B.

#### 4.3.5 Re-Experiment with Configuration Consistency Restored (Autoresearch-v2)

Section 4.3.4 attributed Autoresearch's low performance to a reduced effective search budget caused by duplicate proposals, and Section 5.3(8) argued that those duplicates may have originated **not in agent misjudgement but in configuration inconsistencies between the prompt and the implementation**. If that diagnosis is correct, removing the inconsistencies should restore search diversity. This section reports the re-experiment that tested that prediction.

The original condition (`AutoresearchStrategy`) was left untouched to preserve reproducibility of the first run, and a separate condition, `autoresearch_v2`, was added with the configuration corrected (`src/autoresearch/strategies.py`). **The search space, model, dataset, `max_steps=200`, and repetition count are all identical to the original**; the five changes are as follows.

1. Removed the invalid parameter `epochs` from the search space — the implementation discarded the proposed value.
2. Changed the search-phase schedule from absolute trial numbers to **budget proportions** — this conflicted with the hints the code injected.
3. Permitted rationale generation — the original "JSON only" constraint was lifted.
4. Stated the prohibition on duplicate proposals in the prompt — the code rejected duplicates, but the prompt never said so.
5. Injected the actual budget (`total_trials=20`) at the call site — the default of 40 had truncated both the temperature and the phase schedule at their midpoints.

**The intervention worked as intended.** An exhaustive tally of the execution logs confirms that the truncation was resolved (Table 4.3e).

**Table 4.3e. Effect of restoring configuration consistency (measured from execution logs)**

| Metric | Original Autoresearch | Autoresearch-v2 |
|--------|:--------------------:|:---------------:|
| Unique configurations per repetition | 12.5 / 20 (10–17) | **18.90 / 20** (16–20) |
| Lowest temperature reached | 0.66 (truncated from the planned floor of 0.3) | **0.30** (as planned) |
| `exploitation` phase calls | **0** (exploration 166 / transition 266) | **103** (exploration 51 / transition 122) |
| Responses containing a natural-language rationale | 53 / 200 (26.5%) | **200 / 200 (100%)** |

> Rationale presence was determined by whether the raw agent response (`agent_reasoning`) consists solely of a JSON object. Applying this criterion to the original condition reproduces exactly the 147 JSON-only responses (73.5%) reported in Section 5.1, so the two conditions are compared under an identical criterion.

Unique configurations rose from 12.5 to 18.90, approaching the 20.0 of Random and Optuna. In other words, **the duplicate proposals observed in Section 4.3.4 were a product of configuration inconsistency, not an intrinsic limitation of the agent.** This directly supports the diagnosis in Section 5.3(8).

Lifting the "JSON only" constraint also caused **every response to carry a natural-language rationale** (200/200, mean length 620 characters). The second requirement of RQ3 — the provision of an interpretable search rationale — which Section 5.1 had left unresolved as "a design flaw that permits neither a positive nor a negative answer", is thereby **confirmed at least as to whether rationales are produced at all**. What this section establishes, however, stops at the fact that rationales *were produced*; whether they correspond causally to the actual proposals, or meet a quality an expert would accept as grounds for a decision, requires a separate evaluation design that this study did not undertake (Section 5.4).

The run-level performance of the re-experiment condition is given in Table 4.3f.

**Table 4.3f. Run-level performance of Autoresearch-v2 (n=10)**

| Condition | n | Mean val_accuracy | 95% CI (Bootstrap) | Best single |
|-----------|:-:|:----------------:|:------------------:|:-----------:|
| Optuna (TPE) | 10 | 0.4490 | [0.4368, 0.4594] | 0.4700 |
| **Autoresearch-v2** | 10 | **0.4292** | [0.4124, 0.4478] | **0.4780** |
| Autoresearch (original) | 10 | 0.4184 | [0.4064, 0.4328] | 0.4640 |

The results of the pairwise tests are given in Table 4.3g.

**Table 4.3g. Pairwise Mann-Whitney U tests**

| Comparison (reference vs target) | U | p | rank-biserial r | Verdict |
|---|:-:|:-:|:-:|:-:|
| Autoresearch (original) vs Optuna | 16.00 | **.0112** | −0.68 | Significant (Optuna superior) |
| **Autoresearch-v2 vs Optuna** | 31.50 | .1726 | −0.37 | **Not significant** |
| Autoresearch-v2 vs Autoresearch (original) | 66.00 | .2387 | +0.32 | Not significant |

> The Kruskal-Wallis test across all five conditions gives H = 29.72, p < .001 (H = 27.92 for the four-strategy set). Because the re-experiment differs from the original in prompt and constraints, it is reported separately here rather than folded into the four-strategy same-condition comparison of Table 4.3.

The results point **in two directions, and citing only one of them distorts the finding.**

First, **the significant deficit the original showed against Optuna was eliminated.** The original fell significantly short at p = .0112, r = −0.68, whereas v2 is not significant at p = .1726, r = −0.37. The confidence intervals agree: the original's upper bound (0.4328) failed to reach Optuna's lower bound (0.4368), leaving the two intervals disjoint, whereas v2's upper bound (0.4478) exceeds Optuna's lower bound, so **the intervals overlap.** Moreover, v2's best single result of 0.4780 (repetition 9, trial 1030: rank 64, alpha 256, lr 2.5e-4, batch 4, grad_accum 16, targets full; closed 0.8156 / open 0.1562) is **the highest value across all five conditions**, surpassing Optuna's best (0.4700).

Second, it nevertheless **was not established that v2 outperforms the original** (p = .2387). The effect size r = +0.32 is consistent with the direction of improvement but does not reach significance at n = 10. What this re-experiment demonstrates is therefore not "v2 is better than the original" but **"there is no longer a basis for saying it falls significantly short of Optuna"**. The former claim would require a larger sample (Section 5.3(6)).

Third, **restoring diversity did not translate directly into higher performance.** Although unique configurations rose 51%, from 12.5 to 18.90, the run-level mean improved only from 0.4184 to 0.4292. The per-repetition distribution shows why: of the ten repetitions only two exceeded 0.42 — repetition 2 (0.4303) and repetition 9 (0.4230) — while the remaining eight cluster in 0.39–0.41, and the standard deviation of those two repetitions (0.034–0.040) is more than double that of the rest (0.012–0.018). **Securing search diversity increased the opportunity to reach a high-performing configuration but did not reliably guarantee reaching one.** This is the structure behind v2 holding the single best result across all conditions while its run-level mean still trails Optuna.

Fourth, the upper bound of the IQR for the trial at which the best result is reached still touches the budget limit of 20 in v2 as well (median 15.0, IQR [11.0, 20.0]). Even after configuration consistency is restored, **a budget of 20 trials remains insufficient for this strategy**, sustaining the observation from Section 4.3.3.

In sum, this re-experiment confirms the causal diagnosis of Section 4.3.4 without overturning the answer to RQ3. **The configuration inconsistencies were real and removable, and removing them restored search diversity; that alone did not make the LLM agent surpass TPE.**

#### 4.3.6 Summary

This study's answer to RQ3 is **negative**. Under an identical search space and an identical budget of 20 trials, the LLM agent's autonomous search (Autoresearch) **fell significantly short of Optuna (TPE)** (p = .0112, r = -0.68) and was **statistically indistinguishable from random search** (0.4184 vs 0.4186). That all three automated strategies exceeded the manual configuration nevertheless confirms both the validity of automated search itself and the superiority of the established technique (TPE) within it.

As the underlying cause, Section 4.3.4 points to **a reduced effective search budget caused by duplicate proposals** (12.5 of 20 unique configurations). An agent designed on the premise of sequential self-improvement instead fell into early fixation, and evidence was observed that it failed to distinguish the stochastic variation appearing across re-runs of an identical configuration from a genuine improvement signal.

**The re-experiment in Section 4.3.5 confirmed this diagnosis.** Removing the configuration inconsistencies between prompt and implementation restored unique configurations from 12.5 to 18.90, so the duplicate proposals were a product of the configuration rather than an intrinsic limitation of the agent. The negative answer above must therefore be read as applying **not to "LLM agents in general" but to "an agent operated under this particular prompt configuration"** (Section 5.3(8)).

That qualification does not, however, reverse the conclusion. Even in the corrected v2 condition the run-level mean reached only 0.4292, short of Optuna's 0.4490. What changed is that **the deficit shrank to a level that is no longer significant** (p = .0112 → .1726); that v2 is superior to the original was not itself established in this sample (p = .2387). In short, restoring consistency removed the grounds for asserting inferiority without creating grounds for asserting superiority.

This conclusion is limited to the following conditions: a single model (Qwen3-VL-2B), a single dataset (PathVQA), a budget of 20 trials per strategy, and the control condition `max_steps=200`. In particular, as confirmed in Section 4.3.3, a substantial number of Autoresearch runs were still improving when the budget was exhausted, and **this pattern persisted in v2 as well** (Section 4.3.5), so results under a larger search budget could differ (Section 5.3).

From a practical standpoint, cost must also be considered. Total training time for Autoresearch was 5,450 minutes, somewhat less than Optuna's 5,967 minutes, but Autoresearch incurs an additional LLM API cost on every trial. Since it is a configuration that performs worse while costing more, no grounds were found under these experimental conditions for choosing Autoresearch over Optuna.

---

### 4.4 Overall Analysis and Discussion

Sections 4.1 to 4.3 reported the results of each experimental phase separately. This section consolidates the issues that cut across all three.

#### 4.4.1 Constraints on Cross-Phase Comparison

Before the substantive discussion, the scope of comparability must be made explicit. The three phases differ in their evaluation conditions.

| Phase | Training volume | Evaluation target | Evaluation sample |
|------|--------|-----------|-----------|
| Phase 1 | None (zero-shot) | 3 datasets | Full test split (8,231 items) |
| Phase 2 | `max_steps=500` | 3 datasets × 3 seeds | Full test split |
| Phase 3 | `max_steps=200` | PathVQA only | Validation, up to 500 samples |

**Direct comparison of absolute accuracy across phases is therefore invalid.** The difference between the fine-tuned accuracy in Phase 2 (Qwen3-VL-2B, 0.5845) and the best val_accuracy in Phase 3 (0.4700), for example, does not represent a performance drop but arises from differences in training volume (500 vs 200 steps), evaluation dataset (mean of three vs PathVQA alone), and evaluation split. The discussion below cross-references only **the rankings, patterns, and directions of gaps observed within each phase**, not absolute figures.

#### 4.4.2 The Non-Monotonic Relationship between Inference Resource Consumption and Performance

The most practical implication of Phase 1 is that **resource consumption at inference and performance are not monotonically related**.

| Model | Pooled Acc (rank) | Active / total parameters | Peak VRAM |
|------|:---------------:|:---:|:---------:|
| Qwen3-VL-2B | 0.3843 (1st) | 2B / 2B | ~4,500 MB (lowest) |
| Qwen2.5-VL-3B | 0.3637 (2nd) | 3B / 3B | ~7,580 MB |
| SmolVLM2-2.2B | 0.3391 (3rd) | 2.2B / 2.2B | ~6,000 MB |
| Gemma4-E2B | 0.1708 (4th) | 2.3B / **5.1B (MoE)** | ~13,930 MB (highest) |

**The best-performing model (Qwen3-VL-2B) simultaneously uses the least memory and responds fastest** (Sections 4.1.1, 4.1.2). Conversely, Gemma4-E2B, which uses the most memory, ranks last, and its gap from the other three models is statistically pronounced (McNemar post-hoc tests in Section 4.1.1). At this scale, therefore, architecture and pretraining configuration dominate medical VQA performance more than active parameter count or memory consumption, and a choice that satisfies both performance and efficiency exists for resource-constrained environments.

**The axis of interpretation must, however, be made explicit.** Of the four models, only Gemma4-E2B has a Mixture-of-Experts (MoE) structure in which active parameters (2.3B) differ from stored parameters (5.1B) (Section 3.3). This study's selection of "lightweight VLMs" used **active parameters** as its criterion, in line with the objective of feasibility on consumer GPUs, and the discussion above holds on that criterion. **Measured by stored parameters, however, Gemma4-E2B is the largest model**, so its last-place performance may equally be read as "the model with the largest stored scale ranked last" rather than "a small model beat a large one." The conclusion of this section therefore concerns *performance relative to inference-time resource consumption* and must not be extended to a general relationship between parameter scale as such and performance (Section 5.3(11)).

Because the ranking was preserved after removing samples suspected of pretraining exposure in the Phase 1.5 contamination-robustness verification (Section 4.1.1), this conclusion is not explained by data contamination.

#### 4.4.3 Heterogeneity of Fine-Tuning Effects and the Primacy of Model Selection

The most important finding of Phase 2 is that **the proposition "fine-tuning improves performance" does not hold independently of the model** (Section 4.2.1). Notably, its direction coincides with the zero-shot ranking of Phase 1.

| Model | Phase 1 rank | Phase 2 fine-tuning effect (Cohen's d) |
|------|:-----------:|:--------------------------------:|
| Qwen3-VL-2B | 1st | +1.620 (significant) |
| Qwen2.5-VL-3B | 2nd | +2.646 (significant) |
| SmolVLM2-2.2B | 3rd | -2.284 (significant, deterioration) |
| Gemma4-E2B | 4th | -0.652 (not significant) |

The top two models in zero-shot terms improved significantly under fine-tuning, whereas **the bottom two either failed to improve or deteriorated**. This means fine-tuning did not function as a means of lifting a weak base model, and in practical terms suggests that **base-model selection is a decision that precedes fine-tuning design**. Since this study covered only four models, however, whether this correspondence is a general law remains unconfirmed.

Overlaying the catastrophic forgetting analysis (Section 4.2.5) reveals a trade-off. The Qwen family, which gained substantially in domain performance, gave up 4-7% of general capability on VQAv2; SmolVLM2-2.2B, which almost entirely preserved general capability, deteriorated significantly in the domain; and Gemma4-E2B suffered a double loss, losing 51.5% of general capability without any domain gain. **Domain specialization and preservation of general capability were not achieved simultaneously within the scope of this experiment.** As already stated in Section 4.2.5, this is an observed correlation and no causal relationship was verified.

#### 4.4.4 The Closed-Open Gap: A Constraint Unresolved across All Three Phases

The most substantive limitation identified by this study concerns **open-ended response performance**. The gap is observed consistently in all three phases.

- **Phase 1 (zero-shot)**: PathVQA open accuracy was 0.027-0.061 across all models, more than an order of magnitude below closed accuracy (0.45-0.79). By question type, **all four models scored 0.0000 on the diagnosis and temporal types** (Section 4.1.3).
- **Phase 3 (fine-tuning + optimal HPO configuration)**: even the best configuration among 810 trials reached only **val_open_acc 0.1562** against val_closed_acc 0.8156 (the best configuration of the four-strategy comparison gives 0.8115 / 0.1445 and that of the configuration-consistency re-experiment 0.8156 / 0.1562 — the same level on both sides).

As stated in Section 4.4.1, the absolute values of the two phases cannot be compared directly. Yet **the ratio relationship — open below 20% of closed — holds in every phase** (about 9.5% for Qwen3-VL-2B on PathVQA in Phase 1, about 17.8-19.2% for the best configurations in Phase 3). That this ratio persisted even in the re-experiment, which changed the search strategy and removed the configuration inconsistencies, reinforces that **this gap is not a problem resolvable at the level of hyperparameter search**. When both fine-tuning and hyperparameter optimization were brought to bear, closed accuracy rose from 0.63 to around 0.81, while open accuracy still did not exceed 0.15.

This matters because **the clinically important question types are concentrated in the open-ended category**. As confirmed in Section 4.1.3, the diagnosis type carrying the highest WCA weight and the description type with the largest sample count (41% of the total) are both free-form, and in the zero-shot state they scored 0.0000 and 0.02-0.04 respectively. The yes_no type, which showed relatively high accuracy, is by contrast a binary judgement with limited information content.

This study therefore verified directly **whether the accuracy improvement it achieved was concentrated in types of low clinical value**. All 810 Phase 3 trials evaluated the first 500 items of the PathVQA test split, and those 500 items are a subset of the Phase 1 zero-shot evaluation sample (6,719 items), so a type-level paired comparison on an identical sample is available; question-set identity was confirmed across all 810 trials. Type classification used the same function that produced Table 4.1b in Section 4.1.3.

**Table 4.4. Decomposition of the accuracy gain by clinical question type (Qwen3-VL-2B, PathVQA, 500 evaluation items)**

| Type | WCA weight | n | Zero-shot | Best config | Gain | Share of gain |
|------|:---:|---:|---:|---:|---:|---:|
| diagnosis | 1.0 | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0% |
| location | 0.8 | 21 | 0.2381 | 0.7143 | **+0.4762** | 15.4% |
| measurement | 0.7 | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0% |
| description | 0.6 | 211 | 0.0190 | 0.1137 | +0.0948 | 30.8% |
| temporal | 0.5 | 0 | — | — | — | — |
| yes_no | 0.5 | 244 | 0.6721 | 0.8156 | +0.1434 | **53.8%** |
| unknown | 0.5 | 17 | 0.1765 | 0.0588 | −0.1176 | −3.1% |
| **Overall** | — | **500** | **0.3520** | **0.4780** | **+0.1260** | 100% |

> The best configuration is the best trial of Autoresearch-v2 (Section 4.3.5). Share of gain is (type n ÷ 500) × per-type gain, divided by the sum of positive contributions; the per-type contributions sum to the overall gain of +0.1260. The same pattern appears under the best configuration of the four-strategy comparison (yes_no accounting for 53.4-63.2% of the gain) and persists in the mean over the top ten trials of each condition.

**The concentration is confirmed.** The two lowest-weighted types (yes_no at 0.5 and description at 0.6) account for **84.6%** of the total gain, and those two types make up 91% of the evaluation sample. The highest-weighted type, diagnosis, showed no improvement — 0 of 3 items correct both zero-shot and after fine-tuning. Across all five conditions, including the best configuration of the four-strategy comparison, yes_no accounts for more than half of the gain.

This result must not, however, be generalized to "types of high clinical value do not improve," and three qualifications should be read alongside it. First, **the sample sizes of the highest-weighted types fall short of what a judgement requires**: of the 500 evaluation items, diagnosis has 3, measurement has 4, and temporal has none. Their 0% contribution means less that they failed to improve than that the sample cannot decide whether they did. Second, **the per-type gain is in fact largest among the higher-weighted types**: location, weighted 0.8, tripled from 0.2381 to 0.7143, the largest gain of any type. Its contribution is only 15.4% because its sample is 21 items (4.2%), not because of its performance. Third, **weighting by clinical importance does not cancel the gain**: WCA rose from 0.1527 zero-shot to 0.2627 under the best configuration (+0.1099), an increase comparable to the overall accuracy increase (+0.1260).

In sum, **the concentration is real, but its cause lies in the type composition of the evaluation sample rather than in per-type learning difficulty.** The PathVQA test split places 91% of its items in yes_no and description, while diagnosis accounts for just 23 of all 6,719 items (0.34%). Judging performance on the clinically important types would require an evaluation set that deliberately oversamples them instead of using the natural distribution — a constraint at the dataset level rather than in this study's evaluation design (Sections 5.3(7), 5.4).

#### 4.4.5 What Automated Search Reached, and the Limits of Autonomous Agents

Phases 2 and 3 conducted hyperparameter search in entirely different ways and **arrived independently at the same region**.

- **Phase 2 (manual ablation)**: one axis verified at a time with the others fixed → rank = 64, target = all-linear, ratio = 1.0 (Sections 4.2.2-4.2.4)
- **Phase 3 (free search)**: the best configurations of all three strategies (Random, Optuna, Autoresearch) converged on **`lora_targets=full` with rank in the 32-64 region** (Section 4.3.2)

That two approaches with entirely different designs identified the same region raises confidence in the Phase 2 conclusion. In particular, the Phase 2 ablation had the limitation of verifying the three axes only independently, without validating the simultaneous combination of the three optima (Section 4.2.4); Phase 3's free search partially compensates for this limitation by actually exploring that combined region and reaching the same conclusion.

There was, however, a clear boundary to the level of automation. **Automation of the search algorithm (TPE) surpassed manual configuration by a wide margin** (0.4490 vs 0.3776), yet **the LLM agent's autonomous judgement fell significantly short of that algorithm** (Section 4.3.1). The cause identified in Section 4.3.4 is not an absence of predictive ability but **a problem of search behaviour**: the trial-level mean quality of the agent's proposed configurations was in fact the highest (0.3980), but repeatedly proposing identical configurations meant only 12.5 unique combinations on average out of 20 trials, shrinking the effective budget.

The re-experiment in Section 4.3.5 decomposes this boundary one step further. **The search-behaviour problem proved correctable** — removing the configuration inconsistencies restored unique combinations to 18.90/20 and eliminated the significant deficit against Optuna. **Restored diversity did not, however, translate into performance**: the run-level mean of 0.4292 still trailed Optuna (0.4490), and only 2 of 10 repetitions reached the upper band. The remaining gap therefore resembles **a problem of judgement — identifying promising regions and allocating budget toward them — rather than the procedural defect of repeating the same configuration**. In short, within the scope of this study, **"automation of search" is mature and "the agent's procedural defects" proved correctable, but whether "autonomous research judgement" can replace existing optimization algorithms remains unconfirmed.**

#### 4.4.6 Indirect Comparison with Prior Work

This study did not conduct direct experimental comparison with medical-specialized VLMs (Section 2.5). LLaVA-Med [9], however, reports figures on **the standard test splits of the same three datasets**, so indirect comparison is possible within that scope. Before comparing, it is necessary to define **which axes are comparable and which are not**.

- **Closed-ended is comparable.** Both report simple accuracy based on exact string match against the gold answer, and the evaluation samples are identical at 1,061 items for SLAKE and 451 for VQA-RAD (only PathVQA differs, 6,761 vs 6,719, a gap of 42 items).
- **Open-ended is not comparable.** LLaVA-Med uses as its open metric *the proportion of generated responses containing the gold token (recall)*, whereas this study uses *whether BERTScore F1 ≥ 0.7* (Section 3.8.1). The former is structurally a far more lenient measure, so placing the two figures side by side would itself be misleading. They are noted below for reference with **an explicit statement that they are not objects of comparison**.

**Table 4.4a. Indirect comparison with figures reported in prior work (closed-ended basis)**

| Model | Scale | Training approach | PathVQA | SLAKE | VQA-RAD |
|------|:----:|-----------|:-------:|:-----:|:-------:|
| LLaVA (general) | 7B | No domain training | 63.20 | 63.22 | 65.07 |
| LLaVA-Med | 7B | Biomedical corpus pretraining + per-dataset fine-tuning | **91.21** | 85.34 | **84.19** |
| **This study (Qwen3-VL-2B)** | **2B** | **QLoRA fine-tuning only** | 83.12 | **85.26** | 72.91 |

> The LLaVA and LLaVA-Med figures are quoted from Table 4(a) of the original paper. The figures for this study are `closed_acc` for the Phase 2 main conditions (mean of 3 seeds). **Open-ended figures are excluded from the table because the measures differ** (for reference, LLaVA-Med's open figures are PathVQA 37.95 / SLAKE 83.08 / VQA-RAD 61.52 on a token-recall basis, while this study's are 17.15 / 66.95 / 26.00 on a BERTScore-threshold basis).

The most noteworthy result is that **on SLAKE a 2B model reached practically the same closed accuracy as a 7B medical-specialized model** (85.26 vs 85.34). LLaVA-Med was pretrained on a biomedical corpus of PubMed Central scale and then fine-tuned per dataset, whereas this study performed only QLoRA fine-tuning on a consumer-grade GPU without large-scale domain pretraining. Since all three models greatly exceed general-purpose LLaVA without domain training (63.22), this suggests that **for tasks at the level of SLAKE, the advantage of large-scale domain pretraining can be substantially offset by QLoRA adaptation alone**.

On PathVQA (83.12 vs 91.21) and VQA-RAD (72.91 vs 84.19), by contrast, a gap of 8-11%p remains. Given the character of the two datasets this gap is unsurprising: PathVQA is a highly specialized imaging domain of pathology tissue, and VQA-RAD has only 451 items, which disadvantages small-scale fine-tuning. In other words, **the advantage of large-scale domain pretraining remains clearly visible in proportion to the specialization of the task.**

This comparison carries the following constraints, so its conclusions must not be over-extended. (1) The figures for this study are means over three seeds, whereas LLaVA-Med reports a single run. (2) The training budgets differ greatly (this study applies a `max_steps=500` ceiling). (3) The figures are quoted from a published paper rather than reproduced with identical code in an identical environment, so unreported differences in preprocessing, prompting, and so on may exist. Direct reproduction under an identical protocol remains future work (Section 5.3(2)).

#### 4.4.7 Methodological Discussion: The Unit of Aggregation Changes the Conclusion

On two separate occasions in this study, at different junctures, **the choice of aggregation unit reversed the conclusion itself**. This is not incidental to the interpretation of results but an observation worth recording independently.

**(1) Whether to pool models — Phase 2.** A Mixed-Effects Model pooling the four models reported that the fine-tuning effect was not significant (p = .3629). This was not because there was no effect but because **effects in opposite directions across models cancelled in the pooled mean** (Section 4.2.1). Decomposed by model, three of the four show significant effects, two of them negative. Where heterogeneous effects are expected, reporting only the pooled estimate leads to the erroneous conclusion of "no effect."

**(2) Whether to treat trials as independent observations — Phase 3.** Sequential optimization strategies induce dependence among trials within a run, so trial-level testing is invalid (Section 3.7). The two units in fact present opposite pictures: the trial-level mean is higher for Autoresearch (0.3980) than for Optuna (0.3905), yet the run-level best is significantly higher for Optuna (0.4490) than for Autoresearch (0.4184) (Section 4.3.3). Had the test been conducted at the trial level, the opposite conclusion would have been reached.

The observation in Section 4.3.4 further supports **the necessity of a repeated design** directly. When an identical hyperparameter configuration was re-run, val_accuracy varied between 0.388 and 0.444, and this range (about 0.056) is **larger than the mean difference between strategies detected by this study** (Optuna 0.4490 - Autoresearch 0.4184 = 0.031). Judging the relative merit of strategies from a single trial result is therefore impossible in principle, and this study's design of using 10 independent repeats as the unit of testing (Section 3.7) can be regarded as the minimum requirement given the scale of this noise.

---

## Chapter V. Conclusion

### 5.1 Summary of Findings

This study verified, through a three-stage experiment, the full process of adapting lightweight Vision-Language Models (2-3B) to the medical imaging VQA domain on consumer-grade GPUs. Zero-shot baselines (Phase 1), 75 QLoRA fine-tuning conditions (Phase 2), and an 810-trial programme comparing hyperparameter search strategies and re-testing configuration consistency (Phase 3) were conducted on four models (Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B) and three public datasets (PathVQA, SLAKE, VQA-RAD). The results by research question are as follows.

**RQ1 — Do lightweight VLMs differ significantly across models in zero-shot performance?**
The null hypothesis was rejected. The correctness patterns of the four models differed significantly both on a pooled basis (n = 8,231) and in individual tests on all three datasets (Cochran's Q = 1904.28, df = 3, p < .001). Qwen3-VL-2B achieved the highest pooled accuracy at 0.3843, though it was close enough to Qwen2.5-VL-3B to be statistically indistinguishable on some datasets, while Gemma4-E2B was significantly below all three other models. This ranking was preserved after removing samples suspected of pretraining exposure by Min-K% Probability, making it robust to data contamination (Section 4.1.1).

**RQ2 — Does QLoRA fine-tuning significantly improve performance?**
Whether the null hypothesis was rejected **depended on the model.** Performance improved significantly for Qwen2.5-VL-3B (d = +2.646) and Qwen3-VL-2B (d = +1.620), whereas SmolVLM2-2.2B significantly deteriorated (d = -2.284) and Gemma4-E2B was not significant (d = -0.652). A Mixed-Effects Model pooling all four models reported no significant effect (p = .3629), but this reflects not the absence of an effect but **the cancellation of opposing model-level effects in the pooled mean** (Section 4.2.1). The proposition "QLoRA fine-tuning improves medical VQA performance" therefore does not hold independently of the model.

**RQ3 — Does an LLM agent's autonomous search achieve performance competitive with Bayesian optimization while providing interpretable search rationales?**
The null hypothesis (Autoresearch = Optuna) was rejected, but **in the direction opposite to the one the hypothesis anticipated.** In the run-level comparison, Optuna (0.4490) was significantly superior to Autoresearch (0.4184) (Mann-Whitney U = 16.00, p = .0112, r = -0.68), and Autoresearch was statistically indistinguishable even from Random Search (0.4186), its lower-bound reference. All three automated search strategies did, however, exceed the manual configuration (0.3776), confirming the validity of automated search itself.

The second requirement of RQ3, **interpretable search rationales**, **was not tested by the four-strategy comparison.** Because the system prompt given to the agent (Appendix A) explicitly instructed it to "respond with only a JSON object, without explanation or other text," the production of natural-language rationales was never required in the first place. In practice, **147 of 200 trials (73.5%) returned only the hyperparameter JSON, and only 53 (26.5%) included any natural-language narrative.** This was **a design flaw rather than a result** (Section 5.3(8)).

**This study secured part of the answer through a re-experiment that removed that flaw.** In the `autoresearch_v2` condition, with the constraint lifted, all 200 responses carried a natural-language rationale (100%, mean length 620 characters; Section 4.3.5). **That rationales can be produced at all is therefore confirmed.** No qualitative evaluation was performed, however, of whether those rationales correspond causally to the proposals or meet a quality an expert would accept, so the complete answer to whether the agent provides *interpretable* rationales remains open (Section 5.4).

Search quality was meanwhile poor in the four-strategy comparison. The agent attempted on average only 12.5 unique configurations out of 20 trials (Random and Optuna both achieved 20/20), and a pattern of early fixation with repeated proposal of identical configurations was observed across all 10 repeats (Section 4.3.4). **This phenomenon proved to be a product of configuration inconsistency** — in the corrected re-experiment unique configurations recovered to 18.90/20 and the significant deficit against Optuna was eliminated (p = .0112 → .1726). The run-level mean nevertheless still trailed Optuna (0.4292 vs 0.4490) and improvement over the original was not established (p = .2387), so **the negative answer to the first requirement of RQ3 stands** (Sections 4.3.5, 4.3.6).

### 5.2 Contributions

The contributions of this study are fourfold.

**First, it provides three-stage empirical data on medical-domain adaptation of lightweight VLMs.** The full results of 12 zero-shot conditions, 75 fine-tuning conditions, and 810 hyperparameter search trials were measured and released under a single evaluation protocol. In particular, the finding that performance and resource consumption are not monotonically related — the best-performing model simultaneously has the lowest VRAM and fastest response (Section 4.4.2) — provides direct practical grounds for model selection in resource-constrained environments. The indirect comparison in Section 4.4.6 further shows that **on SLAKE closed-ended accuracy, QLoRA adaptation of a 2B model can reach parity with a 7B medical-specialized model (85.26 vs 85.34)**, while a gap of 8-11%p remains on PathVQA and VQA-RAD, confirming that the advantage of large-scale domain pretraining varies with the specialization of the task.

**Second, it quantifies the model-level heterogeneity of fine-tuning effects and demonstrates the pitfall of pooled analysis.** Through triple verification per model, it shows both that an identical fine-tuning procedure can simultaneously produce large improvement and significant deterioration depending on the model, and that estimating these together leads to the misreading of "no effect" (Sections 4.2.1, 4.4.7).

**Third, it reports a negative result for LLM-based autonomous HPO together with a verification of its failure mechanism.** More useful for subsequent design than the result that the autonomous agent fell short of existing techniques is the diagnosis that the cause was not inferior proposal quality but **a reduced effective search budget caused by duplicate proposals** (12.5 of 20 unique configurations). The agent's trial-level mean performance was in fact the highest of the four strategies (Sections 4.3.3, 4.3.4).

This study did not stop at offering that diagnosis but **verified it through a separate 200-trial re-experiment** (Section 4.3.5). Removing five configuration inconsistencies between the prompt and the implementation restored unique configurations to 18.90 of 20 and eliminated the significant disadvantage against Optuna (p = .0112 → .1726). The duplicate proposals observed were thus **empirically shown to be a product of configuration rather than an intrinsic limitation of the agent**. Recovered diversity did not, however, translate into performance: the run-level mean still fell short of Optuna (0.4292 vs. 0.4490) and no improvement over the original condition was demonstrated (p = .2387). **Reporting a negative result while identifying its cause and settling the question of correctability by experiment** is the core of this contribution, and it indicates that future work on LLM-based HPO must treat prompt–implementation consistency as a controlled variable (Section 5.3(8)).

**Fourth, it records two cases in which the choice of aggregation unit reversed the conclusion, together with the grounds for the judgement.** At two junctures — whether to pool models (Phase 2) and whether to treat trials as independent observations (Phase 3) — opposite conclusions followed from the aggregation unit. In particular, the fact that the variation observed when repeating an identical configuration (about 0.056) exceeds the mean difference between strategies detected here (0.031) constitutes a quantitative counterexample to the practice of comparing techniques from single-run results (Section 4.4.7).

### 5.3 Limitations

**(1) Limits of contamination control.** Because the target datasets were released before the pretraining cut-off of the models, pretraining data contamination is possible. This study identified samples suspected of exposure using Min-K% Probability [15] and confirmed the robustness of its conclusions on a reduced set with those samples removed (Section 4.1.1); Min-K% is, however, only an indirect indicator, and complete control or precise quantification of contamination is not possible.

**(2) Absence of direct comparison with medical-specialized VLMs.** Medical-specialized VLMs such as LLaVA-Med, Med-Flamingo, and CheXagent were not reproduced and compared directly in this study's environment (Section 2.5). Table 4.4a in Section 4.4.6 is an indirect comparison quoting figures that LLaVA-Med reports on **the standard test splits of the same three datasets**, and it carries three constraints. First, **comparison holds only for closed-ended metrics, where the scoring criteria coincide** — for open-ended metrics LLaVA-Med uses the proportion of responses containing the gold token (recall) while this study uses a BERTScore F1 threshold, so the measures differ fundamentally. Second, this study's figures are means over three seeds whereas the quoted figures are single-run reports, and the training budgets differ greatly. Third, because the figures were not reproduced with identical code in an identical environment, unreported differences in preprocessing, prompting, and so on remain. Med-Flamingo and CheXagent were excluded from the table because they do not provide figures on these three datasets under the same protocol. Direct reproduction under an identical protocol remains future work.

**(3) Confounding of effective training volume in Phase 3.** Phase 3 sought to control training volume by fixing `max_steps=200` for all trials, but **fixing the number of steps did not fix the number of training samples actually seen.** Because the search space includes `batch_size` (1/2/4) and `grad_accum_steps` (4/8/16), the effective number of training samples (= batch × grad_accum × max_steps) ranged from 800 to 12,800, **a factor of about 16**. The strategy comparison in Section 4.3 therefore mixes differences in effective training volume with hyperparameter quality. In particular, the inferiority of Manual, which used a fixed configuration (batch 1 × grad_accum 8 → 1,600 samples), may partly reflect a training-volume disadvantage. The best configurations of Optuna and Autoresearch were, however, both batch 4 × grad_accum 16 (12,800 samples), so the difference between these two strategies — the comparison central to RQ3 — is not explained by this confound. Since `results.tsv` records batch_size, grad_accum_steps, and max_steps for every trial, effective training volume can be computed post hoc, though the value itself was not reported as a column.

**(4) Non-determinism of the LLM agent.** Because Autoresearch depends on an external LLM API, complete reproducibility is not guaranteed. **This study originally stated that temperature was fixed at 0; post hoc verification established that this was not the case.** The implementation (`src/autoresearch/agent.py`) applies a schedule that lowers the temperature as the trial budget is consumed, and an exhaustive tally of the execution logs (432 API calls) shows that the temperature actually used ranged **from 1.0 down to 0.66**. The `temperature` column of `results.tsv` reads 0 for every trial because that field is never populated along the recording path, leaving the default value in place; it is not the value actually passed to the API (see (8) for details). Agent calls were therefore stochastic at every trial, and **control of variability rests entirely on the 10 independent repeats.** Beyond this, the possibility of results changing due to model updates on the API side also lies outside its control.

**(5) Limited generalizability of the ablation results.** Ablations A, B, and C in Phase 2 were all conducted under the single condition of PathVQA and Qwen3-VL-2B (Sections 4.2.2-4.2.4). The supplementary rank verification on SLAKE planned at the design stage **could not be carried out** because of GPU time constraints. Whether the conclusion rank = 64, target = all-linear, ratio = 1.0 extends to other datasets and models is therefore unverified. In addition, the three axes were each verified independently with the others fixed, and the simultaneous combination of the three optima was not separately verified (Section 4.2.4). That Phase 3's free search converged independently on the same region partially compensates for this limitation (Section 4.4.5).

**(6) Limits of statistical power.** The test of the fine-tuning effect in Phase 2 has n = 9 (3 datasets × 3 seeds), and the run-level test in Phase 3 has n = 10 per strategy. Robustness was secured through triple verification (BCa bootstrap, mixed-effects, Wilcoxon) and a repeated run-level design, but the sample sizes themselves make the confidence intervals for effect sizes wide (for example, a Cohen's d 95% CI spanning [0.932, 3.153] in Section 4.2.1).

**(7) Indirectness of the clinical significance evaluation.** The per-question-type weights of Weighted Clinical Accuracy (WCA) are a provisional scale assigned by the researcher without external clinical literature or expert consensus (Section 3.8.3) and cannot be interpreted as an absolute scale of clinical importance. Expected Calibration Error (ECE) could not be computed because the current evaluation pipeline does not store per-sample confidence. The possibility raised in Section 4.4.4 — that accuracy gains were concentrated in types of low clinical value — was verified by a type-level decomposition over all 810 trials (Table 4.4), and the concentration itself is confirmed. The decisive constraint on that verification, however, lies in the type composition of the evaluation sample: among the 500 evaluation items, diagnosis has 3, measurement has 4, and temporal has none, so **performance on the highest-weighted types is effectively undetermined**. This constraint originates in the type distribution of PathVQA itself (diagnosis is 23 of all 6,719 items, 0.34%) rather than in this study's evaluation design, and removing it would require a separate evaluation set that oversamples the clinically important types (Section 5.4).

**(8) Internal inconsistencies in the autonomous agent's configuration.** The most serious limitation identified post hoc is that the Autoresearch condition contains the following four configuration inconsistencies. These are stated explicitly because they prevent the negative result of Section 4.3 from being generalized as an "intrinsic limitation of LLM agents."

- **The prompt forbade the production of search rationales.** The system prompt (Appendix A) instructs the agent to "respond with only a JSON object, without explanation, markdown, or other text." As a result, 73.5% of the 200 trials returned only JSON. The "interpretable search rationale" required by RQ3 was excluded by design before it could become an object of measurement.
- **The search schedule conflicts with the actual budget.** The prompt defines search stages by absolute trial number — early exploration 0-5, mid exploitation 5-20 ("take the best configuration and change only 1-2 parameters"), late refinement 20+. The actual budget, however, is 20 trials per repeat, so **the late refinement stage never triggered and roughly 75% of the budget was spent in the interval instructed to "vary only around the best configuration."** The duplicate proposals observed in Section 4.3.4 (12.5 of 20 unique configurations) may therefore be the result of faithfully following this instruction rather than a failure of the agent's judgement. The stage-transition logic on the code side (`src/autoresearch/agent.py`) is based on progress ratios (0.25/0.75) and should in principle adapt to the budget, but because of the un-injected `total_trials` described in the fourth item below, **the code-side schedule was in fact truncated as well** — an exhaustive tally of the 432 calls in the execution logs records the `exploitation` stage **not once** (exploration 166, transition 266). The final stage therefore never triggered on either the prompt side or the code side.
- **Ineffective parameters were presented as search targets.** The prompt includes `epochs` in the search space and even offers the guidance that "more epochs (3-5) help when data are limited," yet the implementation (`src/autoresearch/agent.py`) discards the proposed `epochs` and fixes `max_steps=200`. Cases appear in the actual logs where the agent diagnoses that "all trials have only 200 steps, so this looks like undertraining" or that "epochs were not changed" (Appendix B); these were accurate diagnoses, but the corresponding adjustment lever never worked in the first place.
- **The search budget was never injected into the code.** The strategy object carries a `total_trials` default of 40, and the call site (`src/autoresearch/run_phase3.py`) did not pass the actual budget of 20. Progress consequently stopped at a maximum of 0.487, so **both the temperature schedule and the stage schedule were truncated at their midpoints** — the temperature descended only to 0.66 rather than the intended floor of 0.3 (see (4)), and the `exploitation` stage never triggered. This truncation was invisible in the recorded results because the `phase` and `temperature` fields declared on `TrialResult` are never populated along the construction path (`src/autoresearch/loop.py`), leaving both columns at their defaults (`""`, `0.0`) for every trial; the actual values are recoverable only from the execution logs. Whereas the preceding three items are inconsistencies between prompt and implementation, this one is an inconsistency internal to the implementation, and it rules out explaining the duplicate proposals of Section 4.3.4 by insufficient temperature (reduced sampling diversity) — the actual temperature remained in a high range (1.0-0.66).

All four items apply only to the Autoresearch condition and not to the Random or Optuna conditions. The comparison in Section 4.3.1 must therefore be interpreted narrowly as **"a comparison between an agent operated with this prompt configuration and existing algorithms"** rather than as "an algorithm comparison over an identical search space."

**This study did not merely note the limitation but verified it through a re-experiment** (Section 4.3.5). In the `autoresearch_v2` condition, which corrected the four items above plus the unstated prohibition on duplicates, the truncated temperature schedule descended to its planned floor of 0.30, the `exploitation` phase that had never fired was recorded 103 times, and unique configurations per repetition recovered from 12.5 to 18.90. **The duplicate proposals observed in Section 4.3.4 were thus confirmed to be a product of configuration inconsistency rather than an intrinsic limitation of the agent.** Since the run-level mean still fell short of Optuna even after consistency was restored (0.4292 vs 0.4490), however, this limitation **narrows the interpretive scope of the negative result in Section 4.3 without overturning the result itself.** Because the re-experiment differs from the original in prompt and constraints, it was not folded into the four-strategy same-condition comparison table of Section 4.3.1.

**(9) Absence of multiple-comparison correction.** This study performed more than 20 statistical tests across Phase 1 (Cochran's Q + McNemar), Phase 2 (paired t-test, Wilcoxon, bootstrap, and mixed-effects in parallel), and Phase 3 (Kruskal-Wallis + Mann-Whitney). **Bonferroni correction was applied only to the McNemar post-hoc tests in Phase 1; no integrated multiple-comparison correction spanning the whole pipeline was applied.** Applying a significance level of 0.05 independently to each test accumulates family-wise error rate and increases the risk of chance significance (Type I error), so individual p-values must be interpreted with this in mind. The principal conclusions of this study are, however, supported not by single p-values but by triple verification (Phase 2) or non-overlapping confidence intervals (Phase 3, Section 4.3.1), so the possibility that the absence of correction reverses a conclusion is limited. Pipeline-level FDR correction is left as future analytical work.

**(10) The nature of the cross-dataset results.** The cross-dataset performance change in Section 4.2.5(B) is not catastrophic forgetting in the strict sense (the phenomenon of losing an ability possessed before fine-tuning). Because PathVQA (pathology tissue) and SLAKE/VQA-RAD (radiology) differ in image domain itself, this metric is closer to **a domain generalization gap predictable from domain specialization**. The thesis states this explicitly in Section 4.2.5, and strict determination of catastrophic forgetting is confined to interpretation of the (A) VQAv2 metric.

**(11) Architectural heterogeneity of Gemma4-E2B.** Of the four models evaluated, **only Gemma4-E2B has a Mixture-of-Experts (MoE) structure**, activating 2.3B at inference while its stored parameters reach 5.1B (Section 3.3). The other three are dense structures whose active and total parameters coincide. Since this study's "lightweight VLM" selection criterion took **active parameters** as its basis, in line with the research aim of consumer-GPU feasibility, the selection itself is consistent; but the discussion of the "non-monotonic relationship between scale and performance" in Section 4.4.2 **must not be extended into a scale comparison on a stored-parameter basis** — by stored parameters Gemma4-E2B is the largest model, so its last-place performance may equally be read as "the largest stored scale ranked last." The conclusion of Section 4.4.2 (non-monotonicity with respect to active parameters and VRAM) holds, but the axis of interpretation must be stated.

**(12) Structural constraint of the training-budget ceiling (`max_steps` cap).** Phase 2 applied a ceiling of `max_steps=500` because of GPU time constraints (Section 3.6), which **fixes training volume regardless of dataset size**. Consequently the small VQA-RAD is trained for roughly 2 epochs or more while the large PathVQA does not reach 1 epoch. The dataset-level performance gaps observed in Sections 4.2 and 4.4.6 may therefore partly reflect this difference in effective training volume; in particular, the interpretation in Section 4.4.6 of the PathVQA and VQA-RAD gap in terms of "task specialization" cannot exclude the alternative explanation of differing training budgets. Full-epoch retraining per dataset is left as follow-up work.

**(13) Impact of the reduced Phase 3 search budget.** As described in Section 3.7, the number of search trials per strategy was reduced from the original 40 to 20. Since the number of repeats (10), the unit of run-level testing, was preserved, the validity of the statistical tests is unaffected; however, **halving the number of hyperparameter combinations each strategy could explore may have led to an underestimate of attainable performance relative to the original design**. This impact is expected to be larger for sequential optimization strategies (Optuna, Autoresearch), and indeed a substantial number of Autoresearch runs were still improving when the budget was exhausted (Section 4.3.3).

**(14) Absence of verification on actual 16GB hardware.** This study set out to demonstrate "domain adaptation on consumer-grade GPUs," yet all execution took place on 24GB cards (RTX 4090 for Phases 1-2, RTX 3090 for Phase 3; Section 3.2.1). Feasibility in the 16GB class was **inferred** from the fact that measured peak VRAM (at most 14,373MB during Phase 2 training) falls within the 16GB limit; it was not verified by running directly on a 16GB card. In a real 16GB environment, usable VRAM is smaller than the nominal capacity because of the OS and display output, and fragmentation behaves differently, so an out-of-memory risk cannot be excluded for Gemma4-E2B in particular, whose headroom is under 2GB. The other three models (4,015-7,943MB) have ample margin. Reproduction on actual 16GB hardware is left as follow-up work.

**(15) External service failure during the re-experiment, and re-runs.** In the `autoresearch_v2` execution of Section 4.3.5, **8 of 200 trials failed before training began**. The cause lay neither in the hyperparameters nor in the training code, but in a 120-second timeout that occurred while the fine-tuning library used (Unsloth) transmitted execution-environment statistics to an external hub immediately before training. The model weights required were already in the local cache, so this call is unrelated to training. After disabling that telemetry, **the 8 trials were re-run under identical conditions and all completed normally**; the final aggregation is based on the 200 completed trials. The re-run trials received exactly the temperature and phase schedule they would originally have been assigned (because the schedule is computed from the count of completed trials), and the duplicate check consults the full history, so these re-runs do not distort the unique-configuration metric or the search trajectory of Section 4.3.5. That **dependence on an external service was a real cause of experimental failure**, however, is worth recording for the reproducibility of autonomous HPO pipelines.

### 5.4 Directions for Future Research

**First, autonomous HPO should be re-verified with an expanded search budget.** For Autoresearch, the upper bound of the IQR for the trial at which the best performance was reached touches the budget ceiling (20), meaning that nearly half of the runs were still improving when the budget ran out (Section 4.3.3). Whether the gap against Optuna persists or reverses at a scale of 40-100 trials is a question this study cannot answer.

**Second, autonomous HPO with its consistency restored should be re-evaluated under a sufficient search budget.** The re-experiment correcting the four inconsistencies identified in Section 5.3(8) plus the unstated duplicate prohibition was already carried out in this study (Section 4.3.5), and it restored search diversity from 12.5 to 18.90, **confirming that the duplicate proposals were a product of the configuration**. The remaining question, however, became sharper. While v2 eliminated the significant deficit against Optuna (p = .0112 → .1726), it **did not establish improvement over the original** (p = .2387, r = +0.32), and the upper bound of the IQR for reaching the best result still touched the budget limit of 20. Two things are needed: **a larger sample** (an effect of the magnitude r = +0.32 is not detectable at n = 10; Section 5.3(6)) and **a larger search budget** (which couples with the first item above). Designing a search policy that reliably converts secured diversity into high-performance attainment — thereby reducing the between-repetition variance observed in Section 4.3.5, where only 2 of 10 repetitions reached the upper band — also remains follow-up work. Introducing a decision criterion that distinguishes stochastic variation across re-runs from genuine improvement (see the noise scale in Section 4.4.7) warrants examination as well.

**A qualitative evaluation framework for search rationales is also needed.** The rationale production rate rose from 26.5% to 100% in the re-experiment (Section 4.3.5), but what this study confirmed stops at the fact that rationales *were produced*. Answering whether they are *interpretable*, as RQ3 requires, demands measures for the causal correspondence between a rationale and the actual proposal, for whether it amounts to post-hoc rationalization, and for expert acceptability. No such measures are currently established, and they remain an independent research topic in the evaluation of autonomous research agents.

**Third, improving open-ended response performance is the most urgent task.** Even with both fine-tuning and hyperparameter optimization applied, open accuracy did not exceed 20% of closed accuracy (Section 4.4.4), and the clinically important diagnosis and description types are concentrated there. Redesign of the training objective or evaluation metric to target generative responses themselves is required.

Beyond this, **building an evaluation set that oversamples the clinically important types** was identified as a prerequisite task. The type-level decomposition in Section 4.4.4 showed that 84.6% of the accuracy gain occurred in yes_no and description, the two lowest-weighted types, while at the same time revealing that diagnosis — the highest-weighted type — appears in only 3 of the 500 evaluation items, leaving its performance undeterminable. For an evaluation to reflect clinical value, the protocol must deliberately oversample the important types rather than adopt the dataset's natural distribution.

**Fourth, a re-experiment controlling effective training volume is needed.** Eliminating the 16-fold difference identified in Section 5.3(3) requires fixing the total number of training samples instead of `max_steps`, or including effective training volume as a covariate in the analysis.

**Fifth, comparison with medical-specialized VLMs under an identical protocol and the establishment of a clinically validated evaluation scheme are needed.** The absent comparison in Section 5.3(2) and the provisional WCA weights and uncomputed ECE in Section 5.3(7) are all items to be resolved in follow-up research.

---

## References

This list contains only works actually cited in the text, formatted in IEEE style. The bracketed numbers in the text correspond to the item numbers below, assigned in order of first citation. The authors, venue, volume, pages, and DOI of each entry were verified in August 2026 against the original sources (arXiv abstract pages, publisher pages, PubMed bibliographic records, and the official JMLR pages).

[1] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual instruction tuning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023, arXiv:2304.08485.

[2] S. Bai et al., "Qwen2.5-VL technical report," arXiv:2502.13923, 2025.

[3] A. Marafioti et al., "SmolVLM: Redefining small and efficient multimodal models," arXiv:2504.05299, 2025.

[4] E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2022, arXiv:2106.09685.

[5] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023, arXiv:2305.14314.

[6] X. He, Y. Zhang, L. Mou, E. Xing, and P. Xie, "PathVQA: 30000+ questions for medical visual question answering," arXiv:2003.10286, 2020.

[7] B. Liu, L.-M. Zhan, L. Xu, L. Ma, Y. Yang, and X.-M. Wu, "SLAKE: A semantically-labeled knowledge-enhanced dataset for medical visual question answering," in *Proc. IEEE 18th Int. Symp. Biomedical Imaging (ISBI)*, 2021, pp. 1650-1654, doi: 10.1109/ISBI48211.2021.9434010.

[8] J. J. Lau, S. Gayen, A. Ben Abacha, and D. Demner-Fushman, "A dataset of clinically generated visual questions and answers about radiology images," *Scientific Data*, vol. 5, art. no. 180251, 2018, doi: 10.1038/sdata.2018.251.

[9] C. Li et al., "LLaVA-Med: Training a large language-and-vision assistant for biomedicine in one day," in *Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*, 2023, arXiv:2306.00890.

[10] M. Moor et al., "Med-Flamingo: A multimodal medical few-shot learner," in *Proc. 3rd Machine Learning for Health Symp. (ML4H)*, PMLR, vol. 225, pp. 353-367, 2023, arXiv:2307.15189.

[11] Z. Chen et al., "A vision-language foundation model to enhance efficiency of chest X-ray interpretation," arXiv:2401.12208, 2024.

[12] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *Journal of Machine Learning Research*, vol. 13, no. 10, pp. 281-305, 2012.

[13] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in *Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining (KDD)*, 2019, pp. 2623-2631, doi: 10.1145/3292500.3330701.

[14] L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, and A. Talwalkar, "Hyperband: A novel bandit-based approach to hyperparameter optimization," *Journal of Machine Learning Research*, vol. 18, no. 185, pp. 1-52, 2018.

[15] W. Shi et al., "Detecting pretraining data from large language models," in *Proc. 12th Int. Conf. Learning Representations (ICLR)*, 2024, arXiv:2310.16789.

[16] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, PMLR, vol. 70, pp. 1321-1330, 2017.

> **Note on style**: Entries [2], [3], [4], [9], [10], [11], and [15] have more than six authors and are therefore abbreviated with "et al." per IEEE convention; the complete author lists are available in the original sources. Entries [2], [3], [6], and [11] are technical reports or preprints with no conference version. The first version of [11] was titled "CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation"; the title was changed in the December 2024 revision. The CheXagent referred to in Section 2.5 is this work.

---

## Appendix

### Appendix A. System Prompt of the Autoresearch Agent

The following is the full text of `configs/autoresearch/program.md`. It is reproduced verbatim because it is the evidence for the three prompt-side configuration inconsistencies identified in Section 5.3(8) (the fourth item, the un-injected `total_trials`, is a defect in the calling code and therefore does not appear in this text). **The annotated points are where the problems arise.**

```markdown
# Autonomous HPO Agent - System Prompt

You are an autonomous hyperparameter optimization agent for medical VQA
fine-tuning research.

## Task
Given the history of previous QLoRA fine-tuning experiments, suggest the NEXT
hyperparameter configuration that is most likely to improve validation accuracy
on the PathVQA medical VQA dataset.

## Search Space
| Parameter        | Range                       | Type                     |
|------------------|-----------------------------|--------------------------|
| lora_rank        | {4, 8, 16, 32, 64}          | discrete                 |
| lora_alpha       | rank x {1, 2, 4}            | discrete                 |
| learning_rate    | [1e-5, 5e-4]                | continuous (log-scale)   |
| batch_size       | {1, 2, 4}                   | discrete                 |
| grad_accum_steps | {4, 8, 16}                  | discrete                 |
| warmup_ratio     | [0.0, 0.1]                  | continuous               |
| weight_decay     | [0.0, 0.1]                  | continuous               |
| lora_targets     | {"minimal","medium","full"} | categorical              |
| epochs           | {1, 2, 3, 5}                | discrete   <- (i) inert  |

Where: minimal = [q_proj, v_proj] / medium = [q_proj, k_proj, v_proj, o_proj]
       full = all linear layers

## Strategy Guidelines
1. Early exploration (trials 0-5): Try diverse configurations to map the
   landscape. Vary multiple parameters at once.
2. Mid exploitation (trials 5-20): Focus on promising regions. Take the best
   configuration and vary 1-2 parameters at a time.     <- (ii) budget mismatch
3. Late refinement (trials 20+): Fine-tune around the best configuration with
   small perturbations.                                 <- (ii) never triggered

## Key Insights for Medical VQA
- Medical images benefit from higher LoRA ranks (16-64).
- Learning rate is often the most sensitive parameter.
- `medium` or `full` target modules often outperform `minimal`.
- Effective batch size = batch_size x grad_accum_steps. Keep in 4-16 range.
- More epochs (3-5) help when training data is limited.  <- (i) inert parameter
- Warmup ratio 0.03-0.06 is generally safe.

## Response Format
Respond with ONLY a valid JSON object. No explanation, no markdown fences,
no other text.                                    <- (iii) rationale prohibited
```

**(i) Inert parameter**: `epochs` appears in both the search space and the guidance, yet the implementation (`src/autoresearch/agent.py`) discards the proposed value and fixes `max_steps=200`.
**(ii) Budget mismatch**: the actual budget is 20 trials per repeat, so "Late refinement (trials 20+)" never triggers, and about 75% of the budget falls in the interval instructing the agent to "take the best configuration and vary 1-2 parameters."
**(iii) Rationale prohibited**: the prompt explicitly excludes the "interpretable search rationale" required by RQ3.

**Agent execution settings**: model `claude-sonnet-4-6`, `max_tokens=512`. The temperature was not a fixed value but a schedule that decreases with progress; an exhaustive tally of the execution logs (432 calls) shows it ranged **from 1.0 down to 0.66**. The 0 recorded in the `temperature` column of `results.tsv` is an unpopulated default and not the value actually passed to the API (Section 5.3(8)). Mitigation of API non-determinism rests on the 10 independent repeats (Section 5.3(4)).

### Appendix B. Excerpts from the Autoresearch Rationale Logs

Excerpted from the original text of the `agent_reasoning` column of `results.tsv`. Of the 200 completed trials, **147 (73.5%) returned only JSON as in (1) below**, while 53 (26.5%) included natural-language narrative (Sections 5.1, 5.3(8)).

**(1) Typical response — configuration only, no rationale (73.5% of the total)**

```json
{"lora_rank": 64, "lora_alpha": 256, "learning_rate": 2.0e-4, "batch_size": 2,
 "grad_accum_steps": 8, "warmup_ratio": 0.05, "weight_decay": 0.01,
 "lora_targets": "full", "epochs": 3}
```

Because the prompt prohibited explanation (Appendix A (iii)), this form is the response consistent with the instruction.

**(2) A case including natural-language narrative — accurate diagnosis, but an inoperative lever**

The agent diagnosed that "all trials have only 200 steps, so this looks like undertraining" and that "epochs were not changed." The diagnosis was accurate, but as noted in Appendix A (i) the `epochs` lever it proposed was discarded by the implementation and therefore never took effect.

**(3) Repeated proposal of an identical configuration**

In repeat 8, from trial 602 onward the agent proposed the identical configuration 11 times in succession, while the measured val_accuracy varied between 0.388 and 0.444. This variation reflects the stochastic behaviour of training and evaluation rather than any difference in configuration (Sections 4.3.4, 4.4.7).

> The full logs are in the `agent_reasoning` column of `results/phase3_autoresearch/results.tsv` and in `rationale.md` within each trial directory.

---

## Abstract (in Korean)

### 경량 멀티모달 모델의 의료 영상 VQA 도메인 적응: QLoRA 파인튜닝과 자율 하이퍼파라미터 최적화

황태욱
융합정보기술학과 인공지능전공
건국대학교 정보통신대학원

본 연구는 소비자급 GPU 환경에서 경량 Vision-Language Model(VLM)을 의료 영상 Visual Question Answering(VQA) 도메인에 적응시키는 전 과정을 세 단계 실험으로 검증한다. 20억~30억 파라미터급 공개 모델 4종(Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B)과 공개 의료 VQA 데이터셋 3종(PathVQA, SLAKE, VQA-RAD)을 대상으로, 제로샷 베이스라인 12조건(Phase 1), QLoRA 파인튜닝 75조건(Phase 2), 하이퍼파라미터 탐색 전략 비교와 설정 정합성 재실험 810 trial(Phase 3)을 동일한 평가 프로토콜 아래 수행했다.

제로샷 성능은 모델 간 통계적으로 유의하게 달랐으며(Cochran's Q = 1904.28, df = 3, p < .001), Qwen3-VL-2B가 합산 정확도 0.3843으로 가장 높았다. 이 순위는 Min-K% Probability로 식별한 사전훈련 노출 의심 표본을 제거한 뒤에도 유지되어 데이터 오염에 강건했다. QLoRA 파인튜닝의 효과는 모델에 따라 방향이 엇갈렸다. Qwen2.5-VL-3B(Cohen's d = +2.646)와 Qwen3-VL-2B(d = +1.620)에서는 유의하게 향상된 반면 SmolVLM2-2.2B(d = -2.284)에서는 유의하게 악화되었고, 네 모델을 합산한 혼합효과모형은 상반된 효과가 상쇄되어 "효과 없음"으로 보고했다(p = .3629). 자율 하이퍼파라미터 탐색에서는 대규모 언어모델 에이전트(0.4184)가 베이지안 최적화(0.4490)보다 유의하게 낮았고(Mann-Whitney U = 16.00, p = .0112, r = -0.68) 무작위 탐색(0.4186)과도 구분되지 않았다. 다만 세 자동 탐색 전략 모두 수동 설정(0.3776)을 상회했다. 에이전트의 열세는 제안 품질이 아니라 중복 제안에 따른 실효 탐색 예산 축소에서 비롯되었다(반복당 고유 설정 12.5/20). 이 진단은 프롬프트와 구현의 설정 불일치를 제거한 재실험 200 trial로 확증되었다. 고유 설정은 18.90/20으로 회복되고 Optuna에 대한 유의한 열세도 사라졌으나(p = .1726), run-level 평균은 여전히 Optuna에 미치지 못했고(0.4292 대 0.4490) 원본 대비 개선도 입증되지 않아(p = .2387) 성능 우위는 확인되지 않았다.

본 연구의 의의는 세 가지다. 첫째, 경량 VLM의 의료 도메인 적응에 관한 3단계 실증 데이터를 동일 프로토콜 아래 제공하며, 성능과 자원 소비가 단조 관계를 이루지 않음을 보여 자원 제약 환경의 모델 선택 근거를 제시한다. 둘째, 파인튜닝 효과의 모델별 이질성을 정량화하여 합산 분석이 상반된 효과를 은폐하는 과정을 실증한다. 셋째, 자율 에이전트 기반 최적화의 부정적 결과를 그 실패 기전의 검증과 함께 보고한다. 특히 동일 설정을 반복 실행할 때 관측된 변동폭(0.056)이 전략 간 평균 차이(0.031)보다 크다는 사실은 단일 실행 결과로 기법을 비교하는 관행에 대한 정량적 반례가 된다.

주제어 : 의료 영상 질의응답, 시각-언어 모델, QLoRA, 파라미터 효율적 파인튜닝, 하이퍼파라미터 최적화, 대규모 언어모델 에이전트
