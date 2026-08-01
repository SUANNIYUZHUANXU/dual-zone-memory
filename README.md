# **An Alternative Path to Continual Learning: Frequency-Threshold-Triggered Local Weight Writing on FFN Layers During Inference**

---

> **【About the Priority and Positioning of This Work】**
>
> The author is an independent inventor from China with limited English proficiency. The English version of this article was translated collaboratively by the author and AI. If there are any inaccuracies in expression or technical precision, please refer to the original text in the Chinese community at https://zhuanlan.zhihu.com/p/2065469636003487748 or https://segmentfault.com/a/1190000048094414
>
> This technical solution has been filed as a Chinese invention patent application (Application No.: <u>2026111114738</u>), with the aim of establishing the creator's priority of conception, and is not intended as a malicious patent hoarding that impedes technological progress. The author commits that, during the valid term of the patent, automatic free licensing and fee waiver will be granted for non-commercial academic research, personal learning, and open-source community development.
>
> **To the best of our knowledge, this is the first attempt to achieve continuous memory during the inference phase of a pretrained Transformer, by applying a minimalistic hard-threshold counter to perform bypass-channel, sparse Hebbian physical writes on local weights of the FFN layers.** Current mainstream memory-augmentation methods mostly rely on external caches, segment-level loops, or gated multi-frequency updates. In contrast, our approach chooses a more biologically plastic and engineering-light differentiated path. We believe this direction deserves serious scrutiny and testing by the community.
>
> **What This Document Is**
>
> - An early research report that proposes a novel memory architecture and completes its core logical closure.
> - An open technical document aimed at establishing technical priority and extending an invitation for collaboration to the community.
> - A pilot experiment exploring the paradigm shift of large models from “statistical lookup” to “neuroplastic reshaping.”
>
> **What This Document Is Not**
>
> - Not a full academic paper with comprehensive benchmarking and ablation studies.
> - Does not claim state‑of‑the‑art results on any current standard leaderboard.
> - Due to the limited engineering and compute resources of an independent researcher, the proof‑of‑concept experiments for the mechanisms described herein are yet to be completed. This is precisely the core motivation for making this document public — we seek researchers with reproducibility capability to jointly advance this paradigm to a measurable stage.
>
> We are fully aware that grand narratives without experimental backing can easily degenerate into empty speculation. Therefore, the significance of this document is not to “declare victory,” but to “mark a path that no one has yet walked.” We welcome all forms of examination, questioning, reproduction, and collaboration.

---

## Chapter 1: The Intelligence Illusion and a Sheltered Road

### 1.1 Introduction: AGI Has Arrived, or a Deliberately Crafted Illusion?

When GPT‑4 fluently passes the bar exam, when Claude writes poetry comparable to human poets, and when DeepSeek repeatedly tops mathematical reasoning leaderboards — the world exclaims that “AGI has arrived.”

But are we being deceived by appearances?

If we strip off the outer garment of these models, what we see is not a flickering “spark of thought,” but a **huge, static weight matrix that is completely frozen at the moment of inference**. It has no experiences, no surprises, and no ability to “write” stimuli. It is merely performing the most expensive, context‑based **“advanced fill‑in‑the‑blank”** in human history.

The core thesis of this paper is: **the “intelligence” exhibited by current AI is essentially a high‑order intellectual illusion based on statistical distributions.** To break through this limitation, we must re‑examine a road that has been long shelved in the history of AI.

---

### 1.2 The Freezing Dilemma: The “Cognitive Death Sentence” of Transformer Inference

- How does the human brain work? When stimuli from smell, hearing, vision, taste, and touch arrive at the brain, they are converted into electrical signals, causing certain **physical‑level fixation and changes** on neural synapses — this is the material basis of memory. When new similar stimuli arrive, the brain outputs intuitive experiential feedback based on these already‑“written” physical traces. This feedback is often not a result of logical deduction, but rather “the synapse has already made the choice for you.”
- In contrast, the Transformer’s way of handling “memory” is more like a **pre‑stored static database**. It algorithmically dissects the questions we pose, and then retrieves the most probable combination of responses from this knowledge base. Throughout this process, it neither generates new synaptic connections nor updates any weights.
- **From an engineering perspective**, all current large models based on the Transformer have their billions of parameters **completely stationary** during inference. No matter what “stimulus” you feed in, it only performs one static forward pass. This means: **it reads your pain, but it has never “felt” the pain; it remembers your question, but it has never “written” that experience.**
- This mechanism does not bring it closer to a “human”; it does more of a purely logical judgment, or emotionally simulated responses based on logic — but this cannot be counted as true “intelligence.” It has only “computation,” no “experience.”

---

### 1.3 The Intelligence Illusion: Why Can a “Frozen” System Still “Guess Correctly”?

- Since it is so “rigid,” why can it still give astonishing answers in most scenarios?
- In essence, current AI is still a **super calculator**. It does not truly “understand” mathematics. When you ask it `1 + 1 = ?`, it can answer correctly, not because it understands the concepts of “quantity” and “addition,” but because the pattern `1+1=2` appears millions of times in its training data.
- **From a mathematical essence**, it can “guess correctly” because petabyte‑scale training data compress human knowledge, logic, and conversational patterns into **probability distributions** in high‑dimensional space. When it answers you, it is not “reasoning,” but performing an extremely complex conditional probability calculation:

$$
P(w_t \mid w_1, w_2, ..., w_{t-1})
$$

- When this calculation is sufficiently accurate and covers enough long‑tail scenarios, the observer's brain automatically completes this **“statistical speculation”** into **“logical certainty.”** This is the neuropsychological basis of what we call “intelligence illusion” — **your brain is perfectly deceived by the statistical performance of AI.**

---

### 1.4 A Forgotten Clue: Historical Echoes and Today’s Restart

**Before we propose our solution, it is necessary to acknowledge a historical fact that is easily overlooked.**

- **The urge to “let the model dynamically change its own weights through experience” is not original to this paper.** Its intellectual origin can be traced back to Donald Hebb’s Hebbian learning rule proposed in 1949 — “neurons that fire together, wire together.” In the early days of neural networks, researchers repeatedly attempted to build systems that could continuously update during inference, hoping that the model could “leave traces” in every interaction, much like the human brain.
- **Most of these early attempts did not make it into public literature, not because the idea itself was flawed, but because engineering conditions did not permit it.** At that time, the available model sizes were extremely small (only dozens to hundreds of parameters), stability was poor (any weight update could easily crash the model), and there was a lack of quantitative evaluation benchmarks provided by modern continual learning theory. These attempts failed before producing a working system, and thus were largely unrecorded, uncited, and unforgotten — they became a shelved side road in the history of AI.
- **Today, the conditions for restarting this path are ripe.** The pretrained Transformer provides an extremely stable parameter foundation — with hundreds of billions of parameters, the impact of **local, sparse weight updates** on the whole model is diluted to an acceptable level; modern continual learning benchmarks provide precise quantitative rulers, allowing us to accurately measure the effect of write operations on model generalization.

---

**The essence of this work is not to create a completely new technical path out of thin air, but to conduct a cautious and open re‑exploration of this historically shelved road on a brand new engineering foundation.** We do not consider previous attempts ineffective, but rather believe that technological ideas whose historical conditions were not yet mature deserve to be re‑examined today.

---

### 1.5 The Three Fatal Shackles: The “Wall of Sighs” of Memory, Forgetting, and Compute

It is precisely the misalignment of the above “freezing” and “illusion” that leads to the **three fatal shackles** that large models cannot break through:

**Shackle 1: The “Bottomless Pit” of Memory**
No matter how large the context window is, it cannot remember the “lesson” learned from the previous conversation. Every new session is a “amnesia‑style restart” — the model still treats a returning user as if it were the first meeting.

**Shackle 2: The “Curse” of Forgetting**
Every full update overwrites old knowledge like a “format,” leading to **catastrophic forgetting**. AI can never “grow through conversation” like humans do.

**Shackle 3: The “Wall of Sighs” of Compute**
The inference compute consumption of traditional Transformers grows quadratically with context length $n$:

$$
\text{Cost}_{\text{Transformer}} \propto O(n^2)
$$

The goal of this solution is to break all three shackles simultaneously. The basic idea is: capture stimuli via a short‑term cache, filter information worth remembering via a frequency counter, and consolidate experiences into FFN layer weights via local physical writes. The detailed design will be elaborated in subsequent chapters.

---

### 1.6 Chapter Summary and Roadmap

We have argued three progressive propositions:

1. The Transformer’s weights are frozen during inference, lacking the ability to “write experiences” and sense stimuli;
2. The “intelligence” it exhibits is an intellectual illusion under statistical probability distributions;
3. This frozen architecture naturally leads to three shackles that cannot be solved by simply stacking more compute.

At the same time, we have pointed out a historically shelved path — the attempt of dynamic weight updating — which is worth reopening under today’s engineering conditions.

The subsequent chapters will revolve around this direction:

- **Chapter 2**: Related work review and differentiation positioning;
- **Chapter 3**: Architectural design of “short‑term cache + frequency‑threshold triggering + local physical writing”;
- **Chapter 4**: Discussion and future work.

**(End of Chapter 1)**

---

## Chapter 2: Related Work and Differentiation — Standing on the Shoulders of Giants, Forging Our Own Path

> **Positioning of This Chapter**: Before formally presenting our architecture, we must first clarify one question — “Who exactly are we racing against?” This chapter systematically reviews the most cutting‑edge memory‑augmented architectures and, through precise comparative analysis, clarifies the theoretical innovations and differentiating value of our solution.

---

### 2.1 Related Work: Three Major Schools of Memory‑Augmented Architectures

In recent years, to break through the “freezing dilemma” of Transformers, the academic community has seen a surge of Memory‑Augmented Architectures. These works can be broadly categorized into three schools.

#### School 1: External Memory Retrieval

This is the most mainstream route. The core idea is: **keep the model itself unchanged, and equip it with an “external hard drive”** — compress and vectorize historical conversations, store them in an external database, and retrieve relevant segments to splice into the prompt during inference.

Representative works include MemGPT, the RAG series, and various vector database solutions. The advantage of such methods is their simplicity and plug‑and‑play nature; but in essence, they perform “retrieval” rather than “memory” — the model does not truly “remember” anything, it merely looks up information temporarily for each query. **This is a “clever cheat,” not “genuine growth.”**

#### School 2: Hierarchical Memory Consolidation — Represented by LightMem

**LightMem** (ICLR 2026) is the most influential work in this school. Inspired by the Atkinson‑Shiffrin human memory model, it divides the memory system into three complementary stages:

- **Sensory Memory**: Uses lightweight compression models (e.g., LLMLingua‑2) to quickly filter redundant information and group information by topic;
- **Short‑Term Memory**: Integrates topic‑grouped information into structured units;
- **Long‑Term Memory + Sleep‑time Update**: Uses **offline procedures** to decouple memory consolidation from online inference.

Experimental results show that LightMem achieves up to **29.3%** accuracy improvement on the LongMemEval benchmark, and up to **117×** reduction in token consumption.

**But note a critical detail**: LightMem’s “long‑term memory consolidation” is **offline**. This means it needs to run a separate “sleep” process outside of inference to complete memory writing — memory update and model inference are **two separated phases**.

#### School 3: Multi‑Frequency Updates — Represented by Nested Learning

**Nested Learning** (NeurIPS 2025), proposed by Google Research, is built on the core insight that different components of a model should update their parameters at **different frequencies**.

Specifically, Nested Learning treats the model as a series of **mutually nested optimization problems**, each with its own “update rate.” Based on this paradigm, the research team proposed the **Continuum Memory System (CMS)** , which enables memory components to update at multiple frequencies, forming a smooth spectrum from short‑term to long‑term memory.

The **Hope model**, built on Nested Learning, demonstrates outstanding long‑context memory capability in the “needle‑in‑a‑haystack” test.

**Nested Learning shares the same core intuition as our approach** — that “different frequencies of updating” are key to solving the memory problem. However, his implementation path is fundamentally different from ours.

---

### 2.2 Differentiation Analysis: What Is “New” About Our Architecture?

The following table compares our solution with the above representative works across **six core dimensions**:

| Dimension                     | LightMem (ICLR 2026)                                 | Nested Learning (NeurIPS 2025)           | **Our Solution (Dual‑Zone Memory + Frequency‑Threshold Writing)** |
| :---------------------------- | :--------------------------------------------------- | :--------------------------------------- | :----------------------------------------------------------- |
| **Memory Write Timing**       | Offline “sleep” consolidation                        | Multi‑frequency updates during training  | **Bypass‑channel online immediate writing** (triggered during inference) |
| **Write Trigger Mechanism**   | Offline batch processing                             | Preset update frequencies                | **Frequency‑threshold counting** (triggered when repetition ≥ N) |
| **Write Target**              | External memory storage                              | Model component parameters               | **Model physical weights (FFN layers)**                      |
| **Changes Model Weights?**    | ❌ No (only operates external storage)                | ✅ Yes                                    | ✅ Yes                                                        |
| **Compute Overhead**          | Controllable offline cost                            | High training cost                       | **During inference, constant‑level overhead O(k)**           |
| **Implementation Complexity** | Medium (requires maintaining external memory system) | High (requires re‑engineering optimizer) | **Low (counter + local update)**                             |

#### Differentiation 1: The Fundamental Divide in Write Timing

LightMem’s greatest contribution is “taking memory updates out of online inference” — this does reduce inference latency, but at the cost: **the model cannot “learn” anything instantly during a conversation**. Memory can only be consolidated in batch during “sleep.”

**Our solution is the exact opposite**: writing occurs **during the inference process**. When a conceptual pattern is repeatedly invoked beyond a threshold, local writes are triggered **immediately**. This means the model can **grow in real time** through interaction with users — this is what “online continual learning” truly means.

#### Differentiation 2: Trigger Mechanism — From “Preset Frequency” to “Experience Counting”

The core of Nested Learning is “presetting an update frequency for each component” — some components update quickly, others slowly. This is a **top‑down** design: researchers decide in advance which parts should learn quickly and which slowly.

**Our solution is bottom‑up**: updates are not determined by preset frequencies, but by the frequency of **actual experience**. The more times a concept is repeatedly invoked, the higher its priority for being “written” and promoted to “deep memory.” **This is an “experience‑driven” memory mechanism**, rather than a “design‑driven” one — it is the model that decides “what to remember,” not the researcher.

This is precisely the core insight I mentioned earlier: **“If a token is invoked multiple times, only then is it worth writing into long‑term memory.”**

#### Differentiation 3: The “Physicality” of Writing — From “External Hard Drive” to “Neuroplastic Reshaping”

LightMem’s “memory” is essentially **external storage** — compressed conversations are stored in a vector database, retrieved, and spliced into the prompt during inference. The model’s own weights remain **completely untouched**.

**Our solution pursues true “physical writing”** — directly modifying local weights in the model’s FFN layers, simulating the physical changes of synapses in the human brain. This is a **“neuroplastic reshaping”** memory, rather than a “file‑archiving” memory. The model is not “looking something up,” but “becoming something.”

#### Differentiation 4: The Magnitude Leap in Compute Overhead

This is the hardest‑core differentiation. LightMem can reduce inference‑stage token consumption by 117×, but this is **“saved”** — through compression and offline‑optimization to reduce unnecessary computation.

**Our solution fundamentally changes the growth curve of compute**:

Traditional Transformer:

$$
\text{Cost} \propto O(n^2)
$$

Our solution:

$$
\text{Cost} \propto O(k), \quad k \ll n
$$

where $ k $ is the number of local neurons updated per write trigger (typically < 100), **which does not grow with context length**. This is not “saving,” this is **“circumventing”** — bypassing the entire exponential growth path.

---

### 2.3 Chapter Summary: Three Sentences for Differentiation Positioning

Based on the above analysis, the core differentiation of our solution relative to existing works can be summarized in three sentences:

1. **LightMem does “offline archiving”** — we do **“online writing”** ;
2. **Nested Learning does “preset frequency”** — we do **“experience triggering”** ;
3. **External memory systems do “external hard drives”** — we do **“neuroplastic reshaping”**.

---

### 2.4 Extended Discussion: The Unexplored Blank Zone

The common blind spot of existing work is that: **they all attempt to “optimize memory access,” rather than “change the way memory is generated.”**

LightMem optimizes “how to store and retrieve more cheaply”; Nested Learning optimizes “how to let different components learn at different speeds.” But no one asks a more fundamental question: **“What information is worth remembering?”**

**Our “frequency‑threshold” mechanism answers precisely this question.** This is not an engineering optimization, it is a **cognitive decision** — the model, for the first time, gains the ability to “judge what is worth remembering.” Although this judgment is still primitive (just “repetition count”), it opens up a completely new path: **letting the model decide its own memory**.

In this sense, our architecture is not just “another memory solution” — it may open up a new paradigm of **“granting AI sovereignty over its own memory.”**

Building on this, our architecture naturally possesses several extension potentials. For example, by setting differentiated parameters for different FFN regions, the model can spontaneously develop functional specialization; by reversing the direction of positive writes, active forgetting based on natural language instructions can be realized; going further, if local writes are combined with a large‑scale data middleware, it may fundamentally change the iterative economic model of large models — transforming “pretraining” from a one‑time investment into a continuous evolution process tied to the model’s lifecycle. Specific elaborations of these directions will be presented in Section 4.4 of Chapter 4.

---

**(End of Chapter 2)**

---

## Chapter 3: Architectural Design — Engineering Implementation of the Dual‑Zone Memory System

This chapter elaborates on the core engineering architecture of our solution, consisting of three modules: Short‑Term Cache (3.1), Frequency Counter (3.2), and Write Executor (3.3). Each module is presented with detailed data structure design, algorithmic flow, and complexity analysis.

---

### 3.1 Short‑Term Cache: Temporary Storage and Preliminary Processing of Stimuli

The Short‑Term Cache corresponds to the “electrical signal” stage of the human brain — information briefly resides here for rapid structuring and preliminary association analysis, without triggering any weight modifications.

#### 3.1.1 Storage Structure

The short‑term cache adopts a **“dual‑track storage”** strategy: for each input, it retains both its **raw text trace** and its **semantic vector representation**. The data structure of each memory unit is defined as follows:

**Figure 1: Schematic Diagram of the Short‑Term Cache Data Structure**

<img width="1437" height="1050" alt="2 缓存区数据字段表" src="https://github.com/user-attachments/assets/666604bc-cd5a-4467-90ff-de0f0ffcb29c" />

#### 3.1.2 Three‑Layer Functional Pipeline

After being stored in the cache, the system performs three parallel evaluations on each record: frequency statistics (tracking entity repetition intensity), semantic association mapping (comparing similarity with already‑written concepts), and inference context preloading (if historical inference paths exist, they are injected as heuristic information into the current context). The results of these three evaluations jointly contribute to the subsequent comprehensive write priority scoring.

#### 3.1.3 Weighted Synthesis of the Three Functions

The comprehensive write priority of each record is calculated by the following formula:

$$
Score = a \cdot F + b \cdot A + c \cdot I
$$

where:
- $ F $ = normalized frequency count (`access_count / max_access_count`)
- $ A $ = normalized maximum association similarity
- $ I $ = normalized inference path confidence (if no triggering inference path exists, this term is 0)
- $ a, b, c $ are weight coefficients and $ a + b + c = 1 $ (the preferred ratio is $ a = 0.4, b = 0.35, c = 0.25 $)

A record obtains “write candidate” status **if and only if** $ Score > \theta $ (where $ \theta $ is the global write threshold, default 0.75). In addition, a **dual verification** mechanism must be satisfied: precise match count ≥ 3 and maximum association similarity ≥ 0.8 before the final write can be triggered.

#### 3.1.4 Lifecycle Management

- **Clear‑after‑write**: Once a record is successfully written into long‑term memory (FFN layer), it is immediately removed from the short‑term cache.
- **Clear‑on‑session‑end**: If the current dialogue session ends, all short‑term cache entries except those approaching the threshold are cleared.
- **Clear‑on‑expiry**: Records exceeding a preset survival time (e.g., 60 minutes) are automatically invalidated to prevent cache space occupation.

#### 3.1.5 Complexity Analysis

The core operations of the short‑term cache include storage, query, and association mapping:

- **Storage (write to cache)** : Each record write is an O(1) operation (hash table insertion).
- **Exact match query (frequency counting)** : Traversing all records in the cache for entity‑action pair matching has complexity $ O(M) $, where $ M $ is the current number of records in the cache. $ M $ is limited by the “clear‑on‑session‑end” and “clear‑after‑write” mechanisms, with a default upper limit of 256 entries, making this operation constant‑level in engineering practice.
- **Association mapping (nearest neighbor search)** : Performing k‑NN search of the semantic vector against already‑written concepts in deeper FFN layers, using FAISS indexing, has complexity $ O(\log N) $, where $ N $ is the total number of already‑written concepts. $ N $ grows linearly with usage time, but the search process runs on the GPU with controllable per‑query latency.

Overall, the total complexity for processing a single short‑term cache entry is $ O(M + \log N) $, where $ M $ has an upper bound, and the scale of $ N $ needs to be capped based on hardware conditions in actual deployment (e.g., $ N \le 10^6 $).

---

### 3.2 Frequency Counter: Multi‑Level Measurement of Repetition

#### 3.2.1 Design Objective

The frequency counter is the **primary decision‑making basis for writing into long‑term memory** in our architecture. Its goal is to answer one core question: **“Has a concept been repeated enough to be worth remembering?”**

To answer this question accurately, the counter must possess the following capabilities:

- Identify **exact repetition** (identical entities)
- Identify **repetition with modifiers** (same entity with adjectives)
- Identify **event‑based repetition** (same entity with different actions)
- Identify **derivative associations** (new concepts with the same root) — but without conflating them with the original entity

#### 3.2.2 Four‑Level Matching Rules (Weighting System)

After the `entities_actions` field of each record in the short‑term cache is extracted, it is sequentially matched against existing counter entries. The matching is divided into four levels, each with a different weight coefficient and counting behavior:

| Level                         | Matching Condition                       | Weight  | Counting Behavior                                            | Example                                                      |
| :---------------------------- | :--------------------------------------- | :------ | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **L1 Exact Entity Hit**       | Entity noun exactly matches              | **1.0** | Directly counted in that entity's counter                    | “apple” → hits the “apple” counter                           |
| **L2 Modified Entity Hit**    | Same entity noun with adjective/modifier | **0.8** | Counted in that entity's counter (weighted discount)         | “red apple”, “delicious apple” → hits the “apple” counter with weight 0.8 |
| **L3 Event Memory**           | Same main noun with different verb       | **0.6** | Counted in that entity's counter (weighted discount)         | “eat apple”, “buy apple”, “wash apple” → hits the “apple” event counter |
| **L4 Derivative Association** | Same root but different stem             | **0.5** | Triggers an independent derivative counter (does not share the original counter) | “apple juice”, “apple pie” → triggers independent counters for “apple juice” and “apple pie” respectively |

> **Design Note**: The weight coefficients are currently initial preset values, and can later serve as tunable parameters for “AI personality capability traits.”

**Figure 2: Full Flowchart of the Frequency Counter's Four‑Level Matching Rules and Independent Counter Mechanism**

<img width="2152" height="3492" alt="3 L4层级匹配示意图" src="https://github.com/user-attachments/assets/e363f678-0582-4b94-8a70-1d003c060ae3" />


#### 3.2.3 Independent Counter Mechanism — The Independent Lifecycle of Derivatives

**The L4 level (derivative association) is a distinctive feature of our design and requires special elaboration.**

When “apple juice” is recognized as a derivative, it is **not** counted toward the frequency counter of “apple.” Instead, it triggers a **completely new, independent counter** — the “apple juice” counter.

**Why can't they share a counter?**

- “Apple” and “apple juice” are semantically related, but cognitively distinct concepts.
- A person might mention “apple” every day, but only mention “apple juice” once. Sharing a counter would incorrectly amplify the importance of “apple juice.”
- Independent counters ensure each concept has its own independent “memory threshold.”

**The independent lifecycle of derivatives:**

1. **First occurrence**: The derivative is recognized, an independent counter is created with initial value = 1, but no write is triggered (threshold not yet met).
2. **Multiple repetitions**: Each subsequent occurrence of the derivative increments its independent counter.
3. **Threshold‑reaching write**: When the derivative's independent counter reaches the global threshold $ \theta $ (sharing the same threshold system as entities), it triggers a physical write of the derivative, making it an **independent new concept** in the deeper FFN layers.

```python
# Pseudocode: Derivative independent counter logic
if match_type == "DERIVATIVE":
    # Create or update the independent derivative counter
    if derivative_word not in derivative_counters:
        derivative_counters[derivative_word] = 0
    derivative_counters[derivative_word] += 1
    # Independently determine whether the derivative counter reaches the threshold
    if derivative_counters[derivative_word] >= THRESHOLD:
        trigger_write(derivative_word, source="derivative")
```

#### 3.2.4 Integration Logic with the Short‑Term Cache

Frequency counting results interface with the short‑term cache (Section 3.1) through the following mechanisms:

1. **Score calculation participation (40% weight)** : Frequency count contributes 40% weight to the comprehensive write priority Score of each record.

$$
Score = 0.4 \cdot F + 0.35 \cdot A + 0.25 \cdot I
$$
   
   where
   $ F $ is the normalized frequency value of the corresponding entity (current frequency / global maximum frequency).

3. **Dual verification precondition**: Frequency count serves as a precondition for the “dual verification” mechanism — the exact entity count must be ≥ 3 (i.e., the same entity has been mentioned precisely or with weighted counts at least 3 times) before it can enter the write candidate sequence. This precondition effectively prevents one‑off inputs from being mistakenly written into deep memory (this can later also serve as a “memory strength adjustment knob”).

4. **Independent verification for derivatives**: Derivatives also require their independent counters to reach the threshold (≥ 3 times) before triggering a write. This requirement ensures that derivatives share the same “memory threshold” as original entities.

#### 3.2.5 Setting the Global Write Threshold θ

The global write threshold $ \theta $ controls “how many times a concept needs to be mentioned before it is worth writing into long‑term memory.” The initial setting is $ \theta = 0.75 $.

> **Supplementary Note on the Threshold's Meaning**: $ \theta $ here serves as the threshold for “local write decisions.” Its design value is not arbitrary, but aims to balance “remembering new information as early as possible” against “avoiding being overwhelmed by noise.” This value can be dynamically adjusted during the prototype validation phase, and later can also serve as a “personality adjustment knob” — a higher threshold makes the AI “cautious and conservative,” while a lower threshold makes it “sensitive and easily influenced.”

**Why choose 0.75?**

- Empirical experiments suggest that most meaningful repetition patterns manifest after 3‑5 mentions.
- When converted to an absolute threshold, 0.75 is equivalent to **“≥ 3 exact hits and a comprehensive Score ≥ 0.75”** , filtering out low‑frequency noise while ensuring meaningful repetitions are not missed.
- This value can be adjusted during prototype validation to retain flexibility for optimization under different scenarios.

#### 3.2.6 Complexity Analysis

The core operation of the frequency counter is performing hierarchical matching of the current record's `entities_actions` against all existing counter entries. Let $ E $ be the total number of current entity counters; then the complexity per match is $ O(E) $ (traversing all counters). $ E $ is constrained by the threshold $ \theta $ and the “clear‑after‑write” mechanism, and in the long run stabilizes at the order of “frequently mentioned concepts,” rather than growing indefinitely.

Detailed matching costs per level:

- L1 exact match (string‑identical): O(1) hash lookup.
- L2 modified entity match (stem + modifier parsing): requires tokenization and stemming of the entity, with per‑operation cost O(L), where L is the entity string length (typically ≤ 10).
- L3 event match (main noun + verb combination): requires parsing the main noun and verb, with per‑operation cost O(L).
- L4 derivative match (root recognition): requires lookup in the derivative mapping table, O(1) hash lookup.

Therefore, the total cost for a single record passing through the complete counting pipeline is O(E + L), where $ E $ is the number of counters corresponding to that entity, and $ L $ is the text length.

#### 3.2.7 Section Summary

The frequency counter is the first sieve between “experience” and “memory” in our architecture. It distinguishes repetition signals of different strengths through **four‑level matching rules**, ensures derivatives are not overwhelmed or conflated through the **independent counter mechanism**, and guarantees that write decisions are well‑grounded through the **dual‑verification precondition** and the $ \theta $ threshold.

In the next module (3.3 Write Executor), we will actually write the information judged as “worth remembering” into the physical weights of the FFN layers.

------

### 3.3 Write Executor: Solidifying Experience into Physical Weights

#### 3.3.1 Design Objectives

The Write Executor is the **final execution module** of our architecture — it is responsible for **actually writing** the information judged as “worth remembering” by the frequency counter into the **physical weights of the Transformer's FFN layers**, achieving physical consolidation of long‑term memory.

The Write Executor must simultaneously satisfy the following constraints:

- **Safety**: Write operations must not damage the model's original capabilities (no catastrophic forgetting).
- **Precision**: Only modify the very few neurons most relevant to the current memory (Top‑K sparse writing).
- **Stability**: Each write must be extremely small in magnitude and equipped with a rollback mechanism.

#### 3.3.2 Write Location: Top‑K Neuron Selection in FFN Layers

Each layer of the Transformer contains an FFN (Feed‑Forward Network) module, and each row of its weight matrix corresponds to a “neuron.” Research has shown that FFN layers are the primary physical medium for storing **factual knowledge**.

When a record is judged as “worth writing,” the system needs to answer a core question: **“To which neurons in the FFN should this set of information be written?”**

Our solution adopts **“Activation‑based Selection”** , with the following specific process:

1. **Obtain activations through forward propagation**: Input the record to be written (the current input) into the model, perform one forward pass, and record the activation values of all neurons in the target FFN layer.
2. **Sort and select Top‑K**: Sort by activation values in descending order, and select the top $ K $ neurons with the highest activations ($ K $ is a preset constant, default **10**).
3. **Lock writing targets**: These K neurons become the target weight positions for this write.

> **Why Top‑K?** When the human brain learns new information, it does not reconstruct the entire neural network, but rather **strengthens a specific set of synaptic connections**. Top‑K selection simulates this principle: only modifying the neurons most relevant to the new information, minimizing the impact scope of the write.

#### 3.3.3 Write Formula: Gradient‑Free Local Update

Traditional fine‑tuning relies on **backpropagating gradients**, requiring traversal of the entire model with high cost. Our Write Executor adopts a **gradient‑free heuristic update**, bypassing backpropagation and achieving constant‑level computational overhead.

For each selected target neuron $ i $, the weight update formula is:

$$
w_{i}^{new} = w_{i}^{old} + \eta \cdot a_i
$$

where:

- $ w_i $ is the weight vector of the neuron
- $ a_i $ is the activation value (scalar) of the neuron under the current input
- $ \eta $ is the write learning rate, default **1×10⁻⁵**

**Why must the learning rate be extremely small?** A single write can only cause a weight change on the order of $ 10^{-5} $. Only when the same set of concepts is **written repeatedly many times** (i.e., multiple trigger events) will the weights gradually accumulate noticeable changes — this forms a closed loop with the “frequency threshold” design logic.

**Why is no gradient needed?** Because the goal of our architecture is to **“reinforce existing patterns,”** not to “learn entirely new tasks.” The activation value itself already contains information about “the strength of association between the current input and this neuron” — the larger the activation, the higher the relevance of this neuron to the current concept, and adjusting its weight in that direction constitutes reinforcement.

#### 3.3.4 Three Safety Locks

To prevent write operations from causing model collapse, the system integrates **three safety locks** into the Write Executor:

| Safety Lock                    | Trigger Condition                                            | Action                                                       | Analogy                                             |
| :----------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :-------------------------------------------------- |
| **Lock 1: Magnitude Clipping** | The weight change of a single update exceeds the preset upper limit (preferred: $ \Delta_{max} $) | Linearly scale the update magnitude to within the limit      | Take only a small step each time to avoid falling   |
| **Lock 2: Norm Constraint**    | The Euclidean distance between the modified neuron's weight vector and its original vector exceeds the preset distance upper limit (preferred: $ D_{max} $) | Proportionally shrink the weight vector back to the safe radius | Keep the weight “in a cage” so it can't run far     |
| **Lock 3: Immediate Rollback** | Significant deviation appears in the output of a universal test sample (e.g., “1+1=？”) run immediately after writing | Undo all write operations and restore original weights       | Take a small sip first to confirm it's not poisoned |

**Quantitative standard for Lock 3's trigger condition**: The **change in the output probability distribution (KL divergence)** on the test sample is used as the metric. (Preferred: if the KL divergence before and after writing on a standard test sample exceeds $ 0.01 $), the write is judged as anomalous and rollback is triggered.

**Figure 3: Data Write Flowchart of the Write Executor and Cooperative Lock Mechanism**

<img width="2046" height="1011" alt="4 写入器执行流程示意图" src="https://github.com/user-attachments/assets/db306ceb-bd5b-46fd-93ab-a768f32f9b3c" />

#### 3.3.5 Core Code Implementation (PyTorch Example)

Core operations of the Write Executor:

The following pseudocode illustrates the core logic of the Write Executor (a simplified conceptual implementation, not a final optimized version). The complete engineering implementation and model adaptation details have been disclosed in the patent application; here we show only the core idea. To keep the code concise, this example omits the full implementation of the three safety locks, the specific logic of which has already been described in detail in Section 3.3.4.

```python
# Core write logic (conceptual demonstration, not a complete engineering implementation)
def write_to_ffn(model, layer_idx, neuron_indices, activations, lr):
    # Locate FFN layer weights and perform scalar additive write on the highest‑activated neurons
    weights = model.layers[layer_idx].ffn.weight
    for idx, act in zip(neuron_indices, activations):
        weights[idx] += lr * act
```

#### 3.3.6 Complexity Analysis

The write executor comprises three main operations: forward pass to obtain activation values, Top‑K selection, and weight update.

- **Forward pass**: Requires feeding the current input through the model and computing the activation values of the target FFN layer. The complexity is the same as a single model inference, i.e., $ O(P) $, where $ P $ is the number of model parameters. However, this operation is only executed when a write is triggered (not on every inference), and it can share computation results with the normal inference path.
- **Top‑K selection**: Sorting all neuron activation values and picking the top K. If using a quick‑selection algorithm, the complexity is $ O(D) $, where $ D $ is the total number of neurons in that FFN layer (typically several times the model's hidden dimension). Using `torch.topk`, the actual efficiency is close to $ O(D) $.
- **Weight update**: Modifies only K neurons, each write is $ O(K) $, with K being a constant (default 10), effectively $ O(1) $.

Overall, the total complexity of a single write operation is $ O(P + D) $. Although the absolute values of P and D are large, the write frequency is strictly limited by the frequency counter threshold $ \theta $ (only triggered when a concept has been repeated ≥ 3 times and Score ≥ 0.75). Therefore, the actual compute overhead in practice is far lower than the global backpropagation required during the training phase.

#### 3.3.7 Section Summary

The write executor locks write targets through **Top‑K neuron selection**, achieves constant‑level compute overhead through **gradient‑free heuristic updates**, and guarantees model stability through the **three safety locks**. Together, these three layers of design transform the theoretical idea of “directly modifying FFN layer weights during inference” into a practically safe and engineering‑feasible operation.

---

### Chapter 3 Summary

This chapter has fully presented the three engineering modules of the dual‑zone memory system:

- **3.1 Short‑Term Cache**: Simultaneously stores raw text and semantic vectors, processing information in parallel through three functional pipelines (frequency counting / association mapping / inference triggering).
- **3.2 Frequency Counter**: Measures the repetition strength of concepts through four‑level matching rules (exact / modified / event / derivative) and the independent counter mechanism.
- **3.3 Write Executor**: Safely writes experiences into the physical weights of FFN layers through Top‑K selection and gradient‑free updates.

The three modules form a complete closed loop: **Short‑Term Cache receives stimuli → Frequency Counter filters information worth remembering → Write Executor consolidates information into physical weights**.

**Figure 4: Overall Architecture Closed‑Loop Diagram of the Dual‑Zone Frequency Memory System**

<img width="2082" height="927" alt="1 整体架构图" src="https://github.com/user-attachments/assets/5b92817f-22e8-433d-8dc1-43a0f1a1cf2f" />


**(End of Chapter 3)**

---

## Chapter 4: Discussion and Future Work

This chapter reviews the core contributions of our architecture, discusses its theoretical positioning and limitations, and outlines several promising yet unverified directions for future exploration. These directions are not necessary components of the current architecture, but rather natural extensions built upon its design.

---

### 4.1 Summary of Core Contributions

The core contributions of this architecture can be summarized in four points:

1. **A “dual‑zone memory” cognitive architecture** — Separating short‑term cache (temporary residence) from long‑term physical writing (FFN layer consolidation), simulating the human brain's memory formation pathway from electrical signals to synaptic remodeling.

2. **A “frequency‑threshold‑triggered” write decision mechanism** — Using four‑level matching rules and independent counters to determine “what information is worth remembering,” rather than unconditionally recording everything.

3. **A “gradient‑free local writing” engineering solution** — Through Top‑K neuron selection and three safety locks, safely consolidating experiences into FFN layer physical weights with extremely low (constant‑level) compute overhead.

4. **Identifying a historically shelved path that is now conditionally viable for restart** — Dynamic weight updating, which lay dormant in early neural networks due to immature engineering conditions, is now worth re‑examining on the stable foundation of pretrained Transformers.

---

### 4.2 Positioning Differences from Existing Paradigms

Our architecture differs fundamentally from current mainstream research along the following three dimensions:

| Dimension                                       | Mainstream Paradigm (e.g., LightMem)          | Our Architecture                                  |
| :---------------------------------------------- | :-------------------------------------------- | :------------------------------------------------ |
| **Storage Location of Memory**                  | External vector database / knowledge graph    | The model's own FFN parameters                    |
| **Write Triggering Mechanism**                  | Offline batch processing / preset frequencies | Online experience‑triggered (frequency threshold) |
| **Relationship Between Training and Inference** | Training completes, then frozen               | Continuously evolving during inference            |

This is not a “progressive optimization,” but a **paradigm shift at the architectural level** — moving the AI's “memory system” from outside the model to inside the model, from “completed at training time” to “sustained during inference.” If validated, this architecture will open up a new coordinate system centered on “model self‑evolution” for future research.

---

### 4.3 Limitations and Unverified Assumptions

Our architecture still has several unverified assumptions and known engineering constraints at the theoretical level:

#### 4.3.1 Unverified Core Assumptions

- **Disturbance controllability assumption**: Will the impact of local weight writing on model generalization be as mild as expected? Although the three safety locks provide an engineering‑level buffer, the mapping from “local updates → global behavioral changes” still lacks systematic experimental data support.

- **Memory persistence assumption**: Will knowledge written into FFN layers drift over time or be overwritten by subsequent writes? Human memory decays over time, but our architecture currently only supports “enhancement,” without introducing natural decay mechanisms.

- **Large‑scale interference assumption**: When a large number of new concepts (>10⁴) are written into the same model, what is the capacity limit of the FFN layers? Is there a critical point where “new knowledge crowds out old knowledge”? These questions currently have only theoretical speculation, lacking empirical measurement.

#### 4.3.2 Engineering Constraints

- The current implementation relies on PyTorch's dynamic graph mechanism; directly modifying weights during inference may be restricted in certain deployment environments (e.g., TensorRT, ONNX).
- Write operations require the model state to remain resident in GPU memory, which is not suitable for Serverless or stateless deployment scenarios.

---

### 4.4 Future Work Directions

#### Direction 1: Superposition Awakening of Derivatives — Peripheral Stimuli Nourishing the Original Word

As I speculated in Section 3.2: when derivatives of a concept (e.g., “apple juice,” “apple pie”) are repeatedly written, should the original stem (“apple”) passively receive some cumulative “memory heat”?

**Implementation path**: When a derivative's independent counter reaches the threshold and triggers a write, the system sends a “dilution signal” to the original stem, with intensity proportional to the derivative's write frequency. 

$$
\Delta H_{stem} = \sum_{d \in derivatives} \lambda_{d} \cdot \mathrm{write\_count}_d
$$

where $ \lambda_d $ is the “semantic affinity” coefficient of the derivative to the original word (determined by vector similarity). When the cumulative dilution signal on the original stem exceeds its own threshold, the system triggers a write event for the stem as well.

**Theoretical significance**: If this mechanism holds, it would demonstrate that “knowledge can be indirectly reinforced through peripheral associations,” simulating the human brain's “learning by analogy” ability. Moreover, it extends naturally from our architecture's existing frequency counting and writing logic, without introducing new paradigms.

---

#### Direction 2: Negative Writing and Active Forgetting — Controllable Correction via Verbal Instructions

As I also speculated in Chapter 2, the “positive write” mechanism of our architecture can naturally be reversed to achieve “active forgetting.” Active forgetting refers to the system weakening or deleting specific concepts within the model based on the user's verbal instructions, rather than removing information through retraining.

**Trigger mechanism**: The user initiates a forgetting request via natural language instructions, such as “Forget what I just said” or “Correct your previous wrong answer.” The system parses the instruction semantically and identifies the target concept to weaken or remove.

**Execution mechanism**: Locating the physical coordinates of the target concept in the FFN layer is the core challenge of negative writing. Our preliminary solution is to use the semantic vector in the short‑term cache to perform a nearest‑neighbor search in the FFN layer, locate the Top‑K neurons closest to the concept vector, and then apply the following negative update:

$$
w_{new} = w_{old} - \beta \cdot \text{Activation}
$$

where $ \beta $ is the forgetting strength coefficient, controlling the degree of weakening. This operation does not rely on retraining or RLHF, can be completed entirely during inference, and does not significantly affect the model's other knowledge.

**Applicable scenarios**: The negative writing mechanism is suitable for:

- Correcting factually wrong answers given by the model;
- Immediately correcting specific biases exhibited by the model;
- Removing user privacy‑sensitive or confidential information from the model.

This mechanism provides a low‑cost, interactive path for model safety and controllability, allowing the model to evolve continuously while also accepting effective user corrections.

---

#### Direction 3: Distributed FFN Functional Specialization — Specialized Evolution of Neuron Layers

As I further speculated in Chapter 2: since the FFN layer in the Transformer architecture is not a monolithic whole, but a distributed matrix structure composed of **layer‑by‑layer and neuron‑by‑neuron** components, with different layers handling different levels of abstraction and different neurons within the same layer naturally differing in activation patterns, this raises a question that has not been deeply explored before:

**Will different regions of the FFN spontaneously evolve differentiated functional specializations through different “trigger‑write” histories?**

**Implementation path**: The system performs long‑term tracking of the “concept category distribution” written into each FFN layer and each neuron cluster. When a region's concept category distribution deviates significantly from the global distribution, it is determined that the region has formed a specialization tendency. The system can then set differentiated parameters for that region:

- **Horizontal sharding (differentiation by layer)**: Shallow FFN layers, being closer to the input and responsible for basic syntax and local pattern recognition, can have their write sensitivity appropriately reduced to protect already‑acquired foundational capabilities; deeper FFN layers, being closer to the output and responsible for high‑level semantic and global associations, can have their write sensitivity appropriately increased to absorb new concepts more quickly.

- **Vertical partitioning (differentiation within a layer)**: Different neuron clusters within the same FFN layer, through their respective historical trigger distributions, naturally form specialized divisions. For example, some neuron clusters may specialize in “numeric/logical” concepts, while others specialize in “emotional/descriptive” concepts.

**Theoretical significance**: This mechanism would elevate “physical writing” beyond mere memory enhancement into the realm of **architectural self‑adaptive evolution**. The model would no longer just “remember more things,” but “grow more complex functional structures.” In the long term, this may provide a path for AI systems to break free from “human‑designed architecture” constraints and move toward “self‑organized growth through interaction,” carrying directional exploratory significance.

---

#### Direction 4: From “Single‑Machine Self‑Evolution” to “Federated Continuous Pretraining”

Our architecture is currently positioned for local writing during single or multiple inference sessions. But if this writing mechanism is combined with a large‑scale data middleware, a more ambitious vision emerges:

User interaction data is processed through a middleware (denoising, cleaning, mining, generating high‑quality natural language), and then, during periods of low model compute utilization, continuously injected into the base model using our FFN local writing mechanism.

**Figure 5: Conceptual Diagram of Federated Continuous Pretraining**

<img width="987" height="984" alt="image-20260727192115978" src="https://github.com/user-attachments/assets/c92abe5e-b5d6-410a-af37-3847076c04e5" />


If this direction proves viable, it would produce an unprecedented effect: **pretraining would no longer be a one‑time, permanently frozen expensive process, but a continuous evolution process tied to the service lifecycle.** The model's “learning” would no longer depend on manually triggered full retraining, but would use user interactions as fuel and low‑compute periods as windows to automatically complete knowledge sedimentation and iteration.

**Core value**: This would fundamentally change the economic model of large‑model iteration — from “burning tens of millions of dollars every 1‑2 years for a full retraining” to “automatic daily evolution at extremely low cost.”

---

#### The Intersection of the Four Directions: From “Memory” to “Cognitive Sovereignty”

These four directions collectively point to a deeper proposition: **Can AI possess sovereignty over its own memory — that is, decide what to remember, what to forget, and how to organize its own knowledge structure?**

All current models are “memory systems designed by researchers,” while the extension directions of our architecture attempt to gradually transfer “memory decision rights” to the model itself and its interactors. This is not just an engineering problem, but a **necessary path toward truly autonomous agents**.

---

### 4.5 Experimental Validation Recommendations

Our architecture is currently in the theoretical design and prototype implementation stage; systematic validation at scale has not yet been performed. If experimental conditions become available in the future, we recommend the following validation order:

| Priority | Validation Target                                            | Expected Difficulty | Required Resources                                           |
| :------- | :----------------------------------------------------------- | :------------------ | :----------------------------------------------------------- |
| **P0**   | Can a single write be recorded by the system (log‑level validation) | Very low            | 1 GPU, 1 day                                                 |
| **P1**   | Can the model still pass standard test sets after writing (safety validation) | Low                 | 1 GPU, 2 days                                                |
| **P2**   | Can written memories still be correctly recalled after multiple interactions (persistence validation) | Medium              | 2‑4 GPUs, 1 week                                             |
| **P3**   | Capacity boundary of the model after massive writes (stress validation) | High                | 8+ GPUs, multiple weeks                                      |
| **P4**   | Stability and capacity boundary of federated continuous injection (macro‑architecture validation) | Very high           | Commercial‑grade multiple GPUs, distributed cluster, multiple weeks |

**Special notes**: P0 and P1 validation can be completed in a personal developer environment using existing tools, without requiring large clusters. P2 and P3 validation require more substantial resource preparation. P4 experiments require the first four levels to run smoothly, combined with mature big‑data technologies, and a sufficiently large volume of end‑user interaction data to form noticeable peak usage patterns.

---

### 4.6 Conclusion: Back to the Origin of the Question

This paper began with a simple question: **“When AI receives stimuli during inference, does it truly ‘experience’ anything?”**

Our answer is negative — the weights of current AI are frozen during inference, and the intelligence it exhibits is more like an intellectual illusion based on statistical distributions than genuine cognition.

Our solution is to reopen a road that early neural network researchers once blazed but was forced aside due to immature engineering conditions — **letting the model dynamically, selectively, and safely change its own weights during inference.**

Whether this path will succeed is still uncertain. It requires experimental validation, the test of time, and the scrutiny of peers. But regardless of the outcome, we believe: **re‑examining the historical concept of “dynamic weights” on the stable foundation of Transformers is itself a valuable exploration.**

If this path succeeds, it will provide a possible underlying mechanism for AI's transformation from “trained tool” to “agent that grows through interaction.”

If it does not, we hope to at least leave a clear marker: why it didn't work, and how to try again next time.

**(End of Chapter 4)**

---

## About the Author

The author is a self‑taught AI enthusiast with a bachelor's degree in humanities and arts, who transitioned into the AI field in 2026.

The dual‑zone physical frequency‑based writing architecture proposed in this paper was created solely by the human author. AI tools were used during the writing process to improve clarity of expression; all technical judgments and core logic were led by the author personally. Currently seeking opportunities for AI architecture research assistant positions or technical collaboration.

**Contact**:  

**Chinese-region**

WeChat: l15951766149 (please indicate your purpose)  
Email: 327999898@qq.com

Liu Kechen
