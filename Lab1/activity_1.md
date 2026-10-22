## Activity 1 — Explore LLM Models on Hugging Face

## Objective

To explore three Hugging Face language models of different sizes—small, medium, and large—and compare their creators, parameter counts, generative capabilities, context lengths, local execution suitability, and hardware requirements.

## Model-size classification 
Small: < 1B parameters
Medium: 1B–10B parameters
Large: > 10B parameters

## Models
1. `Qwen/Qwen3-0.6B` - Small
2. `meta-llama/Llama-3.1-8B` - Medium
3. `Aleph-Alpha/Kolibri-1` - Large


## 1. Small Model — Qwen3-0.6B
Model: `Qwen/Qwen3-0.6B`

### 1. Who created the model?

Qwen3-0.6B was developed by the Qwen Team. Qwen3 is the latest generation of models in the Qwen series and includes both dense and Mixture-of-Experts (MoE) models.

### 2. How many parameters does it have?

Qwen3-0.6B contains 0.6 billion parameters (0.6B). It has 0.44 billion non-embedding parameters. The model contains 28 Transformer layers.

It is classified as a small model because its total parameter count is below 1 billion.

### 3. Is it a generative model?

Yes. Qwen3-0.6B is a generative language model. Its model type is Causal Language Model, which means it predicts tokens based on previous tokens and can generate new text.
It supports capabilities such as reasoning, instruction following, dialogue, creative writing, coding, and multilingual tasks.

### 4. What is the context length?

The model has a context length of 32,768 tokens. This represents the amount of tokenized information that the model can process within its context window.

### 5. Is the model suitable for local execution?

Yes. Qwen3-0.6B is relatively suitable for local execution because it has only 0.6 billion parameters. Compared with larger models, it requires considerably fewer computational resources and is therefore more practical for experimentation on local systems.
However, the exact hardware requirement depends on factors such as precision, quantization, and the inference framework being used.

### 6. What hardware does the model card recommend?

The provided model card does not specify a particular GPU or hardware configuration for Qwen3-0.6B. Instead, it directs users to the Qwen blog, GitHub repository, and documentation for detailed information about hardware requirements and inference performance.
Therefore, a specific GPU should not be stated as the model card's official recommendation.




## 2. Medium Model — Llama 3.1-8B

Model: `meta-llama/Llama-3.1-8B`

### 1. Who created the model?

Llama 3.1 was developed by Meta. It is part of the Meta Llama 3.1 family of multilingual large language models, which includes models with 8B, 70B, and 405B parameters.

### 2. How many parameters does it have?

The Llama 3.1 model has contains 8 billion (8B) parameters.
It is classified as a medium-sized model because its parameter count falls between 1B and 10B.

### 3. Is it a generative model?

Yes. Llama 3.1 is a generative language model. The model card describes it as a collection of pretrained and instruction-tuned generative models.
Its architecture is an autoregressive language model based on an optimized Transformer architecture. It can generate multilingual text and code and can be used for natural language generation and assistant-like dialogue.

### 4. What is the context length?

The Llama 3.1-8B model has a 128,000-token(128K) context length.
This means the model can process a very large amount of tokenized text within a single context.

### 5. Is the model suitable for local execution?

Llama 3.1-8B can be used for local inference with suitable computational resources. However, compared with a smaller 0.6B model, it requires substantially more memory and computational resources.
The actual hardware requirement depends on factors such as the numerical precision, quantization, inference framework, and available GPU memory.

Therefore, it can be considered suitable for local execution on a sufficiently capable system, particularly when using optimized or quantized versions.

### 6. What hardware does the model card recommend?

The model card states that Meta used a custom GPU cluster and reports training on H100-80GB GPUs. For the Llama 3.1-8B model, the reported training computation was approximately 1.46 million GPU hours, using hardware with a peak power consumption of 700W per GPU.
However, this information describes the training hardware used by Meta, rather than a specific hardware recommendation for local inference.

Therefore, the model card does not provide a specific consumer GPU recommendation for running Llama 3.1-8B locally.




## 3. Large Model — Kolibri 1

Model: `Aleph-Alpha/Kolibri-1`

### 1. Who created the model?

Kolibri 1 was developed by Aleph Alpha Research GmbH and is provided by Aleph Alpha GmbH. It is a Mixture-of-Experts (MoE) reasoning model focused primarily on German and English. The model supports explicit reasoning mode and tool calling.

### 2. How many parameters does it have?

Kolibri 1 has 78 billion total parameters (78B). Since it uses a Mixture-of-Experts architecture, only 3.46 billion parameters are active per token.
The distinction between total and active parameters is important: the model is classified as a 78B model based on its total parameter count, even though only a smaller portion is activated for each token.

It is classified as a large model because its total parameter count is significantly greater than 10 billion.

### 3. Is it a generative model?

Yes. Kolibri 1 is a generative reasoning model. It is designed to process text input and generate text output in German and English.
The model is intended for tasks including multi-step reasoning, coding, structured extraction, retrieval-augmented generation, long-document processing, and agentic tool calling.

### 4. What is the context length?

Kolibri 1 has a maximum context length of 1,048,576 tokens (1M tokens).
However, the model documentation recommends using at most 262,144 tokens for serving efficiency and complex tasks. The model was trained through a long-context phase using 262,144-token sequences and its quality and serving efficiency have been validated up to 1,048,576 tokens. 

### 5. Is the model suitable for local execution?

Kolibri 1 can technically be executed locally, but it is not suitable for ordinary laptops or standard consumer computers because of its large memory and computational requirements.

The model has a memory footprint of approximately 78 GB even with FP8 weights. Therefore, local execution requires a high-end GPU system with substantial GPU memory.

### 6. What hardware does the model card recommend?

The model card specifies the following hardware requirements:

Minimum hardware:
- 2 × NVIDIA A100 80GB
- 2 × NVIDIA H100 SXM5
- 1 × NVIDIA H200
- 1 × NVIDIA B200
- 1 × NVIDIA B300

Recommended hardware:
- 2 × NVIDIA H100 SXM5
- 2 × NVIDIA H200
- 1 × NVIDIA B200
- 1 × NVIDIA B300

The model has an approximately 78GB memory footprint using FP8 weights.
