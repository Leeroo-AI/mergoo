<h1>Mergoo

<img alt='Leeroo logo' src='https://github.com/Leeroo-AI/mergoo/blob/main/static/logo.png?raw=true' width='148' align='right' />

</h1>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-green.svg)](#python)
[![License: LPGLv3.0](https://img.shields.io/badge/License-LGPLv3.0-yellow.svg)](https://www.gnu.org/licenses/lgpl-3.0.en.html) 
[![Version](https://img.shields.io/pypi/v/mergoo?color=blue)](https://pypi.org/project/mergoo/)



`mergoo` is a library for easily merging multiple LLM experts, and efficiently train the merged LLM. With `mergoo`, you can efficiently integrate the knowledge of different generic or domain-based LLM experts.

<img src='https://github.com/Leeroo-AI/mergoo/blob/main/static/base_light.png?raw=true' />

## 🚀 Features
- Supports recent merging methods including Mixture-of-Experts and Layer-wise merging
- Flexible merging choice for each layer
- Base Models supported : [Llama](https://llama.meta.com/) and [Mistral](https://huggingface.co/docs/transformers/en/model_doc/mistral)
- Trainers supported : 🤗 [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer), [SFTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- Device Supported: CPU, MPS, GPU
- Training choices: Finetune Only Router of MoE layers, Fully fine-tuning of Merged LLM

 
## Installation
Install by pip:
```
pip install mergoo
```
Install latest unstable version on Github:
```
pip install git+https://github.com/Leeroo-AI/mergoo
```
Install it from the source:
```
git clone https://github.com/Leeroo-AI/mergoo
cd mergoo
pip install -e .
``` 

## Quick Start

**Merging Models**  
A sample usage of config and create the merged model
```python
import torch
from mergoo.compose_experts import ComposeExperts

model_id = "data/mistral-math-code-moe"
config = {
    "model_type": "mistral",
    "num_experts_per_tok": 2,
    "experts": [
        {"expert_name": "base_expert", "model_id": "mistralai/Mistral-7B-v0.1"},
        {"expert_name": "expert_1", "model_id": "meta-math/MetaMath-Mistral-7B"},
        {"expert_name": "expert_2", "model_id": "ajibawa-2023/Code-Mistral-7B"}
    ],
    "router_layers": ["gate_proj", "up_proj", "down_proj"]
}

# create checkpoint
expertmerger = ComposeExperts(config, torch_dtype=torch.float16)
expertmerger.compose()
expertmerger.save_checkpoint(model_id)
```
<br>

**Loading / Finetunning Merged models**
```python
from transformers import Trainer
from mergoo.models.modeling_mistral import MistralForCausalLM

model = MistralForCausalLM.from_pretrained("data/mistral-math-code-moe") 
# NOTE: 'gate' / router layers are untrained hence weight loading warning would appeare for them

trainer = Trainer( ... )
trainer.train()
```
## 📚 Learn More:

After finishing the Quick Start guide, you can explore the tutorials below to further familiarize yourself with `mergoo`.

<table>
<thead>
  <tr>
      <th><b>Notebook</b></th>
      <th><b>Details</b></th>
  </tr>
</thead>
<tbody>
    <tr>
    <td><a href="https://github.com/Leeroo-AI/mergoo/blob/main/notebooks/llama_compose_trainer.ipynb"> Unified MoE with Domain Experts </a></td>
    <td>Build a unifined Mixture-of-Experts model with domain-based LLM experts, inspired by <a href=https://arxiv.org/html/2403.07816v1> BTX Research</a>.</td>
  </tr>
</tbody>
</table>

## Mergoo Roadmap and Contributing

As an open-source library in a fast evolving domain, we welcome contributions, whether it is introducing new features, enhancing infrastructure, or improving documentation.

Here is `mergoo` roadmap:

- [X] Support MoE for Transformer Block
- [X] Compatibility with Huggingface 🤗
- [X] Support [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer), [SFTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [X] Loading Unified Checkpoint in BTX
- [X] Feature: Convertible QKV linear layers 
- [X] Feature: Convertible FF linear layers 
- [X] Feature: Routers only for a list of decoder layers indexes
- [X] Sharded [Safetensor](https://github.com/huggingface/safetensors) Saving
- [X] Support experts based on [LLaMa](https://huggingface.co/docs/transformers/en/model_doc/llama) and [Mistral](https://huggingface.co/docs/transformers/en/model_doc/mistral)
- [ ] Router Load balancing loss
- [ ] Lazy loading of tensors for low memory usage in Merging
- [ ] Support Mixture of LORA Expert ( Base model with multiple trained LORAs)
- [ ] Support Layer-wise merging, including [Mergekit](https://github.com/arcee-ai/mergekit)
- [ ] Support experts based on [Gemma](https://blog.google/technology/developers/gemma-open-models) and [Mamba](https://arxiv.org/abs/2312.00752)
- [ ] Support flash-attention
- [ ] Support Mixture of Depths Transformer

Feel free to suggest new features and/or contribute to `mergoo` roadmap!

Join our community!
-------------
🚀 We love to here your feedback, please join Leeroo community:

- [Twitter](https://twitter.com/LeerooAI)
- [LinkedIn](https://www.linkedin.com/company/leeroo)
- [Website](https://www.leeroo.com)
- [Discord](https://discord.gg/tmuXGe3K)

Have a question not listed here? Open a GitHub Issue or send us an [email](support@leeroo.com)!
