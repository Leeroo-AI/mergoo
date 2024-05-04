# Changelog

*This file contains a high-level description of changes that were merged into the mergoo main branch since the last release.

## 🚀 Features
0.0.9:
- Support Phi3  

0.0.8:  
- Support LLaMa3
- Notebook added for [LLaMa3 based LLMs](https://github.com/Leeroo-AI/mergoo/blob/main/notebooks/integrate_llama3_experts.ipynb)  

0.0.7: 
- Supports Mixture of adapters
- Notebook added for [Mixture of adapters](https://github.com/Leeroo-AI/mergoo/blob/main/notebooks/Mistral_lora_compose_trainer.ipynb)

0.0.6:  
- Supports recent merging methods including Mixture-of-Experts and Layer-wise merging
- Flexible merging choice for each layer
- Base Models supported : [Llama](https://llama.meta.com/) and [Mistral](https://huggingface.co/docs/transformers/en/model_doc/mistral)
- Trainers supported : 🤗 [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer), [SFTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- Device Supported: CPU, MPS, GPU
- Training choices: Finetune Only Router of MoE layers, Fully fine-tuning of Merged LLM

## 🔧 Fixes & Refactoring
0.0.8:  
- issue in pip installation ([#8](https://github.com/Leeroo-AI/mergoo/pull/8))