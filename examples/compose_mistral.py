"""
Replaces ff layers using MOE. rest all will be averaged
"""
import torch
import os
from mergoo.compose_experts import ComposeExperts
from mergoo.models.modeling_mistral import MistralForCausalLM

model_id = "data/checkpoint_demo"
config = {
    "model_type": "mistral",
    "num_experts_per_tok": 2,
    "experts": [
        {"expert_name": "base_expert", "model_id": "mistralai/Mistral-7B-v0.1"},
        {"expert_name": "expert_1", "model_id": "meta-math/MetaMath-Mistral-7B"},
    ],
    "router_layers": ["gate_proj", "up_proj", "down_proj"],
    "router_layers_index": [
        0,
        1,
        2,
        3,
        4,
    ],
}

# create checkpoint
if not os.path.exists(model_id):
    expertmerger = ComposeExperts(config, torch_dtype=torch.float16)
    expertmerger.compose()
    expertmerger.save_checkpoint(model_id)

# restart and continue from here if gpu/ram cache is not cleared
device = "cuda:0"
model = MistralForCausalLM.from_pretrained(model_id).to(
    device
)  # 'gate' / router layers are untrained hence loaded warning would appeare for them
out = model(torch.tensor([[1, 2, 3, 33, 44]]).to(device))
print("done")
