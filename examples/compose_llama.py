"""
Replaces ff layers using MOE. rest all will be averaged
"""
import torch
from mergoo.compose_experts import ComposeExperts
from mergoo.models.modeling_llama import LlamaForCausalLM

model_id = "data/checkpoint_demo"
config = {
    "model_type": "llama",
    "num_experts_per_tok": 2,
    "experts": [
        {"expert_name": "base_expert", "model_id": "meta-llama/Llama-2-7b-hf"},
        {"expert_name": "expert_1", "model_id": "codellama/CodeLlama-7b-hf"},
        {
            "expert_name": "expert_2",
            "model_id": "stanford-oval/Llama-2-7b-WikiChat-fused",
        },
    ],
    "router_layers": ["gate_proj", "up_proj", "down_proj"],
}

# create checkpoint
expertmerger = ComposeExperts(config, torch_dtype=torch.float16)
expertmerger.compose()
expertmerger.save_checkpoint(model_id)

# load the merged checkkpoint
model = LlamaForCausalLM.from_pretrained(
    model_id
)  # 'gate' / router layers are untrained hence loaded warning would appeare for them
out = model(torch.tensor([[1, 2, 3, 33, 44]]))
print("done")
