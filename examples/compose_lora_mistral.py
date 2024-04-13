"""
Replaces ff layers using MOE. rest all will be averaged
"""
import torch
from mergoo.compose_experts import ComposeExperts
from mergoo.models.modeling_mistral import MistralForCausalLM

model_id = "data/mistral_lora_moe"
config = {
    "model_type": "mistral",
    "num_experts_per_tok": 2,
    "base_model": "mistralai/Mistral-7B-v0.1",
    "experts": [
        {"expert_name": "adapter_1", "model_id": "predibase/customer_support"},
        {"expert_name": "adapter_2", "model_id": "predibase/customer_support_accounts"},
        {"expert_name": "adapter_3", "model_id": "predibase/customer_support_orders"},
        {"expert_name": "adapter_4", "model_id": "predibase/customer_support_payments"},
    ],
}

# create checkpoint
import os

if not os.path.exists(model_id):
    expertcomposer = ComposeExperts(config)
    expertcomposer.compose()
    expertcomposer.save_checkpoint(model_id)


# load the composed checkkpoint
model = MistralForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)  # 'gate' / router layers are untrained hence loaded warning would appeare for them

out = model(torch.tensor([[1, 2, 3, 33, 44]], device=model.device))
print("done")
