"""
Replaces ff layers using MOE. rest all will be averaged
"""
import torch
from mergoo.compose_experts import ComposeExperts
from mergoo.models.modeling_bert import BertLMHeadModel

model_id = "data/checkpoint_demo"
config = {
    "model_type": "bert",
    "num_experts_per_tok": 2,
    
    "base_model": "google-bert/bert-base-uncased",
    "experts": [
        {"expert_name": "adapter_1", "model_id": "alexdbz/bert-base-peft-Lora-abstracts-6epochs"},
        {"expert_name": "adapter_2", "model_id": "alexdbz/bert-base-peft-Lora-abstracts-2epochs"}
    ],
}

# create checkpoint
import os
expertcomposer = ComposeExperts(config)
expertcomposer.compose()
expertcomposer.save_checkpoint(model_id)



# load the composed checkkpoint
model = BertLMHeadModel.from_pretrained(
    model_id
)  # 'gate' / router layers are untrained hence loaded warning would appeare for them

out = model(torch.tensor([[1, 2, 3, 33, 44]]))
print("done")
