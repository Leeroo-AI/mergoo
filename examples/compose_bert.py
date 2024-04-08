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
    "experts": [
        {"expert_name": "base_expert", "model_id": "google-bert/bert-base-uncased"},
        {
            "expert_name": "expert_1",
            "model_id": "google-bert/bert-base-german-dbmdz-uncased",
        },
    ],
    # "router_layers_index":[None]
}

# create checkpoint
expertcomposer = ComposeExperts(config)
expertcomposer.compose()
expertcomposer.save_checkpoint(model_id)

# load the composed checkkpoint
model = BertLMHeadModel.from_pretrained(
    model_id
)  # 'gate' / router layers are untrained hence loaded warning would appeare for them

out = model(torch.tensor([[1, 2, 3, 33, 44]]))
print("done")
