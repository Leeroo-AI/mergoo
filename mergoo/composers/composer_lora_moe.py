import os
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import PeftConfig
from mergoo.safe_saving import save_pretrained


class ComposeLoraMoeExperts:
    def __init__(
        self,
        config,
        torch_dtype=torch.float16,
        device="cpu",
        device_map="auto",
        max_shard_size="9GB",
        model_cls=AutoModelForCausalLM,
    ):
        """
        Args:
            config (dict): Configuration required to setup the composer. Explore configs/ for examples/
            torch_dtype (torch.dtype, optional): Datatype for loading and saving the weights. Defaults to torch.float16.
            device (str, optional): Defaults to "cpu".
            device_map (str, optional): Defaults to "auto".
            max_shard_size (str, optional): Maximum Shard size checkpoint chuncks. Defaults to "9GB".
            model_cls (type, optional): Change this when using a architecture not registered with transformers. Defaults to AutoModelForCausalLM.
        """
        self.config = config
        self.torch_dtype = torch_dtype
        self._tied_weights_keys = []
        self.device = device
        self.device_map = device_map
        self.max_shard_size = max_shard_size
        self.model_cls = model_cls
        self.config["router_layers_index"] = self.config.get("router_layers_index", [])
        self.moe_layer_index = self.config["router_layers_index"]
        self.select_moe_model_config_idx = 0
        self.config["adapter_configs"] = []
        self._set_moe_layer_index()

    def _set_moe_layer_index(self):
        if len(self.moe_layer_index) == 0:
            self._check_moe_layers = lambda x: True
            print(f"MoE Layer Index : [*]")

        elif len(self.moe_layer_index) >= 1 and self.moe_layer_index[0] is not None:
            self._check_moe_layers = lambda x: x in self.moe_layer_index
            print(f"MoE Layer Index : {self.moe_layer_index}")

        else:
            self._check_moe_layers = lambda x: False
            print(f"No MoE layer indexes.")

    def _is_layer_suitable_for_router(self, layer_identifier, model_layer):
        model_layer_index = [int(x) for x in model_layer.split(".") if x.isdigit()]
        if not model_layer_index:
            valid_layer_index = False
        else:
            if "lora" in model_layer.lower():
                assert len(model_layer_index) == 2
            else:
                assert len(model_layer_index) == 1  # [layer index, adapter index]
            valid_layer_index = self._check_moe_layers(model_layer_index[0])

        if (layer_identifier in model_layer) and valid_layer_index:
            return True
        return False

    def _load_base_model(self, model_id):
        try:
            mps = torch.backends.mps.is_available()
        except:
            mps = False
        config = AutoConfig.from_pretrained(model_id)
        if config.model_type == "bert":
            model = self.model_cls.from_pretrained(
                model_id, torch_dtype=self.torch_dtype
            )
        else:
            model = self.model_cls.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                device_map=None if mps else self.device_map,
            )
        if mps:
            return model.to(self.device)
        return model

    def compose(self):
        """
        Compose all the experts into a single unified checkpoint.
        """
        n = len(self.config["experts"])
        model = self._load_base_model(self.config["base_model"])
        self.model_config = model.config.to_dict()
        count_total_router_layers = 0

        for ix, expert in enumerate(self.config["experts"]):
            adapter_id = expert["model_id"]
            adapter_config = PeftConfig.from_pretrained(adapter_id)
            adapter_config_ = adapter_config.to_dict()
            for k, v in adapter_config_.items():
                try:
                    json.dumps(v)
                except:
                    adapter_config_[k] = str(v)
            self.config["adapter_configs"].append(adapter_config_)
            # check if all the lora are having same target modules
            if "router_layers" in self.config:
                assert self.config["router_layers"] == list(
                    adapter_config.target_modules
                )
            else:
                self.config["router_layers"] = list(adapter_config.target_modules)
            # load the adapter
            model.load_adapter(adapter_id, adapter_name=str(ix))

        if hasattr(model, "_tied_weights_keys"):
            self._tied_weights_keys.extend(model._tied_weights_keys)

        self.state_dict = model.state_dict()

        count_router_layers = 0
        count_averaged_layers = 0
        for layer_name, param in tqdm(model.state_dict().items()):
            if (
                sum(
                    [
                        self._is_layer_suitable_for_router(router_layer, layer_name)
                        for router_layer in self.config["router_layers"]
                    ]
                )
                == 1
            ):
                # Note: Index of adapter in the config are kept as adapter names, while saving.
                # Similar should be case while loading the adapters
                assert layer_name in self.state_dict
                count_total_router_layers += 1
                count_router_layers += 1
            else:
                assert layer_name in self.state_dict
                count_averaged_layers += 1

        print(f"count_averaged_layers : {count_averaged_layers}")
        print(f"count_router_layers : {count_router_layers}")
        print(f"count_total_router_layers : {count_total_router_layers}")
        del model
        gc.collect()

    def save_checkpoint(self, checkpoint_path):
        """
        Save the composed Unified checkpoint.
        Checkpoints are saved as safe tensors in shards(chuncks).
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        config = self.model_config
        config["num_experts"] = len(self.config["experts"])
        config["num_experts_per_tok"] = self.config["num_experts_per_tok"]
        config["router_layers"] = self.config["router_layers"]
        config["adapter_configs"] = self.config["adapter_configs"]

        layer_indexes = list(range(config["num_hidden_layers"]))
        if not self.config["router_layers_index"]:
            config["router_layers_index"] = layer_indexes

        else:
            config["router_layers_index"] = list(
                set(layer_indexes).intersection(set(self.config["router_layers_index"]))
            )

        json.dump(config, open(f"{checkpoint_path}/config.json", "w"), indent=1)
        save_pretrained(
            save_directory=checkpoint_path,
            state_dict=self.state_dict,
            tied_weights_keys=self._tied_weights_keys,
            max_shard_size=self.max_shard_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config["base_model"])
        tokenizer.save_pretrained(checkpoint_path)
        print(f"checkpoint saved at {checkpoint_path}")
