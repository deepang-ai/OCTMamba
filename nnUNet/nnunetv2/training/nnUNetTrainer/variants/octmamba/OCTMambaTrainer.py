from typing import List, Tuple, Union
import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.OCTMamba import OCTMamba


class nnUNetTrainerOCTMamba(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.set_deep_supervision_enabled = lambda enabled: None

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = OCTMamba(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            channels=(16, 32, 64, 128, 256),
            num_heads=(2, 4, 8, 16),
            strides=(4, 2, 2, 1),
            coord=False,
            dropout=0.3,
        )
        self.configuration_manager.configuration["patch_size"] = (96, 96, 96)
        return model
