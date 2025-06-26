"""
MIG-DPG: Multimodal Independent Graph Neural Networks 
with Direct Preference Optimization and Generation

主包初始化文件
"""

from .models.mig_dpg_model import MIG_DPG_Model
from .configs.mig_dpg_default_config import MIG_DPG_DefaultConfig
from .layers.dpo_layer import DPOLayer
from .layers.generation_layer import GenerativeExplanationLayer

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    'MIG_DPG_Model',
    'MIG_DPG_DefaultConfig', 
    'DPOLayer',
    'GenerativeExplanationLayer'
] 