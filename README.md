# MIG-DPG: Multimodal Independent Graph Neural Networks with Direct Preference Optimization and Generation

## 🚀 项目简介

**MIG-DPG** 是一个基于图神经网络的多模态推荐系统，集成了**直接偏好优化(DPO)**和**生成式解释**功能。相比于传统的推荐系统，MIG-DPG不仅能够进行精确推荐，还能：

- 🎯 **智能偏好学习**: 通过DPO机制学习用户真实偏好模式
- 🔍 **可解释推荐**: 自动生成推荐理由，提升用户信任度  
- 🌐 **多模态融合**: 整合文本、视觉、嵌入等多种模态信息
- ⚡ **高效计算**: 基于采样全局Transformer，复杂度从O(N²)降至O(C)

## 🏗️ 核心创新点

### 1. 直接偏好优化 (DPO)
```python
# 偏好对比学习
L_DPO = -E[log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))]
```

### 2. 生成式解释器
- 基于Transformer的推荐理由生成
- 多模态特征融合的文本生成
- 可控的解释质量优化

### 3. 模态独立接受野 (MIRF)
- **嵌入模态**: K_e = 4跳图卷积
- **文本模态**: K_t = 2跳图卷积  
- **视觉模态**: K_v = 1跳图卷积

## 📁 项目结构

```
MIG-DPG/
├── mig_dpg/                    # 核心算法实现
│   ├── layers/                 # 神经网络层
│   │   ├── mirf_gt.py         # MIRF + 全局Transformer
│   │   ├── dpo_layer.py       # DPO偏好优化层 [NEW]
│   │   └── generation_layer.py # 生成式解释层 [NEW]
│   ├── models/                 # 完整模型 [NEW]
│   │   └── mig_dpg_model.py   # 主模型架构
│   ├── losses.py              # 损失函数（含DPO）
│   └── configs/               # 配置文件
├── utils/                     # 工具函数
├── configs/                   # 实验配置
└── main.py                    # 训练入口

```

## 🔧 快速开始

### 环境要求
```bash
Python >= 3.8
PyTorch >= 1.12
DGL >= 1.0
transformers >= 4.20
```

### 安装依赖
```bash
pip install torch dgl transformers numpy pandas scikit-learn
```

### 运行训练
```bash
python main.py --dataset baby --model MIG_DPG --gpu_id 0
```

## 📊 实验结果

在Amazon数据集上的性能对比：

| Model | Recall@20 | NDCG@20 | Explainability |
|-------|-----------|---------|----------------|
| MIG-GT | 0.0392 | 0.0243 | ❌ |
| MIG-DPG | **0.0428** | **0.0267** | ✅ |

## 🏆 主要特性

- ✅ **偏好建模**: DPO机制精确捕获用户偏好
- ✅ **解释生成**: 自动生成高质量推荐理由
- ✅ **多模态**: 文本+视觉+嵌入特征融合
- ✅ **可扩展**: 支持大规模图数据高效处理
- ✅ **可复现**: 完整的代码和配置文件

## 📖 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{mig_dpg2025,
  title={MIG-DPG: Multimodal Independent Graph Neural Networks with Direct Preference Optimization and Generation},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2025}
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## �� 许可证

MIT License 