# MIG-DPG 快速开始指南

## 🚀 快速运行

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 如果遇到DGL安装问题，请参考：
# CPU版本: pip install dgl -f https://data.dgl.ai/wheels/repo.html
# GPU版本: pip install dgl-cuda11.6 -f https://data.dgl.ai/wheels/repo.html (根据CUDA版本调整)
```

### 2. 运行演示

```bash
# 基础演示（无需GPU）
python demo_mig_dpg.py
```

### 3. 完整训练

```bash
# 基础训练（使用默认参数）
python train_mig_dpg.py --dataset baby --epochs 50

# 高级训练（自定义参数）
python train_mig_dpg.py \
    --dataset baby \
    --epochs 100 \
    --batch_size 512 \
    --learning_rate 1e-3 \
    --training_strategy joint \
    --dpo_weight 0.5 \
    --generation_weight 0.3 \
    --synthetic_explanation \
    --save_model

# GPU训练
python train_mig_dpg.py --dataset baby --gpu_id 0
```

## 📊 支持的数据集

- `baby`: Amazon Baby产品数据 (默认)
- `sports`: Amazon Sports & Outdoors数据  
- `clothing`: Amazon Clothing数据
- `elec`: Amazon Electronics数据

## ⚙️ 主要参数

### 模型参数
- `--embedding_size`: 嵌入维度 (默认: 64)
- `--k_e`, `--k_t`, `--k_v`: 各模态的图卷积跳数 (默认: 4,2,1)

### 训练策略
- `--training_strategy`: 训练策略
  - `joint`: 联合训练 (默认)
  - `sequential`: 分阶段训练
  - `curriculum`: 课程学习

### 损失权重
- `--dpo_weight`: DPO损失权重 (默认: 0.5)
- `--generation_weight`: 生成损失权重 (默认: 0.3)

## 📈 预期结果

### 性能提升
相比原始MIG-GT，预期能实现：
- **Recall@20**: 提升 5-10%
- **NDCG@20**: 提升 3-8%
- **可解释性**: 新增推荐理由生成
- **偏好对齐**: 通过DPO优化用户偏好

### 训练时间
- **CPU训练**: ~2-4小时 (50 epochs, baby数据集)
- **GPU训练**: ~30-60分钟 (50 epochs, baby数据集)

## 🔧 自定义数据

如需使用自己的数据，请参考 `mig_dpg/data_processor.py` 中的数据格式：

```python
# 用户-物品交互
user_item_interactions = [(user_id, item_id), ...]

# 偏好数据 (可选)
preference_data = [(user_id, preferred_item, non_preferred_item), ...]

# 解释数据 (可选)
explanation_data = {(user_id, item_id): [token_list], ...}
```

## 📝 输出文件

训练完成后会在 `saved_models/实验名称/` 目录下生成：

- `best_model.pth`: 最佳模型权重
- `training_history.json`: 训练历史
- `final_results.json`: 最终评估结果

## 🐛 常见问题

### Q1: DGL安装失败
**A**: 请根据你的CUDA版本安装对应的DGL版本，或使用CPU版本进行测试。

### Q2: 内存不足
**A**: 尝试减小batch_size参数，如 `--batch_size 256`。

### Q3: 训练速度慢
**A**: 确保使用GPU训练 `--gpu_id 0`，或减少epochs `--epochs 20`。

### Q4: 损失不收敛
**A**: 尝试调整学习率 `--learning_rate 1e-4` 或使用sequential训练策略。

## 🎯 下一步

1. **调参优化**: 根据你的数据特点调整模型参数
2. **评估指标**: 添加更多推荐和解释质量指标
3. **真实数据**: 替换为真实的推荐数据集
4. **生产部署**: 优化模型以适配生产环境

## 📞 技术支持

如遇到问题，请检查：
1. 依赖是否正确安装
2. Python版本 (建议3.8+)
3. 系统资源是否充足

祝您使用愉快！🌟 