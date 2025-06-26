#!/usr/bin/env python3
"""
MIG-DPG 演示脚本
展示如何使用MIG-DPG模型进行训练和推理
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dgl
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'MIG-DPG'))

from mig_dpg.models.mig_dpg_model import MIG_DPG_Model
from mig_dpg.configs.mig_dpg_default_config import MIG_DPG_DefaultConfig
import warnings
warnings.filterwarnings('ignore')

def create_demo_data():
    """创建演示用的模拟数据"""
    num_users = 100
    num_items = 50
    embedding_size = 64
    
    # 模拟用户-物品交互图
    edges = []
    for u in range(num_users):
        # 每个用户随机交互3-8个物品
        num_interactions = np.random.randint(3, 9)
        items = np.random.choice(num_items, num_interactions, replace=False)
        for item in items:
            edges.append((u, num_users + item))  # 用户节点 + 物品节点
    
    # 创建双向边
    edges = edges + [(v, u) for (u, v) in edges]
    
    # 创建DGL图
    src, dst = zip(*edges)
    g = dgl.graph((src, dst))
    
    # 添加自环
    g = dgl.add_self_loop(g)
    
    # 模拟特征
    user_embeddings = torch.randn(num_users, embedding_size)
    item_embeddings = torch.randn(num_items, embedding_size)
    item_text_features = torch.randn(num_items, 384)  # 文本特征
    item_vision_features = torch.randn(num_items, 4096)  # 视觉特征
    
    return {
        'graph': g,
        'user_embeddings': user_embeddings,
        'item_embeddings': item_embeddings,
        'item_text_features': item_text_features,
        'item_vision_features': item_vision_features,
        'num_users': num_users,
        'num_items': num_items
    }

def demo_recommendation():
    """演示推荐功能"""
    print("=" * 60)
    print("🎯 MIG-DPG 推荐系统演示")
    print("=" * 60)
    
    # 1. 准备数据
    print("📊 准备模拟数据...")
    data = create_demo_data()
    
    # 2. 创建配置
    print("⚙️ 创建模型配置...")
    config = MIG_DPG_DefaultConfig()
    config.num_users = data['num_users']
    config.num_items = data['num_items']
    config.embedding_size = 64
    config.text_feat_size = 384
    config.vision_feat_size = 4096
    
    # 3. 创建模型
    print("🧠 初始化MIG-DPG模型...")
    model = MIG_DPG_Model(config)
    model.eval()
    
    print(f"   - 用户数量: {config.num_users}")
    print(f"   - 物品数量: {config.num_items}")
    print(f"   - 嵌入维度: {config.embedding_size}")
    print(f"   - DPO启用: {config.dpo_enabled}")
    print(f"   - 生成解释启用: {config.generation_enabled}")
    
    # 4. 推荐预测
    print("\n🔍 生成推荐列表...")
    with torch.no_grad():
        topk_items = model.predict_recommendations(
            g=data['graph'],
            user_embeddings=data['user_embeddings'],
            item_v_feat=data['item_vision_features'],
            item_t_feat=data['item_text_features'],
            item_embeddings=data['item_embeddings'],
            topk=10
        )
    
    # 显示结果
    print("✅ 推荐结果 (前5个用户):")
    for i in range(min(5, data['num_users'])):
        rec_items = topk_items[i][:5].tolist()
        print(f"   用户 {i}: 推荐物品 {rec_items}")
    
    return model, data

def demo_dpo_training():
    """演示DPO训练功能"""
    print("\n" + "=" * 60)
    print("🎓 DPO偏好优化演示")
    print("=" * 60)
    
    model, data = demo_recommendation()
    
    # 准备训练数据
    batch_size = 32
    user_indices = torch.randint(0, data['num_users'], (batch_size,))
    pos_items = torch.randint(0, data['num_items'], (batch_size,))
    neg_items = torch.randint(0, data['num_items'], (batch_size,))
    
    print(f"📝 准备训练数据 (batch_size={batch_size})...")
    
    # DPO训练
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("🚀 开始DPO训练...")
    for epoch in range(3):  # 简短演示
        outputs = model.forward(
            g=data['graph'],
            user_embeddings=data['user_embeddings'],
            item_v_feat=data['item_vision_features'],
            item_t_feat=data['item_text_features'],
            item_embeddings=data['item_embeddings'],
            positive_items=pos_items,
            negative_items=neg_items,
            mode='joint'
        )
        
        total_loss, loss_dict = model.compute_joint_loss(
            outputs, user_indices, pos_items, neg_items
        )
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"   Epoch {epoch+1}: Total Loss = {total_loss:.4f}")
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                print(f"     - {loss_name}: {loss_value:.4f}")
    
    print("✅ DPO训练完成!")
    return model, data

def demo_explanation_generation():
    """演示解释生成功能"""
    print("\n" + "=" * 60)
    print("💬 生成式解释演示")
    print("=" * 60)
    
    model, data = demo_dpo_training()
    
    # 为推荐生成解释
    print("🔮 为推荐结果生成解释...")
    
    # 选择几个用户-物品对
    target_users = torch.tensor([0, 1, 2, 3, 4])
    target_items = torch.randint(0, data['num_items'], (5,))
    
    model.eval()
    with torch.no_grad():
        explanations = model.generate_explanations(
            g=data['graph'],
            user_embeddings=data['user_embeddings'][target_users],
            item_v_feat=data['item_vision_features'],
            item_t_feat=data['item_text_features'],
            target_items=target_items,
            item_embeddings=data['item_embeddings']
        )
    
    print("✅ 生成的解释 (token序列):")
    for i, (user_id, item_id, explanation) in enumerate(zip(target_users, target_items, explanations)):
        # 截取前10个token用于显示
        tokens_preview = explanation[:10] if len(explanation) > 10 else explanation
        print(f"   用户 {user_id} → 物品 {item_id}: {tokens_preview}...")
    
    print("\n💡 注意: 这些是token ID，实际应用中需要转换为文本")

def demo_model_analysis():
    """演示模型结构分析"""
    print("\n" + "=" * 60)
    print("🔬 模型结构分析")
    print("=" * 60)
    
    config = MIG_DPG_DefaultConfig()
    model = MIG_DPG_Model(config)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📈 模型统计:")
    print(f"   - 总参数数量: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")
    
    # 各组件参数统计
    mig_gt_params = sum(p.numel() for p in model.mig_gt_encoder.parameters())
    dpo_params = sum(p.numel() for p in model.dpo_layer.parameters())
    gen_params = sum(p.numel() for p in model.generation_layer.parameters())
    
    print(f"\n🧩 组件参数分布:")
    print(f"   - MIG-GT编码器: {mig_gt_params:,} ({100*mig_gt_params/total_params:.1f}%)")
    print(f"   - DPO偏好层: {dpo_params:,} ({100*dpo_params/total_params:.1f}%)")
    print(f"   - 生成解释层: {gen_params:,} ({100*gen_params/total_params:.1f}%)")
    
    # 显示训练模式
    print(f"\n⚙️ 训练配置:")
    print(f"   - 推荐权重: {model.recommendation_weight}")
    print(f"   - DPO权重: {model.dpo_weight}")
    print(f"   - 生成权重: {model.generation_weight}")

def main():
    """主演示函数"""
    print("🌟 MIG-DPG: Multimodal Independent Graph Neural Networks")
    print("    with Direct Preference Optimization and Generation")
    print("🌟 欢迎使用MIG-DPG演示系统!")
    
    try:
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 运行各种演示
        demo_model_analysis()
        demo_recommendation()
        demo_dpo_training()
        demo_explanation_generation()
        
        print("\n" + "=" * 60)
        print("🎉 演示完成! MIG-DPG系统运行正常")
        print("=" * 60)
        print("\n📚 下一步:")
        print("1. 准备真实数据集 (Amazon, Yelp等)")
        print("2. 配置模型超参数")
        print("3. 运行完整训练流程")
        print("4. 评估推荐性能和解释质量")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        print("请检查依赖项和代码配置")

if __name__ == "__main__":
    main() 