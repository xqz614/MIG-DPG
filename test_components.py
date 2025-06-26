#!/usr/bin/env python3
"""
MIG-DPG 组件单独测试脚本
分别测试每个组件，避免DGL依赖问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_dpo_layer_direct():
    """直接测试DPO层功能"""
    print("=" * 50)
    print("🧪 直接测试DPO偏好优化层")
    print("=" * 50)
    
    try:
        # 直接导入DPO层类定义
        sys.path.insert(0, os.path.join(current_dir, 'mig_dpg', 'layers'))
        from dpo_layer import DPOLayer
        
        # 创建DPO层
        embedding_dim = 64
        dpo_layer = DPOLayer(
            embedding_dim=embedding_dim,
            hidden_dim=128,
            beta=0.1,
            dropout=0.1,
            num_heads=4
        )
        
        # 创建测试数据
        batch_size = 32
        num_items = 100
        
        user_embeddings = torch.randn(batch_size, embedding_dim)
        item_embeddings = torch.randn(num_items, embedding_dim)
        positive_items = torch.randint(0, num_items, (batch_size,))
        negative_items = torch.randint(0, num_items, (batch_size,))
        
        print(f"输入数据形状:")
        print(f"  用户嵌入: {user_embeddings.shape}")
        print(f"  物品嵌入: {item_embeddings.shape}")
        print(f"  正样本ID: {positive_items.shape}")
        print(f"  负样本ID: {negative_items.shape}")
        
        # 前向传播
        enhanced_user_emb, preference_scores = dpo_layer(
            user_embeddings, item_embeddings, positive_items, negative_items
        )
        
        print(f"输出结果:")
        print(f"  增强用户嵌入: {enhanced_user_emb.shape}")
        print(f"  正样本偏好分数: {preference_scores['pos_policy'].shape}")
        print(f"  负样本偏好分数: {preference_scores['neg_policy'].shape}")
        
        # 计算DPO损失
        dpo_loss = dpo_layer.compute_dpo_loss(preference_scores)
        print(f"✅ DPO损失: {dpo_loss:.4f}")
        
        # 测试偏好排序
        preference_matrix = dpo_layer.get_preference_ranking(user_embeddings[:5], item_embeddings)
        print(f"偏好排序矩阵: {preference_matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ DPO层测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_layer_direct():
    """直接测试生成层功能"""
    print("\n" + "=" * 50)
    print("🧪 直接测试生成式解释层")
    print("=" * 50)
    
    try:
        # 直接导入生成层类定义
        sys.path.insert(0, os.path.join(current_dir, 'mig_dpg', 'layers'))
        from generation_layer import GenerativeExplanationLayer, MultiModalFusion
        
        # 创建生成层
        gen_layer = GenerativeExplanationLayer(
            embedding_dim=64,
            text_dim=384,
            vision_dim=2048,
            hidden_dim=128,
            vocab_size=1000,
            max_length=50,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        # 创建测试数据
        batch_size = 16
        num_items = 50
        
        user_embeddings = torch.randn(batch_size, 64)
        item_embeddings = torch.randn(num_items, 64)
        item_text_features = torch.randn(num_items, 384)
        item_vision_features = torch.randn(num_items, 2048)
        target_items = torch.randint(0, num_items, (batch_size,))
        target_sequence = torch.randint(0, 1000, (batch_size, 20))
        
        print(f"输入数据形状:")
        print(f"  用户嵌入: {user_embeddings.shape}")
        print(f"  物品嵌入: {item_embeddings.shape}")
        print(f"  目标序列: {target_sequence.shape}")
        
        # 训练模式 - 计算损失
        gen_layer.train()
        outputs = gen_layer(
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            item_text_features=item_text_features,
            item_vision_features=item_vision_features,
            target_items=target_items,
            explanation_tokens=target_sequence
        )
        
        # 计算损失
        logits = outputs['logits']
        loss = gen_layer.compute_generation_loss(logits, target_sequence)
        print(f"✅ 生成损失: {loss:.4f}")
        
        # 推理模式 - 生成序列
        gen_layer.eval()
        with torch.no_grad():
            generated_outputs = gen_layer(
                user_embeddings=user_embeddings[:4],
                item_embeddings=item_embeddings,
                item_text_features=item_text_features,
                item_vision_features=item_vision_features,
                target_items=target_items[:4],
                explanation_tokens=None  # 推理模式
            )
        
        print(f"生成的序列:")
        generated_tokens = generated_outputs['generated_tokens']
        for i, seq in enumerate(generated_tokens):
            print(f"  样本 {i}: {seq[:10].tolist()}...")  # 显示前10个token
        
        # 测试多模态融合
        print(f"\n测试多模态融合...")
        fusion = MultiModalFusion(embedding_dim=64, text_dim=384, vision_dim=2048, output_dim=64)
        emb_feat = torch.randn(batch_size, 64)
        text_feat = torch.randn(batch_size, 384)
        vision_feat = torch.randn(batch_size, 2048)
        fused_feat = fusion(emb_feat, text_feat, vision_feat)
        print(f"融合特征形状: {fused_feat.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成层测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_system():
    """测试配置系统"""
    print("\n" + "=" * 50)
    print("🧪 测试配置系统")
    print("=" * 50)
    
    try:
        sys.path.insert(0, os.path.join(current_dir, 'mig_dpg', 'configs'))
        from mig_dpg_default_config import MIG_DPG_DefaultConfig
        
        # 创建配置
        config = MIG_DPG_DefaultConfig()
        
        print(f"默认配置参数:")
        print(f"  嵌入维度: {config.embedding_size}")
        print(f"  学习率: {config.learning_rate}")
        print(f"  批次大小: {config.batch_size}")
        print(f"  DPO启用: {config.dpo_enabled}")
        print(f"  生成启用: {config.generation_enabled}")
        print(f"  训练策略: {config.training_strategy}")
        print(f"  最大序列长度: {config.max_seq_length}")
        print(f"  负采样比例: {config.negative_sampling_ratio}")
        
        # 测试配置修改
        config.num_users = 200
        config.num_items = 100
        config.embedding_size = 128
        
        print(f"\n修改后配置:")
        print(f"  用户数量: {config.num_users}")
        print(f"  物品数量: {config.num_items}")
        print(f"  嵌入维度: {config.embedding_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_dataset():
    """测试简化数据集"""
    print("\n" + "=" * 50)
    print("🧪 测试简化数据集")
    print("=" * 50)
    
    try:
        # 创建简单的数据集类
        class SimpleDataset:
            def __init__(self, num_users, num_items, num_interactions=1000):
                self.num_users = num_users
                self.num_items = num_items
                
                # 生成随机交互
                self.interactions = []
                for _ in range(num_interactions):
                    user = np.random.randint(0, num_users)
                    item = np.random.randint(0, num_items)
                    rating = np.random.choice([1, 2, 3, 4, 5])
                    self.interactions.append([user, item, rating])
                
                # 生成模拟特征
                self.user_features = torch.randn(num_users, 64)
                self.item_features = torch.randn(num_items, 64)
                self.item_text_features = torch.randn(num_items, 384)
                self.item_vision_features = torch.randn(num_items, 2048)
            
            def __len__(self):
                return len(self.interactions)
            
            def __getitem__(self, idx):
                user, item, rating = self.interactions[idx]
                
                # 负采样
                neg_item = np.random.randint(0, self.num_items)
                while neg_item == item:
                    neg_item = np.random.randint(0, self.num_items)
                
                return {
                    'user': user,
                    'pos_item': item,
                    'neg_item': neg_item,
                    'rating': rating,
                    'user_feat': self.user_features[user],
                    'pos_item_feat': self.item_features[item],
                    'neg_item_feat': self.item_features[neg_item],
                    'pos_text_feat': self.item_text_features[item],
                    'pos_vision_feat': self.item_vision_features[item]
                }
        
        # 创建数据集
        dataset = SimpleDataset(num_users=100, num_items=50, num_interactions=500)
        
        print(f"数据集统计:")
        print(f"  用户数量: {dataset.num_users}")
        print(f"  物品数量: {dataset.num_items}")
        print(f"  交互数量: {len(dataset)}")
        
        # 测试数据加载
        sample = dataset[0]
        print(f"\n样本数据:")
        for key, value in sample.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
        
        # 测试批量加载
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        batch = next(iter(dataloader))
        print(f"\n批量数据形状:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 MIG-DPG 组件单独测试")
    print("=" * 60)
    
    results = []
    
    # 基础环境测试
    print("CUDA可用:", torch.cuda.is_available())
    print("PyTorch版本:", torch.__version__)
    print("")
    
    # 组件测试
    results.append(("DPO层", test_dpo_layer_direct()))
    results.append(("生成层", test_generation_layer_direct()))
    results.append(("配置系统", test_config_system()))
    results.append(("简化数据集", test_simple_dataset()))
    
    # 结果总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:15} : {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有核心组件测试通过！")
        print("✨ 可以继续进行训练实验")
    else:
        print("⚠️  部分组件测试失败")
    
    return passed == total

if __name__ == "__main__":
    success = main() 