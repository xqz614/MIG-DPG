#!/usr/bin/env python3
"""
MIG-DPG 简化演示脚本
测试核心组件功能，避开DGL依赖问题
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

def test_dpo_layer():
    """测试DPO层功能"""
    print("=" * 50)
    print("🧪 测试DPO偏好优化层")
    print("=" * 50)
    
    try:
        from mig_dpg.layers.dpo_layer import DPOLayer
        
        # 创建DPO层
        config = {
            'embedding_size': 64,
            'dpo_hidden_dim': 128,
            'dpo_num_heads': 4,
            'dpo_beta': 0.1
        }
        
        dpo_layer = DPOLayer(config)
        
        # 创建测试数据
        batch_size = 32
        user_emb = torch.randn(batch_size, 64)
        pos_item_emb = torch.randn(batch_size, 64)
        neg_item_emb = torch.randn(batch_size, 64)
        
        print(f"输入形状:")
        print(f"  用户嵌入: {user_emb.shape}")
        print(f"  正样本嵌入: {pos_item_emb.shape}")
        print(f"  负样本嵌入: {neg_item_emb.shape}")
        
        # 前向传播
        dpo_loss = dpo_layer(user_emb, pos_item_emb, neg_item_emb)
        
        print(f"✅ DPO损失: {dpo_loss:.4f}")
        print(f"损失形状: {dpo_loss.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ DPO层测试失败: {e}")
        return False

def test_generation_layer():
    """测试生成层功能"""
    print("\n" + "=" * 50)
    print("🧪 测试生成式解释层")
    print("=" * 50)
    
    try:
        from mig_dpg.layers.generation_layer import GenerationLayer
        
        # 创建生成层
        config = {
            'embedding_size': 64,
            'vocab_size': 1000,
            'hidden_dim': 128,
            'num_layers': 2,
            'num_heads': 4,
            'max_seq_length': 50
        }
        
        gen_layer = GenerationLayer(config)
        
        # 创建测试数据
        batch_size = 16
        user_item_emb = torch.randn(batch_size, 64)
        target_tokens = torch.randint(0, 1000, (batch_size, 20))
        
        print(f"输入形状:")
        print(f"  用户-物品嵌入: {user_item_emb.shape}")
        print(f"  目标tokens: {target_tokens.shape}")
        
        # 前向传播 - 训练模式
        gen_layer.train()
        loss = gen_layer(user_item_emb, target_tokens)
        
        print(f"✅ 生成损失: {loss:.4f}")
        
        # 推理模式 - 生成文本
        gen_layer.eval()
        with torch.no_grad():
            generated = gen_layer.generate(user_item_emb[:4], max_length=15)
        
        print(f"生成的token序列:")
        for i, seq in enumerate(generated):
            print(f"  样本 {i}: {seq[:10]}...")  # 显示前10个token
        
        return True
        
    except Exception as e:
        print(f"❌ 生成层测试失败: {e}")
        return False

def test_data_processor():
    """测试数据处理器"""
    print("\n" + "=" * 50)
    print("🧪 测试数据处理器")
    print("=" * 50)
    
    try:
        from mig_dpg.data_processor import MultiTaskDataset
        
        # 创建模拟数据
        num_users, num_items = 100, 50
        interactions = []
        
        # 生成用户-物品交互
        for u in range(num_users):
            num_int = np.random.randint(3, 8)
            items = np.random.choice(num_items, num_int, replace=False)
            for item in items:
                interactions.append([u, item, 1])  # [user, item, rating]
        
        dataset = MultiTaskDataset(
            interactions=interactions,
            num_users=num_users,
            num_items=num_items,
            negative_sampling_ratio=4
        )
        
        print(f"数据集统计:")
        print(f"  交互数量: {len(interactions)}")
        print(f"  用户数量: {num_users}")
        print(f"  物品数量: {num_items}")
        
        # 测试数据加载
        sample = dataset[0]
        print(f"样本数据形状:")
        for key, value in sample.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据处理器测试失败: {e}")
        return False

def test_trainer():
    """测试训练器配置"""
    print("\n" + "=" * 50)
    print("🧪 测试训练器配置")
    print("=" * 50)
    
    try:
        from mig_dpg.trainer import MIG_DPG_Trainer
        from mig_dpg.configs.mig_dpg_default_config import MIG_DPG_DefaultConfig
        
        # 创建配置
        config = MIG_DPG_DefaultConfig()
        config.num_users = 100
        config.num_items = 50
        config.embedding_size = 64
        
        print(f"训练配置:")
        print(f"  学习率: {config.learning_rate}")
        print(f"  批次大小: {config.batch_size}")
        print(f"  嵌入维度: {config.embedding_size}")
        print(f"  DPO启用: {config.dpo_enabled}")
        print(f"  生成启用: {config.generation_enabled}")
        print(f"  训练策略: {config.training_strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练器测试失败: {e}")
        return False

def test_basic_pytorch():
    """测试基础PyTorch功能"""
    print("\n" + "=" * 50)
    print("🧪 测试基础PyTorch环境")
    print("=" * 50)
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    if cuda_available:
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
    
    # 创建简单张量操作
    x = torch.randn(100, 64)
    y = torch.randn(64, 32)
    z = torch.mm(x, y)
    
    print(f"张量运算测试:")
    print(f"  x形状: {x.shape}")
    print(f"  y形状: {y.shape}")
    print(f"  z形状: {z.shape}")
    print(f"  z均值: {z.mean():.4f}")
    
    # 测试神经网络
    net = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.Softmax(dim=1)
    )
    
    output = net(x)
    print(f"神经网络输出形状: {output.shape}")
    print(f"输出和: {output.sum(dim=1)[:5]}")  # softmax应该和为1
    
    return True

def main():
    """主测试函数"""
    print("🚀 MIG-DPG 简化功能测试")
    print("=" * 60)
    
    results = []
    
    # 基础环境测试
    results.append(("PyTorch环境", test_basic_pytorch()))
    
    # 组件测试
    results.append(("DPO层", test_dpo_layer()))
    results.append(("生成层", test_generation_layer()))
    results.append(("数据处理器", test_data_processor()))
    results.append(("训练器配置", test_trainer()))
    
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
        print("🎉 所有测试通过！MIG-DPG环境准备就绪")
    else:
        print("⚠️  部分测试失败，需要检查相关组件")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ 下一步可以运行完整的训练实验")
    else:
        print("\n🔧 请先解决环境问题") 