#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIG-DPG: 简化版全面对比实验与消融研究
避开DGL依赖问题，专注于baseline对比和消融实验

实验设计:
1. Baseline Methods 对比实验
2. Ablation Studies 消融实验
3. Hyperparameter Sensitivity 超参敏感性分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import json
import os
from tqdm import tqdm
import warnings
from sklearn.model_selection import train_test_split
import random
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ========================================
# 1. BASELINE MODELS 基线模型实现
# ========================================

class MLP_Baseline(nn.Module):
    """多层感知机基线"""
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        input_dim = embedding_dim * 2
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.mlp(x).squeeze()

class MatrixFactorization(nn.Module):
    """矩阵分解基线"""
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        scores = (user_emb * item_emb).sum(dim=-1)
        scores += self.user_bias(users).squeeze() + self.item_bias(items).squeeze() + self.global_bias
        return scores

class NeuralCF(nn.Module):
    """Neural Collaborative Filtering"""
    def __init__(self, num_users, num_items, embedding_dim=64, mlp_dims=[128, 64]):
        super().__init__()
        # GMF part
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP part
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        mlp_input_dim = embedding_dim * 2
        mlp_layers = []
        for dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(mlp_input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            mlp_input_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.predict_layer = nn.Linear(embedding_dim + mlp_dims[-1], 1)
        
    def forward(self, users, items):
        # GMF part
        gmf_user_emb = self.gmf_user_embedding(users)
        gmf_item_emb = self.gmf_item_embedding(items)
        gmf_output = gmf_user_emb * gmf_item_emb
        
        # MLP part
        mlp_user_emb = self.mlp_user_embedding(users)
        mlp_item_emb = self.mlp_item_embedding(items)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate GMF and MLP outputs
        concat_output = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.predict_layer(concat_output).squeeze()
        return prediction

class SimpleDPO(nn.Module):
    """简化的DPO组件"""
    def __init__(self, embedding_dim):
        super().__init__()
        self.preference_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
    def forward(self, user_emb, item_emb):
        concat_emb = torch.cat([user_emb, item_emb], dim=-1)
        preference_score = self.preference_head(concat_emb)
        return preference_score.squeeze()

class SimpleGeneration(nn.Module):
    """简化的生成组件"""
    def __init__(self, embedding_dim, vocab_size=1000):
        super().__init__()
        self.vocab_size = vocab_size
        self.generation_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, vocab_size)
        )
        
    def forward(self, user_emb, item_emb):
        concat_emb = torch.cat([user_emb, item_emb], dim=-1)
        logits = self.generation_head(concat_emb)
        return logits

# ========================================
# 2. MIG-DPG 简化版实现
# ========================================

class SimplifiedMIG_DPG(nn.Module):
    """简化版MIG-DPG模型"""
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # 基础嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 多模态特征模拟
        self.text_projection = nn.Linear(64, embedding_dim)
        self.visual_projection = nn.Linear(64, embedding_dim)
        
        # 多模态融合注意力
        self.fusion_attention = nn.MultiheadAttention(embedding_dim, 4, batch_first=True)
        
        # 推荐网络
        self.recommendation_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1)
        )
        
        # DPO组件
        self.dpo_layer = SimpleDPO(embedding_dim)
        
        # 生成组件
        self.generation_layer = SimpleGeneration(embedding_dim)
        
    def forward(self, users, items, mode='recommend'):
        batch_size = users.size(0)
        
        # 基础嵌入
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        
        # 模拟多模态特征
        text_features = torch.randn(batch_size, 64).to(users.device)
        visual_features = torch.randn(batch_size, 64).to(users.device)
        
        text_emb = self.text_projection(text_features)
        visual_emb = self.visual_projection(visual_features)
        
        # 多模态融合
        multimodal_features = torch.stack([user_emb, item_emb, text_emb, visual_emb], dim=1)
        fused_features, _ = self.fusion_attention(multimodal_features, multimodal_features, multimodal_features)
        
        # 取用户和物品的融合特征
        fused_user_emb = fused_features[:, 0]
        fused_item_emb = fused_features[:, 1]
        
        if mode == 'recommend':
            concat_emb = torch.cat([fused_user_emb, fused_item_emb], dim=-1)
            scores = self.recommendation_net(concat_emb).squeeze()
            return scores
        elif mode == 'dpo':
            return self.dpo_layer(fused_user_emb, fused_item_emb)
        elif mode == 'generate':
            return self.generation_layer(fused_user_emb, fused_item_emb)
        else:
            # 联合模式
            rec_scores = self.forward(users, items, 'recommend')
            dpo_scores = self.forward(users, items, 'dpo')
            gen_logits = self.forward(users, items, 'generate')
            
            return {
                'recommendation_scores': rec_scores,
                'dpo_scores': dpo_scores,
                'generation_logits': gen_logits
            }

# ========================================
# 3. ABLATION VARIANTS 消融变体
# ========================================

class MIG_DPG_NoDPO(nn.Module):
    """无DPO组件的变体"""
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 推荐网络
        self.recommendation_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1)
        )
        
    def forward(self, users, items, mode='recommend'):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        concat_emb = torch.cat([user_emb, item_emb], dim=-1)
        scores = self.recommendation_net(concat_emb).squeeze()
        return scores

class MIG_DPG_NoGeneration(nn.Module):
    """无生成组件的变体"""
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 推荐网络
        self.recommendation_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1)
        )
        
        # DPO组件
        self.dpo_layer = SimpleDPO(embedding_dim)
        
    def forward(self, users, items, mode='recommend'):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        
        if mode == 'recommend':
            concat_emb = torch.cat([user_emb, item_emb], dim=-1)
            scores = self.recommendation_net(concat_emb).squeeze()
            return scores
        elif mode == 'dpo':
            return self.dpo_layer(user_emb, item_emb)

class MIG_DPG_NoMultiModal(nn.Module):
    """无多模态组件的变体"""
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 推荐网络
        self.recommendation_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1)
        )
        
        # DPO组件
        self.dpo_layer = SimpleDPO(embedding_dim)
        
        # 生成组件
        self.generation_layer = SimpleGeneration(embedding_dim)
        
    def forward(self, users, items, mode='recommend'):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        
        if mode == 'recommend':
            concat_emb = torch.cat([user_emb, item_emb], dim=-1)
            scores = self.recommendation_net(concat_emb).squeeze()
            return scores
        elif mode == 'dpo':
            return self.dpo_layer(user_emb, item_emb)
        elif mode == 'generate':
            return self.generation_layer(user_emb, item_emb)

# ========================================
# 4. DATA PROCESSING 数据处理
# ========================================

class SimpleDataset(torch.utils.data.Dataset):
    """简化的数据集类"""
    def __init__(self, data):
        self.users = torch.LongTensor(data['user_id'].values)
        self.items = torch.LongTensor(data['item_id'].values)
        self.ratings = torch.FloatTensor(data['rating'].values)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.users[idx],
            'item_id': self.items[idx],
            'rating': self.ratings[idx]
        }

# ========================================
# 5. EXPERIMENT MANAGER 实验管理器
# ========================================

class ComprehensiveExperimentManager:
    def __init__(self):
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 实验设备: {self.device}")
        
    def create_synthetic_data(self, num_users=200, num_items=100, num_interactions=2000):
        """创建合成数据集"""
        print("📊 创建合成数据集...")
        
        # 生成用户-物品交互
        np.random.seed(42)
        users = np.random.randint(0, num_users, num_interactions)
        items = np.random.randint(0, num_items, num_interactions)
        ratings = np.random.choice([1, 2, 3, 4, 5], num_interactions, p=[0.1, 0.1, 0.2, 0.3, 0.3])
        
        # 创建数据框
        data = pd.DataFrame({
            'user_id': users,
            'item_id': items,
            'rating': ratings,
            'timestamp': np.random.randint(1000000000, 1500000000, num_interactions)
        })
        
        # 去重
        data = data.drop_duplicates(subset=['user_id', 'item_id']).reset_index(drop=True)
        
        print(f"✅ 数据集创建完成: {len(data)}个交互, {data['user_id'].nunique()}个用户, {data['item_id'].nunique()}个物品")
        return data
        
    def prepare_datasets(self, data):
        """准备训练、验证、测试数据集"""
        # 划分数据集
        train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        print(f"📊 数据集划分完成:")
        print(f"   训练集: {len(train_data)} 交互")
        print(f"   验证集: {len(val_data)} 交互")
        print(f"   测试集: {len(test_data)} 交互")
        
        return train_data, val_data, test_data
    
    def train_model(self, model, train_loader, epochs=10, lr=0.001):
        """训练模型"""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                users = batch['user_id'].to(self.device)
                items = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                optimizer.zero_grad()
                predictions = model(users, items)
                loss = F.mse_loss(predictions, ratings)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    def evaluate_model(self, model, data_loader):
        """评估模型性能"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                users = batch['user_id'].to(self.device)
                items = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                predictions = model(users, items)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
        
        # 计算评估指标
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # 计算Recall@K和NDCG@K (简化版本)
        recall_10 = self.calculate_recall_at_k(predictions, targets, k=10)
        ndcg_10 = self.calculate_ndcg_at_k(predictions, targets, k=10)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'recall@10': recall_10,
            'ndcg@10': ndcg_10
        }
    
    def calculate_recall_at_k(self, predictions, targets, k=10):
        """计算Recall@K (简化版本)"""
        threshold = np.percentile(predictions, 90)  # 取前10%作为推荐
        recommended = predictions >= threshold
        relevant = targets >= 4  # 评分4分以上认为相关
        
        if np.sum(relevant) == 0:
            return 0.0
        
        return np.sum(recommended & relevant) / min(np.sum(relevant), k)
    
    def calculate_ndcg_at_k(self, predictions, targets, k=10):
        """计算NDCG@K (简化版本)"""
        # 获取预测分数排序
        sorted_indices = np.argsort(predictions)[::-1][:k]
        sorted_targets = targets[sorted_indices]
        
        # 计算DCG
        dcg = np.sum((2**sorted_targets - 1) / np.log2(np.arange(2, len(sorted_targets) + 2)))
        
        # 计算IDCG
        ideal_targets = np.sort(targets)[::-1][:k]
        idcg = np.sum((2**ideal_targets - 1) / np.log2(np.arange(2, len(ideal_targets) + 2)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def run_baseline_experiments(self, train_data, val_data, test_data):
        """运行基线模型对比实验"""
        print("\n" + "="*50)
        print("🏆 BASELINE MODELS COMPARISON")
        print("="*50)
        
        num_users = max(train_data['user_id'].max(), val_data['user_id'].max(), test_data['user_id'].max()) + 1
        num_items = max(train_data['item_id'].max(), val_data['item_id'].max(), test_data['item_id'].max()) + 1
        
        # 定义基线模型
        baseline_models = {
            'MLP': MLP_Baseline(num_users, num_items),
            'MatrixFactorization': MatrixFactorization(num_users, num_items),
            'NeuralCF': NeuralCF(num_users, num_items),
        }
        
        # 准备数据加载器
        train_dataset = SimpleDataset(train_data)
        val_dataset = SimpleDataset(val_data)
        test_dataset = SimpleDataset(test_data)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        baseline_results = {}
        
        for model_name, model in baseline_models.items():
            print(f"\n🔥 训练 {model_name}...")
            model = model.to(self.device)
            
            # 训练模型
            self.train_model(model, train_loader, epochs=10)
            
            # 评估模型
            val_metrics = self.evaluate_model(model, val_loader)
            test_metrics = self.evaluate_model(model, test_loader)
            
            baseline_results[model_name] = {
                'validation': val_metrics,
                'test': test_metrics
            }
            
            print(f"✅ {model_name} 完成:")
            print(f"   验证集 - Recall@10: {val_metrics['recall@10']:.4f}, NDCG@10: {val_metrics['ndcg@10']:.4f}")
            print(f"   测试集 - Recall@10: {test_metrics['recall@10']:.4f}, NDCG@10: {test_metrics['ndcg@10']:.4f}")
        
        self.results['baseline_comparison'] = baseline_results
        return baseline_results
    
    def run_ablation_studies(self, train_data, val_data, test_data):
        """运行消融实验"""
        print("\n" + "="*50)
        print("🔬 ABLATION STUDIES")
        print("="*50)
        
        num_users = max(train_data['user_id'].max(), val_data['user_id'].max(), test_data['user_id'].max()) + 1
        num_items = max(train_data['item_id'].max(), val_data['item_id'].max(), test_data['item_id'].max()) + 1
        
        # 定义消融变体
        ablation_models = {
            'Full_MIG-DPG': SimplifiedMIG_DPG(num_users, num_items),
            'w/o_DPO': MIG_DPG_NoDPO(num_users, num_items),
            'w/o_Generation': MIG_DPG_NoGeneration(num_users, num_items),
            'w/o_MultiModal': MIG_DPG_NoMultiModal(num_users, num_items)
        }
        
        # 准备数据加载器
        train_dataset = SimpleDataset(train_data)
        val_dataset = SimpleDataset(val_data)
        test_dataset = SimpleDataset(test_data)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        ablation_results = {}
        
        for model_name, model in ablation_models.items():
            print(f"\n🧪 训练 {model_name}...")
            model = model.to(self.device)
            
            # 训练模型
            self.train_model(model, train_loader, epochs=10)
            
            # 评估模型
            val_metrics = self.evaluate_model(model, val_loader)
            test_metrics = self.evaluate_model(model, test_loader)
            
            ablation_results[model_name] = {
                'validation': val_metrics,
                'test': test_metrics
            }
            
            print(f"✅ {model_name} 完成:")
            print(f"   验证集 - Recall@10: {val_metrics['recall@10']:.4f}, NDCG@10: {val_metrics['ndcg@10']:.4f}")
            print(f"   测试集 - Recall@10: {test_metrics['recall@10']:.4f}, NDCG@10: {test_metrics['ndcg@10']:.4f}")
        
        self.results['ablation_studies'] = ablation_results
        return ablation_results
    
    def generate_comparison_table(self):
        """生成对比表格"""
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE RESULTS COMPARISON")
        print("="*70)
        
        # Baseline对比表
        if 'baseline_comparison' in self.results:
            print("\n🏆 Baseline Models Performance:")
            print("-" * 60)
            print(f"{'Model':<15} {'Val Recall@10':<12} {'Val NDCG@10':<12} {'Test Recall@10':<13} {'Test NDCG@10':<12}")
            print("-" * 60)
            
            for model_name, results in self.results['baseline_comparison'].items():
                val_recall = results['validation']['recall@10']
                val_ndcg = results['validation']['ndcg@10']
                test_recall = results['test']['recall@10']
                test_ndcg = results['test']['ndcg@10']
                print(f"{model_name:<15} {val_recall:<12.4f} {val_ndcg:<12.4f} {test_recall:<13.4f} {test_ndcg:<12.4f}")
        
        # 消融实验表
        if 'ablation_studies' in self.results:
            print("\n🔬 Ablation Study Results:")
            print("-" * 60)
            print(f"{'Variant':<15} {'Val Recall@10':<12} {'Val NDCG@10':<12} {'Test Recall@10':<13} {'Test NDCG@10':<12}")
            print("-" * 60)
            
            for model_name, results in self.results['ablation_studies'].items():
                val_recall = results['validation']['recall@10']
                val_ndcg = results['validation']['ndcg@10']
                test_recall = results['test']['recall@10']
                test_ndcg = results['test']['ndcg@10']
                print(f"{model_name:<15} {val_recall:<12.4f} {val_ndcg:<12.4f} {test_recall:<13.4f} {test_ndcg:<12.4f}")
                
        # 计算提升度
        if 'baseline_comparison' in self.results and 'ablation_studies' in self.results:
            best_baseline = max(self.results['baseline_comparison'].items(), 
                              key=lambda x: x[1]['test']['recall@10'])
            full_mig_dpg = self.results['ablation_studies'].get('Full_MIG-DPG', {})
            
            if full_mig_dpg:
                improvement_recall = (full_mig_dpg['test']['recall@10'] - best_baseline[1]['test']['recall@10']) / best_baseline[1]['test']['recall@10'] * 100
                improvement_ndcg = (full_mig_dpg['test']['ndcg@10'] - best_baseline[1]['test']['ndcg@10']) / best_baseline[1]['test']['ndcg@10'] * 100
                
                print(f"\n🚀 MIG-DPG vs Best Baseline ({best_baseline[0]}):")
                print(f"   Recall@10 improvement: {improvement_recall:.1f}%")
                print(f"   NDCG@10 improvement: {improvement_ndcg:.1f}%")
    
    def save_results(self):
        """保存实验结果"""
        os.makedirs('experiment_results', exist_ok=True)
        
        # 保存详细结果
        with open('experiment_results/simplified_comprehensive_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print("💾 实验结果已保存至 experiment_results/simplified_comprehensive_results.json")

# ========================================
# 6. MAIN EXPERIMENT EXECUTION
# ========================================

def main():
    """主实验执行函数"""
    print("🚀 启动 MIG-DPG 简化版全面对比实验与消融研究")
    print("=" * 60)
    
    # 创建实验管理器
    experiment_manager = ComprehensiveExperimentManager()
    
    # 1. 创建合成数据
    data = experiment_manager.create_synthetic_data(num_users=200, num_items=100, num_interactions=2000)
    train_data, val_data, test_data = experiment_manager.prepare_datasets(data)
    
    try:
        # 2. 运行基线模型对比实验
        print("\n🏁 开始基线模型对比实验...")
        baseline_results = experiment_manager.run_baseline_experiments(train_data, val_data, test_data)
        
        # 3. 运行消融实验
        print("\n🔬 开始消融实验...")
        ablation_results = experiment_manager.run_ablation_studies(train_data, val_data, test_data)
        
        # 4. 生成对比表格
        experiment_manager.generate_comparison_table()
        
        # 5. 保存实验结果
        experiment_manager.save_results()
        
        print("\n" + "=" * 60)
        print("🎉 简化版全面实验完成！")
        print("📊 结果已保存至 experiment_results/simplified_comprehensive_results.json")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 实验过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()