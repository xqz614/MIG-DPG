"""
MIG-DPG Data Processor
处理MIG-DPG训练所需的多种数据：推荐数据、偏好数据、解释数据
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import dgl
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
from collections import defaultdict
import random


class MIG_DPG_Dataset(Dataset):
    """
    MIG-DPG数据集类
    
    整合三种数据：
    1. 推荐数据：用户-物品交互
    2. 偏好数据：用户对物品的偏好对比
    3. 解释数据：推荐解释文本
    """
    
    def __init__(self,
                 user_item_interactions: List[Tuple[int, int]],
                 user_embeddings: torch.Tensor,
                 item_embeddings: torch.Tensor,
                 item_text_features: torch.Tensor,
                 item_vision_features: torch.Tensor,
                 graph: dgl.DGLGraph,
                 preference_data: Optional[List[Tuple[int, int, int]]] = None,
                 explanation_data: Optional[Dict[Tuple[int, int], List[int]]] = None,
                 negative_sampling_ratio: int = 4,
                 max_explanation_length: int = 128):
        """
        Args:
            user_item_interactions: 用户-物品交互列表 [(user_id, item_id), ...]
            user_embeddings: 用户嵌入 [num_users, embedding_dim]
            item_embeddings: 物品嵌入 [num_items, embedding_dim]  
            item_text_features: 物品文本特征 [num_items, text_dim]
            item_vision_features: 物品视觉特征 [num_items, vision_dim]
            graph: DGL图
            preference_data: 偏好数据 [(user_id, preferred_item, non_preferred_item), ...]
            explanation_data: 解释数据 {(user_id, item_id): token_list, ...}
            negative_sampling_ratio: 负采样比例
            max_explanation_length: 最大解释长度
        """
        self.user_item_interactions = user_item_interactions
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.item_text_features = item_text_features
        self.item_vision_features = item_vision_features
        self.graph = graph
        self.preference_data = preference_data or []
        self.explanation_data = explanation_data or {}
        self.negative_sampling_ratio = negative_sampling_ratio
        self.max_explanation_length = max_explanation_length
        
        # 构建用户交互字典
        self.user_items = defaultdict(set)
        for user_id, item_id in user_item_interactions:
            self.user_items[user_id].add(item_id)
        
        # 预计算负样本池
        self.num_users = user_embeddings.size(0)
        self.num_items = item_embeddings.size(0)
        self.all_items = set(range(self.num_items))
        
        # 准备训练样本
        self._prepare_samples()
        
    def _prepare_samples(self):
        """准备训练样本"""
        self.samples = []
        
        for user_id, item_id in self.user_item_interactions:
            # 基础推荐样本
            sample = {
                'user_id': user_id,
                'pos_item': item_id,
                'neg_items': self._sample_negative_items(user_id),
                'has_preference': False,
                'has_explanation': False
            }
            
            # 检查是否有偏好数据
            for pref_user, pref_item, non_pref_item in self.preference_data:
                if pref_user == user_id and pref_item == item_id:
                    sample['neg_items'] = [non_pref_item]  # 使用偏好负样本
                    sample['has_preference'] = True
                    break
            
            # 检查是否有解释数据
            if (user_id, item_id) in self.explanation_data:
                sample['explanation_tokens'] = self._process_explanation(
                    self.explanation_data[(user_id, item_id)]
                )
                sample['has_explanation'] = True
            
            self.samples.append(sample)
    
    def _sample_negative_items(self, user_id: int) -> List[int]:
        """为用户采样负样本物品"""
        user_positive_items = self.user_items[user_id]
        candidate_items = self.all_items - user_positive_items
        
        if len(candidate_items) < self.negative_sampling_ratio:
            return list(candidate_items)
        
        return random.sample(list(candidate_items), self.negative_sampling_ratio)
    
    def _process_explanation(self, explanation_tokens: List[int]) -> torch.Tensor:
        """处理解释文本tokens"""
        # 截断或填充到指定长度
        if len(explanation_tokens) > self.max_explanation_length:
            explanation_tokens = explanation_tokens[:self.max_explanation_length]
        else:
            # 填充padding token (假设0是padding)
            explanation_tokens = explanation_tokens + [0] * (
                self.max_explanation_length - len(explanation_tokens)
            )
        
        return torch.tensor(explanation_tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        result = {
            'user_id': sample['user_id'],
            'pos_item': sample['pos_item'],
            'neg_items': torch.tensor(sample['neg_items'][:1], dtype=torch.long),  # 只取第一个负样本
            'graph': self.graph,
            'user_embeddings': self.user_embeddings,
            'item_embeddings': self.item_embeddings,
            'item_text_features': self.item_text_features,
            'item_vision_features': self.item_vision_features,
            'has_preference': sample['has_preference'],
            'has_explanation': sample['has_explanation']
        }
        
        if sample['has_explanation']:
            result['explanation_tokens'] = sample['explanation_tokens']
        
        return result


class MIG_DPG_DataProcessor:
    """
    MIG-DPG数据处理器
    
    负责：
    1. 加载和预处理原始数据
    2. 生成偏好数据
    3. 生成合成解释数据
    4. 创建训练/验证/测试数据集
    """
    
    def __init__(self, config):
        self.config = config
        
        # 特殊token定义
        self.PAD_TOKEN = 0
        self.BOS_TOKEN = 1 
        self.EOS_TOKEN = 2
        self.UNK_TOKEN = 3
        
        # 解释模板（用于合成解释数据）
        self.explanation_templates = [
            [self.BOS_TOKEN, 100, 101, 102, self.EOS_TOKEN],  # "推荐这个商品因为..."
            [self.BOS_TOKEN, 200, 201, 202, self.EOS_TOKEN],  # "您可能喜欢..."
            [self.BOS_TOKEN, 300, 301, 302, self.EOS_TOKEN],  # "基于您的历史..."
        ]
    
    def load_data(self, dataset_name: str) -> Dict[str, Any]:
        """
        加载数据集
        
        Args:
            dataset_name: 数据集名称 ('baby', 'sports', 'clothing', 'elec')
            
        Returns:
            包含所有数据的字典
        """
        print(f"正在加载 {dataset_name} 数据集...")
        
        # 这里应该加载真实数据，暂时使用模拟数据
        data = self._create_mock_data(dataset_name)
        
        print(f"数据加载完成:")
        print(f"  - 用户数量: {data['num_users']}")
        print(f"  - 物品数量: {data['num_items']}")
        print(f"  - 交互数量: {len(data['user_item_interactions'])}")
        
        return data
    
    def _create_mock_data(self, dataset_name: str) -> Dict[str, Any]:
        """创建模拟数据用于演示"""
        
        # 根据数据集设置参数
        if dataset_name == 'baby':
            num_users, num_items = 1000, 500
        elif dataset_name == 'sports':
            num_users, num_items = 1500, 800
        elif dataset_name == 'clothing':
            num_users, num_items = 2000, 1000
        else:  # elec
            num_users, num_items = 1200, 600
        
        embedding_dim = self.config.embedding_size
        
        # 生成用户-物品交互
        interactions = []
        for user_id in range(num_users):
            num_interactions = np.random.randint(5, 20)  # 每个用户5-20个交互
            items = np.random.choice(num_items, num_interactions, replace=False)
            for item_id in items:
                interactions.append((user_id, item_id))
        
        # 创建图
        edges = [(u, num_users + i) for u, i in interactions]
        edges += [(num_users + i, u) for u, i in interactions]  # 双向边
        
        src, dst = zip(*edges)
        graph = dgl.graph((src, dst))
        graph = dgl.add_self_loop(graph)
        
        # 生成特征
        user_embeddings = torch.randn(num_users, embedding_dim)
        item_embeddings = torch.randn(num_items, embedding_dim)
        item_text_features = torch.randn(num_items, self.config.text_feat_size)
        item_vision_features = torch.randn(num_items, self.config.vision_feat_size)
        
        return {
            'num_users': num_users,
            'num_items': num_items,
            'user_item_interactions': interactions,
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings,
            'item_text_features': item_text_features,
            'item_vision_features': item_vision_features,
            'graph': graph
        }
    
    def generate_preference_data(self, 
                               interactions: List[Tuple[int, int]],
                               ratio: float = 0.3) -> List[Tuple[int, int, int]]:
        """
        生成偏好数据
        
        Args:
            interactions: 用户-物品交互
            ratio: 生成偏好数据的比例
            
        Returns:
            偏好三元组列表 [(user_id, preferred_item, non_preferred_item), ...]
        """
        print(f"正在生成偏好数据 (比例: {ratio})...")
        
        # 构建用户交互字典
        user_items = defaultdict(list)
        for user_id, item_id in interactions:
            user_items[user_id].append(item_id)
        
        preference_data = []
        target_count = int(len(interactions) * ratio)
        
        # 随机选择交互生成偏好对
        selected_interactions = random.sample(interactions, target_count)
        
        for user_id, preferred_item in selected_interactions:
            # 从用户未交互的物品中选择负样本
            user_positive_items = set(user_items[user_id])
            all_items = set(range(max(item_id for _, item_id in interactions) + 1))
            candidate_negative = list(all_items - user_positive_items)
            
            if candidate_negative:
                non_preferred_item = random.choice(candidate_negative)
                preference_data.append((user_id, preferred_item, non_preferred_item))
        
        print(f"生成了 {len(preference_data)} 个偏好三元组")
        return preference_data
    
    def generate_synthetic_explanations(self,
                                      interactions: List[Tuple[int, int]],
                                      ratio: float = 0.5) -> Dict[Tuple[int, int], List[int]]:
        """
        生成合成解释数据
        
        Args:
            interactions: 用户-物品交互
            ratio: 生成解释的比例
            
        Returns:
            解释数据字典 {(user_id, item_id): token_list, ...}
        """
        print(f"正在生成合成解释数据 (比例: {ratio})...")
        
        explanation_data = {}
        target_count = int(len(interactions) * ratio)
        
        # 随机选择交互生成解释
        selected_interactions = random.sample(interactions, target_count)
        
        for user_id, item_id in selected_interactions:
            # 随机选择解释模板
            template = random.choice(self.explanation_templates)
            
            # 添加一些随机变化
            explanation = template.copy()
            if len(explanation) < self.config.max_explanation_length - 5:
                # 随机添加一些token
                additional_tokens = [random.randint(4, 999) for _ in range(random.randint(1, 5))]
                explanation = explanation[:-1] + additional_tokens + [explanation[-1]]
            
            explanation_data[(user_id, item_id)] = explanation
        
        print(f"生成了 {len(explanation_data)} 个解释")
        return explanation_data
    
    def create_datasets(self, data: Dict[str, Any]) -> Tuple[MIG_DPG_Dataset, MIG_DPG_Dataset, MIG_DPG_Dataset]:
        """
        创建训练/验证/测试数据集
        
        Args:
            data: 原始数据
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        print("正在创建数据集...")
        
        interactions = data['user_item_interactions']
        
        # 数据集分割
        random.shuffle(interactions)
        train_size = int(len(interactions) * self.config.train_ratio)
        val_size = int(len(interactions) * self.config.val_ratio)
        
        train_interactions = interactions[:train_size]
        val_interactions = interactions[train_size:train_size + val_size]
        test_interactions = interactions[train_size + val_size:]
        
        # 生成偏好数据和解释数据
        preference_data = self.generate_preference_data(
            train_interactions, self.config.preference_data_ratio
        )
        
        explanation_data = {}
        if self.config.synthetic_explanation:
            explanation_data = self.generate_synthetic_explanations(
                train_interactions, 0.5
            )
        
        # 创建数据集
        train_dataset = MIG_DPG_Dataset(
            train_interactions,
            data['user_embeddings'],
            data['item_embeddings'],
            data['item_text_features'],
            data['item_vision_features'],
            data['graph'],
            preference_data,
            explanation_data,
            self.config.negative_sampling_ratio,
            self.config.max_explanation_length
        )
        
        val_dataset = MIG_DPG_Dataset(
            val_interactions,
            data['user_embeddings'],
            data['item_embeddings'],
            data['item_text_features'],
            data['item_vision_features'],
            data['graph'],
            negative_sampling_ratio=1  # 验证时只需要一个负样本
        )
        
        test_dataset = MIG_DPG_Dataset(
            test_interactions,
            data['user_embeddings'],
            data['item_embeddings'],
            data['item_text_features'],
            data['item_vision_features'],
            data['graph'],
            negative_sampling_ratio=1
        )
        
        print(f"数据集创建完成:")
        print(f"  - 训练集: {len(train_dataset)} 样本")
        print(f"  - 验证集: {len(val_dataset)} 样本")
        print(f"  - 测试集: {len(test_dataset)} 样本")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, 
                          train_dataset: MIG_DPG_Dataset,
                          val_dataset: MIG_DPG_Dataset,
                          test_dataset: MIG_DPG_Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Returns:
            (train_dataloader, val_dataloader, test_dataloader)
        """
        def collate_fn(batch):
            """自定义collate函数"""
            # 由于每个样本都包含完整的图和特征，直接返回第一个样本的图和特征
            sample = batch[0]
            
            batch_data = {
                'graph': sample['graph'],
                'user_embeddings': sample['user_embeddings'], 
                'item_embeddings': sample['item_embeddings'],
                'item_v_feat': sample['item_vision_features'],
                'item_t_feat': sample['item_text_features'],
                'user_indices': torch.tensor([item['user_id'] for item in batch]),
                'pos_items': torch.tensor([item['pos_item'] for item in batch]),
                'neg_items': torch.stack([item['neg_items'] for item in batch]).squeeze(1)
            }
            
            # 处理解释数据
            has_explanations = [item['has_explanation'] for item in batch]
            if any(has_explanations):
                explanation_tokens = []
                for item in batch:
                    if item['has_explanation']:
                        explanation_tokens.append(item['explanation_tokens'])
                    else:
                        # 填充空解释
                        explanation_tokens.append(
                            torch.zeros(self.config.max_explanation_length, dtype=torch.long)
                        )
                batch_data['explanation_tokens'] = torch.stack(explanation_tokens)
            
            return batch_data
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Windows下设为0
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        return train_dataloader, val_dataloader, test_dataloader 