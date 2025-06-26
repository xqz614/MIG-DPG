"""
DPO (Direct Preference Optimization) Layer Implementation
DPO层用于学习用户的直接偏好，通过对比学习的方式优化推荐质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DPOLayer(nn.Module):
    """
    直接偏好优化层
    
    基于DPO论文的实现，通过对比学习的方式学习用户偏好
    相比传统的BPR损失，DPO能够更好地建模用户的真实偏好分布
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 hidden_dim: int = 256,
                 beta: float = 0.1,
                 dropout: float = 0.1,
                 num_heads: int = 4):
        """
        Args:
            embedding_dim: 输入嵌入维度
            hidden_dim: 隐藏层维度
            beta: DPO温度参数，控制偏好强度
            dropout: dropout比率
            num_heads: 多头注意力头数
        """
        super(DPOLayer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.num_heads = num_heads
        
        # 偏好编码器：将用户-物品表征转换为偏好分数
        self.preference_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),  # 用户+物品拼接
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 输出偏好分数
        )
        
        # 参考策略网络（用于DPO计算）
        self.reference_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 多头注意力机制用于偏好建模
        self.preference_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 偏好融合层
        self.preference_fusion = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, 
                user_embeddings: torch.Tensor,
                item_embeddings: torch.Tensor,
                positive_items: torch.Tensor,
                negative_items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            user_embeddings: 用户嵌入 [batch_size, embedding_dim]
            item_embeddings: 物品嵌入 [num_items, embedding_dim] 
            positive_items: 正样本物品ID [batch_size]
            negative_items: 负样本物品ID [batch_size]
            
        Returns:
            enhanced_user_emb: 增强的用户嵌入
            preference_scores: 偏好分数
        """
        batch_size = user_embeddings.size(0)
        
        # 获取正负样本的物品嵌入
        pos_item_emb = item_embeddings[positive_items]  # [batch_size, embedding_dim]
        neg_item_emb = item_embeddings[negative_items]  # [batch_size, embedding_dim]
        
        # 计算用户偏好注意力
        # 使用物品嵌入作为query，用户嵌入作为key和value
        item_queries = torch.stack([pos_item_emb, neg_item_emb], dim=1)  # [batch_size, 2, embedding_dim]
        user_keys = user_embeddings.unsqueeze(1).repeat(1, 2, 1)  # [batch_size, 2, embedding_dim]
        
        attended_preferences, attention_weights = self.preference_attention(
            query=item_queries,
            key=user_keys, 
            value=user_keys
        )  # [batch_size, 2, embedding_dim]
        
        # 增强用户嵌入（融合偏好信息）
        preference_context = attended_preferences.mean(dim=1)  # [batch_size, embedding_dim]
        enhanced_user_emb = user_embeddings + self.preference_fusion(preference_context)
        
        # 计算偏好分数用于DPO损失
        pos_pair = torch.cat([user_embeddings, pos_item_emb], dim=-1)  # [batch_size, 2*embedding_dim]
        neg_pair = torch.cat([user_embeddings, neg_item_emb], dim=-1)  # [batch_size, 2*embedding_dim]
        
        pos_preference = self.preference_encoder(pos_pair).squeeze(-1)  # [batch_size]
        neg_preference = self.preference_encoder(neg_pair).squeeze(-1)  # [batch_size]
        
        # 参考策略分数（冻结参数）
        with torch.no_grad():
            pos_reference = self.reference_encoder(pos_pair).squeeze(-1)
            neg_reference = self.reference_encoder(neg_pair).squeeze(-1)
        
        preference_scores = {
            'pos_policy': pos_preference,
            'neg_policy': neg_preference,
            'pos_reference': pos_reference,
            'neg_reference': neg_reference,
            'attention_weights': attention_weights
        }
        
        return enhanced_user_emb, preference_scores
    
    def compute_dpo_loss(self, preference_scores: dict) -> torch.Tensor:
        """
        计算DPO损失
        
        Args:
            preference_scores: 偏好分数字典
            
        Returns:
            DPO损失值
        """
        # 提取分数
        pos_policy = preference_scores['pos_policy']
        neg_policy = preference_scores['neg_policy']
        pos_reference = preference_scores['pos_reference']
        neg_reference = preference_scores['neg_reference']
        
        # 计算log比率
        pos_log_ratio = self.beta * (pos_policy - pos_reference)
        neg_log_ratio = self.beta * (neg_policy - neg_reference)
        
        # DPO损失：-log(sigmoid(log_ratio_pos - log_ratio_neg))
        dpo_loss = -F.logsigmoid(pos_log_ratio - neg_log_ratio).mean()
        
        return dpo_loss
    
    def get_preference_ranking(self, 
                             user_embeddings: torch.Tensor,
                             item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        获取所有物品的偏好排序分数
        
        Args:
            user_embeddings: 用户嵌入 [batch_size, embedding_dim]
            item_embeddings: 物品嵌入 [num_items, embedding_dim]
            
        Returns:
            preference_matrix: 偏好分数矩阵 [batch_size, num_items]
        """
        batch_size = user_embeddings.size(0)
        num_items = item_embeddings.size(0)
        
        # 扩展维度进行批量计算
        user_expanded = user_embeddings.unsqueeze(1).expand(-1, num_items, -1)  # [batch_size, num_items, embedding_dim]
        item_expanded = item_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_items, embedding_dim]
        
        # 拼接用户-物品对
        user_item_pairs = torch.cat([
            user_expanded.reshape(-1, self.embedding_dim),
            item_expanded.reshape(-1, self.embedding_dim)
        ], dim=-1)  # [batch_size * num_items, 2 * embedding_dim]
        
        # 计算偏好分数
        preference_scores = self.preference_encoder(user_item_pairs)  # [batch_size * num_items, 1]
        preference_matrix = preference_scores.reshape(batch_size, num_items)  # [batch_size, num_items]
        
        return preference_matrix 