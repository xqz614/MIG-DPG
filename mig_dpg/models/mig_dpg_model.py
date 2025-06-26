"""
MIG-DPG Model: Multimodal Independent Graph Neural Networks 
with Direct Preference Optimization and Generation

集成了原始MIG-GT、DPO偏好优化和生成式解释的完整模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List
import dgl

# 导入原始MIG-GT组件
from ..layers.mirf_gt import MIGGT
from ..layers.dpo_layer import DPOLayer
from ..layers.generation_layer import GenerativeExplanationLayer


class MIG_DPG_Model(nn.Module):
    """
    MIG-DPG主模型
    
    结合了：
    1. MIG-GT的多模态图神经网络
    2. DPO的直接偏好优化
    3. 生成式解释器
    """
    
    def __init__(self, config):
        """
        Args:
            config: 配置对象，包含所有超参数
        """
        super(MIG_DPG_Model, self).__init__()
        
        # 保存配置
        self.config = config
        self.embedding_size = config.embedding_size
        self.num_users = config.num_users
        self.num_items = config.num_items
        
        # 任务权重
        self.recommendation_weight = getattr(config, 'recommendation_weight', 1.0)
        self.dpo_weight = getattr(config, 'dpo_weight', 0.5)
        self.generation_weight = getattr(config, 'generation_weight', 0.3)
        
        # =============== 核心组件 ===============
        
        # 1. 原始MIG-GT编码器
        self.mig_gt_encoder = MIGGT(
            k_e=config.k_e,
            k_t=config.k_t, 
            k_v=config.k_v,
            alpha=config.alpha,
            beta=config.beta,
            n_layers=config.n_layers,
            edge_drop_rate=config.edge_drop_rate,
            message_drop_rate=config.message_drop_rate,
            embedding_size=config.embedding_size,
            num_samples=config.num_samples,
            text_feat_size=config.text_feat_size,
            vision_feat_size=config.vision_feat_size,
            dropout_p=config.dropout
        )
        
        # 2. DPO偏好优化层
        self.dpo_layer = DPOLayer(
            embedding_dim=config.embedding_size,
            hidden_dim=getattr(config, 'dpo_hidden_dim', 256),
            beta=getattr(config, 'dpo_beta', 0.1),
            dropout=config.dropout,
            num_heads=getattr(config, 'dpo_num_heads', 4)
        )
        
        # 3. 生成式解释层
        self.generation_layer = GenerativeExplanationLayer(
            embedding_dim=config.embedding_size,
            text_dim=config.text_feat_size,
            vision_dim=config.vision_feat_size,
            hidden_dim=getattr(config, 'gen_hidden_dim', 512),
            vocab_size=getattr(config, 'vocab_size', 10000),
            max_length=getattr(config, 'max_explanation_length', 128),
            num_layers=getattr(config, 'gen_num_layers', 6),
            num_heads=getattr(config, 'gen_num_heads', 8),
            dropout=config.dropout
        )
        
        # =============== 任务头 ===============
        
        # 推荐任务头（与原始一致）
        self.recommendation_head = nn.Identity()  # MIG-GT直接输出用于推荐
        
        # 偏好预测头
        self.preference_head = nn.Sequential(
            nn.Linear(config.embedding_size, config.embedding_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_size // 2, 1)
        )
        
        # =============== 模式控制 ===============
        self.training_mode = 'joint'  # 'recommendation', 'dpo', 'generation', 'joint'
        
    def forward(self,
                g: dgl.DGLGraph,
                user_embeddings: torch.Tensor,
                item_v_feat: torch.Tensor,
                item_t_feat: torch.Tensor,
                item_embeddings: Optional[torch.Tensor] = None,
                positive_items: Optional[torch.Tensor] = None,
                negative_items: Optional[torch.Tensor] = None,
                explanation_tokens: Optional[torch.Tensor] = None,
                mode: str = 'recommendation') -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            g: DGL图
            user_embeddings: 用户嵌入
            item_v_feat: 物品视觉特征
            item_t_feat: 物品文本特征
            item_embeddings: 物品嵌入
            positive_items: 正样本物品ID (用于DPO)
            negative_items: 负样本物品ID (用于DPO)
            explanation_tokens: 解释文本tokens (用于生成)
            mode: 运行模式
            
        Returns:
            包含各种输出的字典
        """
        # 1. MIG-GT编码器 - 获取所有中间表示
        mig_gt_outputs = self.mig_gt_encoder(
            g, user_embeddings, item_v_feat, item_t_feat, 
            item_embeddings, return_all=True
        )
        
        # 解析MIG-GT输出
        combined_h, emb_h, t_h, v_h, encoded_t, encoded_v, z_memory_h = mig_gt_outputs
        
        # 分离用户和物品嵌入
        num_users = user_embeddings.size(0)
        enhanced_user_emb = combined_h[:num_users]  # [num_users, embedding_size]
        enhanced_item_emb = combined_h[num_users:]  # [num_items, embedding_size]
        
        outputs = {
            'user_embeddings': enhanced_user_emb,
            'item_embeddings': enhanced_item_emb,
            'mig_gt_outputs': {
                'combined_h': combined_h,
                'emb_h': emb_h,
                't_h': t_h, 
                'v_h': v_h,
                'encoded_t': encoded_t,
                'encoded_v': encoded_v,
                'z_memory_h': z_memory_h
            }
        }
        
        # 2. 根据模式执行不同的任务
        if mode in ['recommendation', 'joint']:
            # 推荐任务 - 计算用户-物品相似度
            rec_scores = torch.matmul(enhanced_user_emb, enhanced_item_emb.t())
            outputs['recommendation_scores'] = rec_scores
        
        if mode in ['dpo', 'joint'] and positive_items is not None and negative_items is not None:
            # DPO偏好优化
            batch_size = positive_items.size(0)
            sampled_user_emb = enhanced_user_emb[:batch_size]  # 采样用户
            
            enhanced_user_emb_dpo, preference_scores = self.dpo_layer(
                sampled_user_emb,
                enhanced_item_emb,
                positive_items,
                negative_items
            )
            
            outputs['enhanced_user_emb_dpo'] = enhanced_user_emb_dpo
            outputs['preference_scores'] = preference_scores
            outputs['dpo_loss'] = self.dpo_layer.compute_dpo_loss(preference_scores)
        
        if mode in ['generation', 'joint'] and positive_items is not None:
            # 生成式解释
            batch_size = positive_items.size(0)
            sampled_user_emb = enhanced_user_emb[:batch_size]
            
            generation_outputs = self.generation_layer(
                sampled_user_emb,
                enhanced_item_emb,
                item_t_feat,
                item_v_feat,
                positive_items,
                explanation_tokens
            )
            
            outputs['generation_outputs'] = generation_outputs
            
            # 如果有目标tokens，计算生成损失
            if explanation_tokens is not None and 'logits' in generation_outputs:
                # 目标tokens应该是shift过的（用于teacher forcing）
                target_tokens = explanation_tokens[:, 1:]  # 去掉BOS token
                input_logits = generation_outputs['logits'][:, :-1]  # 去掉最后一个位置
                
                gen_loss = self.generation_layer.compute_generation_loss(
                    input_logits, target_tokens
                )
                outputs['generation_loss'] = gen_loss
        
        return outputs
    
    def compute_joint_loss(self,
                          outputs: Dict[str, Any],
                          user_indices: torch.Tensor,
                          pos_items: torch.Tensor,
                          neg_items: torch.Tensor) -> torch.Tensor:
        """
        计算联合训练损失
        
        Args:
            outputs: 模型输出
            user_indices: 用户索引
            pos_items: 正样本物品
            neg_items: 负样本物品
            
        Returns:
            总损失
        """
        total_loss = 0
        loss_dict = {}
        
        # 1. 推荐损失（InfoBPR）
        if 'recommendation_scores' in outputs:
            rec_scores = outputs['recommendation_scores']
            batch_size = user_indices.size(0)
            
            # 采样对应的分数
            user_scores = rec_scores[user_indices]  # [batch_size, num_items]
            pos_scores = user_scores.gather(1, pos_items.unsqueeze(1)).squeeze(1)
            neg_scores = user_scores.gather(1, neg_items.unsqueeze(1)).squeeze(1)
            
            # BPR损失
            rec_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
            total_loss += self.recommendation_weight * rec_loss
            loss_dict['recommendation_loss'] = rec_loss
        
        # 2. DPO损失
        if 'dpo_loss' in outputs:
            dpo_loss = outputs['dpo_loss']
            total_loss += self.dpo_weight * dpo_loss
            loss_dict['dpo_loss'] = dpo_loss
        
        # 3. 生成损失
        if 'generation_loss' in outputs:
            gen_loss = outputs['generation_loss']
            total_loss += self.generation_weight * gen_loss
            loss_dict['generation_loss'] = gen_loss
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
    
    def predict_recommendations(self,
                              g: dgl.DGLGraph,
                              user_embeddings: torch.Tensor,
                              item_v_feat: torch.Tensor,
                              item_t_feat: torch.Tensor,
                              item_embeddings: Optional[torch.Tensor] = None,
                              topk: int = 20) -> torch.Tensor:
        """
        预测推荐列表
        
        Args:
            g: DGL图
            user_embeddings: 用户嵌入
            item_v_feat: 物品视觉特征
            item_t_feat: 物品文本特征
            item_embeddings: 物品嵌入
            topk: 返回top-k推荐
            
        Returns:
            推荐物品ID [num_users, topk]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                g, user_embeddings, item_v_feat, item_t_feat,
                item_embeddings, mode='recommendation'
            )
            
            rec_scores = outputs['recommendation_scores']
            _, topk_items = torch.topk(rec_scores, topk, dim=1)
            
        return topk_items
    
    def generate_explanations(self,
                            g: dgl.DGLGraph,
                            user_embeddings: torch.Tensor,
                            item_v_feat: torch.Tensor,
                            item_t_feat: torch.Tensor,
                            target_items: torch.Tensor,
                            item_embeddings: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        为给定的用户-物品对生成解释
        
        Args:
            g: DGL图
            user_embeddings: 用户嵌入
            item_v_feat: 物品视觉特征
            item_t_feat: 物品文本特征
            target_items: 目标物品ID
            item_embeddings: 物品嵌入
            
        Returns:
            生成的解释token序列
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                g, user_embeddings, item_v_feat, item_t_feat,
                item_embeddings, positive_items=target_items,
                mode='generation'
            )
            
            generation_outputs = outputs['generation_outputs']
            generated_tokens = generation_outputs['generated_tokens']
            
        return generated_tokens.tolist()
    
    def set_training_mode(self, mode: str):
        """设置训练模式"""
        assert mode in ['recommendation', 'dpo', 'generation', 'joint']
        self.training_mode = mode
        
    def freeze_mig_gt(self):
        """冻结MIG-GT参数，只训练DPO和生成组件"""
        for param in self.mig_gt_encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True 