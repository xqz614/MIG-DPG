"""
Generative Explanation Layer Implementation
生成式解释层用于为推荐结果生成自然语言解释
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiModalFusion(nn.Module):
    """多模态特征融合模块"""
    
    def __init__(self, 
                 embedding_dim: int,
                 text_dim: int = 384,
                 vision_dim: int = 4096,
                 output_dim: int = 512):
        super().__init__()
        
        # 模态特征投影
        self.embedding_proj = nn.Linear(embedding_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim) 
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        
        # 模态注意力权重
        self.modal_attention = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, 
                embedding_feat: torch.Tensor,
                text_feat: torch.Tensor, 
                vision_feat: torch.Tensor) -> torch.Tensor:
        """
        多模态特征融合
        
        Args:
            embedding_feat: 嵌入特征 [batch_size, embedding_dim]
            text_feat: 文本特征 [batch_size, text_dim]
            vision_feat: 视觉特征 [batch_size, vision_dim]
            
        Returns:
            fused_feat: 融合后的特征 [batch_size, output_dim]
        """
        # 特征投影
        emb_proj = self.embedding_proj(embedding_feat)  # [batch_size, output_dim]
        text_proj = self.text_proj(text_feat)  # [batch_size, output_dim]
        vision_proj = self.vision_proj(vision_feat)  # [batch_size, output_dim]
        
        # 计算模态注意力权重
        concat_feats = torch.cat([emb_proj, text_proj, vision_proj], dim=-1)
        modal_weights = self.modal_attention(concat_feats)  # [batch_size, 3]
        
        # 加权融合
        fused_feat = (modal_weights[:, 0:1] * emb_proj + 
                     modal_weights[:, 1:2] * text_proj + 
                     modal_weights[:, 2:3] * vision_proj)
        
        # 进一步处理
        fused_feat = self.fusion_layer(fused_feat)
        
        return fused_feat


class GenerativeExplanationLayer(nn.Module):
    """
    生成式解释层
    
    基于Transformer生成推荐解释文本
    集成多模态信息生成连贯、准确的推荐理由
    """
    
    def __init__(self,
                 embedding_dim: int = 64,
                 text_dim: int = 384,
                 vision_dim: int = 4096,
                 hidden_dim: int = 512,
                 vocab_size: int = 10000,
                 max_length: int = 128,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Args:
            embedding_dim: 用户/物品嵌入维度
            text_dim: 文本特征维度
            vision_dim: 视觉特征维度
            hidden_dim: Transformer隐藏维度
            vocab_size: 词汇表大小
            max_length: 最大生成长度
            num_layers: Transformer层数
            num_heads: 注意力头数
            dropout: dropout比率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # 多模态融合模块
        self.multimodal_fusion = MultiModalFusion(
            embedding_dim=embedding_dim,
            text_dim=text_dim, 
            vision_dim=vision_dim,
            output_dim=hidden_dim
        )
        
        # 用户-物品交互编码器
        self.interaction_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 词嵌入和位置编码
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_length)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # 特殊token定义
        self.pad_token_id = 0
        self.bos_token_id = 1  # 开始token
        self.eos_token_id = 2  # 结束token
        
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成causal mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self,
                user_embeddings: torch.Tensor,
                item_embeddings: torch.Tensor, 
                item_text_features: torch.Tensor,
                item_vision_features: torch.Tensor,
                target_items: torch.Tensor,
                explanation_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            user_embeddings: 用户嵌入 [batch_size, embedding_dim]
            item_embeddings: 物品嵌入 [num_items, embedding_dim]
            item_text_features: 物品文本特征 [num_items, text_dim]
            item_vision_features: 物品视觉特征 [num_items, vision_dim]
            target_items: 目标物品ID [batch_size]
            explanation_tokens: 解释文本token [batch_size, seq_len] (训练时使用)
            
        Returns:
            outputs: 包含logits和attention权重的字典
        """
        batch_size = user_embeddings.size(0)
        
        # 获取目标物品的特征
        target_item_emb = item_embeddings[target_items]  # [batch_size, embedding_dim]
        target_text_feat = item_text_features[target_items]  # [batch_size, text_dim]
        target_vision_feat = item_vision_features[target_items]  # [batch_size, vision_dim]
        
        # 多模态特征融合
        item_multimodal_feat = self.multimodal_fusion(
            target_item_emb, target_text_feat, target_vision_feat
        )  # [batch_size, hidden_dim]
        
        # 用户-物品交互编码
        user_item_interaction = torch.cat([user_embeddings, target_item_emb], dim=-1)
        interaction_feat = self.interaction_encoder(user_item_interaction)  # [batch_size, hidden_dim]
        
        # 构建记忆上下文（用户偏好 + 物品特征）
        memory_context = torch.stack([interaction_feat, item_multimodal_feat], dim=1)  # [batch_size, 2, hidden_dim]
        
        if explanation_tokens is not None:
            # 训练模式：使用teacher forcing
            seq_len = explanation_tokens.size(1)
            
            # Token嵌入 + 位置编码
            token_emb = self.token_embedding(explanation_tokens)  # [batch_size, seq_len, hidden_dim]
            token_emb = self.positional_encoding(token_emb.transpose(0, 1)).transpose(0, 1)
            
            # 生成causal mask
            causal_mask = self._generate_square_subsequent_mask(seq_len).to(token_emb.device)
            
            # Transformer解码
            decoder_output = self.transformer_decoder(
                tgt=token_emb,
                memory=memory_context,
                tgt_mask=causal_mask
            )  # [batch_size, seq_len, hidden_dim]
            
            # 输出投影
            logits = self.output_projection(decoder_output)  # [batch_size, seq_len, vocab_size]
            
            return {
                'logits': logits,
                'memory_context': memory_context,
                'multimodal_features': item_multimodal_feat
            }
            
        else:
            # 推理模式：自回归生成
            return self._generate_explanation(memory_context, batch_size)
    
    def _generate_explanation(self, 
                            memory_context: torch.Tensor,
                            batch_size: int) -> Dict[str, torch.Tensor]:
        """
        自回归生成解释文本
        
        Args:
            memory_context: 记忆上下文 [batch_size, 2, hidden_dim]
            batch_size: 批次大小
            
        Returns:
            生成结果字典
        """
        device = memory_context.device
        
        # 初始化输入（BOS token）
        input_ids = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        
        generated_tokens = []
        
        for step in range(self.max_length - 1):
            # Token嵌入 + 位置编码
            token_emb = self.token_embedding(input_ids)
            token_emb = self.positional_encoding(token_emb.transpose(0, 1)).transpose(0, 1)
            
            # Causal mask
            seq_len = input_ids.size(1)
            causal_mask = self._generate_square_subsequent_mask(seq_len).to(device)
            
            # Transformer解码
            decoder_output = self.transformer_decoder(
                tgt=token_emb,
                memory=memory_context,
                tgt_mask=causal_mask
            )  # [batch_size, seq_len, hidden_dim]
            
            # 获取最后一个时间步的输出
            last_hidden = decoder_output[:, -1, :]  # [batch_size, hidden_dim]
            
            # 输出投影
            logits = self.output_projection(last_hidden)  # [batch_size, vocab_size]
            
            # 采样下一个token
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [batch_size, 1]
            
            generated_tokens.append(next_token)
            
            # 更新输入序列
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 检查是否生成了结束token
            if (next_token == self.eos_token_id).all():
                break
        
        # 合并生成的tokens
        if generated_tokens:
            generated_sequence = torch.cat(generated_tokens, dim=1)  # [batch_size, generated_len]
            full_sequence = torch.cat([input_ids[:, :1], generated_sequence], dim=1)
        else:
            full_sequence = input_ids
        
        return {
            'generated_tokens': full_sequence,
            'memory_context': memory_context,
            'generation_length': full_sequence.size(1)
        }
    
    def compute_generation_loss(self, 
                              logits: torch.Tensor,
                              target_tokens: torch.Tensor,
                              ignore_index: int = -100) -> torch.Tensor:
        """
        计算生成损失
        
        Args:
            logits: 模型输出logits [batch_size, seq_len, vocab_size]
            target_tokens: 目标tokens [batch_size, seq_len]
            ignore_index: 忽略的索引（如padding）
            
        Returns:
            生成损失
        """
        # 重塑为二维
        logits_2d = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
        targets_1d = target_tokens.view(-1)  # [batch_size * seq_len]
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits_2d, targets_1d, ignore_index=ignore_index)
        
        return loss 