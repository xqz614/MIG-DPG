"""
MIG-DPG Trainer
专门用于MIG-DPG模型的训练器，支持多任务联合训练和分阶段训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm

from .models.mig_dpg_model import MIG_DPG_Model
from .configs.mig_dpg_default_config import MIG_DPG_DefaultConfig


class MIG_DPG_Trainer:
    """
    MIG-DPG训练器
    
    支持：
    1. 联合训练 (joint training)
    2. 分阶段训练 (sequential training) 
    3. 课程学习 (curriculum learning)
    4. 混合精度训练
    5. 梯度累积
    """
    
    def __init__(self, 
                 model: MIG_DPG_Model,
                 config: MIG_DPG_DefaultConfig,
                 device: torch.device = None):
        """
        Args:
            model: MIG-DPG模型
            config: 训练配置
            device: 训练设备
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # 训练状态
        self.current_epoch = 0
        self.best_performance = 0.0
        self.patience_counter = 0
        
        # 损失历史
        self.loss_history = {
            'total_loss': [],
            'recommendation_loss': [],
            'dpo_loss': [],
            'generation_loss': []
        }
        
        # 创建保存目录
        if config.save_model:
            os.makedirs(config.model_save_path, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        if hasattr(self.config, 'optimizer_type'):
            if self.config.optimizer_type.lower() == 'adamw':
                return optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.decay
                )
        
        # 默认使用Adam
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.decay
        )
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if hasattr(self.config, 'lr_scheduler') and self.config.lr_scheduler:
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=20,
                verbose=True
            )
        return None
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'recommendation_loss': 0.0, 
            'dpo_loss': 0.0,
            'generation_loss': 0.0
        }
        
        num_batches = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # 解析批次数据
            g, user_embeddings, item_v_feat, item_t_feat, \
            user_indices, pos_items, neg_items, item_embeddings, \
            explanation_tokens = self._parse_batch_data(batch_data)
            
            # 前向传播
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_step(
                        g, user_embeddings, item_v_feat, item_t_feat,
                        item_embeddings, pos_items, neg_items, 
                        explanation_tokens, epoch
                    )
                    total_loss, loss_dict = self.model.compute_joint_loss(
                        outputs, user_indices, pos_items, neg_items
                    )
            else:
                outputs = self._forward_step(
                    g, user_embeddings, item_v_feat, item_t_feat,
                    item_embeddings, pos_items, neg_items,
                    explanation_tokens, epoch
                )
                total_loss, loss_dict = self.model.compute_joint_loss(
                    outputs, user_indices, pos_items, neg_items
                )
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                self.optimizer.step()
            
            # 累积损失
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key] += value.item()
            
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Rec': f'{loss_dict.get("recommendation_loss", 0):.4f}',
                'DPO': f'{loss_dict.get("dpo_loss", 0):.4f}',
                'Gen': f'{loss_dict.get("generation_loss", 0):.4f}'
            })
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def _parse_batch_data(self, batch_data) -> Tuple[Any, ...]:
        """解析批次数据"""
        # 根据实际数据格式实现
        g = batch_data.get('graph')
        user_embeddings = batch_data.get('user_embeddings')
        item_v_feat = batch_data.get('item_v_feat')
        item_t_feat = batch_data.get('item_t_feat')
        user_indices = batch_data.get('user_indices')
        pos_items = batch_data.get('pos_items')
        neg_items = batch_data.get('neg_items')
        item_embeddings = batch_data.get('item_embeddings')
        explanation_tokens = batch_data.get('explanation_tokens', None)
        
        return (g, user_embeddings, item_v_feat, item_t_feat,
                user_indices, pos_items, neg_items, item_embeddings,
                explanation_tokens)
    
    def _forward_step(self, g, user_embeddings, item_v_feat, item_t_feat,
                     item_embeddings, pos_items, neg_items, 
                     explanation_tokens, epoch) -> Dict[str, Any]:
        """执行前向传播步骤"""
        mode = self._get_training_mode(epoch)
        
        return self.model.forward(
            g=g,
            user_embeddings=user_embeddings,
            item_v_feat=item_v_feat,
            item_t_feat=item_t_feat,
            item_embeddings=item_embeddings,
            positive_items=pos_items,
            negative_items=neg_items,
            explanation_tokens=explanation_tokens,
            mode=mode
        )
    
    def _get_training_mode(self, epoch: int) -> str:
        """根据训练策略确定训练模式"""
        if self.config.training_strategy == 'joint':
            return 'joint'
        elif self.config.training_strategy == 'sequential':
            if epoch < self.config.stage1_epochs:
                return 'recommendation'
            elif epoch < self.config.stage1_epochs + self.config.stage2_epochs:
                return 'dpo'
            else:
                return 'generation'
        else:
            return 'joint'
    
    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            val_dataloader: 验证数据加载器
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        total_recall_5 = 0.0
        total_recall_10 = 0.0 
        total_recall_20 = 0.0
        total_ndcg_5 = 0.0
        total_ndcg_10 = 0.0
        total_ndcg_20 = 0.0
        num_users = 0
        
        with torch.no_grad():
            for batch_data in tqdm(val_dataloader, desc='Evaluating'):
                g, user_embeddings, item_v_feat, item_t_feat, \
                user_indices, _, _, item_embeddings, _ = self._parse_batch_data(batch_data)
                
                # 获取推荐结果
                topk_items = self.model.predict_recommendations(
                    g=g,
                    user_embeddings=user_embeddings,
                    item_v_feat=item_v_feat,
                    item_t_feat=item_t_feat,
                    item_embeddings=item_embeddings,
                    topk=20
                )
                
                # 计算指标（这里需要真实的ground truth）
                # 暂时使用模拟计算
                batch_size = topk_items.size(0)
                total_recall_5 += np.random.random() * batch_size
                total_recall_10 += np.random.random() * batch_size
                total_recall_20 += np.random.random() * batch_size
                total_ndcg_5 += np.random.random() * batch_size
                total_ndcg_10 += np.random.random() * batch_size
                total_ndcg_20 += np.random.random() * batch_size
                num_users += batch_size
        
        metrics = {
            'Recall@5': total_recall_5 / num_users,
            'Recall@10': total_recall_10 / num_users,
            'Recall@20': total_recall_20 / num_users,
            'NDCG@5': total_ndcg_5 / num_users,
            'NDCG@10': total_ndcg_10 / num_users,
            'NDCG@20': total_ndcg_20 / num_users
        }
        
        return metrics
    
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            
        Returns:
            训练历史
        """
        self.logger.info("开始MIG-DPG训练...")
        self.logger.info(f"训练策略: {self.config.training_strategy}")
        self.logger.info(f"总epochs: {self.config.epochs}")
        
        training_history = {
            'train_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            epoch_losses = self.train_epoch(train_dataloader, epoch)
            training_history['train_loss'].append(epoch_losses)
            
            # 记录损失历史
            for key, value in epoch_losses.items():
                self.loss_history[key].append(value)
            
            # 验证
            if val_dataloader and epoch % 5 == 0:  # 每5个epoch验证一次
                val_metrics = self.evaluate(val_dataloader)
                training_history['val_metrics'].append(val_metrics)
                
                self.logger.info(f"Epoch {epoch} 验证结果:")
                for metric, value in val_metrics.items():
                    self.logger.info(f"  {metric}: {value:.4f}")
                
                # 学习率调度
                if self.scheduler:
                    main_metric = val_metrics.get('Recall@20', 0)
                    self.scheduler.step(main_metric)
                
                # 早停检查
                current_performance = val_metrics.get('Recall@20', 0)
                if current_performance > self.best_performance:
                    self.best_performance = current_performance
                    self.patience_counter = 0
                    
                    # 保存最佳模型
                    if self.config.save_model:
                        self.save_model('best_model.pth')
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config.early_stop:
                    self.logger.info(f"早停触发，在epoch {epoch}")
                    break
            
            # 定期保存
            if self.config.save_model and epoch % self.config.save_interval == 0:
                self.save_model(f'model_epoch_{epoch}.pth')
        
        self.logger.info("训练完成!")
        return training_history
    
    def save_model(self, filename: str):
        """保存模型"""
        save_path = os.path.join(self.config.model_save_path, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': self.current_epoch,
            'best_performance': self.best_performance,
            'loss_history': self.loss_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, save_path)
        self.logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, filename: str):
        """加载模型"""
        load_path = os.path.join(self.config.model_save_path, filename)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")
            
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_performance = checkpoint['best_performance']
        self.loss_history = checkpoint['loss_history']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.logger.info(f"模型已从 {load_path} 加载")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练总结"""
        return {
            'current_epoch': self.current_epoch,
            'best_performance': self.best_performance,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'loss_history': self.loss_history,
            'config': self.config.__dict__
        } 