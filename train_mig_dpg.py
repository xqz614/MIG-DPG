#!/usr/bin/env python3
"""
MIG-DPG 完整训练脚本
包含数据加载、模型创建、训练和评估的完整流程
"""

import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from mig_dpg.models.mig_dpg_model import MIG_DPG_Model
from mig_dpg.configs.mig_dpg_default_config import MIG_DPG_DefaultConfig
from mig_dpg.trainer import MIG_DPG_Trainer
from mig_dpg.data_processor import MIG_DPG_DataProcessor


def set_seed(seed: int):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MIG-DPG Training Script')
    
    # 基础参数
    parser.add_argument('--dataset', type=str, default='baby', 
                       choices=['baby', 'sports', 'clothing', 'elec'],
                       help='Dataset name')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    
    # 模型参数
    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--k_e', type=int, default=4, help='Embedding modality hops')
    parser.add_argument('--k_t', type=int, default=2, help='Text modality hops')
    parser.add_argument('--k_v', type=int, default=1, help='Vision modality hops')
    
    # 训练策略
    parser.add_argument('--training_strategy', type=str, default='joint',
                       choices=['joint', 'sequential', 'curriculum'],
                       help='Training strategy')
    parser.add_argument('--dpo_weight', type=float, default=0.5, 
                       help='DPO loss weight')
    parser.add_argument('--generation_weight', type=float, default=0.3,
                       help='Generation loss weight')
    
    # 数据参数
    parser.add_argument('--preference_data_ratio', type=float, default=0.3,
                       help='Ratio of preference data to generate')
    parser.add_argument('--synthetic_explanation', action='store_true',
                       help='Use synthetic explanation data')
    
    # 保存和日志
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='Save model checkpoints')
    parser.add_argument('--model_save_path', type=str, default='./saved_models/',
                       help='Path to save models')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')
    
    return parser.parse_args()


def create_experiment_name(args):
    """创建实验名称"""
    if args.experiment_name:
        return args.experiment_name
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"MIG_DPG_{args.dataset}_{args.training_strategy}_{timestamp}"
    return experiment_name


def main():
    """主训练函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建实验名称
    experiment_name = create_experiment_name(args)
    print(f"实验名称: {experiment_name}")
    
    # 创建配置
    config = MIG_DPG_DefaultConfig()
    
    # 更新配置
    config.dataset = args.dataset
    config.gpu_id = args.gpu_id
    config.seed = args.seed
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.embedding_size = args.embedding_size
    config.k_e = args.k_e
    config.k_t = args.k_t
    config.k_v = args.k_v
    config.training_strategy = args.training_strategy
    config.dpo_weight = args.dpo_weight
    config.generation_weight = args.generation_weight
    config.preference_data_ratio = args.preference_data_ratio
    config.synthetic_explanation = args.synthetic_explanation
    config.save_model = args.save_model
    config.model_save_path = os.path.join(args.model_save_path, experiment_name)
    
    # 根据数据集调整配置
    config.get_dataset_specific_config(args.dataset)
    
    # 打印配置
    print("\n" + "="*60)
    print("MIG-DPG 训练配置")
    print("="*60)
    config.print_config()
    
    # 创建数据处理器
    print("\n" + "="*60)
    print("数据准备")
    print("="*60)
    data_processor = MIG_DPG_DataProcessor(config)
    
    # 加载数据
    data = data_processor.load_data(config.dataset)
    
    # 更新配置中的数据统计
    config.num_users = data['num_users']
    config.num_items = data['num_items']
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset = data_processor.create_datasets(data)
    
    # 创建数据加载器
    train_dataloader, val_dataloader, test_dataloader = data_processor.create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # 创建模型
    print("\n" + "="*60)
    print("模型创建")
    print("="*60)
    model = MIG_DPG_Model(config)
    
    # 模型统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建训练器
    trainer = MIG_DPG_Trainer(model, config, device)
    
    # 开始训练
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    try:
        training_history = trainer.train(train_dataloader, val_dataloader)
        
        # 训练完成后的最终评估
        print("\n" + "="*60)
        print("最终评估")
        print("="*60)
        
        final_metrics = trainer.evaluate(test_dataloader)
        print("测试集最终结果:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 保存训练历史
        if config.save_model:
            import json
            history_path = os.path.join(config.model_save_path, 'training_history.json')
            
            # 转换为可序列化的格式
            serializable_history = {}
            for key, value in training_history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        # 处理嵌套字典
                        serializable_history[key] = [
                            {k: float(v) if torch.is_tensor(v) else v for k, v in item.items()}
                            for item in value
                        ]
                    else:
                        serializable_history[key] = [float(v) if torch.is_tensor(v) else v for v in value]
                else:
                    serializable_history[key] = value
            
            with open(history_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
            print(f"训练历史已保存到: {history_path}")
        
        # 保存最终指标
        final_results = {
            'experiment_name': experiment_name,
            'config': config.__dict__,
            'final_metrics': final_metrics,
            'best_performance': trainer.best_performance,
            'total_epochs': trainer.current_epoch
        }
        
        if config.save_model:
            results_path = os.path.join(config.model_save_path, 'final_results.json')
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            print(f"最终结果已保存到: {results_path}")
        
        print("\n" + "="*60)
        print("训练完成! 🎉")
        print("="*60)
        print(f"最佳性能: {trainer.best_performance:.4f}")
        print(f"总训练轮数: {trainer.current_epoch}")
        
        # 生成一些推荐示例
        print("\n推荐示例:")
        model.eval()
        with torch.no_grad():
            sample_data = next(iter(test_dataloader))
            sample_users = sample_data['user_embeddings'][:5]  # 取前5个用户
            
            recommendations = model.predict_recommendations(
                g=sample_data['graph'],
                user_embeddings=sample_users,
                item_v_feat=sample_data['item_v_feat'],
                item_t_feat=sample_data['item_t_feat'],
                item_embeddings=sample_data['item_embeddings'],
                topk=10
            )
            
            for i, rec_list in enumerate(recommendations):
                top5_items = rec_list[:5].tolist()
                print(f"  用户 {i}: {top5_items}")
                
        # 生成解释示例
        if config.generation_enabled:
            print("\n解释生成示例:")
            target_items = recommendations[:3, 0]  # 每个用户的第一个推荐
            
            explanations = model.generate_explanations(
                g=sample_data['graph'],
                user_embeddings=sample_users[:3],
                item_v_feat=sample_data['item_v_feat'],
                item_t_feat=sample_data['item_t_feat'],
                target_items=target_items,
                item_embeddings=sample_data['item_embeddings']
            )
            
            for i, explanation in enumerate(explanations):
                tokens_preview = explanation[:10]
                print(f"  用户 {i} → 物品 {target_items[i].item()}: {tokens_preview}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        if trainer.best_performance > 0:
            print(f"当前最佳性能: {trainer.best_performance:.4f}")
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n程序结束")


if __name__ == "__main__":
    main() 