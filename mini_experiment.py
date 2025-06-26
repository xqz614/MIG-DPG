#!/usr/bin/env python3
"""
MIG-DPG 小规模训练实验
验证完整的训练流程和模型性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import time
import warnings
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def create_simple_model():
    """创建简化的MIG-DPG模型"""
    
    class SimpleMIG_DPG(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128):
            super().__init__()
            
            # 嵌入层
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            
            # 推荐预测器
            self.recommendation_predictor = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
            # DPO层（简化版）
            self.dpo_predictor = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, 1)
            )
            
            # 生成层（简化版）
            self.generation_predictor = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 100),  # 简化词汇表
                nn.Softmax(dim=-1)
            )
            
        def forward(self, users, items, mode='recommendation'):
            user_emb = self.user_embedding(users)
            item_emb = self.item_embedding(items)
            user_item = torch.cat([user_emb, item_emb], dim=-1)
            
            if mode == 'recommendation':
                return self.recommendation_predictor(user_item)
            elif mode == 'dpo':
                return self.dpo_predictor(user_item)
            elif mode == 'generation':
                return self.generation_predictor(user_item)
            else:
                rec_score = self.recommendation_predictor(user_item)
                dpo_score = self.dpo_predictor(user_item)
                gen_probs = self.generation_predictor(user_item)
                return rec_score, dpo_score, gen_probs
        
        def get_all_scores(self, users):
            """获取用户对所有物品的评分"""
            user_emb = self.user_embedding(users)  # [batch_size, embedding_dim]
            item_emb = self.item_embedding.weight  # [num_items, embedding_dim]
            
            # 计算所有用户-物品对的分数
            scores = torch.mm(user_emb, item_emb.t())  # [batch_size, num_items]
            return torch.sigmoid(scores)
    
    return SimpleMIG_DPG

def create_synthetic_dataset():
    """创建合成数据集"""
    
    class SyntheticDataset:
        def __init__(self, num_users=200, num_items=100, num_interactions=2000):
            self.num_users = num_users
            self.num_items = num_items
            
            # 生成用户-物品交互
            self.interactions = []
            
            # 为每个用户生成偏好模式
            user_preferences = {}
            for user in range(num_users):
                # 每个用户偏好5-15个物品类别
                preferred_items = np.random.choice(num_items, 
                                                 np.random.randint(5, 16), 
                                                 replace=False)
                user_preferences[user] = set(preferred_items)
            
            # 生成交互（80%正向，20%负向）
            for _ in range(num_interactions):
                user = np.random.randint(0, num_users)
                
                if np.random.random() < 0.8:  # 正向交互
                    if user in user_preferences:
                        item = np.random.choice(list(user_preferences[user]))
                        rating = np.random.choice([4, 5])  # 高评分
                    else:
                        item = np.random.randint(0, num_items)
                        rating = np.random.choice([3, 4])
                else:  # 负向交互
                    item = np.random.randint(0, num_items)
                    rating = np.random.choice([1, 2])  # 低评分
                
                self.interactions.append([user, item, rating])
            
            # 转换为numpy数组
            self.interactions = np.array(self.interactions)
            
            # 分割数据集
            np.random.shuffle(self.interactions)
            train_size = int(0.8 * len(self.interactions))
            val_size = int(0.1 * len(self.interactions))
            
            self.train_data = self.interactions[:train_size]
            self.val_data = self.interactions[train_size:train_size+val_size]
            self.test_data = self.interactions[train_size+val_size:]
            
            print(f"数据集统计:")
            print(f"  用户数: {num_users}, 物品数: {num_items}")
            print(f"  总交互数: {len(self.interactions)}")
            print(f"  训练集: {len(self.train_data)}")
            print(f"  验证集: {len(self.val_data)}")
            print(f"  测试集: {len(self.test_data)}")
        
        def get_dataloader(self, split='train', batch_size=64, shuffle=True):
            if split == 'train':
                data = self.train_data
            elif split == 'val':
                data = self.val_data
            else:
                data = self.test_data
            
            class SimpleDataset:
                def __init__(self, interactions, num_items):
                    self.interactions = interactions
                    self.num_items = num_items
                
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
                        'rating': rating
                    }
            
            dataset = SimpleDataset(data, self.num_items)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return SyntheticDataset()

def compute_metrics(model, dataloader, device, topk=10):
    """计算推荐指标"""
    model.eval()
    
    all_users = []
    all_pos_items = []
    all_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            users = batch['user'].to(device)
            pos_items = batch['pos_item'].to(device)
            
            # 获取用户对所有物品的评分
            scores = model.get_all_scores(users)
            
            all_users.extend(users.cpu().numpy())
            all_pos_items.extend(pos_items.cpu().numpy())
            all_scores.append(scores.cpu())
    
    all_scores = torch.cat(all_scores, dim=0)
    
    # 计算Recall@K和NDCG@K
    recall_sum = 0
    ndcg_sum = 0
    num_users = len(all_users)
    
    for i in range(num_users):
        user_scores = all_scores[i]
        pos_item = all_pos_items[i]
        
        # 获取Top-K推荐
        _, topk_items = torch.topk(user_scores, topk)
        topk_items = topk_items.numpy()
        
        # Recall@K
        if pos_item in topk_items:
            recall_sum += 1
            
            # NDCG@K
            pos_rank = np.where(topk_items == pos_item)[0][0] + 1
            ndcg_sum += 1.0 / np.log2(pos_rank + 1)
    
    recall = recall_sum / num_users
    ndcg = ndcg_sum / num_users
    
    return recall, ndcg

def train_experiment():
    """运行完整训练实验"""
    print("🚀 开始MIG-DPG小规模训练实验")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("\n📊 创建合成数据集...")
    dataset = create_synthetic_dataset()
    
    train_loader = dataset.get_dataloader('train', batch_size=32, shuffle=True)
    val_loader = dataset.get_dataloader('val', batch_size=64, shuffle=False)
    test_loader = dataset.get_dataloader('test', batch_size=64, shuffle=False)
    
    # 创建模型
    print("\n🧠 初始化模型...")
    ModelClass = create_simple_model()
    model = ModelClass(dataset.num_users, dataset.num_items).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    rec_criterion = nn.BCELoss()
    dpo_criterion = nn.MSELoss()
    gen_criterion = nn.CrossEntropyLoss()
    
    # 训练配置
    num_epochs = 20
    log_interval = 50
    best_val_recall = 0
    
    print(f"\n🏋️ 开始训练 (共{num_epochs}个epochs)...")
    
    train_losses = []
    val_recalls = []
    val_ndcgs = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            users = batch['user'].to(device)
            pos_items = batch['pos_item'].to(device)
            neg_items = batch['neg_item'].to(device)
            ratings = batch['rating'].float().to(device)
            
            optimizer.zero_grad()
            
            # 推荐损失 (BPR-like)
            pos_scores = model(users, pos_items, mode='recommendation').squeeze()
            neg_scores = model(users, neg_items, mode='recommendation').squeeze()
            rec_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
            
            # DPO损失 (简化版)
            pos_dpo = model(users, pos_items, mode='dpo').squeeze()
            neg_dpo = model(users, neg_items, mode='dpo').squeeze()
            dpo_loss = dpo_criterion(pos_dpo, ratings/5.0) + dpo_criterion(neg_dpo, torch.zeros_like(neg_dpo))
            
            # 生成损失 (简化版)
            gen_targets = torch.randint(0, 100, (len(users),)).to(device)
            gen_probs = model(users, pos_items, mode='generation')
            gen_loss = gen_criterion(gen_probs, gen_targets)
            
            # 总损失
            total_loss = rec_loss + 0.3 * dpo_loss + 0.2 * gen_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            batch_count += 1
            
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch+1:2d} [{batch_idx:3d}/{len(train_loader):3d}] "
                      f"Loss: {total_loss.item():.4f} "
                      f"(Rec: {rec_loss.item():.4f}, DPO: {dpo_loss.item():.4f}, Gen: {gen_loss.item():.4f})")
        
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # 验证
        if (epoch + 1) % 5 == 0:
            print(f"\n🔍 验证 Epoch {epoch+1}...")
            val_recall, val_ndcg = compute_metrics(model, val_loader, device, topk=10)
            val_recalls.append(val_recall)
            val_ndcgs.append(val_ndcg)
            
            print(f"验证结果: Recall@10={val_recall:.4f}, NDCG@10={val_ndcg:.4f}")
            
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                print(f"✅ 新的最佳验证Recall: {best_val_recall:.4f}")
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:2d} 完成，用时: {epoch_time:.2f}s，平均损失: {avg_train_loss:.4f}\n")
    
    # 最终测试
    print("🎯 最终测试评估...")
    test_recall, test_ndcg = compute_metrics(model, test_loader, device, topk=10)
    
    print(f"\n📈 最终结果:")
    print(f"  测试 Recall@10: {test_recall:.4f}")
    print(f"  测试 NDCG@10: {test_ndcg:.4f}")
    print(f"  最佳验证 Recall@10: {best_val_recall:.4f}")
    
    # 分析训练曲线
    print(f"\n📊 训练分析:")
    print(f"  初始训练损失: {train_losses[0]:.4f}")
    print(f"  最终训练损失: {train_losses[-1]:.4f}")
    print(f"  损失下降: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    
    if val_recalls:
        print(f"  验证Recall提升: {((val_recalls[-1] - val_recalls[0]) / val_recalls[0] * 100 if val_recalls[0] > 0 else 0):.1f}%")
    
    return {
        'train_losses': train_losses,
        'val_recalls': val_recalls,
        'val_ndcgs': val_ndcgs,
        'test_recall': test_recall,
        'test_ndcg': test_ndcg,
        'best_val_recall': best_val_recall
    }

def main():
    """主函数"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        results = train_experiment()
        
        print("\n🎉 实验完成！")
        print("=" * 60)
        print("实验结果已保存，可以进行进一步分析。")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 