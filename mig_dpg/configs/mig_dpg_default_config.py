"""
MIG-DPG Default Configuration
集成了原始MIG-GT配置和新增的DPO、生成式配置
"""

class MIG_DPG_DefaultConfig:
    """MIG-DPG模型的默认配置"""
    
    def __init__(self):
        # =============== 基础配置 ===============
        self.model = 'MIG_DPG'
        self.dataset = 'baby'  # baby, sports, clothing, elec
        self.gpu_id = 0
        self.seed = 2022
        
        # =============== 原始MIG-GT配置 ===============
        # 模型维度
        self.embedding_size = 64
        self.text_feat_size = 384
        self.vision_feat_size = 4096
        
        # 图卷积参数 (MIRF: Modality-Independent Receptive Fields)
        self.k_e = 4  # 嵌入模态的跳数
        self.k_t = 2  # 文本模态的跳数  
        self.k_v = 1  # 视觉模态的跳数
        
        # MGDCF参数
        self.alpha = 0.2  # 消息聚合权重
        self.beta = 0.5   # 层间连接权重
        self.n_layers = 2 # GCN层数
        
        # Dropout和正则化
        self.dropout = 0.1
        self.edge_drop_rate = 0.1
        self.message_drop_rate = 0.1
        
        # 全局Transformer
        self.num_samples = 32  # 采样节点数
        
        # =============== DPO配置 ===============
        self.dpo_enabled = True
        self.dpo_hidden_dim = 256
        self.dpo_beta = 0.1  # DPO温度参数
        self.dpo_num_heads = 4
        self.dpo_weight = 0.5  # DPO损失权重
        
        # =============== 生成式配置 ===============
        self.generation_enabled = True
        self.vocab_size = 10000  # 词汇表大小
        self.max_explanation_length = 128  # 最大解释长度
        self.max_seq_length = 128  # 最大序列长度（别名）
        self.gen_hidden_dim = 512  # 生成器隐藏维度
        self.gen_num_layers = 6  # Transformer层数
        self.gen_num_heads = 8   # 注意力头数
        self.generation_weight = 0.3  # 生成损失权重
        
        # =============== 训练配置 ===============
        self.learning_rate = 1e-3
        self.decay = 1e-4  # L2正则化
        self.batch_size = 1024
        self.test_batch_size = 2048
        self.epochs = 1000
        self.early_stop = 100
        
        # 损失权重
        self.recommendation_weight = 1.0
        
        # =============== 评估配置 ===============
        self.topk = [5, 10, 20]
        self.metrics = ['Recall', 'NDCG']
        
        # =============== 多任务训练策略 ===============
        self.training_strategy = 'joint'  # 'sequential', 'joint', 'curriculum'
        
        # 顺序训练配置（如果使用sequential）
        self.stage1_epochs = 100  # 先训练推荐任务
        self.stage2_epochs = 50   # 再训练DPO
        self.stage3_epochs = 50   # 最后训练生成
        
        # 课程学习配置（如果使用curriculum）
        self.curriculum_milestones = [50, 100, 150]  # 任务切换节点
        self.curriculum_weights = {
            'stage1': {'rec': 1.0, 'dpo': 0.0, 'gen': 0.0},
            'stage2': {'rec': 0.7, 'dpo': 0.3, 'gen': 0.0}, 
            'stage3': {'rec': 0.5, 'dpo': 0.3, 'gen': 0.2},
            'stage4': {'rec': 0.4, 'dpo': 0.3, 'gen': 0.3}
        }
        
        # =============== 数据配置 ===============
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        
        # 偏好数据配置
        self.preference_data_ratio = 0.3  # 用于DPO的数据比例
        self.negative_sampling_ratio = 4  # 负采样比例
        
        # 解释数据配置  
        self.explanation_data_path = None  # 如果有预训练的解释数据
        self.synthetic_explanation = True   # 是否使用合成解释数据
        
        # =============== 高级配置 ===============
        self.mixed_precision = True  # 混合精度训练
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0  # 梯度裁剪
        
        # 模型保存
        self.save_model = True
        self.model_save_path = './saved_models/'
        self.save_interval = 10  # 每多少个epoch保存一次
        
        # 日志配置
        self.log_interval = 100  # 每多少个batch记录一次
        self.tensorboard_enabled = True
        self.wandb_enabled = False
        
    def get_dataset_specific_config(self, dataset_name: str):
        """根据数据集调整特定配置"""
        if dataset_name == 'baby':
            self.k_e, self.k_t, self.k_v = 4, 2, 1
            self.num_samples = 32
            self.batch_size = 1024
        elif dataset_name == 'sports':
            self.k_e, self.k_t, self.k_v = 3, 2, 1  
            self.num_samples = 64
            self.batch_size = 512
        elif dataset_name == 'clothing':
            self.k_e, self.k_t, self.k_v = 4, 3, 2
            self.num_samples = 48
            self.batch_size = 768
        elif dataset_name == 'elec':
            self.k_e, self.k_t, self.k_v = 3, 2, 1
            self.num_samples = 40
            self.batch_size = 896
        
        self.dataset = dataset_name
        
    def get_config_dict(self):
        """返回配置字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid configuration parameter")
    
    def print_config(self):
        """打印当前配置"""
        print("=" * 50)
        print("MIG-DPG Configuration")
        print("=" * 50)
        
        for key, value in self.get_config_dict().items():
            print(f"{key:30s}: {value}")
        
        print("=" * 50) 