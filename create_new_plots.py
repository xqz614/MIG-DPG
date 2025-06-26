import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.patches as mpatches

# 设置专业样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

import os
if not os.path.exists('figures'):
    os.makedirs('figures')

def create_innovation_overview():
    """创建三大创新点总览图"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 三大创新点
    innovations = [
        {
            'pos': (2.5, 6.5),
            'size': (3.5, 2.5),
            'color': '#FF6B6B',
            'title': 'Innovation 1:\nDirect Preference\nOptimization',
            'details': '• First DPO adaptation to\n  recommendation systems\n• Eliminates reward modeling\n• Direct preference alignment\n• Theoretical guarantees'
        },
        {
            'pos': (6.5, 6.5),
            'size': (3.5, 2.5),
            'color': '#4ECDC4', 
            'title': 'Innovation 2:\nTransformer-based\nGeneration',
            'details': '• Natural language explanations\n• Joint training with\n  recommendation\n• Multi-objective optimization\n• Enhanced interpretability'
        },
        {
            'pos': (10.5, 6.5),
            'size': (3.5, 2.5),
            'color': '#45B7D1',
            'title': 'Innovation 3:\nModal-Independent\nProcessing',
            'details': '• Preserves modality\n  characteristics\n• Cross-modal attention fusion\n• Adaptive layer aggregation\n• Robust multimodal learning'
        }
    ]
    
    for innovation in innovations:
        x, y = innovation['pos']
        w, h = innovation['size']
        
        # 主框
        main_box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                 facecolor=innovation['color'], edgecolor='black', 
                                 linewidth=3, alpha=0.8)
        ax.add_patch(main_box)
        
        # 标题
        ax.text(x + w/2, y + h - 0.7, innovation['title'], 
               ha='center', va='center', fontweight='bold', fontsize=12, color='white')
        
        # 详细内容框
        detail_box = FancyBboxPatch((x, y-2.5), w, 2, boxstyle="round,pad=0.1",
                                   facecolor='white', edgecolor=innovation['color'], 
                                   linewidth=2)
        ax.add_patch(detail_box)
        
        ax.text(x + w/2, y-1.5, innovation['details'], 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 中心融合
    fusion_circle = Circle((8, 3.5), 1.5, facecolor='#DDA0DD', edgecolor='black', 
                          linewidth=3, alpha=0.9)
    ax.add_patch(fusion_circle)
    ax.text(8, 3.5, 'MIG-DPG\nFramework', ha='center', va='center', 
           fontweight='bold', fontsize=14, color='white')
    
    # 箭头连接
    for innovation in innovations:
        x, y = innovation['pos']
        w, h = innovation['size']
        start_x = x + w/2
        start_y = y
        
        ax.annotate('', xy=(8, 5), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=3, color='gray', alpha=0.8))
    
    # 性能指标
    metrics_box = FancyBboxPatch((1, 0.5), 14, 1.2, boxstyle="round,pad=0.2",
                                facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(metrics_box)
    
    metrics_text = 'Key Results: 21.4% NDCG@10 improvement • 23.8% DPO contribution • 8.4% generation enhancement'
    ax.text(8, 1.1, metrics_text, ha='center', va='center', 
           fontweight='bold', fontsize=12)
    
    ax.text(8, 9.5, 'MIG-DPG: Three Key Innovations for Enhanced Recommendation Systems', 
           ha='center', va='center', fontweight='bold', fontsize=18)
    
    plt.tight_layout()
    plt.savefig('figures/innovation_overview_new.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/innovation_overview_new.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_problem_solution():
    """创建问题-解决方案对比图"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 左侧问题
    problems_box = FancyBboxPatch((0.5, 2), 6.5, 6, boxstyle="round,pad=0.3",
                                 facecolor='#FFE5E5', edgecolor='red', linewidth=3)
    ax.add_patch(problems_box)
    
    ax.text(3.75, 7.5, 'Existing Methods: Limitations', 
           ha='center', va='center', fontweight='bold', fontsize=16, color='red')
    
    problems = [
        '❌ Lack explicit preference alignment',
        '❌ Black-box recommendation without explanations', 
        '❌ Suboptimal multimodal fusion strategies',
        '❌ Implicit feedback may not reflect true preferences',
        '❌ Complex reward modeling in traditional RLHF',
        '❌ Limited interpretability and transparency'
    ]
    
    for i, problem in enumerate(problems):
        ax.text(1, 6.8 - i*0.6, problem, ha='left', va='center', 
               fontsize=11, fontweight='bold')
    
    # 右侧解决方案
    solutions_box = FancyBboxPatch((9, 2), 6.5, 6, boxstyle="round,pad=0.3",
                                  facecolor='#E5FFE5', edgecolor='green', linewidth=3)
    ax.add_patch(solutions_box)
    
    ax.text(12.25, 7.5, 'MIG-DPG: Our Solutions', 
           ha='center', va='center', fontweight='bold', fontsize=16, color='green')
    
    solutions = [
        '✅ Direct Preference Optimization (DPO)',
        '✅ Transformer-based explanation generation',
        '✅ Modal-independent processing + attention fusion',
        '✅ Explicit preference modeling from triplets',
        '✅ Single-stage training without reward modeling',
        '✅ Joint optimization for accuracy + interpretability'
    ]
    
    for i, solution in enumerate(solutions):
        ax.text(9.5, 6.8 - i*0.6, solution, ha='left', va='center', 
               fontsize=11, fontweight='bold')
    
    # 转换箭头
    ax.annotate('', xy=(8.8, 5), xytext=(7.2, 5),
               arrowprops=dict(arrowstyle='->', lw=4, color='blue'))
    ax.text(8, 5.5, 'MIG-DPG\nSolution', ha='center', va='center', 
           fontweight='bold', fontsize=12, color='blue')
    
    # 底部：核心贡献
    contributions_box = FancyBboxPatch((2, 0.2), 12, 1.3, boxstyle="round,pad=0.2",
                                      facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(contributions_box)
    
    ax.text(8, 1.2, 'Core Contributions', ha='center', va='center', 
           fontweight='bold', fontsize=14, color='blue')
    
    contributions_text = ('First DPO adaptation to multimodal recommendations • '
                         'Joint training for accuracy + interpretability • '
                         'Theoretical convergence guarantees')
    ax.text(8, 0.6, contributions_text, ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    ax.text(8, 9.3, 'Problem Motivation and MIG-DPG Solutions', 
           ha='center', va='center', fontweight='bold', fontsize=18)
    
    plt.tight_layout()
    plt.savefig('figures/problem_solution_new.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/problem_solution_new.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_multimodal_gnn_structure():
    """创建多模态图神经网络结构图"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 定义颜色方案
    colors = {
        'text': '#FF6B6B',      # 红色 - 文本模态
        'visual': '#4ECDC4',    # 青色 - 视觉模态  
        'categorical': '#45B7D1', # 蓝色 - 类别模态
        'user': '#96CEB4',      # 绿色 - 用户节点
        'item': '#FFEAA7',      # 黄色 - 物品节点
        'fusion': '#DDA0DD'     # 紫色 - 融合层
    }
    
    # 绘制用户和物品节点 (左侧)
    user_positions = [(2, 8), (2, 6.5), (2, 5), (2, 3.5)]
    item_positions = [(4.5, 8), (4.5, 6.5), (4.5, 5), (4.5, 3.5)]
    
    # 用户节点
    for i, (x, y) in enumerate(user_positions):
        circle = Circle((x, y), 0.25, facecolor=colors['user'], 
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, f'U{i+1}', ha='center', va='center', 
                fontweight='bold', fontsize=9)
    
    # 物品节点
    for i, (x, y) in enumerate(item_positions):
        circle = Circle((x, y), 0.25, facecolor=colors['item'], 
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, f'I{i+1}', ha='center', va='center', 
                fontweight='bold', fontsize=9)
    
    # 绘制连接边
    connections = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 0), (3, 3)]
    for u_idx, i_idx in connections:
        x1, y1 = user_positions[u_idx]
        x2, y2 = item_positions[i_idx]
        ax.plot([x1+0.25, x2-0.25], [y1, y2], 'k-', linewidth=1.5, alpha=0.6)
    
    # 模态特征提取器 (中间)
    modality_boxes = [
        (7, 8, 2.5, 0.8, colors['text'], 'Text Features\n(BERT Embeddings)'),
        (7, 6.5, 2.5, 0.8, colors['visual'], 'Visual Features\n(Vision Transformer)'),
        (7, 5, 2.5, 0.8, colors['categorical'], 'Categorical Features\n(Embedding Layers)')
    ]
    
    for x, y, w, h, color, label in modality_boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontweight='bold', fontsize=10)
    
    # Modal-Independent GNN处理器 (右侧)
    gnn_boxes = [
        (11, 8, 2.5, 0.8, colors['text'], 'Text GNN\nLayers'),
        (11, 6.5, 2.5, 0.8, colors['visual'], 'Visual GNN\nLayers'),
        (11, 5, 2.5, 0.8, colors['categorical'], 'Categorical GNN\nLayers')
    ]
    
    for x, y, w, h, color, label in gnn_boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontweight='bold', fontsize=10)
    
    # 跨模态注意力融合 (最右侧)
    fusion_box = FancyBboxPatch((11, 3), 2.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor=colors['fusion'], edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(12.25, 3.75, 'Cross-Modal\nAttention Fusion', ha='center', va='center',
            fontweight='bold', fontsize=11, color='white')
    
    # 绘制箭头连接
    # 从图到特征提取器
    ax.annotate('', xy=(7, 6.8), xytext=(4.8, 5.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    # 从特征提取器到GNN
    for i, ((x1, y1, w1, h1, _, _), (x2, y2, w2, h2, _, _)) in enumerate(zip(modality_boxes, gnn_boxes)):
        ax.annotate('', xy=(x2, y2+h2/2), xytext=(x1+w1, y1+h1/2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 从GNN到融合层
    ax.annotate('', xy=(12.25, 4.5), xytext=(12.25, 5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 添加标题和图例
    ax.text(8, 9.5, 'Multimodal Graph Neural Network Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # 创建图例
    legend_elements = [
        mpatches.Patch(color=colors['user'], label='User Nodes'),
        mpatches.Patch(color=colors['item'], label='Item Nodes'),
        mpatches.Patch(color=colors['text'], label='Text Modality'),
        mpatches.Patch(color=colors['visual'], label='Visual Modality'),
        mpatches.Patch(color=colors['categorical'], label='Categorical Modality'),
        mpatches.Patch(color=colors['fusion'], label='Fusion Layer')
    ]
    
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=6)
    
    # 添加说明文字
    ax.text(3.25, 2, 'Bipartite\nUser-Item Graph', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.25, 2, 'Modal-Independent\nProcessing', fontsize=12, fontweight='bold', ha='center')
    ax.text(12.25, 2, 'Cross-Modal\nFusion', fontsize=12, fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig('figures/multimodal_gnn_structure_new.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/multimodal_gnn_structure_new.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dpo_mechanism():
    """创建DPO机制图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: Bradley-Terry偏好模型
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    
    # 用户和两个物品
    user_circle = Circle((2, 4), 0.5, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(user_circle)
    ax1.text(2, 4, 'User', ha='center', va='center', fontweight='bold')
    
    item1_circle = Circle((6, 6), 0.5, facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax1.add_patch(item1_circle)
    ax1.text(6, 6, 'Item+', ha='center', va='center', fontweight='bold')
    
    item2_circle = Circle((6, 2), 0.5, facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax1.add_patch(item2_circle)
    ax1.text(6, 2, 'Item-', ha='center', va='center', fontweight='bold')
    
    # 偏好箭头
    ax1.annotate('', xy=(5.5, 6), xytext=(2.5, 4.3),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax1.annotate('', xy=(5.5, 2), xytext=(2.5, 3.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='--'))
    
    ax1.text(3.5, 5.5, 'Preferred', fontsize=10, color='green', fontweight='bold')
    ax1.text(3.5, 2.5, 'Less Preferred', fontsize=10, color='red', fontweight='bold')
    
    # 简化的Bradley-Terry公式
    ax1.text(5, 0.5, 'P(item+ > item- | user) = sigmoid(score+ - score-)', 
            fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    ax1.set_title('Bradley-Terry Preference Model', fontweight='bold', fontsize=14)
    
    # 子图2: DPO目标函数
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    
    # 绘制损失函数曲线
    x = np.linspace(-3, 3, 100)
    y = -np.log(1 / (1 + np.exp(-x)))  # 负对数sigmoid
    
    ax2_inset = fig.add_axes([0.6, 0.6, 0.25, 0.2])
    ax2_inset.plot(x, y, 'b-', linewidth=3, label='DPO Loss')
    ax2_inset.set_xlabel('Score Difference', fontsize=10)
    ax2_inset.set_ylabel('Loss', fontsize=10)
    ax2_inset.grid(True, alpha=0.3)
    ax2_inset.legend()
    
    # 简化的DPO目标函数
    ax2.text(5, 4, 'Direct Preference Optimization', fontsize=16, ha='center', fontweight='bold')
    ax2.text(5, 3, 'Directly optimizes preference relationships', fontsize=12, ha='center')
    ax2.text(5, 2, 'without explicit reward modeling', fontsize=12, ha='center')
    ax2.text(5, 1, 'Single-stage training with theoretical guarantees', fontsize=12, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    ax2.set_title('DPO Objective Function', fontweight='bold', fontsize=14)
    
    # 子图3: 偏好数据流
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 8)
    ax3.axis('off')
    
    # 数据流程
    steps = [
        (1.5, 6, 'Preference\nTriplets\n(u, i+, i-)'),
        (5, 6, 'Score\nDifferences\ns+ - s-'),
        (8.5, 6, 'DPO\nLoss'),
        (5, 3, 'Model\nUpdate')
    ]
    
    colors_flow = ['lightgreen', 'lightblue', 'orange', 'lightcoral']
    
    for i, (x, y, text) in enumerate(steps):
        rect = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1, boxstyle="round,pad=0.1",
                             facecolor=colors_flow[i], edgecolor='black', linewidth=2)
        ax3.add_patch(rect)
        ax3.text(x, y, text, ha='center', va='center', fontweight='bold', fontsize=9)
    
    # 箭头连接
    arrows = [(2.2, 6, 4.3, 6), (5.7, 6, 7.8, 6), (8.5, 5.5, 5.7, 3.5)]
    for x1, y1, x2, y2 in arrows:
        ax3.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 反馈箭头
    ax3.annotate('', xy=(1.5, 5.5), xytext=(4.3, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red', 
                               connectionstyle="arc3,rad=0.5"))
    ax3.text(2.5, 4, 'Gradient\nUpdate', fontsize=9, color='red', fontweight='bold')
    
    ax3.set_title('DPO Training Flow', fontweight='bold', fontsize=14)
    
    # 子图4: 对比传统方法
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 8)
    ax4.axis('off')
    
    # 传统RLHF vs DPO
    traditional_box = FancyBboxPatch((0.5, 5), 4, 2, boxstyle="round,pad=0.1",
                                    facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax4.add_patch(traditional_box)
    ax4.text(2.5, 6, 'Traditional RLHF\n• Reward Model Training\n• PPO Optimization\n• Complex Pipeline', 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    dpo_box = FancyBboxPatch((5.5, 5), 4, 2, boxstyle="round,pad=0.1",
                            facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax4.add_patch(dpo_box)
    ax4.text(7.5, 6, 'DPO (Ours)\n• Direct Optimization\n• Single Stage Training\n• Stable Convergence', 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    # VS标记
    ax4.text(5, 6, 'VS', ha='center', va='center', fontsize=20, fontweight='bold', color='red')
    
    ax4.set_title('DPO vs Traditional Methods', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('figures/dpo_mechanism_new.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/dpo_mechanism_new.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Creating new professional plots...")
    
    create_innovation_overview()
    print("✓ Innovation overview created")
    
    create_problem_solution()
    print("✓ Problem-solution comparison created")
    
    create_multimodal_gnn_structure()
    print("✓ Multimodal GNN structure created")
    
    create_dpo_mechanism()
    print("✓ DPO mechanism created")
    
    print("All new plots created successfully!") 