import warnings
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# 位置编码，与原模型相同
class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 新增: 跨模态注意力模块
class CrossModalAttention(nn.Module):
    """跨模态注意力模块：让时间序列特征关注文本信息"""

    def __init__(self, hidden_dim, nhead=4, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, query, key_value):
        """
        参数:
        - query: 需要增强的特征 [seq_len, batch_size, hidden_dim]
        - key_value: 提供信息的特征 [kv_len, batch_size, hidden_dim]

        返回:
        - output: 注意力增强后的特征 [seq_len, batch_size, hidden_dim]
        """
        # 应用多头注意力
        attn_output, _ = self.multihead_attn(query, key_value, key_value)

        # 残差连接和层归一化
        query = query + self.dropout(attn_output)
        query = self.norm1(query)

        # 前馈网络
        ffn_output = self.ffn(query)
        query = query + self.dropout(ffn_output)
        query = self.norm2(query)

        return query


class EnhancedMultiModalFusionClassifier(nn.Module):
    """
    增强版多模态融合分类器，使用跨模态注意力机制
    让时间序列特征可以从文本和预测中获取上下文信息
    """

    def __init__(self,
                 time_feature_dim,  # 时间序列特征维度
                 hidden_dim=256,  # 隐藏层维度
                 dropout=0.1,  # Dropout率
                 text_dim=768,  # 文本嵌入维度，默认是BERT维度
                 nhead=2,  # Transformer头数
                 nlayers=4,  # Transformer层数
                 seq_len=24,  # 时间序列长度
                 pred_len=12,  # 预测长度
                 use_temporal_attention=True,  # 是否使用时间注意力机制
                 use_cross_modal_attention=True,  # 是否使用跨模态注意力
                 ):
        super(EnhancedMultiModalFusionClassifier, self).__init__()

        # 文本处理模块 - 处理已有的文本嵌入
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # 时间序列特征处理模块
        self.time_feature_proj = nn.Linear(time_feature_dim, hidden_dim)

        # 预测值处理模块 (传统模型和LLM)
        self.pred_encoder = nn.Linear(2 * pred_len, hidden_dim)

        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # 时序注意力机制
        self.use_temporal_attention = use_temporal_attention
        if use_temporal_attention:
            encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                     dim_feedforward=hidden_dim * 4, dropout=dropout)
            self.temporal_transformer = TransformerEncoder(encoder_layers, nlayers)

        # 新增: 跨模态注意力机制
        self.use_cross_modal_attention = use_cross_modal_attention
        if use_cross_modal_attention:
            # 文本到时间序列的注意力
            self.text_to_time_attention = CrossModalAttention(hidden_dim, nhead, dropout)

            # 预测到时间序列的注意力
            self.pred_to_time_attention = CrossModalAttention(hidden_dim, nhead, dropout)

            # 融合层处理增强后的时间序列特征
            self.enhanced_time_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # 多模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # 文本、时间序列和预测融合
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 权重预测层 - 输出每个时间步的权重
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim, pred_len),
            nn.Sigmoid()  # 输出范围[0,1]的权重
        )

        # 保存配置参数
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

    def process_text_embeddings(self, text_embeddings):
        """
        处理已有的文本嵌入

        参数:
        - text_embeddings: 以下形状之一:
            [batch_size, text_dim]
            [batch_size, text_dim, 1]
            [batch_size, seq_len, text_dim]

        返回:
        - text_features: [batch_size, hidden_dim]
        - text_seq_features: [seq_len, batch_size, hidden_dim] (如果是序列输入)
        """
        batch_size = text_embeddings.shape[0]

        # 处理多种可能的输入形状
        if text_embeddings.dim() == 3:  # [batch_size, seq_len, dim] 或 [batch_size, dim, 1]
            if text_embeddings.shape[2] == 1:  # [batch_size, dim, 1]
                text_embeddings = text_embeddings.squeeze(2)  # [batch_size, dim]

                # 投影到隐藏维度
                text_features = self.text_proj(text_embeddings)  # [batch_size, hidden_dim]

                # 为了跨模态注意力，创建序列形式 [seq_len=1, batch_size, hidden_dim]
                text_seq_features = text_features.unsqueeze(0)

            else:  # [batch_size, seq_len, dim]
                seq_len = text_embeddings.shape[1]

                # 为每个时间步投影到隐藏维度
                text_seq = self.text_proj(text_embeddings)  # [batch_size, seq_len, hidden_dim]

                # 为了跨模态注意力，调整维度顺序 [seq_len, batch_size, hidden_dim]
                text_seq_features = text_seq.transpose(0, 1)

                # 使用平均池化得到整体文本特征
                text_features = torch.mean(text_seq, dim=1)  # [batch_size, hidden_dim]
        else:  # [batch_size, dim]
            # 投影到隐藏维度
            text_features = self.text_proj(text_embeddings)  # [batch_size, hidden_dim]

            # 为了跨模态注意力，创建序列形式 [seq_len=1, batch_size, hidden_dim]
            text_seq_features = text_features.unsqueeze(0)

        return text_features, text_seq_features

    def encode_time_features(self, time_features, text_seq_features=None, pred_seq_features=None):
        """
        编码时间序列特征，可选择性地应用跨模态注意力

        参数:
        - time_features: [batch_size, seq_len, time_feature_dim]
        - text_seq_features: [text_seq_len, batch_size, hidden_dim] 用于跨模态注意力
        - pred_seq_features: [pred_seq_len, batch_size, hidden_dim] 用于跨模态注意力

        返回:
        - temporal_features: [batch_size, hidden_dim]
        - time_seq_features: [seq_len, batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = time_features.shape

        # 投影到隐藏维度
        projected_features = self.time_feature_proj(time_features)  # [batch_size, seq_len, hidden_dim]

        # 调整维度顺序以适应Transformer [seq_len, batch_size, hidden_dim]
        time_seq = projected_features.transpose(0, 1)

        # 添加位置编码
        time_seq_with_pos = self.pos_encoder(time_seq)

        # 保存原始的时间序列特征
        original_time_seq = time_seq_with_pos

        # 应用时序注意力（如果启用）
        if self.use_temporal_attention:
            time_seq = self.temporal_transformer(time_seq_with_pos)  # [seq_len, batch_size, hidden_dim]
        else:
            time_seq = time_seq_with_pos

        # 应用跨模态注意力（如果启用）
        if self.use_cross_modal_attention and text_seq_features is not None and pred_seq_features is not None:
            # 让时间序列关注预测信息
            time_with_pred = self.pred_to_time_attention(time_seq, pred_seq_features)
            # 让时间序列关注文本信息
            time_with_text = self.text_to_time_attention(time_with_pred, text_seq_features)



            # 融合三种时间序列特征（原始、带文本信息、带预测信息）
            # 首先调整维度顺序回 [batch_size, seq_len, hidden_dim]
            orig_time = time_seq.transpose(0, 1)
            text_enhanced = time_with_text.transpose(0, 1)
            pred_enhanced = time_with_pred.transpose(0, 1)

            # 按特征维度拼接
            combined_time = torch.cat([orig_time, text_enhanced, pred_enhanced],
                                      dim=2)  # [batch_size, seq_len, 3*hidden_dim]

            # 融合不同来源的时间特征
            enhanced_time = self.enhanced_time_fusion(combined_time)  # [batch_size, seq_len, hidden_dim]

            # 调整回 [seq_len, batch_size, hidden_dim]
            time_seq = enhanced_time.transpose(0, 1)

        # 将维度顺序调整回来 [batch_size, seq_len, hidden_dim]
        time_seq_features = time_seq.transpose(0, 1)

        # 仅使用与预测相关的最后pred_len个时间步
        relevant_features = time_seq_features[:, -self.pred_len:, :]  # [batch_size, pred_len, hidden_dim]

        # 平均池化得到整体时间特征
        temporal_features = torch.mean(relevant_features, dim=1)  # [batch_size, hidden_dim]

        return temporal_features, time_seq_features

    def encode_predictions(self, traditional_preds, llm_preds):
        """
        编码预测值

        参数:
        - traditional_preds: [batch_size, pred_len] 或 [batch_size, pred_len, 1]
        - llm_preds: [batch_size, pred_len] 或 [batch_size, pred_len, 1]

        返回:
        - pred_features: [batch_size, hidden_dim]
        - pred_seq_features: [pred_len, batch_size, hidden_dim]
        """
        # 处理多种可能的输入形状
        if traditional_preds.dim() == 3:  # [batch_size, pred_len, 1]
            traditional_preds = traditional_preds.squeeze(-1)  # [batch_size, pred_len]

        if llm_preds.dim() == 3:  # [batch_size, pred_len, 1]
            llm_preds = llm_preds.squeeze(-1)  # [batch_size, pred_len]

        batch_size = traditional_preds.shape[0]
        pred_len = traditional_preds.shape[1]

        # 拼接两种预测
        combined_preds = torch.cat([traditional_preds, llm_preds], dim=1)  # [batch_size, 2*pred_len]

        # 编码为隐藏特征
        pred_features = self.pred_encoder(combined_preds)  # [batch_size, hidden_dim]

        # 为了跨模态注意力，创建序列形式的预测特征
        # 这里我们简单地复制相同的特征pred_len次
        pred_seq = pred_features.unsqueeze(1).expand(-1, pred_len, -1)  # [batch_size, pred_len, hidden_dim]
        pred_seq_features = pred_seq.transpose(0, 1)  # [pred_len, batch_size, hidden_dim]

        return pred_features, pred_seq_features

    def forward(self, time_features, text_embeddings, traditional_preds, llm_preds):
        """
        前向传播

        参数:
        - time_features: 时间序列特征 [batch_size, seq_len, time_feature_dim]
        - text_embeddings: 已处理的文本嵌入，可以是多种形状
        - traditional_preds: 传统模型预测 [batch_size, pred_len] 或 [batch_size, pred_len, 1]
        - llm_preds: LLM预测 [batch_size, pred_len] 或 [batch_size, pred_len, 1]

        返回:
        - weights: 每个时间步的权重 [batch_size, pred_len] 或 [batch_size, pred_len, 1]
        """
        # 1. 处理文本嵌入
        text_features, text_seq_features = self.process_text_embeddings(text_embeddings)

        # 2. 编码预测值
        pred_features, pred_seq_features = self.encode_predictions(traditional_preds, llm_preds)

        # 3. 使用跨模态注意力编码时间特征
        temporal_features, _ = self.encode_time_features(
            time_features,
            text_seq_features=text_seq_features,
            pred_seq_features=pred_seq_features
        )

        # 4. 融合不同模态的特征
        combined_features = torch.cat([text_features, temporal_features, pred_features],
                                      dim=1)  # [batch_size, 3*hidden_dim]
        fused_features = self.fusion_layer(combined_features)  # [batch_size, hidden_dim]

        # 5. 预测权重
        weights = self.weight_predictor(fused_features)  # [batch_size, pred_len]

        # 调整输出形状以匹配输入
        if traditional_preds.dim() == 3:  # 如果输入是3D的，输出也应该是3D的
            weights = weights.unsqueeze(-1)  # [batch_size, pred_len, 1]

        return weights

    def predict_fusion(self, time_features, text_embeddings, traditional_preds, llm_preds):
        """
        预测并融合结果

        参数:
        - time_features: 时间序列特征 [batch_size, seq_len, time_feature_dim]
        - text_embeddings: 已处理的文本嵌入 [batch_size, text_dim] 或其他兼容形状
        - traditional_preds: 传统模型预测 [batch_size, pred_len] 或 [batch_size, pred_len, 1]
        - llm_preds: LLM预测 [batch_size, pred_len] 或 [batch_size, pred_len, 1]

        返回:
        - fused_preds: 融合后的预测结果 [batch_size, pred_len] 或 [batch_size, pred_len, 1]
        - weights: 融合权重 [batch_size, pred_len] 或 [batch_size, pred_len, 1]
        """
        weights = self.forward(time_features, text_embeddings, traditional_preds, llm_preds)

        # 权重调整传统模型预测的贡献度 (0表示完全信任LLM, 1表示完全信任传统模型)
        fused_preds = weights * traditional_preds + (1 - weights) * llm_preds

        return fused_preds, weights