import warnings
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

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



class CrossModalAttention(nn.Module):


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

        attn_output, _ = self.multihead_attn(query, key_value, key_value)

        query = query + self.dropout(attn_output)
        query = self.norm1(query)


        ffn_output = self.ffn(query)
        query = query + self.dropout(ffn_output)
        query = self.norm2(query)

        return query


class EnhancedMultiModalFusionClassifier(nn.Module):


    def __init__(self,

                 ):
        super(EnhancedMultiModalFusionClassifier, self).__init__()


        self.text_proj = nn.Linear(text_dim, hidden_dim)


        self.time_feature_proj = nn.Linear(time_feature_dim, hidden_dim)


        self.pred_encoder = nn.Linear(2 * pred_len, hidden_dim)

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)


        self.use_temporal_attention = use_temporal_attention
        if use_temporal_attention:
            encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                     dim_feedforward=hidden_dim * 4, dropout=dropout)
            self.temporal_transformer = TransformerEncoder(encoder_layers, nlayers)

        self.use_cross_modal_attention = use_cross_modal_attention
        if use_cross_modal_attention:
  
            self.text_to_time_attention = CrossModalAttention(hidden_dim, nhead, dropout)


            self.pred_to_time_attention = CrossModalAttention(hidden_dim, nhead, dropout)

  
            self.enhanced_time_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )


        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), 
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

 
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim, pred_len),
            nn.Sigmoid()  
        )

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

    def process_text_embeddings(self, text_embeddings):

        batch_size = text_embeddings.shape[0]


        if text_embeddings.dim() == 3:  
            if text_embeddings.shape[2] == 1: 
                text_embeddings = text_embeddings.squeeze(2) 

                text_features = self.text_proj(text_embeddings) 

                text_seq_features = text_features.unsqueeze(0)

            else:  # [batch_size, seq_len, dim]
                seq_len = text_embeddings.shape[1]


                text_seq = self.text_proj(text_embeddings)  # [batch_size, seq_len, hidden_dim] [seq_len, batch_size, hidden_dim]
                text_seq_features = text_seq.transpose(0, 1)

                text_features = torch.mean(text_seq, dim=1)  # [batch_size, hidden_dim]
        else:  # [batch_size, dim]

            text_features = self.text_proj(text_embeddings)  # [batch_size, hidden_dim]

            #  [seq_len=1, batch_size, hidden_dim]
            text_seq_features = text_features.unsqueeze(0)

        return text_features, text_seq_features

    def encode_time_features(self, time_features, text_seq_features=None, pred_seq_features=None):

        batch_size, seq_len, _ = time_features.shape


        projected_features = self.time_feature_proj(time_features)  # [batch_size, seq_len, hidden_dim]

        # Transformer [seq_len, batch_size, hidden_dim]
        time_seq = projected_features.transpose(0, 1)

        # 
        time_seq_with_pos = self.pos_encoder(time_seq)

        # 
        original_time_seq = time_seq_with_pos

        # 
        if self.use_temporal_attention:
            time_seq = self.temporal_transformer(time_seq_with_pos)  # [seq_len, batch_size, hidden_dim]
        else:
            time_seq = time_seq_with_pos

        # 
        if self.use_cross_modal_attention and text_seq_features is not None and pred_seq_features is not None:
            # 
            time_with_pred = self.pred_to_time_attention(time_seq, pred_seq_features)
            # 
            time_with_text = self.text_to_time_attention(time_with_pred, text_seq_features)




            orig_time = time_seq.transpose(0, 1)
            text_enhanced = time_with_text.transpose(0, 1)
            pred_enhanced = time_with_pred.transpose(0, 1)

            # 
            combined_time = torch.cat([orig_time, text_enhanced, pred_enhanced],
                                      dim=2)  # [batch_size, seq_len, 3*hidden_dim]

            # 
            enhanced_time = self.enhanced_time_fusion(combined_time)  # [batch_size, seq_len, hidden_dim]

   
            time_seq = enhanced_time.transpose(0, 1)

 
        time_seq_features = time_seq.transpose(0, 1)

        # 
        relevant_features = time_seq_features[:, -self.pred_len:, :]  # [batch_size, pred_len, hidden_dim]

        # 
        temporal_features = torch.mean(relevant_features, dim=1)  # [batch_size, hidden_dim]

        return temporal_features, time_seq_features

    def encode_predictions(self, traditional_preds, llm_preds):


        if traditional_preds.dim() == 3:  # [batch_size, pred_len, 1]
            traditional_preds = traditional_preds.squeeze(-1)  # [batch_size, pred_len]

        if llm_preds.dim() == 3:  # [batch_size, pred_len, 1]
            llm_preds = llm_preds.squeeze(-1)  # [batch_size, pred_len]

        batch_size = traditional_preds.shape[0]
        pred_len = traditional_preds.shape[1]

        combined_preds = torch.cat([traditional_preds, llm_preds], dim=1)  # [batch_size, 2*pred_len]

        pred_features = self.pred_encoder(combined_preds)  # [batch_size, hidden_dim]

        pred_seq = pred_features.unsqueeze(1).expand(-1, pred_len, -1)  # [batch_size, pred_len, hidden_dim]
        pred_seq_features = pred_seq.transpose(0, 1)  # [pred_len, batch_size, hidden_dim]

        return pred_features, pred_seq_features

    def forward(self, time_features, text_embeddings, traditional_preds, llm_preds):


        text_features, text_seq_features = self.process_text_embeddings(text_embeddings)


        pred_features, pred_seq_features = self.encode_predictions(traditional_preds, llm_preds)


        temporal_features, _ = self.encode_time_features(
            time_features,
            text_seq_features=text_seq_features,
            pred_seq_features=pred_seq_features
        )

        combined_features = torch.cat([text_features, temporal_features, pred_features],
                                      dim=1)  # [batch_size, 3*hidden_dim]
        fused_features = self.fusion_layer(combined_features)  # [batch_size, hidden_dim]

        weights = self.weight_predictor(fused_features)  # [batch_size, pred_len]

        if traditional_preds.dim() == 3: 
            weights = weights.unsqueeze(-1)  # [batch_size, pred_len, 1]

        return weights

    def predict_fusion(self, time_features, text_embeddings, traditional_preds, llm_preds):

        weights = self.forward(time_features, text_embeddings, traditional_preds, llm_preds)
        fused_preds = weights * traditional_preds + (1 - weights) * llm_preds

        return fused_preds, weights