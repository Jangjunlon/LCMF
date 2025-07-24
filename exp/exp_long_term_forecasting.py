from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer, LlamaForCausalLM
import datetime
from datetime import datetime, timedelta
import os
import time
import warnings
import numpy as np
import pdb
from newmodel.MultiModalFusionClassifier import MultiModalFusionClassifier
from newmodel.EnhancedMultiModalFusionClassifier import EnhancedMultiModalFusionClassifier
from newmodel.XiaoEnhancedMultiModalFusionClassifier import XiaoEnhancedMultiModalFusionClassifier

def norm(input_emb):
    input_emb = input_emb - input_emb.mean(1, keepdim=True).detach()
    input_emb = input_emb / torch.sqrt(
        torch.var(input_emb, dim=1, keepdim=True, unbiased=False) + 1e-5)

    return input_emb


class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


warnings.filterwarnings('ignore')


class Dual_Stage(Exp_Basic):
    def __init__(self, args):
        super(Dual_Stage, self).__init__(args)
        configs = args
        self.text_path = configs.text_path
        self.prompt_weight = configs.prompt_weight

        self.d_llm = configs.llm_dim
        self.pred_len = configs.pred_len
        self.text_embedding_dim = configs.text_emb
        self.pool_type = configs.pool_type
        self.use_fullmodel = configs.use_fullmodel

        mlp_sizes = [self.d_llm, int(self.d_llm / 8), self.text_embedding_dim]
        self.Doc2Vec = False
        self.mlp = MLP(mlp_sizes, dropout_rate=0.3)

        # pdb.set_trace()  # 在这里暂停，准备进行单步调试

        # 初始化融合模型
        self._init_fusion_model()

        if configs.llm_model == 'Doc2Vec':
            print('Now using Doc2Vec')
        else:
            if configs.llm_model == 'BERT':
                self.bert_config = BertConfig.from_pretrained(
                    'D:/Jang')

                self.bert_config.num_hidden_layers = configs.llm_layers
                self.bert_config.output_attentions = True
                self.bert_config.output_hidden_states = True
                try:
                    self.llm_model = BertModel.from_pretrained(
                        'D:/Jang',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.bert_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = BertModel.from_pretrained(
                        'D:/Jang',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.bert_config,
                    )

                try:
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'D:/Jang',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'D:/Jang',
                        trust_remote_code=True,
                        local_files_only=False
                    )

            else:
                raise Exception('LLM model is not defined')

            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.llm_model = self.llm_model.to(self.device)
        if args.init_method == 'uniform':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.uniform_(self.weight1.weight)
            nn.init.uniform_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True

        # self.tokenizer=self.tokenizer.to(self.device)
        self.mlp = self.mlp.to(self.device)
        # self.mlp_proj=self.mlp_proj.to(self.device)
        self.learning_rate2 = 1e-2
        self.learning_rate3 = 1e-3

    def _init_fusion_model(self):
        """初始化融合模型"""
        time_feature_dim = self._get_feature_dim()  # 时间特征维度

        # 文本嵌入维度应与您的文本处理结果匹配
        text_dim = self.text_embedding_dim  # 使用已定义的文本嵌入维度
        # pdb.set_trace()  # 在这里暂停，准备进行单步调试
        # self.fusion_model = MultiModalFusionClassifier(
        #     time_feature_dim=time_feature_dim,  # 时间特征维度
        #     hidden_dim=128,  # 隐藏层维度
        #     dropout=0.1,  # Dropout率
        #     text_dim=text_dim,  # 文本嵌入维度
        #     nhead=4,  # Transformer头数n
        #     nlayers=2,  # Transformer层数
        #     seq_len=self.args.seq_len,  # 时间序列长度
        #     pred_len=self.args.pred_len,  # 预测长度
        #     use_temporal_attention=True  # 使用时间注意力机制
        # ).to(self.device)
        # 确保所有参数需要梯度
        self.fusion_model = XiaoEnhancedMultiModalFusionClassifier(
            time_feature_dim=time_feature_dim,
            hidden_dim=512,
            text_dim=text_dim,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            nhead=4,
            nlayers=2,
            dropout=0.1,
            use_temporal_attention=True,  # 使用时序自注意力
            use_cross_modal_attention=True,  # 使用跨模态注意力
        ).to(self.device)
        # for param in self.fusion_model.parameters():
        #     param.requires_grad = True

        # 打印模型状态
        trainable_params = sum(p.numel() for p in self.fusion_model.parameters() if p.requires_grad)
        print(f"融合模型可训练参数: {trainable_params}")

    def _select_fusion_optimizer(self):
        """为融合模型选择优化器"""
        fusion_optim = torch.optim.Adam(
            self.fusion_model.parameters(),
            lr=self.args.fusion_lr if hasattr(self.args, 'fusion_lr') else 1e-4
        )
        return fusion_optim

    def _get_feature_dim(self):
        """获取特征维度"""
        # 根据您的数据结构确定时间序列特征的维度
        # 这里只是示例，请根据实际情况实现
        return self.args.enc_in

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_optimizer_mlp(self):
        model_optim = optim.Adam(self.mlp.parameters(), lr=self.args.learning_rate2)
        return model_optim

    def _select_optimizer_proj(self):
        model_optim = optim.Adam(self.mlp_proj.parameters(), lr=self.args.learning_rate3)
        return model_optim

    def _select_optimizer_weight(self):
        model_optim = optim.Adam([{'params': self.weight1.parameters()},
                                  {'params': self.weight2.parameters()}], lr=self.args.learning_rate_weight)
        return model_optim

    # 确保融合模型参数需要梯度
    def ensure_requires_grad(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def compute_mse(self, predictions, targets):
        """计算MSE"""
        return torch.mean((predictions - targets) ** 2).item()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # 初始化最佳联合性能
        best_joint_performance = float('inf')

        model_optim = self._select_optimizer()
        model_optim_mlp = self._select_optimizer_mlp()
        # model_optim_proj = self._select_optimizer_proj()
        criterion = self._select_criterion()

        # 融合模型的优化器
        fusion_optim = self._select_fusion_optimizer()
        fusion_criterion = nn.MSELoss()  # 用于权重预测的损失函数
        fusion_pred_criterion = nn.MSELoss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            # 融合模型的损失函数
            fusion_losses = []
            fusion_pred_losses = []  # 【新增】收集融合预测损失

            # 用于收集各种预测的MSE
            trad_mse_values = []
            llm_mse_values = []

            self.model.train()
            self.mlp.train()
            # self.mlp_proj.train()
            # 融合模型train
            self.fusion_model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                # print("batch_x",batch_x.shape)
                # print("batch_y", batch_y.shape)
                iter_count += 1
                # 清零梯度
                model_optim.zero_grad()
                if model_optim_mlp is not None:
                    model_optim_mlp.zero_grad()
                # if model_optim_proj is not None:
                #     model_optim_proj.zero_grad()
                fusion_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                prior_y = torch.from_numpy(train_data.get_prior_y(index)).float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # 加载文本特征
                batch_text = train_data.get_text(index)
                # pdb.set_trace()  # 在这里暂停，准备进行单步调试
                if self.Doc2Vec == False:
                    prompt = [
                        f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                        for text_info in batch_text]
                    prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                            max_length=1024).input_ids
                    prompt_embeddings = self.llm_model.get_input_embeddings()(
                        prompt.to(self.device))  # (batch, prompt_token, dim)
                else:
                    prompt = batch_text
                    prompt_embeddings = torch.tensor([self.text_model.infer_vector(text) for text in prompt]).to(
                        self.device)
                if self.use_fullmodel:
                    prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
                else:
                    prompt_emb = prompt_embeddings
                prompt_emb = self.mlp(prompt_emb)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # 1. 训练传统模型
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                if self.Doc2Vec == False:
                    if self.pool_type == "avg":
                        global_avg_pool = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_avg_pool.unsqueeze(-1)
                    elif self.pool_type == "max":
                        global_max_pool = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_max_pool.unsqueeze(-1)
                    elif self.pool_type == "min":
                        global_min_pool = F.adaptive_max_pool1d(-1.0 * prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_min_pool.unsqueeze(-1)
                    elif self.pool_type == "attention":
                        outputs_reshaped = outputs  # .transpose(1, 2)
                        outputs_norm = F.normalize(outputs_reshaped, p=2, dim=1)
                        prompt_emb_norm = F.normalize(prompt_emb, p=2, dim=2)
                        attention_scores = torch.bmm(prompt_emb_norm, outputs_norm)
                        attention_weights = F.softmax(attention_scores, dim=1)
                        weighted_prompt_emb = torch.sum(prompt_emb * attention_weights, dim=1)
                        prompt_emb = weighted_prompt_emb.unsqueeze(-1)
                else:
                    prompt_emb = prompt_emb.unsqueeze(-1)
                prompt_y = norm(prompt_emb) + prior_y

                outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                test = outputs

                # for name, param in self.model.named_parameters():
                #     print(f"{name} has grad: {has_grad(param)}")

                # 传统模型损失
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # pdb.set_trace()  # 在这里暂停，准备进行单步调试

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # 传统模型反向传播和参数更新
                    loss.backward()
                    model_optim.step()
                    model_optim_mlp.step()
                    # model_optim_proj.step()

                # 2. 训练融合模型
                if self.fusion_model is not None:
                    llm_preds_full = train_data.get_llm_predictions(index)

                    if llm_preds_full is not None:
                        # 根据当前预测长度截取相应的预测结果
                        pred_len = self.args.pred_len  # 获取当前预测长度

                        # # 假设llm_preds_full的第二维是时间步，根据pred_len截取相应长度
                        if pred_len == 6:
                            llm_preds = llm_preds_full[:, :6]  # 截取前12个时间步
                        elif pred_len == 8:
                            llm_preds = llm_preds_full[:, :8]  # 截取前24个时间步
                        elif pred_len == 10:
                            llm_preds = llm_preds_full[:, :10]  # 截取前36个时间步
                        elif pred_len == 12:
                            llm_preds = llm_preds_full  # 使用全部48个时间步
                        # 假设llm_preds_full的第二维是时间步，根据pred_len截取相应长度
                        # if pred_len == 12:
                        #     llm_preds = llm_preds_full[:, :12]  # 截取前12个时间步
                        # elif pred_len == 24:
                        #     llm_preds = llm_preds_full[:, :24]  # 截取前24个时间步
                        # elif pred_len == 36:
                        #     llm_preds = llm_preds_full[:, :36]  # 截取前36个时间步
                        # elif pred_len == 48:
                        #     llm_preds = llm_preds_full  # 使用全部48个时间步
                        else:
                            # 对于其他预测长度，可以添加额外的处理逻辑
                            llm_preds = llm_preds_full[:, :pred_len]  # 默认截取到pred_len

                        # 转换为张量并移到设备
                        llm_preds = torch.tensor(llm_preds, dtype=torch.float32).to(self.device)
                        llm_preds = llm_preds.unsqueeze(-1)

                        # 可以添加日志来验证截取是否正确
                        # print(f"Using LLM predictions for pred_len={pred_len}, shape={llm_preds.shape}")

                        # pdb.set_trace()  # 如果需要调试，可以保留

                        # llm_preds = torch.tensor(llm_preds, dtype=torch.float32).to(self.device)
                        #
                        # llm_preds= llm_preds.unsqueeze(-1)
                        # pdb.set_trace()  # 在这里暂停，准备进行单步调试

                    # 此时不需要更新传统模型的参数
                    with torch.no_grad():
                        outputs_detached = outputs.clone().detach()  # 完全分离输出
                        # 获取文本特征
                        batch_text_desc = prompt_emb.detach()  # 确保文本特征已分离
                        # 确保预测形状一致
                    if llm_preds.shape != outputs_detached.shape:
                        llm_preds = self._reshape_llm_predictions(llm_preds, outputs_detached.shape)
                        # 训练融合模型
                    fusion_optim.zero_grad()
                    # 使用融合模型预测权重
                    weights = self.fusion_model(
                        batch_x,  # 时间序列特征
                        batch_text_desc,  # 文本描述
                        outputs_detached,  # 传统模型预测，detach确保不影响传统模型
                        llm_preds  # LLM预测
                    )
                    # 计算LLM预测的MSE
                    llm_mse = self.compute_mse(llm_preds, batch_y)
                    llm_mse_values.append(llm_mse)
                    # 计算理想权重（哪个模型更准确）
                    trad_error = torch.abs(outputs_detached - batch_y)
                    llm_error = torch.abs(llm_preds - batch_y)
                    ideal_weights = llm_error / (trad_error + llm_error + 1e-6)  # 误差越大，权重越小

                    # # fusion
                    fused_prediction = weights * outputs_detached + (1 - weights) * llm_preds
                    # 计算融合预测的损失
                    fusion_pred_loss = fusion_pred_criterion(fused_prediction, batch_y)
                    fusion_pred_losses.append(fusion_pred_loss.item())  # 【新增】收集融合预测损失

                    # 可以结合两种损失进行优化
                    # total_fusion_loss = fusion_loss + fused_prediction_loss
                    # total_fusion_loss.backward()

                    # 检查 weights 是否需要梯度
                    # print("weights requires_grad:", weights.requires_grad)
                    # print("batch_text_desc requires_grad:", batch_text_desc.requires_grad)
                    # print("llm_preds requires_grad:", llm_preds.requires_grad)
                    # print("outputs_detached requires_grad:", outputs_detached.requires_grad)
                    # print("batch_y requires_grad:", batch_y.requires_grad)
                    #     # 检查 weights 是否需要梯度
                    # print("ideal_weights requires_grad:", ideal_weights.requires_grad)
                    # print("batch_x has grad:", batch_x.requires_grad)
                    # print("batch_text_desc has grad:", batch_text_desc.requires_grad)

                    # 计算融合模型损失
                    fusion_loss = fusion_criterion(weights, ideal_weights)
                    fusion_losses.append(fusion_loss.item())
                    # 融合模型反向传播和参数更新

                    # 修改这一行
                    fusion_loss.backward()

                    # 检查哪些张量已经有梯度
                    # def has_grad(tensor):
                    #     return tensor.grad is not None if tensor.requires_grad else False

                    # # 检查融合模型参数是否在传统模型训练中被意外更新
                    # for name, param in self.fusion_model.named_parameters():
                    #     print(f"{name} has grad: {has_grad(param)}")

                    fusion_optim.step()
                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            fusion_loss = np.average(fusion_losses)
            avg_llm_mse = np.mean(llm_mse_values)
            fusion_pred_loss_avg = np.average(fusion_pred_losses)  # 【新增】融合预测损失平均值
            # 验证
            vali_loss, vali_fusion_loss, fusion_pred_vali_loss_avg = self.vali(vali_data, vali_loader, criterion,
                                                                               fusion_criterion, fusion_pred_criterion)
            # print(fusion_pred_vali_loss_avg)
            # pdb.set_trace()  # 在这里暂停，准备进行单步调试
            # pdb.set_trace()  # 在这里暂停，准备进行单步调试
            test_loss, test_fusion_loss, fusion_pred_test_loss_avg = self.vali(test_data, test_loader, criterion,
                                                                               fusion_criterion, fusion_pred_criterion)

            print(
                "Epoch: {0}, Steps: {1} |LLM模型 MSE : {2:.7f} Train Loss: {3:.7f} Fusion Loss: {4:.7f} fusion_pred_loss_avg Loss: {5:.7f}| Vali Loss: {6:.7f} Fusion Loss: {7:.7f} fusion_pred_vali_loss_avg Loss: {8:.7f}| Test Loss: {9:.7f} Fusion Loss: {10:.7f} fusion_pred_test_loss_avg Loss: {11:.7f}".format(
                    epoch + 1, train_steps, avg_llm_mse, train_loss, fusion_loss, fusion_pred_loss_avg, vali_loss,
                    vali_fusion_loss, fusion_pred_vali_loss_avg, test_loss,
                    test_fusion_loss, fusion_pred_test_loss_avg))

            # 使用模型损失进行早停
            early_stopping(vali_loss, self.model, path)

            # 【新增】基于融合预测损失保存整体最佳模型
            if fusion_pred_vali_loss_avg < best_joint_performance:
                best_joint_performance = fusion_pred_vali_loss_avg
                print(f"Saving best joint models with joint performance: {fusion_pred_vali_loss_avg:.7f}")

                # 保存传统模型
                joint_model_path = path + '/' + 'joint_traditional_model.pth'
                torch.save(self.model.state_dict(), joint_model_path)

                # 保存融合模型
                joint_fusion_path = path + '/' + 'joint_fusion_model.pth'
                torch.save(self.fusion_model.state_dict(), joint_fusion_path)

                # 保存MLP模型
                joint_mlp_path = path + '/' + 'joint_mlp_model.pth'
                torch.save(self.mlp.state_dict(), joint_mlp_path)

                # 保存当前epoch信息
                with open(path + '/' + 'joint_best_epoch.txt', 'w') as f:
                    f.write(
                        f"Best joint model saved at epoch {epoch + 1} with performance {fusion_pred_vali_loss_avg:.7f}")

            # 保存融合模型
            fusion_model_path = path + '/' + 'fusion_checkpoint.pth'
            torch.save(self.fusion_model.state_dict(), fusion_model_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, self.fusion_model

    def vali(self, vali_data, vali_loader, criterion, fusion_criterion, fusion_pred_criterion):
        # pdb.set_trace()  # 在这里暂停，准备进行单步调试
        total_loss = []
        fusion_losses = []

        self.model.eval()
        self.mlp.eval()
        # self.mlp_proj.eval()

        # 用于收集各种预测的MSE
        trad_mse_values = []
        llm_mse_values = []
        fusion_mse_values = []
        fusion_pred_losses = []  # 【新增】收集融合预测损失

        # print(1)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                prior_y = torch.from_numpy(vali_data.get_prior_y(index)).float().to(self.device)

                batch_text = vali_data.get_text(index)

                if self.Doc2Vec == False:
                    prompt = [
                        f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                        for text_info in batch_text]

                    prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                            max_length=1024).input_ids
                    prompt_embeddings = self.llm_model.get_input_embeddings()(
                        prompt.to(self.device))  # (batch, prompt_token, dim)
                else:
                    prompt = batch_text
                    prompt_embeddings = torch.tensor([self.text_model.infer_vector(text) for text in prompt]).to(
                        self.device)
                if self.use_fullmodel:
                    prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
                else:
                    prompt_emb = prompt_embeddings
                prompt_emb = self.mlp(prompt_emb)  # (batch, prompt_token, text_embedding_dim)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                if self.Doc2Vec == False:
                    if self.pool_type == "avg":
                        global_avg_pool = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_avg_pool.unsqueeze(-1)
                    elif self.pool_type == "max":
                        global_max_pool = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_max_pool.unsqueeze(-1)
                    elif self.pool_type == "min":
                        global_min_pool = F.adaptive_max_pool1d(-1.0 * prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_min_pool.unsqueeze(-1)
                    elif self.pool_type == "attention":
                        outputs_reshaped = outputs
                        attention_scores = torch.bmm(prompt_emb, outputs_reshaped)
                        attention_weights = F.softmax(attention_scores, dim=1)
                        weighted_prompt_emb = torch.sum(prompt_emb * attention_weights, dim=1)
                        prompt_emb = weighted_prompt_emb.unsqueeze(-1)
                else:
                    prompt_emb = prompt_emb.unsqueeze(-1)

                prompt_y = norm(prompt_emb) + prior_y
                outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 融合模型验证
                if self.fusion_model is not None and fusion_criterion is not None:
                    # 获取LLM预测
                    llm_preds_full = vali_data.get_llm_predictions(index)

                    if llm_preds_full is not None:
                        # 根据当前预测长度截取相应的预测结果
                        pred_len = self.args.pred_len  # 获取当前预测长度

                        # # 假设llm_preds_full的第二维是时间步，根据pred_len截取相应长度
                        if pred_len == 6:
                            llm_preds = llm_preds_full[:, :6]  # 截取前12个时间步
                        elif pred_len == 8:
                            llm_preds = llm_preds_full[:, :8]  # 截取前24个时间步
                        elif pred_len == 10:
                            llm_preds = llm_preds_full[:, :10]  # 截取前36个时间步
                        elif pred_len == 12:
                            llm_preds = llm_preds_full  # 使用全部48个时间步
                        # if pred_len == 12:
                        #     llm_preds = llm_preds_full[:, :12]  # 截取前12个时间步
                        # elif pred_len == 24:
                        #     llm_preds = llm_preds_full[:, :24]  # 截取前24个时间步
                        # elif pred_len == 36:
                        #     llm_preds = llm_preds_full[:, :36]  # 截取前36个时间步
                        # elif pred_len == 48:
                        #     llm_preds = llm_preds_full  # 使用全部48个时间步
                        else:
                            # 对于其他预测长度，可以添加额外的处理逻辑
                            llm_preds = llm_preds_full[:, :pred_len]  # 默认截取到pred_len

                        # 转换为张量并移到设备
                        llm_preds = torch.tensor(llm_preds, dtype=torch.float32).to(self.device)
                        llm_preds = llm_preds.unsqueeze(-1)

                        # 可以添加日志来验证截取是否正确
                        # print(f"vali for pred_len={pred_len}, shape={llm_preds.shape}")

                        # 确保预测形状一致
                        if llm_preds.shape != outputs.shape:
                            llm_preds = self._reshape_llm_predictions(llm_preds, outputs.shape)

                        # # 计算LLM预测的MSE
                        # llm_mse = self.compute_mse(llm_preds, batch_y)
                        # llm_mse_values.append(llm_mse)

                        # 使用融合模型获取权重
                        batch_text_desc = prompt_emb.detach()
                        weights = self.fusion_model(
                            batch_x,
                            batch_text_desc,
                            outputs,
                            llm_preds
                        )

                        # 计算理想权重
                        trad_error = torch.abs(outputs - batch_y)
                        llm_error = torch.abs(llm_preds - batch_y)
                        ideal_weights = llm_error / (trad_error + llm_error + 1e-6)
                        # pdb.set_trace()  # 在这里暂停，准备进行单步调试

                        # 计算融合模型损失
                        fusion_loss = fusion_criterion(weights, ideal_weights)
                        fusion_losses.append(fusion_loss.item())

                        # 计算融合预测的MSE

                        fused_prediction = weights * outputs + (1 - weights) * llm_preds
                        # 计算融合预测的损失
                        fusion_pred_loss = fusion_pred_criterion(fused_prediction, batch_y)
                        fusion_pred_losses.append(fusion_pred_loss.item())  # 【新增】收集融合预测损失
                        # print("fusion_pred_loss",fusion_pred_loss)
                        # print("fusion_loss", fusion_loss)
                        # print("trad_error", trad_error)
                        # print("llm_error", llm_error)
                        # pdb.set_trace()  # 在这里暂停，准备进行单步调试
                        # fusion_mse = self.compute_mse(fusion_preds, batch_y)
                        # fusion_mse_values.append(fusion_mse)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        fusion_loss = np.average(fusion_losses) if fusion_losses else 0
        fusion_pred_loss_avg = np.average(fusion_pred_losses) if fusion_pred_losses else 0
        self.model.train()
        self.mlp.train()
        # self.mlp_proj.train()
        self.fusion_model.train()
        # print(fusion_pred_loss_avg)
        # pdb.set_trace()  # 在这里暂停，准备进行单步调试

        return total_loss, fusion_loss, fusion_pred_loss_avg

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        use_joint = True

        if test:
            print('loading model')

            # 检查是否使用联合优化的模型
            if use_joint:
                # 尝试加载联合优化的模型
                joint_traditional_path = os.path.join('./checkpoints/' + setting, 'joint_traditional_model.pth')
                joint_fusion_path = os.path.join('./checkpoints/' + setting, 'joint_fusion_model.pth')
                joint_mlp_path = os.path.join('./checkpoints/' + setting, 'joint_mlp_model.pth')

                # 检查联合模型文件是否存在
                if os.path.exists(joint_traditional_path) and os.path.exists(joint_fusion_path) and os.path.exists(
                        joint_mlp_path):
                    print('Loading joint optimized models...')

                    # 加载传统模型
                    self.model.load_state_dict(torch.load(joint_traditional_path))
                    print('Joint traditional model loaded')

                    # 加载融合模型
                    if self.fusion_model is not None:
                        self.fusion_model.load_state_dict(torch.load(joint_fusion_path))
                        print('Joint fusion model loaded')

                    # 加载MLP模型
                    if hasattr(self, 'mlp') and self.mlp is not None:
                        self.mlp.load_state_dict(torch.load(joint_mlp_path))
                        print('Joint MLP model loaded')

                    # 可选：显示最佳联合模型的信息
                    joint_info_path = os.path.join('./checkpoints/' + setting, 'joint_best_epoch.txt')
                    if os.path.exists(joint_info_path):
                        with open(joint_info_path, 'r') as f:
                            print(f.read())
                else:
                    print('Warning: Joint model checkpoints not found. Falling back to individual models.')
                    use_joint = False

            # 如果不使用联合模型或联合模型加载失败，使用原有加载方式
            if not use_joint:
                # 加载传统模型
                traditional_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
                self.model.load_state_dict(torch.load(traditional_path))
                print('Traditional model loaded')

                # 加载融合模型
                if self.fusion_model is not None:
                    fusion_model_path = os.path.join('./checkpoints/' + setting, 'fusion_checkpoint.pth')
                    if os.path.exists(fusion_model_path):
                        self.fusion_model.load_state_dict(torch.load(fusion_model_path))
                        print('Fusion model loaded')
                    else:
                        print('Warning: Fusion model checkpoint not found')

        # 保存三种预测结果
        preds = []
        trues = []
        llm_preds_list = []  # LLM预测
        fusion_preds_list = []  # 融合模型预测
        fusion_pred_loss_avg = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.mlp.eval()
        # self.mlp_proj.eval()
        if self.fusion_model is not None:
            self.fusion_model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # 0523
                prior_y = torch.from_numpy(test_data.get_prior_y(index)).float().to(self.device)
                # input_start_dates,input_end_dates=test_data.get_date(index)
                # 0523
                batch_text = test_data.get_text(index)

                prompt = [
                    f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                    for text_info in batch_text]

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if self.Doc2Vec == False:
                    prompt = [
                        f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                        for text_info in batch_text]

                    prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                            max_length=1024).input_ids
                    prompt_embeddings = self.llm_model.get_input_embeddings()(
                        prompt.to(self.device))  # (batch, prompt_token, dim)
                else:
                    prompt = batch_text
                    prompt_embeddings = torch.tensor([self.text_model.infer_vector(text) for text in prompt]).to(
                        self.device)
                if self.use_fullmodel:
                    prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
                else:
                    prompt_emb = prompt_embeddings
                prompt_emb = self.mlp(prompt_emb)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.Doc2Vec == False:
                    if self.pool_type == "avg":
                        global_avg_pool = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_avg_pool.unsqueeze(-1)
                    elif self.pool_type == "max":
                        global_max_pool = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_max_pool.unsqueeze(-1)
                    elif self.pool_type == "min":
                        global_min_pool = F.adaptive_max_pool1d(-1.0 * prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_min_pool.unsqueeze(-1)
                    elif self.pool_type == "attention":
                        outputs_reshaped = outputs  # .transpose(1, 2)
                        outputs_norm = F.normalize(outputs_reshaped, p=2, dim=1)
                        prompt_emb_norm = F.normalize(prompt_emb, p=2, dim=2)
                        attention_scores = torch.bmm(prompt_emb_norm, outputs_norm)
                        attention_weights = F.softmax(attention_scores, dim=1)
                        weighted_prompt_emb = torch.sum(prompt_emb * attention_weights, dim=1)
                        prompt_emb = weighted_prompt_emb.unsqueeze(-1)
                        # 0523
                else:
                    prompt_emb = prompt_emb.unsqueeze(-1)
                prompt_y = norm(prompt_emb) + prior_y
                outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                # 获取LLM预测
                llm_pred_array = None
                fusion_pred_array = None
                if self.fusion_model is not None:
                    # 获取LLM预测
                    llm_preds_data = test_data.get_llm_predictions(index)
                    if llm_preds_data is not None:
                        pred_len = self.args.pred_len  # 获取当前预测长度

                        # 假设llm_preds_full的第二维是时间步，根据pred_len截取相应长度
                        if pred_len == 6:
                            llm_preds_tensor = llm_preds_data[:, :6]  # 截取前12个时间步
                        elif pred_len == 8:
                            llm_preds_tensor = llm_preds_data[:, :8]  # 截取前24个时间步
                        elif pred_len == 10:
                            llm_preds_tensor = llm_preds_data[:, :10]  # 截取前36个时间步
                        elif pred_len == 12:
                            llm_preds_tensor = llm_preds_data  # 使用全部48个时间步
                        # if pred_len == 12:
                        #     llm_preds_tensor = llm_preds_data[:, :12]  # 截取前12个时间步
                        # elif pred_len == 24:
                        #     llm_preds_tensor = llm_preds_data[:, :24]  # 截取前24个时间步
                        # elif pred_len == 36:
                        #     llm_preds_tensor = llm_preds_data[:, :36]  # 截取前36个时间步
                        # elif pred_len == 48:
                        #     llm_preds_tensor = llm_preds_data  # 使用全部48个时间步
                        else:
                            # 对于其他预测长度，可以添加额外的处理逻辑
                            llm_preds_tensor = llm_preds_data[:, :pred_len]  # 默认截取到pred_len

                        # 转换为张量并移到设备
                        llm_preds_tensor = torch.tensor(llm_preds_tensor, dtype=torch.float32).to(self.device)
                        llm_preds_tensor = llm_preds_tensor.unsqueeze(-1)

                        # 可以添加日志来验证截取是否正确
                        # print(f"test for pred_len={pred_len}, shape={llm_preds_tensor.shape}")
                        # 调整维度以匹配outputs
                        if llm_preds_tensor.dim() == 2 and outputs.dim() == 3:
                            llm_preds_tensor = llm_preds_tensor.unsqueeze(-1)
                        # 确保形状一致
                        if llm_preds_tensor.shape != outputs.shape:
                            llm_preds_tensor = self._reshape_llm_predictions(llm_preds_tensor,
                                                                             outputs.shape)
                        # 使用融合模型预测权重
                        weights = self.fusion_model(
                            batch_x,
                            prompt_emb,
                            outputs,
                            llm_preds_tensor
                        )
                        # 融合预测

                        fusion_preds = weights * outputs + (1 - weights) * llm_preds_tensor
                        # 保存LLM预测和融合预测
                        llm_pred_array = llm_preds_tensor.detach().cpu().numpy()
                        fusion_pred_array = fusion_preds.detach().cpu().numpy()

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                # 如果需要反归一化
                if test_data.scale and self.args.inverse:
                    try:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                        if llm_pred_array is not None:
                            llm_shape = llm_pred_array.shape
                            llm_pred_array = test_data.inverse_transform(llm_pred_array.squeeze(0)).reshape(llm_shape)

                        if fusion_pred_array is not None:
                            fusion_shape = fusion_pred_array.shape
                            fusion_pred_array = test_data.inverse_transform(fusion_pred_array.squeeze(0)).reshape(
                                fusion_shape)
                    except Exception as e:
                        print(f"反归一化错误: {e}")

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if llm_pred_array is not None:
                    llm_preds_list.append(llm_pred_array)

                if fusion_pred_array is not None:
                    fusion_preds_list.append(fusion_pred_array)

                if i % 20 == 0:
                    input_array = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        try:
                            shape = input_array.shape
                            input_array = test_data.inverse_transform(input_array.squeeze(0)).reshape(shape)
                        except Exception as e:
                            print(f"输入反归一化错误: {e}")

                    # 绘制传统模型预测
                    gt = np.concatenate((input_array[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input_array[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, f'traditional_{i}.pdf'))

                    # 如果有融合模型预测，也绘制
                    if fusion_pred_array is not None:
                        pd_fusion = np.concatenate((input_array[0, :, -1], fusion_pred_array[0, :, -1]), axis=0)
                        visual(gt, pd_fusion, os.path.join(folder_path, f'fusion_{i}.pdf'))

                    # 如果有LLM预测，也绘制
                    if llm_pred_array is not None:
                        pd_llm = np.concatenate((input_array[0, :, -1], llm_pred_array[0, :, -1]), axis=0)
                        visual(gt, pd_llm, os.path.join(folder_path, f'llm_{i}.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('Traditional model test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 处理LLM和融合模型预测
        if llm_preds_list:
            llm_preds_array = np.array(llm_preds_list)
            llm_preds_array = llm_preds_array.reshape(-1, llm_preds_array.shape[-2], llm_preds_array.shape[-1])
            print('LLM model test shape:', llm_preds_array.shape)
        else:
            llm_preds_array = None

        if fusion_preds_list:
            fusion_preds_array = np.array(fusion_preds_list)
            fusion_preds_array = fusion_preds_array.reshape(-1, fusion_preds_array.shape[-2],
                                                            fusion_preds_array.shape[-1])
            print('Fusion model test shape:', fusion_preds_array.shape)
        else:
            fusion_preds_array = None

        # 创建结果保存文件夹
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation

        dtw = -999

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open(self.args.save_name, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        # 计算评估指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('Traditional Model:')
        print('MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, MSPE: {:.4f}%'.format(mse, mae, rmse, mape, mspe))

        # 如果有LLM预测，计算其指标
        if llm_preds_array is not None:
            llm_mae, llm_mse, llm_rmse, llm_mape, llm_mspe = metric(llm_preds_array, trues)
            print('LLM Model:')
            print('MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, MSPE: {:.4f}%'.format(
                llm_mse, llm_mae, llm_rmse, llm_mape, llm_mspe))

        # 如果有融合预测，计算其指标
        if fusion_preds_array is not None:
            fusion_mae, fusion_mse, fusion_rmse, fusion_mape, fusion_mspe = metric(fusion_preds_array, trues)
            print('Fusion Model:')
            print('MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, MSPE: {:.4f}%'.format(
                fusion_mse, fusion_mae, fusion_rmse, fusion_mape, fusion_mspe))

            # 保存结果到文件
        f = open(self.args.save_name, 'a')
        f.write(setting + "  \n")
        f.write('Traditional Model:\n')
        f.write(
            'MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, MSPE: {:.4f}%'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')

        if llm_preds_array is not None:
            f.write('LLM Model:\n')
            f.write('MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, MSPE: {:.4f}%'.format(
                llm_mse, llm_mae, llm_rmse, llm_mape, llm_mspe))
            f.write('\n')

        if fusion_preds_array is not None:
            f.write('Fusion Model:\n')
            f.write('MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, MSPE: {:.4f}%'.format(
                fusion_mse, fusion_mae, fusion_rmse, fusion_mape, fusion_mspe))
            f.write('\n')

        f.write('\n')
        f.close()

        # 保存指标和预测结果
        np.save(folder_path + 'traditional_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'traditional_pred.npy', preds)

        if llm_preds_array is not None:
            np.save(folder_path + 'llm_metrics.npy', np.array([llm_mae, llm_mse, llm_rmse, llm_mape, llm_mspe]))
            np.save(folder_path + 'llm_pred.npy', llm_preds_array)

        if fusion_preds_array is not None:
            np.save(folder_path + 'fusion_metrics.npy',
                    np.array([fusion_mae, fusion_mse, fusion_rmse, fusion_mape, fusion_mspe]))
            np.save(folder_path + 'fusion_pred.npy', fusion_preds_array)

        np.save(folder_path + 'true.npy', trues)

        return mse
