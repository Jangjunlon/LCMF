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
from exp.MultiModal import EnhancedMultiModalFusionClassifier

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

        self._init_fusion_model()

        if configs.llm_model == 'Doc2Vec':
            print('Now using Doc2Vec')
        else:
            if configs.llm_model == 'BERT':
                self.bert_config = BertConfig.from_pretrained(
                    '')

                self.bert_config.num_hidden_layers = configs.llm_layers
                self.bert_config.output_attentions = True
                self.bert_config.output_hidden_states = True
                try:
                    self.llm_model = BertModel.from_pretrained(
                        '',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.bert_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = BertModel.from_pretrained(
                        '',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.bert_config,
                    )

                try:
                    self.tokenizer = BertTokenizer.from_pretrained(
                        '',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = BertTokenizer.from_pretrained(
                        '',
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


        self.learning_rate2 = 1e-2
        self.learning_rate3 = 1e-3

    def _init_fusion_model(self):
        time_feature_dim = self._get_feature_dim() 
  
        text_dim = self.text_embedding_dim  

        self.fusion_model = EnhancedMultiModalFusionClassifier(

        ).to(self.device)
        trainable_params = sum(p.numel() for p in self.fusion_model.parameters() if p.requires_grad)


    def _select_fusion_optimizer(self):

        fusion_optim = torch.optim.Adam(
            self.fusion_model.parameters(),
            lr=self.args.fusion_lr if hasattr(self.args, 'fusion_lr') else 1e-4
        )
        return fusion_optim

    def _get_feature_dim(self):

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


    def ensure_requires_grad(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def compute_mse(self, predictions, targets):
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

        best_joint_performance = float('inf')

        model_optim = self._select_optimizer()
        model_optim_mlp = self._select_optimizer_mlp()
        # model_optim_proj = self._select_optimizer_proj()
        criterion = self._select_criterion()


        fusion_optim = self._select_fusion_optimizer()
        fusion_criterion = nn.MSELoss()  
        fusion_pred_criterion = nn.MSELoss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            fusion_losses = []
            fusion_pred_losses = []  

            trad_mse_values = []
            llm_mse_values = []

            self.model.train()
            self.mlp.train()

            self.fusion_model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                # print("batch_x",batch_x.shape)
                # print("batch_y", batch_y.shape)
                iter_count += 1

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
    
                batch_text = train_data.get_text(index)
                if self.Doc2Vec == False:
                    prompt = [
                        f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                        for text_info in batch_text]
                    prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                            max_length=1024).input_ids
                    prompt_embeddings = self.llm_model.get_input_embeddings()(
                        prompt.to(self.device))  
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
                else:
                    prompt_emb = prompt_emb.unsqueeze(-1)
                prompt_y = norm(prompt_emb) + prior_y

                outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                test = outputs
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())


                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    model_optim_mlp.step()
                    # model_optim_proj.step()
                if self.fusion_model is not None:
                    llm_preds_full = train_data.get_llm_predictions(index)

                    if llm_preds_full is not None:

                        pred_len = self.args.pred_len
                  
                        llm_preds = torch.tensor(llm_preds, dtype=torch.float32).to(self.device)
                        llm_preds = llm_preds.unsqueeze(-1)


                    with torch.no_grad():
                        outputs_detached = outputs.clone().detach() 
                  
                        batch_text_desc = prompt_emb.detach()
                
                    if llm_preds.shape != outputs_detached.shape:
                        llm_preds = self._reshape_llm_predictions(llm_preds, outputs_detached.shape)
                       
                    fusion_optim.zero_grad()

             
                    llm_mse = self.compute_mse(llm_preds, batch_y)
                    llm_mse_values.append(llm_mse)
            
                    trad_error = torch.abs(outputs_detached - batch_y)
                    llm_error = torch.abs(llm_preds - batch_y)
                    ideal_weights = llm_error / (trad_error + llm_error + 1e-6)  

                 
                    fused_prediction = weights * outputs_detached + (1 - weights) * llm_preds
                
                    fusion_pred_loss = fusion_pred_criterion(fused_prediction, batch_y)
                    fusion_pred_losses.append(fusion_pred_loss.item()) 

                    
                    fusion_loss = fusion_criterion(weights, ideal_weights)
                    fusion_losses.append(fusion_loss.item())
                    fusion_loss.backward()
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
      
            fusion_pred_loss_avg = np.average(fusion_pred_losses)  

            vali_loss, vali_fusion_loss, fusion_pred_vali_loss_avg = self.vali(vali_data, vali_loader, criterion,
                                                                               fusion_criterion, fusion_pred_criterion)
            test_loss, test_fusion_loss, fusion_pred_test_loss_avg = self.vali(test_data, test_loader, criterion,
                                                                               fusion_criterion, fusion_pred_criterion)

            print(
                "Epoch: {0}, Steps: {1} |Train Loss: {3:.7f} Fusion Loss: {4:.7f} fusion_pred_loss_avg Loss: {5:.7f}| Vali Loss: {6:.7f} Fusion Loss: {7:.7f} fusion_pred_vali_loss_avg Loss: {8:.7f}| Test Loss: {9:.7f} Fusion Loss: {10:.7f} fusion_pred_test_loss_avg Loss: {11:.7f}".format(
                    epoch + 1, train_steps, train_loss, fusion_loss, fusion_pred_loss_avg, vali_loss,
                    vali_fusion_loss, fusion_pred_vali_loss_avg, test_loss,
                    test_fusion_loss, fusion_pred_test_loss_avg))

            early_stopping(vali_loss, self.model, path)

            if fusion_pred_vali_loss_avg < best_joint_performance:
                best_joint_performance = fusion_pred_vali_loss_avg
                print(f"Saving best joint models with joint performance: {fusion_pred_vali_loss_avg:.7f}")
                joint_model_path = path + '/' + 'joint_traditional_model.pth'
                torch.save(self.model.state_dict(), joint_model_path)
                joint_fusion_path = path + '/' + 'joint_fusion_model.pth'
                torch.save(self.fusion_model.state_dict(), joint_fusion_path)

                joint_mlp_path = path + '/' + 'joint_mlp_model.pth'
                torch.save(self.mlp.state_dict(), joint_mlp_path)
                with open(path + '/' + 'joint_best_epoch.txt', 'w') as f:
                    f.write(
                        f"Best joint model saved at epoch {epoch + 1} with performance {fusion_pred_vali_loss_avg:.7f}")
            fusion_model_path = path + '/' + 'fusion_checkpoint.pth'
            torch.save(self.fusion_model.state_dict(), fusion_model_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, self.fusion_model

    def vali(self, vali_data, vali_loader, criterion, fusion_criterion, fusion_pred_criterion):

        total_loss = []
        fusion_losses = []

        self.model.eval()
        self.mlp.eval()

        fusion_pred_losses = [] 


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


                if self.fusion_model is not None and fusion_criterion is not None:
  
                    llm_preds_full = vali_data.get_llm_predictions(index)

                    if llm_preds_full is not None:

                        llm_preds = torch.tensor(llm_preds, dtype=torch.float32).to(self.device)
                        llm_preds = llm_preds.unsqueeze(-1)
                        if llm_preds.shape != outputs.shape:
                            llm_preds = self._reshape_llm_predictions(llm_preds, outputs.shape)

                        batch_text_desc = prompt_emb.detach()
                        trad_error = torch.abs(outputs - batch_y)
                        llm_error = torch.abs(llm_preds - batch_y)
                        ideal_weights = llm_error / (trad_error + llm_error + 1e-6)
 
                        fusion_loss = fusion_criterion(weights, ideal_weights)
                        fusion_losses.append(fusion_loss.item())
                        fused_prediction = weights * outputs + (1 - weights) * llm_preds

                        fusion_pred_loss = fusion_pred_criterion(fused_prediction, batch_y)
                        fusion_pred_losses.append(fusion_pred_loss.item()) 

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        fusion_loss = np.average(fusion_losses) if fusion_losses else 0
        fusion_pred_loss_avg = np.average(fusion_pred_losses) if fusion_pred_losses else 0
        self.model.train()
        self.mlp.train()

        self.fusion_model.train()
        return total_loss, fusion_loss, fusion_pred_loss_avg

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        use_joint = True

        if test:
            print('loading model')


            if use_joint:

                joint_traditional_path = os.path.join('./checkpoints/' + setting, 'joint_traditional_model.pth')
                joint_fusion_path = os.path.join('./checkpoints/' + setting, 'joint_fusion_model.pth')
                joint_mlp_path = os.path.join('./checkpoints/' + setting, 'joint_mlp_model.pth')


                if os.path.exists(joint_traditional_path) and os.path.exists(joint_fusion_path) and os.path.exists(
                        joint_mlp_path):
                    print('Loading joint optimized models...')
   
                    self.model.load_state_dict(torch.load(joint_traditional_path))
                    print('Joint traditional model loaded')

                    if self.fusion_model is not None:
                        self.fusion_model.load_state_dict(torch.load(joint_fusion_path))
                        print('Joint fusion model loaded')

                    if hasattr(self, 'mlp') and self.mlp is not None:
                        self.mlp.load_state_dict(torch.load(joint_mlp_path))
                        print('Joint MLP model loaded')


                    joint_info_path = os.path.join('./checkpoints/' + setting, 'joint_best_epoch.txt')
                    if os.path.exists(joint_info_path):
                        with open(joint_info_path, 'r') as f:
                            print(f.read())
                else:
                    print('Warning: Joint model checkpoints not found. Falling back to individual models.')
                    use_joint = False

            if not use_joint:

                traditional_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
                self.model.load_state_dict(torch.load(traditional_path))
                print('Traditional model loaded')


                if self.fusion_model is not None:
                    fusion_model_path = os.path.join('./checkpoints/' + setting, 'fusion_checkpoint.pth')
                    if os.path.exists(fusion_model_path):
                        self.fusion_model.load_state_dict(torch.load(fusion_model_path))
                        print('Fusion model loaded')
                    else:
                        print('Warning: Fusion model checkpoint not found')


        preds = []
        trues = []

        fusion_preds_list = []  
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


                llm_pred_array = None
                fusion_pred_array = None
                if self.fusion_model is not None:

                    llm_preds_data = test_data.get_llm_predictions(index)
                    if llm_preds_data is not None:
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
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
                        print(f": {e}")

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
                            print(f": {e}")
                    gt = np.concatenate((input_array[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input_array[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, f'traditional_{i}.pdf'))

     
                    if fusion_pred_array is not None:
                        pd_fusion = np.concatenate((input_array[0, :, -1], fusion_pred_array[0, :, -1]), axis=0)
                        visual(gt, pd_fusion, os.path.join(folder_path, f'fusion_{i}.pdf'))

                    if llm_pred_array is not None:
                        pd_llm = np.concatenate((input_array[0, :, -1], llm_pred_array[0, :, -1]), axis=0)
                        visual(gt, pd_llm, os.path.join(folder_path, f'llm_{i}.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('Traditional model test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        if fusion_preds_list:
            fusion_preds_array = np.array(fusion_preds_list)
            fusion_preds_array = fusion_preds_array.reshape(-1, fusion_preds_array.shape[-2],
                                                            fusion_preds_array.shape[-1])
            print('Fusion model test shape:', fusion_preds_array.shape)
        else:
            fusion_preds_array = None


        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open(self.args.save_name, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('Traditional Model:')
        print('MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, MSPE: {:.4f}%'.format(mse, mae, rmse, mape, mspe))
        if fusion_preds_array is not None:
            fusion_mae, fusion_mse, fusion_rmse, fusion_mape, fusion_mspe = metric(fusion_preds_array, trues)
            print('Fusion Model:')
            print('MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, MSPE: {:.4f}%'.format(
                fusion_mse, fusion_mae, fusion_rmse, fusion_mape, fusion_mspe))

        f = open(self.args.save_name, 'a')
        f.write(setting + "  \n")
        f.write('Traditional Model:\n')
        f.write(
            'MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, MSPE: {:.4f}%'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        if fusion_preds_array is not None:
            f.write('Fusion Model:\n')
            f.write('MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, MSPE: {:.4f}%'.format(
                fusion_mse, fusion_mae, fusion_rmse, fusion_mape, fusion_mspe))
            f.write('\n')

        f.write('\n')
        f.close()
        np.save(folder_path + 'traditional_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'traditional_pred.npy', preds)
        if fusion_preds_array is not None:
            np.save(folder_path + 'fusion_metrics.npy',
                    np.array([fusion_mae, fusion_mse, fusion_rmse, fusion_mape, fusion_mspe]))
            np.save(folder_path + 'fusion_pred.npy', fusion_preds_array)

        np.save(folder_path + 'true.npy', trues)
        return mse
