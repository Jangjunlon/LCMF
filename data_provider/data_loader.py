
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer

import warnings
from utils.augmentation import run_augmentation_single
import pdb
warnings.filterwarnings('ignore')

from torch.utils.data import Sampler


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), encoding='latin-1')
        # df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path)), encoding='gbk'

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # 获取基本列
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')

        # 确定文本预测列
        if self.args.use_closedllm == 0:
            text_name = "text_data"
        else:
            print("!!!!!!!!!!!!Using output of closed source llm and Bert as encoder!!!!!!!!!!!!!!!")
            text_name = "text_data"

        # 获取LLM预测列
        llm_pred_cols = [col for col in cols if col.startswith('llm_pred_week_')]

        # 检查是否有LLM预测列
        # if llm_pred_cols:
        #     print(f"找到 {len(llm_pred_cols)} 个LLM预测列: {llm_pred_cols}")
        # else:
        #     print("警告: 未找到LLM预测列!")

        # 选择需要的列
        selected_cols = ['date'] + cols + [self.target] + ['prior_history_avg'] + ['start_date'] + ['end_date'] + [
            text_name]
        # 添加LLM预测列
        # selected_cols.extend(llm_pred_cols)


        # pdb.set_trace()  # 在这里暂停，准备进行单步调试


        # 确保所有列都存在于数据集中
        available_cols = [col for col in selected_cols if col in df_raw.columns]
        df_raw = df_raw[available_cols]

        # 分割数据集
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 准备主要特征数据
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_data_prior = df_raw[['prior_history_avg']]

        # 数据缩放
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            data_prior = self.scaler.transform(df_data_prior.values[:, -1].reshape(-1, 1))
        else:
            data = df_data.values
            data_prior = df_data_prior.values

        # 准备时间戳数据
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 保存数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_prior = data_prior[border1:border2]
        self.data_stamp = data_stamp

        # 保存日期信息
        self.date = df_raw[['date']][border1:border2].values
        self.start_date = df_raw[['start_date']][border1:border2].values
        self.end_date = df_raw[['end_date']][border1:border2].values
        self.text = df_raw[[text_name]][border1:border2].values

        # 提取LLM预测结果
        if llm_pred_cols:
            llm_preds = df_raw[llm_pred_cols][border1:border2].values
            self.llm_predictions = llm_preds





            print(f"LLM预测数组形状: {self.llm_predictions.shape}")
        else:
            self.llm_predictions = None

    def get_prior_y(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends
        r_ends = r_begins + self.pred_len
        prior_y = np.array([self.data_prior[r_beg:r_end] for r_beg, r_end in zip(r_begins, r_ends)])
        return prior_y

    def get_text(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len

        text = np.array([self.text[s_end] for s_end in s_ends])
        return text

    def get_date(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends - self.label_len
        r_ends = r_begins + self.label_len + self.pred_len

        x_start_dates = np.array([self.start_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        x_end_dates = np.array([self.end_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])

        return x_start_dates, x_end_dates

    # def get_llm_predictions(self, indices):
    #     """
    #     获取LLM预测结果 - 修改版
    #
    #     参数:
    #     - indices: 数据索引
    #
    #     返回:
    #     - llm_preds: LLM预测数组, 形状为 [batch_size, pred_len]
    #     """
    #     if not hasattr(self, 'llm_predictions') or self.llm_predictions is None:
    #         return None
    #
    #     if isinstance(indices, torch.Tensor):
    #         indices = indices.numpy()
    #
    #     # 计算序列结束索引
    #     s_begins = indices % self.tot_len
    #     s_ends = s_begins + self.seq_len
    #
    #     # 对于每个样本，提取序列末尾时间点的LLM预测
    #     # 注意：我们需要的是每个序列的最后一个时间点的预测
    #     llm_preds = np.array([self.llm_predictions[s_end - 1] for s_end in s_ends])
    #
    #     return llm_preds

    def get_llm_predictions(self, indices):
        """获取并归一化LLM预测结果"""
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len

        # 提取LLM预测
        llm_preds = np.array([self.llm_predictions[s_end - 1] for s_end in s_ends])

        # 对预测值应用相同的归一化
        if self.scale:
            # 将预测数组重塑为合适的形状以便归一化
            original_shape = llm_preds.shape

            # pdb.set_trace()  # 在这里暂停，准备进行单步调试

            llm_preds_flat = llm_preds.reshape(-1, 1)

            # 应用相同的scaler
            llm_preds_norm = self.scaler.transform(llm_preds_flat)

            # 恢复原始形状
            llm_preds = llm_preds_norm.reshape(original_shape)

        return llm_preds

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)