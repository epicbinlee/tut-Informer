from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')


class ExpInformer(Exp_Basic):
    """
    继承 实验基础抽象类
    """

    def __init__(self, args):
        super(ExpInformer, self).__init__(args)

    def _build_model(self):
        # 通过模型字典声明创建模型
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }

        if self.args.model == 'informer' or self.args.model == 'informerstack':
            # 选择模型
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            # 读取参数加入到模型字典
            informer_model = model_dict[self.args.model]
            model = informer_model(
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        # 使用多GPU
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # 返回模型
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        # 数据封装成DataSet
        # Namespace(activation='gelu', attn='prob', batch_size=32, c_out=7, checkpoints='./checkpoints/',
        # cols=None, d_ff=2048, d_layers=1, d_model=512, data='ETTh1', data_path='ETTh1.csv', dec_in=7,
        # des='test', detail_freq='h', devices='0,1~,2,3', distil=True, do_predict=False, dropout=0.05, e_layers=2,
        # embed='timeF', enc_in=7, factor=5, features='M', freq='h', gpu=0, inverse=False, itr=2, label_len=48, learning_rate=0.0001,
        # loss='mse', lradj='type1', mix=True, model='informer', n_heads=8, num_workers=0, output_attention=False, padding=0,
        # patience=3, pred_len=24, root_path='./data/ETT/', s_layers=[3, 2, 1], seq_len=96, target='OT', train_epochs=6, use_amp=False,
        # use_gpu=False, use_multi_gpu=False)
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            # 定义seq长度、标签长度、预测长度
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        # 获取数据完成
        print(flag, len(data_set))

        # 数据封装成DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        # 作者没有使用自定义的优化器，直接使用了Adam优化器
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 回归问题的基本损失函数
        criterion = nn.MSELoss()
        return criterion

    # noinspection PyMethodOverriding
    def vali(self, vali_data, vali_loader, criterion):
        """
        评估：验证集和测试集在该方法中进行评估
        """
        # 声明如下代码是评估代码，BN和Dropout和训练不一致
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        # 评估完成设置回训练模式
        self.model.train()
        return total_loss

    # noinspection PyMethodOverriding
    def train(self, setting):
        # 获取数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        # 数据的路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # 训练计时器
        time_now = time.time()
        # 训练计步器
        train_steps = len(train_loader)
        # 早停策略
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # 优化器
        model_optim = self._select_optimizer()
        # 评估标准
        criterion = self._select_criterion()
        # 在cuda上使用自动混合精度计算
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # 训练，epoches 默认为6 @see main_informer.py line46
        for epoch in tqdm(range(self.args.train_epochs), desc='epoch progress'):
            iter_count = 0
            train_loss = []
            # model.train() 和 model.eval()的区别：
            # 如果模型中有BN(Batch Normalization)和Dropout，训练需要加入model.train()，测试需要加入model.eval()；
            # 其中针对BN，model.train()保证BN用每个批次数据的均值和方差，model.eval()保证BN用全部的数据的均值和方差；
            # 其中针对Dropout, model.train()随机选取一部分链接更新网络参数，model.eval()是利用所有网络连接。
            self.model.train()
            epoch_time = time.time()
            # =========================================================================================
            # 训练的模板：
            # 1. 优化器梯度置零 model_optimizer.zero_grad()
            # 2. 前向传播 outputs = net(inputs)
            # 3. 计算损失 loss = criterion(outputs, labels)
            # 4. 反向传播求梯度 loss.backward()
            # 5. 更新参数 model_optim.step()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                # 优化器梯度置0， loss对weight的倒数置0，训练不需要将两个Batch的梯度累乘
                model_optim.zero_grad()
                # 前向传播
                pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                # 计算损失
                loss = criterion(pred, true)
                # 记录各个步骤的损失， 方便后续计算平均损失
                train_loss.append(loss.item())
                # 每个Batch打印参数：batch的下标，epoch的下标，当前Batch的损失
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                # 是否使用混合精度，使用混合精度计算时，反向传播和参数更新不一致
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            # =========================================================================================
            # 每个epoch完成，打印训练时间、训练集/验证集/测试集三种平均损失
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # 进入验证阶段
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # 进入测试阶段
            test_loss = self.vali(test_data, test_loader, criterion)
            # 每个epoch打印一次
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # 每个epoch完成后评估是否进行早停
            early_stopping(vali_loss, self.model, path)
            # 需要早停则停止epoch循环
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # 每个epoch完成需要调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        # 保存模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        前向传播代码，在网络上做一次计算
        """
        # 输入数据
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        # 标定数据
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            # 输出前向传播的结果还是attention的结果
            # parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                # 32*24*7
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # 将输出翻转，后处理操作
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        # 将数据移到CPU上
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return outputs, batch_y
