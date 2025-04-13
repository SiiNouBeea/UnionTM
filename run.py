# -*- coding: utf-8 -*-
import argparse
import os
import torch
import numpy as np
import random
import matplotlib
from statsmodels.graphics.tukeyplot import results
from sympy import false

# from statsmodels.sandbox.panel.sandwich_covariance_generic import kernel

from PatchTST.PatchTST_supervised.exp.exp_main import Exp_Main as Exp_Main_2

from Pathformer.exp.exp_main import Exp_Main as Exp_Main_1

from SimpleTM.experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast as Exp_Long_Term_Forecast_Simple

from TimeMixer.exp.exp_anomaly_detection import Exp_Anomaly_Detection
from TimeMixer.exp.exp_classification import Exp_Classification
from TimeMixer.exp.exp_imputation import Exp_Imputation
from TimeMixer.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from TimeMixer.exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
import time
from PatchTST.PatchTST_supervised.utils.metrics import metric
from sklearn.linear_model import LinearRegression
from tools.optimize_weights import optimize_weights
from tools.weightThresholdScreening import weightThresholdScreening
import matplotlib.pyplot as plt


matplotlib.use('Agg')  # 使用非交互式后端

# 合并三个模型的参数
def parse_args():
    parser = argparse.ArgumentParser(description='Multivariate Time Series Forecasting')

    # Basic Config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='PathFormer',
                        help='model name, options: [PathFormer, Autoformer, NewModel, SimpleTM, PatchTST]')
    parser.add_argument('--model_id', type=str, default="ETT.sh")
    parser.add_argument('--random_seed', type=int, default=1024, help='random seed')
    parser.add_argument('--pre_train', type=bool, default=False, help='Enable pre train')
    parser.add_argument('--mix_weight', type=float, nargs='+', default=[0.25,0.25,0.25,0.25],
                        help='the weight of models')  # 各模型的权重
    parser.add_argument('--sample_rate', type=float, default=0.5, help='sample rate')
    parser.add_argument('--dynamic_delta', type=float, default=None, help='dynamic delta')
    parser.add_argument('--plot', type=bool, default=True, help='whether to plot')
    parser.add_argument('--super_ensemble', type=bool, default=True, help='Use the latest update feature: Ultimate Learner')

    # Data Loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/weather', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Forecasting Task Parameters
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')

    # Model Parameters
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--num_nodes', type=int, default=21)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at every layer ')
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2])
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--metric', type=str, default='mae')
    parser.add_argument('--batch_norm', type=int, default=0)

    # Additional Parameters for Transformer-Family Models
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    parser.add_argument('--embed_type', type=int, default=0)
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--des', type=str, default='test', help='exp description')

    # Optimization Parameters
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU Parameters
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # New Parameters
    parser.add_argument('--new_model_param1', type=float, default=0.1)
    parser.add_argument('--new_model_param2', type=int, default=32)
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--comment', type=str, default='none', help='com')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                        help='whether to use future_temporal_feature; True 1 False 0')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # SimpleTM
    parser.add_argument('--requires_grad', type=bool, default=True, help='Set to True to enable learnable wavelets')
    parser.add_argument('--wv', type=str, default='db1',
                        help='Wavelet filter type. Supports all wavelets available in PyTorch Wavelets')
    parser.add_argument('--m', type=int, default=3, help='Number of levels for the stationary wavelet transform')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Weight of the inner product score in geometric attention')
    parser.add_argument('--l1_weight', type=float, default=5e-4, help='Weight of L1 loss')
    parser.add_argument('--fix_seed', type=int, default=2025, help='gpu')
    parser.add_argument('--geomattn_dropout', type=float, default=0.5,
                        help='dropout rate of the projection layer in the geometric attention')
    parser.add_argument('--compile', type=bool, default=False,
                        help='Set to True to enable compilation, which can accelerate speed but may slightly impact performance')

    return parser.parse_args()


def main():
    args = parse_args()
    time_Total = time.time()  # 总计时

    # Set random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    kernel_size = args.kernel_size  # 窗口长度

    if args.super_ensemble:
        args.pre_train = False

    # GPU settings
    if not torch.cuda.is_available():
        args.use_gpu = False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if 'patch_size_list' in args:
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    #WTS-Mechanism
    run_flag, single_model, single_model_name, weight = weightThresholdScreening(args.mix_weight)

    #Start train
    if run_flag[0] == 1:
        # Model 1: SimpleTM
        model_a = {
            'model': 'SimpleTM',
            'kernel_size': None
        }

        # Update args with model_a parameters
        for key, value in model_a.items():
            setattr(args, key, value)

        # Run Model 1
        print('\n\n>>>>>>>Running Model: SimpleTM<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        print('Args in experiment:')
        print(args)
        time_start = time.time()  # 开始计时
        Exp = Exp_Long_Term_Forecast_Simple

        if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                    args.model_id,
                    args.data,
                    args.seq_len,
                    args.pred_len,
                    args.d_model,
                    args.d_ff,
                    args.e_layers,
                    args.wv,
                    args.kernel_size,
                    args.m,
                    args.alpha,
                    args.l1_weight,
                    args.learning_rate,
                    args.lradj,
                    args.batch_size,
                    args.fix_seed,
                    args.use_norm,
                    ii)

                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                if single_model:
                    preds_A, trues, result_A = exp.test(setting)
                    print('>>>>>>>Retrain Model: SimpleTM<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    exp.retrain(setting)
                    preds_B, trues, result_B = exp.test(setting)
                    print('>>>>>>>Retrain Model: SimpleTM<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    exp.retrain(setting)
                    preds_C, trues, result_C = exp.test(setting)
                else:
                    preds_A, trues, result_A = exp.test(setting)
                print('Preds has been recorded.')

                print('Trues has been recorded.')

                cost_time_A = time.time() - time_start  # 计时结束
                print(f'>>>>>>>Cost Time : {cost_time_A}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

                torch.cuda.empty_cache()
                del exp
        else:

            ii = 0
            setting = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.data,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.e_layers,
                args.wv,
                args.kernel_size,
                args.m,
                args.alpha,
                args.l1_weight,
                args.learning_rate,
                args.lradj,
                args.batch_size,
                args.fix_seed,
                args.use_norm,
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if single_model:
                preds_A, trues, result_A = exp.test(setting)
                print('>>>>>>>Retrain Model: SimpleTM<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                exp.retrain(setting)
                preds_B, trues, result_B = exp.test(setting)
                print('>>>>>>>Retrain Model: SimpleTM<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                exp.retrain(setting)
                preds_C, trues, result_C = exp.test(setting)
            else:
                preds_A, trues, result_A = exp.test(setting)
            print('Preds has been recorded.')

            print('Trues has been recorded.')

            cost_time_A = time.time() - time_start  # 计时结束
            print(f'>>>>>>>Cost Time : {cost_time_A}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

            torch.cuda.empty_cache()
            del exp
    else:
        preds_A = 0
        results_A = None
        cost_time_A = 0
        print('\n\n>>>>>>>Model SimpleTM is not selected<<<<<<<<<<<<<')



    if run_flag[1] == 1:
        # Model 2: PatchTST
        model_b = {
            'model': 'PatchTST',
            'kernel_size': kernel_size
        }

        # Update args with model_b parameters
        for key, value in model_b.items():
            setattr(args, key, value)

        # Run Model 2
        print('\n\n>>>>>>>Running Model: PatchTST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        time_start = time.time()  # 开始计时
        # random seed
        fix_seed = args.random_seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu and args.use_multi_gpu:
            args.dvices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

        print('Args in experiment:')
        print(args)

        Exp = Exp_Main_2

        if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des, ii)

                time_start = time.time()
                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                if single_model:
                    preds_A, trues, result_A = exp.test(setting, test=1)
                    print('>>>>>>>Retrain Model: PatchTST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    exp.retrain(setting)
                    preds_B, trues, result_B = exp.test(setting, test=1)
                    exp.retrain(setting)
                    preds_C, trues, result_C = exp.test(setting, test=1)
                else:
                    preds_B, trues, result_B = exp.test(setting, test=1)
                print('Preds has been recorded.')

                print('Trues has been recorded.')


                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)
                cost_time_B = time.time() - time_start  # 计时结束
                print(f'>>>>>>>Cost Time : {cost_time_B}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

                torch.cuda.empty_cache()
                del exp
        else:
            ii = 0
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                          args.model,
                                                                                                          args.data,
                                                                                                          args.features,
                                                                                                          args.seq_len,
                                                                                                          args.label_len,
                                                                                                          args.pred_len,
                                                                                                          args.d_model,
                                                                                                          args.n_heads,
                                                                                                          args.e_layers,
                                                                                                          args.d_layers,
                                                                                                          args.d_ff,
                                                                                                          args.factor,
                                                                                                          args.embed,
                                                                                                          args.distil,
                                                                                                          args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if single_model:
                preds_A, trues, result_A = exp.test(setting, test=1)
                exp.retrain(setting)
                preds_B, trues, result_B = exp.test(setting, test=1)
                exp.retrain(setting)
                preds_C, trues, result_C = exp.test(setting, test=1)
            else:
                preds_B, trues, result_B = exp.test(setting, test=1)
            print('Preds has been recorded.')

            print('Trues has been recorded.')

            cost_time_B = time.time() - time_start  # 计时结束
            print(f'>>>>>>>Cost Time : {cost_time_B}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

            torch.cuda.empty_cache()
            del exp
    else:
        preds_B = 0
        results_B = None
        cost_time_B = 0
        print('\n\n>>>>>>>Model PatchTST is not selected<<<<<<<<<<<<<')


    if run_flag[2] == 1:
        # Model 3: TimeMixer
        model_c = {
            'model': 'TimeMixer',
            'kernel_size': kernel_size
        }

        # Update args with model_c parameters
        for key, value in model_c.items():
            setattr(args, key, value)

        # Run Model 3
        print('\n\n>>>>>>>Running Model: TimeMixer<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        time_start = time.time()  # 开始计时
        print('Args in experiment:')
        print(args)

        if args.task_name == 'long_term_forecast':
            Exp = Exp_Long_Term_Forecast
        elif args.task_name == 'short_term_forecast':
            Exp = Exp_Short_Term_Forecast
        elif args.task_name == 'imputation':
            Exp = Exp_Imputation
        elif args.task_name == 'anomaly_detection':
            Exp = Exp_Anomaly_Detection
        elif args.task_name == 'classification':
            Exp = Exp_Classification
        else:
            Exp = Exp_Long_Term_Forecast

        if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.comment,
                    args.model,
                    args.data,
                    args.seq_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des, ii)

                time_start = time.time()
                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                if single_model:
                    preds_A, trues, result_A = exp.test(setting)
                    exp.train(setting)
                    preds_B, trues, result_B = exp.test(setting)
                    exp.train(setting)
                    preds_C, trues, result_C = exp.test(setting)
                else:
                    preds_C, trues, result_C = exp.test(setting)
                print('Preds has been recorded.')

                print('Trues has been recorded.')

                cost_time_C = time.time() - time_start  # 计时结束
                print(f'>>>>>>>Cost Time : {cost_time_C}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

                torch.cuda.empty_cache()
                del exp
        else:
            ii = 0
            setting = '{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.comment,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if single_model:
                preds_A, trues, result_A = exp.test(setting)
                exp.retrain(setting)
                preds_B, trues, result_B = exp.test(setting)
                exp.retrain(setting)
                preds_C, trues, result_C = exp.test(setting)
            else:
                preds_C, trues, result_C = exp.test(setting)
            print('Preds has been recorded.')

            print('Trues has been recorded.')

            cost_time_C = time.time() - time_start  # 计时结束
            print(f'>>>>>>>Cost Time : {cost_time_C}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

            torch.cuda.empty_cache()
            del exp
    else:
        preds_C = 0
        results_C = None
        cost_time_C = 0
        print('\n\n>>>>>>>Model TimeMixer is not selected<<<<<<<<<<<<<')



    if run_flag[3] == 1:
        # Model 4: PathFormer
        model_a = {
            'model': 'PathFormer',
        }

        # Update args with model_a parameters
        for key, value in model_a.items():
            setattr(args, key, value)
        # Run Model 4
        print('\n\n>>>>>>>Running Model : PathFormer<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print('Args in experiment:')
        print(args)
        time_start = time.time()  # 开始计时
        Exp = Exp_Main_1
        if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_ft{}_sl{}_pl{}_{}'.format(
                    args.model_id,
                    args.model,
                    args.data_path[:-4],
                    args.features,
                    args.seq_len,
                    args.pred_len, ii)

                exp = Exp_Main_1(args)
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                time_now = time.time()
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                if single_model:
                    preds_A, trues, result_A = exp.test(setting)
                    exp.retrain(setting)
                    preds_B, trues, result_B = exp.test(setting)
                    exp.retrain(setting)
                    preds_C, trues, result_C = exp.test(setting)
                else:
                    preds_D, trues, result_D = exp.test(setting)
                print('Preds Preds has been recorded.')
                print('Inference time: ', time.time() - time_now)

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

                cost_time_D = time.time() - time_start  # 计时结束
                print(f'>>>>>>>Cost Time : {cost_time_D}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

                torch.cuda.empty_cache()

        else:
            ii = 0
            setting = '{}_{}_ft{}_sl{}_pl{}_{}'.format(
                args.model_id,
                args.model,
                args.data_path[:-4],
                args.features,
                args.seq_len,
                args.pred_len, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if single_model:
                preds_A, trues, result_A = exp.test(setting)
                exp.retrain(setting)
                preds_B, trues, result_B = exp.test(setting)
                exp.retrain(setting)
                preds_C, trues, result_C = exp.test(setting)
            else:
                preds_D, trues, result_D = exp.test(setting)
            print('Preds Preds has been recorded.')
            cost_time_D = time.time() - time_start  # 计时结束
            print(f'>>>>>>>Cost Time : {cost_time_D}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

            torch.cuda.empty_cache()
    else:
        preds_D = 0
        results_D = None
        cost_time_D = 0
        print('\n\n>>>>>>>Model PathFormer is not selected<<<<<<<<<<<<<')


    print('>>>>>>>Final Treatment<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    time_now = time.time()
    #求平均
    weights = args.mix_weight
    preds_AVG = (
            preds_A * weights[0] +
            preds_B * weights[1] +
            preds_C * weights[2] +
            preds_D * weights[3]
    )

    if args.pre_train and single_model != True:
        print(">>>>AWO-EMP: Start Learning Weight<<<<<<<<<<<<<")
        weights = optimize_weights([preds_A, preds_B, preds_C, preds_D], trues, initial_learning_rate=0.1,
                                   loss_function='mse', epochs=300, output_epochs=20)

        # 测试集融合预测
        preds_AVG = (
                preds_A * weights[0] +
                preds_B * weights[1] +
                preds_C * weights[2] +
                preds_D * weights[3]
        )
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds_AVG, trues)
        print("Weights:", weights)
        print(">>>>AWO-EMP: Over Learning Weight<<<<<<<<<<<<<")

    elif args.super_ensemble:
        print(">>>>>AWO-EMP: Ultimate Learner >>> Start Learning<<<<<<<<<<<<<")
        weights = optimize_weights([preds_A, preds_B, preds_C, preds_D, preds_AVG], trues, initial_learning_rate=0.1,
                                   loss_function='mse', epochs=300, output_epochs=20, max_patience=25)

        # 测试集融合预测
        preds_AVG = (
                preds_A * weights[0] +
                preds_B * weights[1] +
                preds_C * weights[2] +
                preds_D * weights[3] +
                preds_AVG * weights[4]
        )
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds_AVG, trues)
        print(">>>>AWO-EMP: Ultimate Learner >>> Over Learning<<<<<<<<<<<<<\n")

    if single_model:
        print(">>>>AWO-EMP: Start Learning<<<<<<<<<<<<<")
        weights = optimize_weights([preds_C, preds_B, preds_A], trues, initial_learning_rate=0.1,
                                   loss_function='mse', epochs=100, output_epochs=20)
        print(">>>>AWO-EMP: Over Learning Weight<<<<<<<<<<<<<")
        # 测试集融合预测
        preds_AVG = (
                preds_A * weights[2] +
                preds_B * weights[1] +
                preds_C * weights[0]
        )
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds_AVG, trues)
        time_final_treatment = time.time() - time_now


        print('Evaluation of model')  # 各模型评估
        print(f'{single_model_name}_1 >> mse:{result_A[0]}  mae:{result_A[1]} rse:{result_A[2]}')
        print(f'{single_model_name}_2 >> mse:{result_B[0]}  mae:{result_B[1]} rse:{result_B[2]}')
        print(f'{single_model_name}_3 >> mse:{result_C[0]}  mae:{result_C[1]} rse:{result_C[2]}')
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds_AVG, trues)
        print('Final Evaluation >>> mse:{}, mae:{}, rse:{}\nThis Step Cost time >>> {}'.format(mse, mae, rse,
                                                                                               time_final_treatment))
        print('>>>>>>>Spend Time Summary<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        cost_total = time.time() - time_Total  # 最终用时
        if run_flag[0] == 1:
            print(f'SimpleTM >> {cost_time_A}')
        if run_flag[1] == 1:
            print(f'PatchTST >> {cost_time_B}')
        if run_flag[2] == 1:
            print(f'TimeMixer >> {cost_time_C}')
        if run_flag[3] == 1:
            print(f'PathFormer >> {cost_time_D}')
        print(f'Final Treatment >> {time_final_treatment}')
        print(f'>>>>>>>TOTAL TIME >> {cost_total}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    else:
        time_final_treatment = time.time() - time_now
        print('Evaluation of model')  # 各模型评估
        if run_flag[0] == 1:
            print(f'SimpleTM[{weights[0]}] >> mse:{result_A[0]}  mae:{result_A[1]} rse:{result_A[2]}')
        if run_flag[1] == 1:
            print(f'PatchTST[{weights[1]}] >> mse:{result_B[0]}  mae:{result_B[1]} rse:{result_B[2]}')
        if run_flag[2] == 1:
            print(f'TimeMixer[{weights[2]}] >> mse:{result_C[0]}  mae:{result_C[1]} rse:{result_C[2]}')
        if run_flag[3] == 1:
            print(f'PathFormer[{weights[3]}] >> mse:{result_D[0]}  mae:{result_D[1]} rse:{result_D[2]}')
        print('Final Evaluation >>> mse:{}, mae:{}, rse:{}\nThis Step Cost time >>> {}'.format(mse, mae, rse, time_final_treatment))
        print('>>>>>>>Spend Time Summary<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        cost_total = time.time() - time_Total  # 最终用时
        # print(f'PathFormer >> {cost_time_A} \nPatchTST >> {cost_time_B} \nTimeMixer >> {cost_time_C} \nFinal Treatment >> {time_final_treatment}')
        if run_flag[0] == 1:
            print(f'SimpleTM >> {cost_time_A}')
        if run_flag[1] == 1:
            print(f'PatchTST >> {cost_time_B}')
        if run_flag[2] == 1:
            print(f'TimeMixer >> {cost_time_C}')
        if run_flag[3] == 1:
            print(f'PathFormer >> {cost_time_D}')
        print(f'Final Treatment >> {time_final_treatment}')
        print(f'>>>>>>>TOTAL TIME >> {cost_total}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    f = open("result.txt", 'a')
    # folder_path = './results/' + setting + '/'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    f.write('Final Evaluation >>> mse:{}, mae:{}, rse:{}\nCost time >>> {}'.format(mse, mae, rse, time_final_treatment))
    f.write('\n\n\n\n')
    f.close()

    if args.plot:
        # 创建保存目录
        plot_dir = "./plots/"
        os.makedirs(plot_dir, exist_ok=True)

        # 绘制前100个时间点的对比图（可根据需要调整）
        plt.plot(trues[:600, 0][:, -1], label='True Values')  # 取第一个特征绘制  -1
        plt.plot(preds_AVG[:600, 0][:, -1], label='Predicted Values')  # 取第一个特征绘制
        plt.title(f"{args.model_id} Forecast Comparison (seq={args.seq_len} pred={args.pred_len})")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        # 保存图像
        filename = f"{args.model_id}_{args.seq_len}_{args.pred_len}.png"
        plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {os.path.join(plot_dir, filename)}")










if __name__ == '__main__':
    main()
