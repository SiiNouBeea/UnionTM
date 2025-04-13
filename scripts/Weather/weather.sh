if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi



seq_len=96
model_name=SimpleTM
root_path_name=./dataset/weather/
data_path_name=weather.csv
model_id_name=Weather
data_name=custom
dynamic_delt=1.6316809212295418

for pred_len in 96
do
  python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --dynamic_delt $dynamic_delt \
  --model_id $model_id_name \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 4 \
  --d_model 32 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --batch_size 128 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db4" \
  --m 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --alpha 0.9 \
  --l1_weight 5e-05 \
  --lradj 'TST' \
  --patience 3 \
  --task_name long_term_forecast \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --label_len 0 \
  --n_heads 16 \
  --patch_len 16 \
  --stride 8 \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --pct_start 0.4 \
  --train_epochs 12 \
  --random_seed 2021 \
  --d_layers 1 \
  --mix_weight 0.47379644 0.25559652 0.27060703 0\
  --factor 3 >logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 192 336
do
  python -u run.py \
  --is_training 1 \
  --dynamic_delt $dynamic_delt \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 1 \
  --d_model 32 \
  --d_ff 32 \
  --learning_rate 0.009 \
  --batch_size 128 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db4" \
  --m 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --alpha 0.9 \
  --l1_weight 5e-05 \
  --lradj 'TST' \
  --patience 3 \
  --task_name long_term_forecast \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --label_len 0 \
  --n_heads 16 \
  --patch_len 16 \
  --stride 8 \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --pct_start 0.4 \
  --train_epochs 12 \
  --random_seed 2021 \
  --d_layers 1 \
  --mix_weight 0.47379644 0.25559652 0.27060703 0\
  --factor 3 >logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 720
do
  python -u run.py \
  --is_training 1 \
  --dynamic_delt $dynamic_delt \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 1 \
  --d_model 32 \
  --d_ff 32 \
  --learning_rate 0.02 \
  --batch_size 128 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db4" \
  --m 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --alpha 0.9 \
  --l1_weight 5e-05 \
  --lradj 'TST' \
  --patience 3 \
  --task_name long_term_forecast \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --label_len 0 \
  --n_heads 16 \
  --patch_len 16 \
  --stride 8 \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --pct_start 0.4 \
  --train_epochs 12 \
  --random_seed 2021 \
  --d_layers 1 \
  --mix_weight 0.47379644 0.25559652 0.27060703 \
  --factor 3 >logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len.log
done
