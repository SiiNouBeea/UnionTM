if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=96
model_name=SimpleTM
root_path_name=./dataset/solar/
data_path_name=solar_AL.csv
model_id_name=Solar
data_name=Solar
dynamic_delta=1.6111095439660712


for pred_len in 96 192
do
  python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 0.006 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 0 \
  --wv "db8" \
  --m 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --alpha 0.0 \
  --l1_weight 0.005 \
  --lradj 'TST' \
  --patience 4 \
  --task_name long_term_forecast \
  --down_sampling_layers 2 \
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
  --train_epochs 20 \
  --random_seed 2021 \
  --dynamic_delta $dynamic_delta \
  --mix_weight 0.72512417 0.27487583 0.0 >logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 336 720
do
  python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 1024 \
  --learning_rate 0.02 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 0 \
  --wv "db8" \
  --m 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --alpha 0.0 \
  --l1_weight 0.005 \
  --lradj 'TST' \
  --patience 6 \
  --task_name long_term_forecast \
  --down_sampling_layers 2 \
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
  --train_epochs 25 \
  --random_seed 2021 \
  --dynamic_delta $dynamic_delta \
  --mix_weight 0.72512417 0.27487583 0.0 >logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len.log
done