if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi


seq_len=96
model_name=SimpleTM
root_path_name=./dataset/Wind/
data_path_name=wind.csv
model_id_name=Wind
data_name=custom
dynamic_delta=1.6004820523714062


for pred_len in 96 192 336 720
do
      python -u run.py \
      --target None \
      --is_training 1 \
      --dynamic_delta $dynamic_delta \
      --target 'target' \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name \
      --model $model_name \
      --data $data_name \
      --freq '15min' \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 3 \
      --d_model 32 \
      --d_ff 64 \
      --learning_rate 0.006 \
      --batch_size 256 \
      --fix_seed 2025 \
      --use_norm 1 \
      --wv "db1" \
      --m 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --alpha 0.3 \
      --l1_weight 0.0005 \
      --lradj 'TST' \
      --patience 6 \
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
      --pct_start 0.2 \
      --train_epochs 40 \
      --random_seed 2021 \
      --mix_weight 0.5 0.5 0 \
      --plot 'true' \
      --find_weight 'true' >logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len.log
done
