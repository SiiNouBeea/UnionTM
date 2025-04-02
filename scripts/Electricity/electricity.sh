if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi


seq_len=336
model_name=SimpleTM
root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

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
      --e_layers 3 \
      --d_model 128 \
      --d_ff 512 \
      --learning_rate 0.0001 \
      --batch_size 32 \
      --fix_seed 2025 \
      --use_norm 1 \
      --wv "db1" \
      --m 4 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --itr 1 \
      --alpha 0.3 \
      --l1_weight 0.0005 \
      --lradj 'TST' \
      --patience 4 \
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
      --train_epochs 30 \
      --random_seed 2021 \
      --mix_weight 1 0 0 >logs/LongForecasting/$model_id_name'_'$seq_len_len'_'$pred_len.log
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
      --e_layers 3 \
      --d_model 256 \
      --d_ff 2048 \
      --learning_rate 0.0001 \
      --batch_size 32 \
      --fix_seed 2025 \
      --use_norm 1 \
      --wv "db1" \
      --m 4 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
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
      --train_epochs 30 \
      --random_seed 2021 \
      --mix_weight 1 0 0 >logs/LongForecasting/$model_id_name'_'$seq_len_len'_'$pred_len.log
done