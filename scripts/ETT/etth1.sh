if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi


seq_len=96
model_name=SimpleTM
root_path_name=./dataset/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

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
  --num_nodes 7 \
  --layer_nums 3 \
  --batch_norm 0 \
  --residual_connection 1 \
  --k 3 \
  --d_model 32 \
  --d_ff 128 \
  --patch_len 16 \
  --patience 8 \
  --lradj 'TST' \
  --patience 3 \
  --itr 1 \
  --batch_size 256 \
  --learning_rate 0.02 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --task_name 'long_term_forecast' \
  --e_layers 1 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --label_len 0 \
  --wv db1 \
  --m 3 \
  --kernel_size 25 \
  --des Exp \
  --alpha 0.3 \
  --train_epochs 40 \
  --fix_seed 2025 \
  --use_norm 1 \
  --l1_weight 0.0005 \
  --mix_weight 0.46439425 0.29939022 0.23621554 0.0 >logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len.log
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
  --num_nodes 7 \
  --layer_nums 3 \
  --batch_norm 0 \
  --residual_connection 1 \
  --k 3 \
  --d_model 8 \
  --d_ff 512 \
  --patch_len 16 \
  --patience 3 \
  --lradj 'TST' \
  --patience 10 \
  --itr 1 \
  --batch_size 256 \
  --learning_rate 0.02 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --task_name 'long_term_forecast' \
  --e_layers 1 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --label_len 0 \
  --wv db1 \
  --m 3 \
  --kernel_size 25 \
  --des Exp \
  --alpha 0.3 \
  --train_epochs 60 \
  --fix_seed 2025 \
  --use_norm 1 \
  --l1_weight 0.0005 \
  --mix_weight 0.46439425 0.29939022 0.23621554 0.0 >logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len.log
done
