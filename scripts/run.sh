python /mnt/results/crc/crc_poc3_plus/analysis7/bin/deliver_to_hj/MLGenie_prediction.py \
    --model_path /mnt/results/crc/crc_poc3_plus/analysis7/bin/deliver_to_hj/model.pkl \
    --test_data_path //mnt/results/crc/crc_poc3_plus/analysis7/bin/deliver_to_hj/POC3_plus_Target_inflammation_combine_batch1_batch2_filtered_matrix.csv \
    --test_label_int_path /mnt/results/crc/crc_poc3_plus/analysis7/bin/deliver_to_hj/POC3_plus_Target_inflammation_combine_batch1_batch2_filtered_label_int.csv \
    --test_label_path /mnt/results/crc/crc_poc3_plus/analysis7/bin/deliver_to_hj/POC3_plus_Target_inflammation_combine_batch1_batch2_filtered_label.csv \
    --threshold 0.46 \
    --output_dir /mnt/results/crc/crc_poc3_plus/analysis7/bin/deliver_to_hj/output