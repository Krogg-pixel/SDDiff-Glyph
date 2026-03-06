accelerate launch training/train_text_to_image_lora.py \
 --pretrained_model_name_or_path pretrained/sd15 \
 --train_data_dir dataset/semantic_levels/s3 \
 --resolution 512 \
 --train_batch_size 1 \
 --learning_rate 1e-4 \
 --rank 4 \
 --max_train_steps 1500 \
 --checkpointing_steps 250