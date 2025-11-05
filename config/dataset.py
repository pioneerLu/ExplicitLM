from hydra_zen import builds

DatasetConf = builds(
    dict,
    dataset_path="data/database/merged_pretrain.jsonl", # 预训练数据集路径
    val_dataset_path="data/benchmarks/eval_data.jsonl", # 验证数据集路径
    max_subject_len=8,
    max_predicate_len=4,
    max_object_len=8,
    populate_full_signature=False,
    # ---- sft 相关字段 ----
    pretrained_sft_model_path="out/pretrain_latest.pth", # 预训练 SFT 模型路径
    sft_dataset_path="data/database/merged_sft.jsonl", # SFT 数据集路径
    sft_val_dataset_path="data/benchmarks/sft_eval_data.jsonl", # SFT 验证数据集路径
)