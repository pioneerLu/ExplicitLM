"""
预训练主程序

功能：
- 初始化分布式训练环境（Accelerator + DeepSpeed）
- 配置混合精度训练
- 设置随机种子保证可复现性
"""

import os
from typing import Optional, Dict, Any

from utils.config_utils import setup_config
from utils.logger import logger
from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
from accelerate.utils import set_seed

try:
    import swanlab
except ImportError:
    swanlab = None


def main():
    """
    预训练主函数

    实现流程：
    1. 加载超参数配置
    2. 初始化分布式训练环境
    3. 设置随机种子
    4. 创建输出目录
    5. TODO: 模型初始化
    6. TODO: 数据加载器初始化
    7. TODO: 训练循环
    """

    #########################################################
    # 第一阶段：配置初始化
    #########################################################
    args = setup_config()

    #########################################################
    # 第二阶段：初始化 Accelerator 和 DeepSpeed
    #########################################################
    # 配置 DDP 参数：允许未使用的参数（用于部分模型组件可能不参与梯度计算的情况）
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # 配置 DeepSpeed 插件
    # 注意：大部分配置在 ds_config.json 中已定义，这里只设置关键参数
    # ds_config.json 中已配置：
    #   - ZeRO-2 优化
    #   - CPU offload（优化器和参数）
    #   - bf16 混合精度
    #   - gradient clipping 和 accumulation（设为 auto）
    ds_plugin = DeepSpeedPlugin(
        zero_stage=2,  # 使用 ZeRO-2 优化，与 ds_config.json 保持一致
    )

    # 初始化 Accelerator
    # 混合精度、梯度累积等参数由 ds_config.json 控制
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=ds_plugin,
    )

    #########################################################
    # 第三阶段：设置随机种子
    #########################################################
    # 为每个进程设置不同的随机种子，确保数据采样的多样性
    set_seed(1337 + accelerator.process_index)

    #########################################################
    # 第四阶段：创建输出目录
    #########################################################
    # 配置保存路径：args.save_dir 作为模型检查点和训练输出的根目录
    args.save_dir = args.out_dir

    # 仅主进程负责创建目录，避免多进程竞争
    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)

        # 打印配置信息
        print(f"初始化完成:")
        print(f"  - 进程数: {accelerator.num_processes}")
        print(f"  - 混合精度: {accelerator.mixed_precision}")
        print(f"  - DeepSpeed ZeRO Stage: 2")
        print(f"  - 设备: {accelerator.device}")
        print(f"  - 输出目录: {args.out_dir}")

    #########################################################
    # 第五阶段：配置SwanLab实验追踪
    #########################################################
    swanlab_run: Optional[Any] = None

    if hasattr(args, 'use_swanlab') and args.use_swanlab and accelerator.is_main_process:
        if swanlab is None:
            logger("警告: SwanLab未安装，跳过实验追踪配置", accelerator)
        else:
            # 配置实验运行名称
            args.swanlab_run_name = (
                f"MiniMind-Pretrain-Epoch-{args.epochs}-"
                f"BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
            )

            # 创建配置字典用于SwanLab追踪
            config_dict: Dict[str, Any] = vars(args).copy()

            # 根据配置选择在线或离线模式
            mode = "online" if getattr(args, 'swanlab_online', False) else "offline"

            # 初始化SwanLab实验实例
            swanlab_run = swanlab.init(
                project=getattr(args, 'swanlab_project', 'MiniMind'),
                experiment_name=args.swanlab_run_name,
                description="MiniMind预训练实验追踪",
                config=config_dict,
                mode=mode
            )

            logger(f"SwanLab初始化完成 (模式: {mode})", accelerator)
            logger(f"  - 项目: {getattr(args, 'swanlab_project', 'MiniMind')}", accelerator)
            logger(f"  - 实验名称: {args.swanlab_run_name}", accelerator)

    #########################################################
    # 第六阶段：模型初始化
    #########################################################
    from utils.model_initializer import init_model

    logger("开始初始化模型...", accelerator)

    # 初始化模型和tokenizer
    model, tokenizer = init_model(args)

    logger(f"模型初始化完成，准备使用Accelerator...", accelerator)

    # 使用Accelerator准备模型
    model = accelerator.prepare(model)

    logger(f"模型已准备完毕，设备: {accelerator.device}", accelerator)

    #########################################################
    # TODO: 第七阶段 - 数据加载器初始化
    #########################################################
    # train_dataloader = ...
    # train_dataloader = accelerator.prepare(train_dataloader)

    #########################################################
    # TODO: 第八阶段 - 优化器初始化
    #########################################################
    # optimizer = ...
    # optimizer = accelerator.prepare(optimizer)

    #########################################################
    # TODO: 第九阶段 - 训练循环
    #########################################################
    # for epoch in range(config.num_epochs):
    #     for batch in train_dataloader:
    #         ...


if __name__ == '__main__':
    main()
