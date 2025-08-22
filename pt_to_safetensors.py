import torch
from safetensors.torch import save_file
import os
import logging

# --- 可选：设置日志记录 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_pt_to_safetensors(pt_file_path, safetensors_file_path=None, prefix_to_remove="llm.model."):
    """
    将 PyTorch .pt/.pth 文件转换为 .safetensors 文件。
    处理共享内存张量（通过复制）并移除指定的键前缀。

    Args:
        pt_file_path (str): 输入的 PyTorch 文件路径 (.pt 或 .pth)。
        safetensors_file_path (str, optional): 输出的 safetensors 文件路径。
                                               如果为 None，则自动在输入文件同目录下生成同名 .safetensors 文件。
                                               Defaults to None.
        prefix_to_remove (str, optional): 要从 state_dict 键中移除的前缀。
                                          Defaults to "llm.model.".

    Returns:
        str: 成功保存的 safetensors 文件路径。
        None: 如果转换失败。
    """
    try:
        # 1. 确定输出文件路径
        if safetensors_file_path is None:
            base_name = os.path.splitext(pt_file_path)[0] # 获取不带扩展名的文件名
            safetensors_file_path = f"{base_name}.safetensors"

        # 2. 加载 PyTorch 文件
        logger.info(f"正在加载 PyTorch 文件: {pt_file_path}")
        # 使用 weights_only=True (PyTorch >= 1.13) 可以提高安全性
        try:
            state_dict = torch.load(pt_file_path, map_location='cpu', weights_only=True)
        except TypeError:
            # 如果 weights_only 参数不被支持 (旧版本 PyTorch)
            logger.warning("当前 PyTorch 版本可能不支持 'weights_only' 参数，将尝试不带该参数加载。")
            state_dict = torch.load(pt_file_path, map_location='cpu')

        # 3. 检查加载的内容是否为 state_dict (字典)
        if not isinstance(state_dict, dict):
            logger.error(f"加载的文件 {pt_file_path} 不包含 state_dict (字典)。它可能是整个模型对象。")
            logger.error("此脚本仅处理包含 state_dict 的文件。")
            return None

        # 4. --- 处理键名前缀 ---
        if prefix_to_remove:
            logger.info(f"正在移除键名前缀: '{prefix_to_remove}'")
            new_state_dict = {}
            keys_removed = 0
            for key, value in state_dict.items():
                if key.startswith(prefix_to_remove):
                    new_key = key[len(prefix_to_remove):]
                    new_state_dict[new_key] = value
                    keys_removed += 1
                else:
                    new_state_dict[key] = value
            logger.info(f"已移除 {keys_removed} 个键的前缀。")
            state_dict = new_state_dict

        # 5. --- 处理共享内存张量 ---
        # 创建一个字典来跟踪张量的存储 (storage) 和对应的键
        storage_to_keys = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                storage_ptr = tensor.storage().data_ptr() # 获取底层存储的指针
                if storage_ptr not in storage_to_keys:
                    storage_to_keys[storage_ptr] = []
                storage_to_keys[storage_ptr].append(key)

        # 找出共享存储的键组
        shared_key_groups = [keys for keys in storage_to_keys.values() if len(keys) > 1]
        if shared_key_groups:
            logger.info(f"检测到 {len(shared_key_groups)} 组共享内存张量。")
            for i, group in enumerate(shared_key_groups):
                logger.info(f"  组 {i+1}: {group}")

            # 为共享的张量创建副本
            logger.info("正在为共享张量创建独立副本...")
            processed_state_dict = {}
            copied_count = 0
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    storage_ptr = tensor.storage().data_ptr()
                    # 如果这个存储被多个键共享
                    if len(storage_to_keys.get(storage_ptr, [])) > 1:
                        # 创建副本
                        processed_state_dict[key] = tensor.clone()
                        copied_count += 1
                        logger.debug(f"    已为键 '{key}' 创建副本。")
                    else:
                        # 不共享，直接使用原张量
                        processed_state_dict[key] = tensor
                else:
                    # 非张量，直接使用
                    processed_state_dict[key] = tensor
            logger.info(f"共为 {copied_count} 个共享张量创建了副本。")
            state_dict = processed_state_dict # 使用处理后的 state_dict
        else:
            logger.info("未检测到共享内存张量。")

        # 6. 保存为 safetensors 格式
        logger.info(f"正在将处理后的权重保存为 safetensors 格式: {safetensors_file_path}")
        save_file(state_dict, safetensors_file_path) # 现在应该可以成功保存了

        logger.info(f"转换成功! safetensors 文件已保存至: {safetensors_file_path}")
        return safetensors_file_path

    except FileNotFoundError:
        logger.error(f"错误：找不到输入文件 {pt_file_path}")
    except Exception as e:
        logger.error(f"转换过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return None

# --- 使用示例 ---
if __name__ == "__main__":
    # --- 请修改为你自己的 .pt 或 .pth 文件路径 ---
    input_pt_file = "pretrained_models/CosyVoice2-0.5B/llm.pt" # 修改为实际路径和文件名
    # --- 可选：指定输出文件路径 ---
    output_safetensors_file = "pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN/model.safetensors"
    # output_safetensors_file = None # 如果为 None，则使用默认命名规则

    # --- 执行转换 ---
    result_path = convert_pt_to_safetensors(input_pt_file, output_safetensors_file)

    if result_path:
        print(f"\n[完成] 文件已成功转换并保存到: {result_path}")
    else:
        print(f"\n[失败] 文件转换失败。请检查日志信息。")
