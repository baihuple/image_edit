from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import json
from tqdm import tqdm
import boto3
import tempfile
from botocore.client import Config
import io
import PIL.Image as Image
from pathlib import Path
import os

# ============================================================
# 配置区域 - 根据你的网络环境修改
# ============================================================

# S3 访问密钥（Access Key ID）
AWS_ACCESS_KEY = '6EB0792D927D458584E29958B958D22E'
# S3 秘密密钥（Secret Access Key）
AWS_SECRET_KEY = '9D407B59553C44C6AD0080967F344C3E'
# 桶名称
BUCKET_NAME = 'edit-data'
# S3 端点地址
# 商汤内网机器使用内网端点（速度快，免流量）
S3_ENDPOINT_INTERNAL = 'https://aoss-internal.cn-fz-01.fjscmsapi-oss.com'
# 外部机器使用外网端点
S3_ENDPOINT_EXTERNAL = 'https://aoss.cn-fz-01.fjscmsapi-oss.com'
# 【重要】根据你的网络环境选择端点
# 如果你在商汤内部机器上，使用内网端点
# 如果你在其他机器上，请改为外网端点
S3_ENDPOINT = S3_ENDPOINT_INTERNAL  # 默认使用内网端点

# 列出 S3 目录下的所有 key
def list_s3_keys(s3_client, bucket_name, prefix):
    """
    列出 s3://bucket/prefix 下所有对象 key（递归）
    prefix 例子: "relighting-image/images_1/007518_seed7560_2x3_img1_Add/"
    """
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # 排除“目录占位符”
            if not key.endswith("/"):
                keys.append(key)
    return keys


# ============================================================
# 从桶中下载文件到临时文件
# ============================================================
def download_to_temp(s3_client, bucket_name, s3_key):
    """
    从 S3 桶中下载文件到本地临时文件

    参数:
        s3_client: S3 客户端对象
        bucket_name: 桶名称
        s3_key: 文件在桶中的 key（路径）

    返回:
        临时文件的路径
    """
    # 获取文件扩展名（如 .png, .jpg）
    # 这样临时文件也会有正确的扩展名
    _, ext = os.path.splitext(s3_key)

    # 创建一个临时文件
    # delete=False 表示关闭文件后不自动删除，我们稍后手动删除
    # suffix=ext 表示临时文件使用原文件的扩展名
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = temp_file.name  # 获取临时文件路径
    temp_file.close()  # 关闭文件句柄，以便 boto3 可以写入

    # 从 S3 下载文件到临时路径
    # bucket_name: 桶名称
    # s3_key: 文件在桶中的 key
    # temp_path: 本地保存路径
    s3_client.download_file(bucket_name, s3_key, temp_path)
    return temp_path

def load_image_from_s3(s3_client, bucket_name, s3_key):
    """
    从 S3 桶中读取图像，返回 PIL Image 对象
    使用临时文件方式，读取完成后自动删除临时文件

    参数:
        s3_client: S3 客户端对象
        bucket_name: 桶名称
        s3_key: 图像在桶中的 key（路径）

    返回:
        PIL.Image 对象
    """
    temp_path = None  # 初始化临时文件路径变量

    try:
        # 步骤1: 下载图像到临时文件
        temp_path = download_to_temp(s3_client, bucket_name, s3_key)

        # 步骤2: 使用 PIL 打开图像
        image = Image.open(temp_path).convert("RGB")

        # 步骤3: 将图像数据加载到内存中
        # 这一步很重要！load() 会将图像数据完全读入内存
        # 这样即使删除临时文件，图像数据仍然可用
        image.load()

        # 返回图像对象
        return image

    finally:
        # 步骤4: 清理临时文件
        # finally 块确保无论是否出错都会执行清理
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)  # 删除临时文件
def create_s3_client():
    """
    创建并返回一个 S3 客户端对象

    boto3 是 AWS 官方的 Python SDK，也兼容其他 S3 协议的对象存储服务
    """
    # 创建 S3 客户端
    s3_client = boto3.client(
        's3',                                    # 服务类型为 S3
        endpoint_url=S3_ENDPOINT,                # S3 服务端点地址
        aws_access_key_id=AWS_ACCESS_KEY,        # 访问密钥 ID
        aws_secret_access_key=AWS_SECRET_KEY,    # 秘密访问密钥
        config=Config(signature_version='s3v4')    # 使用 S3 签名版本
    )
    return s3_client
def init_s3_client():
    # ========== 步骤1: 创建 S3 客户端 ==========
    print("正在创建 S3 客户端...")
    print(f"使用端点: {S3_ENDPOINT}")
    s3_client = create_s3_client()
    print("S3 客户端创建成功！\n")
    return s3_client



def main():

    split_num = 3
    input_path = f"/mnt/cache/yangquan/datasets/image_edit/split_seg_json_diff/add_only_seg_V2_mv_part_0_{split_num}.json"
    output_json = input_path.replace(".json", "_filter.json")

    print("正在处理：", input_path)

    model_path = "/mnt/cache/yangquan/cache/huggingface/hub/Qwen3-VL-32B-Instruct"
    text_path = "/mnt/cache/yangquan/users/bhc/Qwen3-VL/filter.txt"

    s3_client = init_s3_client()

    with open(input_path, "r") as f:
        data = json.load(f)

    text = Path(text_path).read_text(encoding="utf-8").strip()

    # default: Load the model on the available device(s)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = AutoModelForImageTextToText.from_pretrained(
    #     model_path,
    #     dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    processor = AutoProcessor.from_pretrained(model_path)

    filter_result = []

    for item in tqdm(data):
        # origin [seg images]
        origin_img = load_image_from_s3(s3_client,BUCKET_NAME,item["edited_item_image_url"])
        # diff-view
        diff_view_paths = list_s3_keys(s3_client,BUCKET_NAME,item["diff-view"])
        diff_view_filter = []
        for diff_view_path in diff_view_paths:
            target_img = load_image_from_s3(s3_client,BUCKET_NAME,diff_view_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": origin_img,
                        },
                        {
                            "type": "image",
                            "image": target_img,
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]

            # Preparation for inference
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text)
            if output_text[0] == "yes":
                diff_view_filter.append(diff_view_path)
        item["diff-view-filter"] = diff_view_filter
        filter_result.append(item)
    with open(output_json, "w") as f:
        json.dump(filter_result, f, indent=2)

if __name__ == "__main__":
    main()