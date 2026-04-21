# download_qwen3_8b_full.py
import os
import sys
import traceback
from huggingface_hub import snapshot_download

def main():
    # 官方仓库
    repo_id = "Qwen/Qwen3-8B"

    # 本地保存目录
    local_dir = os.path.abspath("./models/Qwen3-8B")

    # 可选：如果你有 HF_TOKEN，可以自动使用；没有也通常能下载这个公开仓库
    token = os.getenv("HF_TOKEN", None)

    # 可选：固定到某个 revision（分支/commit/tag）
    revision = "main"

    print("=" * 60)
    print(f"开始下载模型: {repo_id}")
    print(f"保存目录: {local_dir}")
    print(f"revision : {revision}")
    print("=" * 60)

    os.makedirs(local_dir, exist_ok=True)

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            token=token,
            resume_download=True,
            max_workers=8,
        )

        print("\n下载完成。")
        print("本地路径:", downloaded_path)

        print("\n目录下的关键文件通常应包括：")
        expected_files = [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ]
        for name in expected_files:
            path = os.path.join(local_dir, name)
            print(f"[{'OK' if os.path.exists(path) else 'MISSING'}] {name}")

        print("\n所有 .safetensors 分片：")
        safetensor_files = sorted(
            [f for f in os.listdir(local_dir) if f.endswith(".safetensors")]
        )
        if safetensor_files:
            for f in safetensor_files:
                print(" -", f)
        else:
            print("未找到 .safetensors 文件，请检查下载是否完整。")

    except Exception as e:
        print("\n下载失败。")
        print("错误类型:", type(e).__name__)
        print("错误信息:", str(e))
        print("\n详细堆栈：")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()