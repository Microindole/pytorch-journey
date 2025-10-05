import os
import requests
import time
import argparse
from duckduckgo_search import DDGS

# --- 设置 ---
# 默认保存图片的目录
DEFAULT_SAVE_DIR = "./test_images_found"
# 请求头，模拟浏览器
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def search_and_download(query, limit, save_dir):
    """
    根据给定的查询词，搜索并下载图片。

    :param query: 搜索关键词 (例如 "Pikachu", "Greymon")
    :param limit: 要下载的图片数量
    :param save_dir: 图片保存目录
    """
    print(f"\n🔍 正在为关键词 '{query}' 搜索 {limit} 张图片...")

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    download_count = 0
    try:
        # 使用 DDGS().images() 进行图片搜索
        with DDGS() as ddgs:
            search_results = [r for r in ddgs.images(query, max_results=limit * 2)]  # 多获取一些结果以备下载失败

        if not search_results:
            print("❌ 未找到任何图片结果。")
            return

        print(f"✅ 找到了 {len(search_results)} 个结果，开始下载前 {limit} 个...")

        for i, result in enumerate(search_results):
            if download_count >= limit:
                break

            image_url = result.get('image')
            if not image_url:
                continue

            try:
                # 生成文件名
                # 获取文件后缀名，如果没有则默认为 .jpg
                file_ext = os.path.splitext(os.path.basename(image_url))[1]
                if not file_ext:
                    file_ext = '.jpg'

                # 清理查询词作为文件名的一部分
                safe_query = "".join(c for c in query if c.isalnum() or c in " _-").rstrip()
                filename = f"{safe_query}_{download_count + 1}{file_ext}"
                save_path = os.path.join(save_dir, filename)

                # 下载图片
                response = requests.get(image_url, headers=HEADERS, stream=True, timeout=10)
                response.raise_for_status()

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"  ({download_count + 1}/{limit}) 下载成功: {filename}")
                download_count += 1
                time.sleep(0.2)  # 短暂延时

            except Exception as e:
                print(f"  (跳过) 下载失败: {image_url} (原因: {e})")

    except Exception as e:
        print(f"❌ 搜索时发生错误: {e}")

    print(f"\n下载完成！共下载了 {download_count} 张图片到 '{os.path.abspath(save_dir)}' 目录。")


# --- 主程序入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从网上搜索并下载测试图片。")
    parser.add_argument('-q', '--query', type=str, required=True,
                        help='要搜索的图片关键词，例如 "Charizard" 或 "Agumon"。')
    parser.add_argument('-l', '--limit', type=int, default=5, help='要下载的图片数量 (默认: 5)。')
    parser.add_argument('-o', '--output_dir', type=str, default=DEFAULT_SAVE_DIR,
                        help=f'图片保存目录 (默认: {DEFAULT_SAVE_DIR})。')
    args = parser.parse_args()

    search_and_download(args.query, args.limit, args.output_dir)