import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# --- 全局设置 ---
# 设置每个类别要下载的图片数量上限 (设置为 None 则不限制)
IMAGE_LIMIT = 150
# 保存图片的总目录
BASE_SAVE_DIR = "../dataset_scraped"
# 请求头，模拟浏览器
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def download_image(image_url, save_path):
    """下载单张图片并保存"""
    try:
        if os.path.exists(save_path):
            print(f"  [跳过] 文件已存在: {os.path.basename(save_path)}")
            return True

        response = requests.get(image_url, headers=HEADERS, stream=True, timeout=15)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  [成功] 已下载: {os.path.basename(save_path)}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  [失败] 下载错误: {e}")
        return False


def scrape_pokemon():
    """抓取宝可梦图片"""
    print("\n--- 开始抓取宝可梦图片 ---")
    list_url = "https://pokemondb.net/pokedex/national"
    save_dir = os.path.join(BASE_SAVE_DIR, "pokemon")
    os.makedirs(save_dir, exist_ok=True)

    try:
        print(f"正在从 {list_url} 获取宝可梦列表...")
        response = requests.get(list_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 找到所有宝可梦卡片
        pokemon_cards = soup.select('div.infocard-list div.infocard')
        count = 0

        for card in pokemon_cards:
            if IMAGE_LIMIT and count >= IMAGE_LIMIT:
                print(f"已达到 {IMAGE_LIMIT} 张图片的下载上限。")
                break

            # 提取详情页链接
            detail_link = card.select_one('a.ent-name')
            if not detail_link:
                continue

            name = detail_link.text
            detail_page_url = urljoin(list_url, detail_link['href'])

            print(f"[{count + 1}] 正在处理: {name}")

            try:
                # 访问详情页
                detail_response = requests.get(detail_page_url, headers=HEADERS)
                detail_response.raise_for_status()
                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

                # 查找主图片
                image_tag = detail_soup.select_one('main .grid-col.span-md-6.span-lg-4 p a img')
                if not image_tag or not image_tag.has_attr('src'):
                    print(f"  [警告] 在 {name} 的页面未找到图片。")
                    continue

                image_url = image_tag['src']
                filename = os.path.basename(image_url)
                save_path = os.path.join(save_dir, filename)

                if download_image(image_url, save_path):
                    count += 1

                time.sleep(0.5)  # 友好抓取延时

            except requests.exceptions.RequestException as e:
                print(f"  [失败] 无法访问 {name} 的详情页: {e}")

    except Exception as e:
        print(f"抓取宝可梦时发生严重错误: {e}")


def scrape_digimon():
    """抓取数码宝贝图片"""
    print("\n--- 开始抓取数码宝贝图片 ---")
    list_url = "https://digimon.fandom.com/wiki/List_of_Digimon"
    save_dir = os.path.join(BASE_SAVE_DIR, "digimon")
    os.makedirs(save_dir, exist_ok=True)

    try:
        print(f"正在从 {list_url} 获取数码宝贝列表...")
        response = requests.get(list_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Fandom wiki的列表在表格中
        digimon_rows = soup.select('table.sortable tr')
        count = 0

        for row in digimon_rows:
            if IMAGE_LIMIT and count >= IMAGE_LIMIT:
                print(f"已达到 {IMAGE_LIMIT} 张图片的下载上限。")
                break

            # 从表格第二列找到名字和链接
            link_cell = row.select_one('td:nth-of-type(2) a')
            if not link_cell or not link_cell.has_attr('href'):
                continue

            name = link_cell.text.strip()
            detail_page_url = urljoin(list_url, link_cell['href'])

            print(f"[{count + 1}] 正在处理: {name}")

            try:
                # 访问详情页
                detail_response = requests.get(detail_page_url, headers=HEADERS)
                detail_response.raise_for_status()
                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

                # 在右侧信息框中找到图片链接
                image_tag = detail_soup.select_one('aside.portable-infobox .pi-image a img')
                if not image_tag or not image_tag.has_attr('src'):
                    print(f"  [警告] 在 {name} 的页面未找到图片。")
                    continue

                # Fandom的图片URL有时需要处理一下，去掉尺寸后缀
                image_url = image_tag['src'].split('/revision/')[0]

                # 使用名字来命名文件，避免URL中没有文件名
                extension = os.path.splitext(urlparse(image_url).path)[1]
                if not extension: extension = '.png'  # 默认后缀
                filename = f"{name.replace(' ', '_')}{extension}"
                save_path = os.path.join(save_dir, filename)

                if download_image(image_url, save_path):
                    count += 1

                time.sleep(0.5)  # 友好抓取延时

            except requests.exceptions.RequestException as e:
                print(f"  [失败] 无法访问 {name} 的详情页: {e}")

    except Exception as e:
        print(f"抓取数码宝贝时发生严重错误: {e}")


if __name__ == '__main__':
    print("自动图片抓取脚本已启动。")
    print(f"图片数量上限: {'无限制' if IMAGE_LIMIT is None else IMAGE_LIMIT}")
    print(f"图片将保存到: {os.path.abspath(BASE_SAVE_DIR)}")

    # 执行抓取任务
    scrape_pokemon()
    scrape_digimon()

    print("\n所有抓取任务完成！")
    print(f"请检查 '{BASE_SAVE_DIR}' 文件夹中的内容。")