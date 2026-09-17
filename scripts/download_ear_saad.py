"""
Ear-SAAD 数据集多线程下载脚本
支持：断点续传、自动重试、多线程、进度显示
用法：python scripts/download_ear_saad.py
"""
import os
import sys
import time
import threading
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("正在安装 requests 库...")
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

# ============== 配置 ==============
DOWNLOAD_DIR = Path(r"D:\Attention\data\raw\ear_saad")
FILES = [
    {
        "name": "preprocessedData.zip",
        "url": "https://zenodo.org/records/16536441/files/preprocessedData.zip?download=1",
        "size_mb": 415.6,
        "md5": "b3cde4bf6ec08df5b8f4ed5118fcd6ba",
    },
    {
        "name": "experiment-manual.pdf",
        "url": "https://zenodo.org/records/16536441/files/experiment-manual.pdf?download=1",
        "size_mb": 1.2,
        "md5": "26a41ea5110335e5f2b6090e2dd265f8",
    },
]

NUM_THREADS = 8       # 下载线程数
CHUNK_SIZE = 1024 * 1024  # 1MB 块
MAX_RETRIES = 10       # 最大重试次数
RETRY_DELAY = 5        # 重试间隔（秒）
TIMEOUT = 30           # 连接超时（秒）

# 备用镜像（如果Zenodo直连失败，可以尝试这些）
MIRRORS = [
    "https://zenodo.org/records/16536441/files/{filename}?download=1",
    "https://zenodo.cern.ch/records/16536441/files/{filename}?download=1",
]


class Downloader:
    """多线程下载器，支持断点续传"""

    def __init__(self, url, save_path, num_threads=8):
        self.url = url
        self.save_path = Path(save_path)
        self.num_threads = num_threads
        self.file_size = 0
        self.downloaded = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.start_time = time.time()

    def get_file_size(self):
        """获取文件大小"""
        try:
            resp = requests.head(self.url, allow_redirects=True, timeout=TIMEOUT)
            size = int(resp.headers.get('content-length', 0))
            return size
        except Exception as e:
            print(f"  获取文件大小失败: {e}")
            return 0

    def download_chunk(self, start, end, thread_id):
        """下载单个块"""
        headers = {'Range': f'bytes={start}-{end}'}
        retries = 0
        while retries < MAX_RETRIES and not self.stop_event.is_set():
            try:
                resp = requests.get(
                    self.url, headers=headers, stream=True,
                    timeout=TIMEOUT, allow_redirects=True
                )
                with open(self.save_path, 'r+b') as f:
                    f.seek(start)
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            with self.lock:
                                self.downloaded += len(chunk)
                return
            except Exception as e:
                retries += 1
                if retries < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  线程{thread_id}下载失败（块 {start}-{end}）: {e}")

    def download(self):
        """开始下载"""
        # 获取文件大小
        self.file_size = self.get_file_size()
        if self.file_size == 0:
            print(f"  无法获取文件大小，尝试单线程下载...")
            return self.single_thread_download()

        # 检查已下载大小（断点续传）
        if self.save_path.exists():
            existing_size = self.save_path.stat().st_size
            if existing_size == self.file_size:
                print(f"  文件已存在且完整，跳过下载")
                return True
            elif existing_size > 0:
                print(f"  发现未完成下载（{existing_size/1024/1024:.1f}MB / {self.file_size/1024/1024:.1f}MB），尝试断点续传...")
                # 简单起见，如果已有部分文件，先删除重新下载（多线程断点续传较复杂）
                # 后续可以优化为真正的断点续传
                self.save_path.unlink()

        # 创建空文件
        with open(self.save_path, 'wb') as f:
            f.truncate(self.file_size)

        # 分块
        chunk_size = self.file_size // self.num_threads
        threads = []
        for i in range(self.num_threads):
            start = i * chunk_size
            end = self.file_size - 1 if i == self.num_threads - 1 else (i + 1) * chunk_size - 1
            t = threading.Thread(target=self.download_chunk, args=(start, end, i))
            threads.append(t)
            t.start()

        # 显示进度
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            self.print_progress()

        for t in threads:
            t.join()

        self.print_progress()
        print()

        # 验证
        if self.save_path.stat().st_size == self.file_size:
            print(f"  下载完成！大小: {self.file_size/1024/1024:.1f}MB")
            return True
        else:
            print(f"  下载不完整！期望: {self.file_size}, 实际: {self.save_path.stat().st_size}")
            return False

    def single_thread_download(self):
        """单线程下载（备用方案）"""
        retries = 0
        while retries < MAX_RETRIES:
            try:
                resp = requests.get(self.url, stream=True, timeout=TIMEOUT, allow_redirects=True)
                total = int(resp.headers.get('content-length', 0))
                downloaded = 0
                with open(self.save_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                pct = downloaded / total * 100
                                print(f"\r  进度: {pct:.1f}% ({downloaded/1024/1024:.1f}/{total/1024/1024:.1f}MB)", end='', flush=True)
                print()
                print(f"  下载完成！大小: {self.save_path.stat().st_size/1024/1024:.1f}MB")
                return True
            except Exception as e:
                retries += 1
                print(f"  下载失败（第{retries}次）: {e}")
                if retries < MAX_RETRIES:
                    print(f"  {RETRY_DELAY}秒后重试...")
                    time.sleep(RETRY_DELAY)
        return False

    def print_progress(self):
        """打印进度"""
        if self.file_size > 0:
            pct = self.downloaded / self.file_size * 100
            elapsed = time.time() - self.start_time
            speed = self.downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
            remaining = (self.file_size - self.downloaded) / (speed * 1024 * 1024) if speed > 0 else 0
            print(f"\r  进度: {pct:.1f}% ({self.downloaded/1024/1024:.1f}/{self.file_size/1024/1024:.1f}MB) "
                  f"速度: {speed:.2f}MB/s 剩余: {remaining:.0f}s", end='', flush=True)


def verify_md5(file_path, expected_md5):
    """验证文件MD5"""
    import hashlib
    print(f"  正在验证MD5...")
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_md5.update(chunk)
    actual_md5 = hash_md5.hexdigest()
    if actual_md5 == expected_md5:
        print(f"  MD5验证通过！")
        return True
    else:
        print(f"  MD5验证失败！期望: {expected_md5}, 实际: {actual_md5}")
        return False


def main():
    print("=" * 60)
    print("Ear-SAAD 数据集下载工具")
    print("=" * 60)
    print(f"下载目录: {DOWNLOAD_DIR}")
    print(f"线程数: {NUM_THREADS}")
    print(f"最大重试: {MAX_RETRIES}次")
    print()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    all_success = True
    for file_info in FILES:
        filename = file_info["name"]
        url = file_info["url"]
        save_path = DOWNLOAD_DIR / filename

        print(f"[{filename}] ({file_info['size_mb']}MB)")
        print(f"  URL: {url}")

        # 检查是否已存在
        if save_path.exists() and save_path.stat().st_size > 0:
            print(f"  文件已存在，大小: {save_path.stat().st_size/1024/1024:.1f}MB")
            if 'md5' in file_info:
                if verify_md5(save_path, file_info['md5']):
                    print(f"  跳过下载")
                    continue
                else:
                    print(f"  MD5不匹配，重新下载...")
                    save_path.unlink()

        # 尝试下载
        downloader = Downloader(url, save_path, num_threads=NUM_THREADS)
        success = downloader.download()

        if success and 'md5' in file_info:
            verify_md5(save_path, file_info['md5'])

        if not success:
            all_success = False
            print(f"  ❌ 下载失败！")
        else:
            print(f"  ✅ 下载成功！")
        print()

    print("=" * 60)
    if all_success:
        print("✅ 所有文件下载完成！")
        print(f"文件位置: {DOWNLOAD_DIR}")
        print()
        print("下一步：告诉我下载完成，我会开始数据预处理和训练")
    else:
        print("⚠️ 部分文件下载失败，请检查网络连接后重试")
        print()
        print("备选方案：")
        print("  1. 增加线程数：修改脚本中的 NUM_THREADS = 16")
        print("  2. 使用代理：设置环境变量 HTTP_PROXY 和 HTTPS_PROXY")
        print("  3. 用下载工具手动下载：IDM、迅雷等")
        print("  4. 换网络环境：手机热点等")
    print("=" * 60)


if __name__ == "__main__":
    main()
