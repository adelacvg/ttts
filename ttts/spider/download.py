import os
import requests
from multiprocessing.pool import ThreadPool

def download_file(url, output_dir):
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")

def download_files(urls, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pool = ThreadPool(processes=os.cpu_count())
    for url in urls:
        pool.apply_async(download_file, (url, output_dir))
    pool.close()
    pool.join()

if __name__ == '__main__':
    urls_file = 'urls.txt'  # 按照你的描述，音频文件的URL存储在名为urls.txt的txt文件中
    output_directory = 'dataset'  # 下载的音频文件将保存在名为dataset的文件夹中

    with open(urls_file, 'r') as file:
        urls = file.read().splitlines()

    download_files(urls, output_directory)
