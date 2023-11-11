import requests
from bs4 import BeautifulSoup
import subprocess
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
from webdriver_manager.chrome import ChromeDriverManager


def get_page_content(url,driver):
    driver.get(url)
    html_content = driver.page_source
    return html_content

def parse_album_links(content):
    soup = BeautifulSoup(content, 'html.parser')
    album_elements = soup.find_all('a', class_='album-cover')
    album_links = [element['href'] for element in album_elements]
    return album_links

def get_all_album_links(base_url, num_pages):
    service = Service(executable_path=ChromeDriverManager().install())

    op = webdriver.ChromeOptions()
    # op.add_argument('headless')
    op.add_argument("--log-level=3")
    driver = webdriver.Chrome(options=op,service=service)
    album_links = []

    page_url = base_url
    page_count = 1
    while page_url and page_count < num_pages:
        content = get_page_content(page_url,driver)
        links = parse_album_links(content)
        album_links.extend(links)

        next_page_link = parse_next_page_link(content, page_count+1)
        if next_page_link:
            page_url = f"https://www.ximalaya.com{next_page_link}"
        else:
            page_url = None

        page_count += 1

    return album_links

def parse_next_page_link(content, page_count):
    soup = BeautifulSoup(content, 'html.parser')
    next_page_element = soup.find('a', class_='page-link', text=str(page_count))
    if next_page_element:
        return next_page_element['href']
    return None

def download_album_audio(album_id):
    out_path = './xmly'
    command = f'xmlyfetcher -o {out_path} {album_id} all '
    subprocess.run(command, shell=True)

def download_all_albums(album_links):
    for link in album_links:
        album_id = link.split('/')[-1]
        download_album_audio(album_id)
def save_links_to_file(links, file_path):
    with open(file_path, 'w') as file:
        for link in links:
            file.write(link + '\n')
def read_links_from_file(file_path):
    links = []

    with open(file_path, 'r') as file:
        for line in file:
            link = line.strip()  # 去除行末尾的换行符
            links.append(link)

    return links
if __name__ == '__main__':
    # base_url = 'https://www.ximalaya.com/category/a1001'
    # num_pages = 50

    # album_links = get_all_album_links(base_url, num_pages)
    # save_links_to_file(album_links, 'album_links.txt')
    album_links = read_links_from_file('album_links.txt')
    download_all_albums(album_links)