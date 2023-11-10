import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
from webdriver_manager.chrome import ChromeDriverManager
service = Service(executable_path=ChromeDriverManager().install())

op = webdriver.ChromeOptions()
# op.add_argument('headless')
op.add_argument("--log-level=3")
driver = webdriver.Chrome(options=op,service=service)

websites = [
    "https://zh.player.fm/series/fm-59854",
    "https://zh.player.fm/series/series-1288180",
    "https://zh.player.fm/series/series-1952287",
    "https://zh.player.fm/series/re-men-shu-ji-jie-du",
    "https://zh.player.fm/series/xue-qiu-cai-jing-you-shen-du",
    "https://zh.player.fm/series/series-1540278",
    "https://zh.player.fm/series/2435157",
    "https://zh.player.fm/series/gu-shi-fm-1496859"

]


urls_file = "urls.txt"
for website in websites:
    driver.get(website)

    scrolls = 50
    for _ in range(scrolls):
        body = driver.find_element(By.TAG_NAME,'html')
        body.send_keys(Keys.END)
        time.sleep(2)
        body.send_keys(Keys.PAGE_UP)  # 模拟按下“Page Up”键，将页面稍微向上滑动
        time.sleep(1)  # 等待一段时间，确保页面加载完成

    html_content = driver.page_source

    soup = BeautifulSoup(html_content, "html.parser")

    target_tags = soup.select('a[href$=".m4a"]')
    # audio_links = soup.find_all("audio")
    i = 0
    for tag in target_tags:
        i = 1-i
        if i==0:
            continue
        audio_url = tag['href']
        with open(urls_file, "a") as file:
            file.write(audio_url + "\n")

driver.quit()
