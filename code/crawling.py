"""
crawling.py
-----------
네이버 이미지 검색 결과를 스크롤하며 지정한 키워드의 사진을 수집하는 크롤러.
연예인마다 --keyword 인자를 바꿔가며 실행해서 data/raw/{키워드} 폴더에 원본 이미지를 모은다.
수집된 이미지는 이후 수동 검수를 거쳐 data/{연예인}/{neutral,smile}로 분류된다.
"""
import argparse
import os
import time

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

from config import RAW_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="네이버 이미지 검색 크롤러")
    parser.add_argument("--keyword", type=str, required=True, help="검색할 키워드 (예: 아이유)")
    parser.add_argument("--target_count", type=int, default=500, help="목표 수집 장수")
    parser.add_argument("--scroll_count", type=int, default=30, help="이미지 추가 로딩을 위한 스크롤 횟수")
    parser.add_argument("--save_dir", type=str, default=None, help="저장 경로 (미지정 시 data/raw/{keyword})")
    return parser.parse_args()


def collect_images(keyword, target_count, scroll_count, save_dir):
    """네이버 이미지 검색에서 keyword로 target_count장까지 이미지를 수집해 save_dir에 저장."""
    os.makedirs(save_dir, exist_ok=True)

    driver = webdriver.Chrome()
    driver.implicitly_wait(5)

    try:
        url = f"https://search.naver.com/search.naver?where=image&query={keyword}"
        driver.get(url)
        time.sleep(2)

        print(f"{keyword} 사진 수집을 위한 스크롤 시작")
        for _ in range(scroll_count):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1.5)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        imgs = soup.select(".image_wrap img[src]")
        # dict.fromkeys로 중복을 제거하면서 등장 순서는 그대로 유지
        img_urls = list(dict.fromkeys(img["src"] for img in imgs if not img["src"].startswith("data")))
        print(f"총 {len(img_urls)}개의 이미지 주소 확보")
    finally:
        driver.quit()

    headers = {"User-Agent": "Mozilla/5.0"}
    success_count = 0

    for url in img_urls:
        if success_count >= target_count:
            break

        file_name = f"{keyword}_{success_count + 1:03}.jpg"
        path = os.path.join(save_dir, file_name)

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                with open(path, "wb") as f:
                    f.write(response.content)
                success_count += 1
                if success_count % 50 == 0:
                    print(f"{success_count}장 저장 완료")
        except Exception as e:
            print(f"다운로드 실패 ({url}): {e}")
            continue

    print(f"최종 {success_count}장 저장 완료")


if __name__ == "__main__":
    args = parse_args()
    target_save_dir = args.save_dir or os.path.join(RAW_DIR, args.keyword)
    collect_images(args.keyword, args.target_count, args.scroll_count, target_save_dir)
