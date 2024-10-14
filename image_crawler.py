# image_crawler.py
from icrawler.builtin import GoogleImageCrawler
import os

class ImageCrawler:
    def __init__(self, storage_root='food_images'):
        self.storage_root = storage_root
        if not os.path.exists(self.storage_root):
            os.makedirs(self.storage_root)
    
    def crawl_images(self, keyword, max_num=20):
        # Create a subdirectory for each keyword
        storage_dir = os.path.join(self.storage_root, keyword.replace(' ', '_'))
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        else:
            # Skip if images already exist
            print(f"Images for '{keyword}' already exist. Skipping crawling.")
            return storage_dir
        
        google_crawler = GoogleImageCrawler(storage={'root_dir': storage_dir})
        google_crawler.crawl(keyword=keyword, max_num=max_num)
        return storage_dir

