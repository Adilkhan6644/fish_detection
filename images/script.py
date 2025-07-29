from icrawler.builtin import BingImageCrawler
# :magnifying_glass: Set your search keyword here
search_term = "Mackerel Tuna a popular Australian fish, with the scientific name Euthynnus affinis"  # CHANGE this to anything you want
# :file_folder: Folder where images will be saved
output_dir = ".Mackerel Tuna"  # CHANGE this to your desired folder name
# :camera_with_flash: Create crawler and start downloading
crawler = BingImageCrawler(storage={"root_dir": output_dir})
crawler.crawl(keyword=search_term, max_num=100)
print(f":white_tick: Download complete! Check the folder: {output_dir}")