import scrapy
from scrapy.selector import Selector
import html2text
import os

BASE_DIR = '/home/ani/Projects/xianxiabot/stories/'
class XianxiaSpider(scrapy.Spider):
    name = "xianxia"
    
    website = 'http://www.wuxiaworld.com/'
    start_urls=[website]
    output_ext = '.txt'

    def clean_text(self,text):

        text = text.replace("\n"," ")
        text = text.encode('ascii', 'ignore').decode('unicode_escape', 'ignore')
        return text

    
    def parse_chapter(self,response):
        meta = response.request.meta
        name = meta["name"]
        directory = meta["dir"]
        os.chdir(directory)
        hxs = Selector(response)
        converter = html2text.HTML2Text()
        converter.ignore_links = True
        article = hxs.xpath("//div[@itemprop='articleBody']/p").extract()
        title = converter.handle(article.pop(0))
        text = ''
        for section in article:
            text += self.clean_text(converter.handle(section))
        with open(title + self.output_ext, 'w+') as f:
            f.write(text)

    def parse_story(self, response):
        name = response.url[len(self.website):-len('-index/')]
        directory = BASE_DIR + name
        if not os.path.exists(directory):
            os.makedirs(directory)
        links = response.xpath("//div[@class='entry-content']//a/@href").extract()
        chapter_links = [link for link in links if 'chapter' in link]
        print(chapter_links[0], chapter_links[-1])
        for ch in chapter_links:
            yield scrapy.Request(ch, callback=self.parse_chapter, 
                    meta = {"name" : name, "dir" : directory}) 
        


    def parse(self,response):
        for url in response.xpath("//nav[@id='site-navigation']//a/@href").extract():
            if 'index' in url:
                
                yield scrapy.Request(url, callback=self.parse_story)
