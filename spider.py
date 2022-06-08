import scrapy
import time
import json


class SchoolSpider(scrapy.Spider):
    name = 'schoolspider'
    start_urls = ['https://www.greatschools.org/california/schools?page={}&view=table'.format(page) for page in
                  range(1, 240)]

    def parse(self, response):
        schools = json.loads(response.css('script::text').get().split(';')[20].replace('gon.search=', ''))['schools']
        for school in schools:
            yield {'Name': school['name'],
                   'City': school['districtCity'],
                   'Latitude': school['lat'],
                   'Longitude': school['lon'],
                   'Address': school['zipcode'],
                   'Rating': school['rating'],
                   'SchoolType': school['schoolType']}
    # scrapy runspider /Users/irina/PycharmProjects/pythonFinalProject/spider.py -O table.csv