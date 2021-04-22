from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
#specify driver path
DRIVER_PATH = '/opt/homebrew/bin/chromedriver'


driver = webdriver.Chrome(executable_path = DRIVER_PATH)

driver.get('https://indeed.com')


initial_search_button = driver.find_element_by_xpath('//*[@id="whatWhereFormId"]/div[3]/button')
initial_search_button.click()

close_popup = driver.find_element_by_id('popover-close-link')
close_popup.click()

advanced_search = driver.find_element_by_xpath('//*[@id="jobsearch"]/table/tbody/tr/td[4]/div/a')
advanced_search.click()

#search data science
search_job = driver.find_element_by_xpath('//input[@id="as_and"]')
search_job.send_keys(['data science'])
#set display limit of 30 results per page
display_limit = driver.find_element_by_xpath('//select[@id="limit"]//option[@value="30"]')
display_limit.click()
#sort by date
sort_option = driver.find_element_by_xpath('//select[@id="sort"]//option[@value="date"]')
sort_option.click()
search_button = driver.find_element_by_xpath('//*[@id="fj"]')
search_button.click()

# let the driver wait 3 seconds to locate the element before exiting out
driver.implicitly_wait(3)

titles = []
companies = []
locations = []
links = []
reviews = []
salaries = []
descriptions = []

for i in range(0, 20):

    job_card = driver.find_elements_by_xpath('//div[contains(@class,"clickcard")]')

    for job in job_card:

        # .  not all companies have review
        try:
            review = job.find_element_by_xpath('.//span[@class="ratingsContent"]').text
        except:
            review = "None"
        reviews.append(review)
        # .   not all positions have salary
        try:
            salary = job.find_element_by_xpath('.//span[@class="salaryText"]').text
        except:
            salary = "None"
        # .  tells only to look at the element
        salaries.append(salary)

        try:
            location = job.find_element_by_xpath('.//span[contains(@class,"location")]').text
        except:
            location = "None"
        # .  tells only to look at the element
        locations.append(location)

        try:
            title = job.find_element_by_xpath('.//h2[@class="title"]//a').text
        except:
            title = job.find_element_by_xpath('.//h2[@class="title"]//a').get_attribute(name="title")
        titles.append(title)
        links.append(job.find_element_by_xpath('.//h2[@class="title"]//a').get_attribute(name="href"))
        companies.append(job.find_element_by_xpath('.//span[@class="company"]').text)

    try:
        next_page = driver.find_element_by_xpath('//a[@aria-label={}]//span[@class="pn"]'.format(i + 2))
        next_page.click()

    except:
        next_page = driver.find_element_by_xpath('//a[@aria-label="Next"]//span[@class="np"]')
        next_page.click()
    # except:
    # next_page = driver.find_element_by_xpath('//a[.//span[contains(text(),"Next")]]')
    # next_page.click()

    print("Page: {}".format(str(i + 2)))

descriptions = []
for link in links:
    driver.get(link)
    jd = driver.find_element_by_xpath('//div[@id="jobDescriptionText"]').text
    descriptions.append(jd)

df_da=pd.DataFrame()
df_da['Title']=titles
df_da['Company']=companies
df_da['Location']="Palo Alto, CA"
df_da['Link']=links
df_da['Review']=reviews
df_da['Salary']=salaries
df_da['Description']=descriptions