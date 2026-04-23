from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import random
import re
import pandas as pd
import os

# ==========================================
# 1. Browser Configuration
# ==========================================
chrome_options = Options()
chrome_options.add_argument("--headless") 
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

keywords = [
    "Backend", "Frontend", "Data Analyst", "Machine Learning", 
    "Software Engineer", "DevOps", "Full Stack", "Data Engineer", "Python"
]
# تم التعديل لـ 50 صفحة لجمع داتا ضخمة
pages_per_keyword = 50 

csv_filename = "wuzzuf_big_data_complete.csv"

# ==========================================
# 2. Resumable Scraping Logic
# ==========================================
# 🌟 السطر ده اللي كان ممسوح عندك وعمل الإيرور 🌟
scraped_urls = set()

if os.path.exists(csv_filename):
    try:
        df_existing = pd.read_csv(csv_filename)
        scraped_urls = set(df_existing['URL'].tolist())
        print(f"📌 Found existing database with {len(scraped_urls)} jobs. These will be skipped.")
    except Exception as e:
        print("Existing file found but it might be empty. Starting fresh.")

# ==========================================
# 3. Phase 1: Fast Link Collection
# ==========================================
all_new_links = []
print("\n--- Phase 1: Collecting Links ---")

for keyword in keywords:
    print(f"\nSearching for: '{keyword}'")
    for page in range(pages_per_keyword):
        url = f"https://wuzzuf.net/search/jobs/?q={keyword}&start={page}"
        try:
            driver.get(url)
            time.sleep(random.uniform(1.5, 3)) 
            
            elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/jobs/p/']")
            links = [el.get_attribute("href") for el in elements]
            
            if not links:
                print(f"⏩ Reached the end of results for '{keyword}' at page {page}.")
                break 
                
            all_new_links.extend(links)
            
        except Exception as e:
            print(f"Error extracting links from search page: {e}")

all_new_links = list(set(all_new_links))

# الفلترة الذكية (هنا كان بيحصل الإيرور لو scraped_urls مش متعرفة)
all_new_links = [link for link in all_new_links if link not in scraped_urls]

print(f"\n✅ Total NEW unique jobs to scrape deeply: {len(all_new_links)}")

if len(all_new_links) == 0:
    print("🎉 No new jobs found! The database is fully up-to-date. Closing script.")
    driver.quit()
    exit()

# ==========================================
# 4. Phase 2: Deep Data Extraction
# ==========================================
print("\n--- Phase 2: Deep Data Extraction ---")

if not os.path.exists(csv_filename) or len(scraped_urls) == 0:
    df_empty = pd.DataFrame(columns=["Title", "Company", "Location", "Salary", "Experience", "Job_Type", "Skills", "URL"])
    df_empty.to_csv(csv_filename, index=False, encoding="utf-8-sig")

for index, link in enumerate(all_new_links):
    try:
        driver.get(link)
        time.sleep(random.uniform(2, 3.5)) 
        
        body_text = driver.find_element(By.TAG_NAME, "body").text
        page_title = driver.title 
        
        title, company, location = "N/A", "N/A", "N/A"
        try:
            if " job at " in page_title:
                parts = page_title.split(" job at ")
                title = parts[0].strip()
                if " in " in parts[1]:
                    company_loc = parts[1].split(" in ")
                    company = company_loc[0].strip()
                    location = company_loc[1].split(" – ")[0].split(" - ")[0].strip()
                else:
                    company = parts[1].split(" – ")[0].split(" - ")[0].strip()
            else:
                title = page_title.split('|')[0].strip()
        except:
            title = page_title.split('|')[0].strip()

        salary_match = re.search(r'(\d[\d,]*\s*(?:to|-)?\s*\d[\d,]*\s*(?:EGP|USD|\$))', body_text, re.IGNORECASE)
        salary = salary_match.group(0) if salary_match else "Confidential"
        
        exp_match = re.search(r'Experience Needed:\s*(.*?)(?:\n|$)', body_text)
        experience = exp_match.group(1).strip() if exp_match else "N/A"
        
        type_match = re.search(r'Job Type:\s*(.*?)(?:\n|$)', body_text)
        job_type = type_match.group(1).strip() if type_match else "N/A"
        
        skills_match = re.search(r'Skills And Tools:\s*(.*?)(?:\nJob Description|\n|$)', body_text, re.DOTALL)
        skills = skills_match.group(1).replace('\n', ' - ').strip() if skills_match else "N/A"

        job_data = {
            "Title": title, "Company": company, "Location": location, 
            "Salary": salary, "Experience": experience, "Job_Type": job_type, 
            "Skills": skills, "URL": link
        }
        
        pd.DataFrame([job_data]).to_csv(csv_filename, mode='a', header=False, index=False, encoding="utf-8-sig")
        
        print(f"[{index+1}/{len(all_new_links)}] Saved: {title[:15]}... | Sal: {salary}")
        
    except Exception as e:
        print(f"❌ Failed to extract data at link {index+1}: {e}")

driver.quit()
print(f"\n🎉 Mission Accomplished! All new data appended successfully to {csv_filename}")