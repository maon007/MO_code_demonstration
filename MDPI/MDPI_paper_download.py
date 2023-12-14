import requests
import webbrowser
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
import time

class MDPIArticleScraper:
    def __init__(self, base_url, year_from, year_to, page_count, file_path, page):
        self.base_url = base_url
        self.year_from = year_from
        self.year_to = year_to
        self.page_count = page_count
        self.file_path = file_path
        self.page = page
        self.scan_urls_output = ""


    def extract_links_from_class(self, website_url):
        links_list = []
        # Send a GET request to the website
        response = requests.get(website_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all links with class "title-link"
            title_links = soup.find_all('a', class_='title-link')

            # Extract and store the href values of the links
            for link in title_links:
                href = link.get('href')
                if href:
                    full_link = 'https://www.mdpi.com/' + href
                    links_list.append(full_link)
                    print(full_link)
        else:
            print(f"Error: Could not retrieve the website. Response status code: {response.status_code}")

        return links_list


    def check_if_file_exists(self, file_path):
        return os.path.exists(file_path)


    def download_pdf_from_link(self, link, file_path):
        try:
            # Send a GET request to the link
            response = requests.get(link)

            # Check if the request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                pdf_link_element = soup.find('a', class_='UD_ArticlePDF')

                if pdf_link_element:
                    pdf_href = pdf_link_element.get('href')

                    if pdf_href:
                        # Extend the pdf_href with "https://www.mdpi.com" if it doesn't contain it already
                        if not pdf_href.startswith("https://www.mdpi.com"):
                            pdf_href = "https://www.mdpi.com" + pdf_href
                            print("pdf_href", pdf_href)

                        # Find the PDF file name from div class="bib-identity"
                        bib_identity_div = soup.find('div', class_='bib-identity')
                        if bib_identity_div:
                            bib_text = bib_identity_div.text.strip()
                            doi_start_index = bib_text.find("https://doi.org/")
                            if doi_start_index != -1:
                                doi_link = bib_text[doi_start_index + len("https://doi.org/"):].split()[0]
                                doi_link = doi_link.replace("/", '___')
                                pdf_file_name = doi_link
                            else:
                                doi_link = None
                                pdf_file_name = "unknown_pdf"

                            # Create a file path with the PDF file name for saving the PDF file
                            pdf_file_path = os.path.join(file_path, f"{pdf_file_name}.pdf")

                            # Check if the PDF file has already been downloaded
                            if not self.check_if_file_exists(pdf_file_path):
                                retry_count = 0
                                max_retries = 2
                                while retry_count < max_retries:
                                    pdf_response = requests.get(pdf_href)

                                    if pdf_response.status_code == 200:
                                        # Save the PDF content to the specified file path
                                        with open(pdf_file_path, 'wb') as f:
                                            f.write(pdf_response.content)
                                        print(f"PDF downloaded successfully and saved to: {pdf_file_path}")
                                        time.sleep(4)
                                        break  # Exit the loop if successful

                                    elif pdf_response.status_code == 429:
                                        print(f"Error: Too many requests (status code 429). Retrying in 15 minutes...")
                                        time.sleep(900)  # Wait for 10 minutes
                                        retry_count += 1
                                    else:
                                        print(f"Error: Could not download the PDF. Response status code: {pdf_response.status_code}")
                                        break  # Exit the loop if not successful
                            else:
                                print(f"PDF file already exists: {pdf_file_path}")

                            if doi_link:
                                print(f"DOI link found: {doi_link}")
                        else:
                            print(f"Error: 'bib-identity' div not found in the page: {link}")
                    else:
                        print(f"Error: PDF link not found in the page: {link}")
                else:
                    print(f"Error: Link with class 'UD_ArticlePDF' not found in the page: {link}")
            else:
                print(f"Error: Could not retrieve the page. Response status code: {response.status_code}")

        except Exception as e:
            print(f"Error while downloading PDF: {e}")


    def find_metadata_elements(self, link):
        try:
            # Send a GET request to the link
            response = requests.get(link)

            # Check if the request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Define the list of meta names to search for and their corresponding keys
                meta_mapping = {
                    "citation_doi": "doi",
                    "citation_abstract_html_url": "url",
                    "dc.date": "date",
                    "dc.publisher": "publisher",
                    "prism.volume": "volume",
                    "prism.number": "number",
                    "dc.creator": "author",
                    "dc.title": "title",
                    "citation_journal_title": "journal"
                }

                # Create a dictionary to store the metadata
                metadata_dict = {}
                dc_creator_list = []

                for name, key in meta_mapping.items():
                    meta_elements = soup.find_all('meta', attrs={'name': name})
                    if meta_elements:
                        for meta_element in meta_elements:
                            content = meta_element.get('content')
                            if name == "dc.creator":
                                dc_creator_list.append(content)
                            else:
                                metadata_dict[key] = content

                # Merge and add the dc.creator content to the metadata dictionary
                if dc_creator_list:
                    dc_creator_merged = ' and '.join(dc_creator_list)
                    metadata_dict["author"] = dc_creator_merged

                # Extract year and month from the "date" and add them after the "url" in the metadata dictionary
                date_content = metadata_dict.get("date", "")
                if date_content and len(date_content) >= 6:
                    year = date_content[:4]
                    month = date_content[5:7]
                    metadata_dict["year"] = year
                    metadata_dict["month"] = month

                # Remove the "date" key from the metadata dictionary
                metadata_dict.pop("date", None)
             #   print("metadata_dict: ", metadata_dict)
                return metadata_dict


            else:
                print(f"Error: Could not retrieve the page. Response status code: {response.status_code}")

        except Exception as e:
            print(f"Error while finding metadata elements: {e}")


    def generate_bib_id(self, doi):
        return doi.replace("/", "___")


    def write_bib_file(self, metadata_dict, filename):
        with open(filename, 'w') as bib_file:
            bib_file.write("@article{")
            bib_file.write("\n")

            for name, content in metadata_dict.items():
                bib_file.write(f"  {name} = {{{content}}},")
                bib_file.write("\n")

            bib_file.write("}")
            bib_file.write("\n")


    def scan_urls(self):
        while True:
            link = f"{self.base_url}&page_no={self.page}&page_count={self.page_count}&year_from={self.year_from}&year_to={self.year_to}&view=default"
            response = requests.get(link)

            if response.status_code == 200:
                links = self.extract_links_from_class(link)
                if not links:
                    break

                # Store page and year_from information in a text file
                info_text = f"Page: {self.page}, Year: {self.year_from}"
                with open(os.path.join(self.file_path, r"F:\MDPI_run\checkpoint.txt"), "w") as info_file:
                    info_file.write(info_text)

                print('Articles from year:', self.year_from, ' - Page:', self.page)
                print(f"The link {link} is valid")

                for link in links:
                    self.download_pdf_from_link(link=link, file_path=self.file_path)
                    metadata = self.find_metadata_elements(link=link)

                    if metadata:
                        bib_id = self.generate_bib_id(metadata.get("doi", "No_DOI"))
                        bib_filename = os.path.join(self.file_path, f"{bib_id}.bib")

                        with open(bib_filename, 'w', encoding='utf-8') as bib_file:
                            bib_file.write(f"@article{{{bib_id},")
                            bib_file.write("\n")

                            sorted_metadata = {k: metadata.get(k, "Not found") for k in ["doi", "url", "year", "month", "publisher", "volume", "number", "author", "title", "journal"]}

                            for key, value in sorted_metadata.items():
                                bib_file.write(f"  {key} = {{{value}}},")
                                bib_file.write("\n")

                            bib_file.write("}")
                            bib_file.write("\n")

                        print(f"Metadata saved to {bib_filename}")

                    print('-' * 100)

                self.page += 1
            else:
                print(f"The link {link} is not valid")
                self.year_from -= 1

                break


if __name__ == "__main__":
    base_url = "https://www.mdpi.com/search?sort=pubdate"
    year_from = 2017
    year_to = 2021
    page_count = 10
    file_path = r'F:\MDPI'
    page = 40080

    scraper = MDPIArticleScraper(base_url, year_from, year_to, page_count, file_path, page)
    scraper.scan_urls()
