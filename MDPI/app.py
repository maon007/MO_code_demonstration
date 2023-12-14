import streamlit as st
from MDPI_paper_download import MDPIArticleScraper
import sys

# Redirect the print output to Streamlit
def st_stdout_redirector(text):
    st.text(text)

sys.stdout.write = st_stdout_redirector

# Define the Streamlit app
def main():
    st.title("MDPI Article Scraper")

    # Sidebar inputs
    st.sidebar.header("Scraper Configuration")
    base_url = st.sidebar.text_input("Base URL", "https://www.mdpi.com/search?sort=pubdate")
    year_from = st.sidebar.number_input("Year From", min_value=1990, max_value=2023, value=2017)
    year_to = st.sidebar.number_input("Year To", min_value=1990, max_value=2023, value=2021)
    file_path = st.sidebar.text_input("File Path", "S:/MDPI_from_48437")
    page_count = st.sidebar.selectbox("Page Count", [10, 50, 100, 200])
    page = int(st.sidebar.text_input("Starting Page", "1"))

    # Create an instance of your scraper
    scraper = MDPIArticleScraper(base_url, year_from, year_to, page_count, file_path, page)

    # Add a button to start scraping
    if st.sidebar.button("Start Scraping"):
        with st.spinner("Scraping in progress..."):
            # Call your scan_urls method
            scraper.scan_urls()

        # Display the output of scan_urls with scrolling
        st.subheader("Scan URLs Output:")
        st.text_area(
            "Output Log",
            scraper.scan_urls_output,
            height=400,  # Set the height parameter for scrolling
            key="output_log",
        )  # Enable scrolling

# Call the main function directly
if __name__ == "__main__":
    main()
