# import libraries
import psycopg2
import pandas as pd
import requests
import os
import sqlite3
from datetime import datetime
import json
import subprocess
import argparse


def order_by_release_edit_date(df):
    """
    This function orders the rows in a pandas dataframe based on the date in the `release_edit_date` column.
    Returns: A pandas dataframe with the rows ordered by the date in the `release_edit_date` column.
    """
    return df.sort_values(by=['release_edit_date'], ascending=True)


# Function to create folders based on c_rev_publisher and journal
def create_folders(output_folder, doi):
    # Extract the folder name from the doi column (all before the first "/")
    folder_name = doi.split('/')[0]

    # Create the folder path
    folder_path = os.path.join(output_folder, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


# Function to download PDFs from URLs in a DataFrame with retry
def download_pdfs_with_retry(df, output_folder):
    max_retries = len(df)
    downloaded_rows = []  # Initialize a list to store rows with status
    successfully_downloaded = False  # Flag to track successful download

    for index, row in df.iterrows():
        url = row['url']
        doi = row['doi']

        # Check if doi is null
        if pd.isnull(doi):
            # Find the next available folder with a unique number
            folder_number = 0
            while True:
                doi_folder = os.path.join(output_folder, f"10.xxxx{folder_number}")
                if not os.path.exists(doi_folder):
                    os.makedirs(doi_folder)
                    break
                folder_number += 1
            file_name = os.path.join(doi_folder, f"{row['release_rev_id']}.pdf")
        else:
            # Use doi as the folder name and create folders if needed
            doi_folder = create_folders(output_folder, doi)
            file_name = os.path.join(doi_folder, f"{row['release_rev_id']}.pdf")

        try:
            # Send a GET request to the URL to download the PDF
            response = requests.get(url)

            if response.status_code == 200:
                # Save the PDF to the specified output folder
                with open(file_name, 'wb') as file:
                    file.write(response.content)
                print("PDF downloaded successfully.")
                print(f"Downloaded PDF: release_rev_id: {row['release_rev_id']}, title: {row['title']}")
                # Modify the DataFrame columns based on success
                row['downloaded'] = "YES"
                row['status'] = str(response.status_code) + ":" + response.reason

                downloaded_rows.append(row)  # Add the row to the list

                # Run pdftotext on the downloaded PDF
                try:
                    process = subprocess.run(['pdftotext', file_name], stdout=subprocess.PIPE)
                    if process.returncode == 0:
                        print("pdftotext completed successfully.")
                        row['txt_generated'] = "YES"
                    else:
                        print(f"pdftotext failed with return code {process.returncode}")
                        row['txt_generated'] = "NO"
                except Exception as e:
                    print(f"Error running pdftotext: {e}")

                successfully_downloaded = True  # Set the flag to True upon successful download
                break

            else:
                print(f"Failed to download: {row['release_rev_id']} (Status reason: {response.reason})")
                # Modify the DataFrame columns based on failure
                row['downloaded'] = "NO"
                row['status'] = str(response.status_code) + ":" + response.reason

                downloaded_rows.append(row)  # Add the row to the list

        except Exception as e:
            print(f"Error while downloading PDF: {e}")
            # Modify the DataFrame columns based on exception
            row['downloaded'] = "NO"
            row['status'] = str(e)

            downloaded_rows.append(row)  # Add the row to the list

        # Check the number of PDFs in the folder and create a new folder if needed
        if len(os.listdir(doi_folder)) >= 200000:
            folder_number += 1
            doi_folder = os.path.join(output_folder, f"10.xxxx{folder_number}")
            if not os.path.exists(doi_folder):
                os.makedirs(doi_folder)

    # Create a DataFrame containing all rows (including failures and errors)
    downloaded_df = pd.DataFrame(downloaded_rows)

    if successfully_downloaded:
        # If at least one PDF was successfully downloaded, return only that row
        return downloaded_df[downloaded_df['downloaded'] == "YES"].head(1)
    else:
        # If no PDFs were successfully downloaded, return all rows
        return downloaded_df


def aggregate_dataframe(df):
    # Check if all rows in the "downloaded" column are "NO"
    if all(df['downloaded'] == 'NO'):
        # Take the first row and aggregate URLs and statuses in JSON format
        urls = df['url'].tolist()
        statuses = df['status'].tolist()
        aggregated_status = json.dumps(dict(zip(urls, statuses)))
        aggregated_urls = ', '.join(urls)

        # Update the first row with the aggregated data
        df.loc[df.index[0], 'url'] = aggregated_urls
        df.loc[df.index[0], 'status'] = aggregated_status

        return df.head(1)
    else:
        # Filter rows with "downloaded" equal to "YES"
        yes_rows = df[df['downloaded'] == 'YES']
        if not yes_rows.empty:
            return yes_rows.head(1)
        else:
            # If there are no rows with "YES," return the entire DataFrame
            return df


def add_month_column(df):
    # Convert release_date to datetime format
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Extract month and create a new column 'month'
    df['month'] = df['release_date'].apply(lambda x: datetime.strftime(x, "%B") if not pd.isna(x) else '')

    return df



def generate_bibtex_entries(df, output_folder):
    try:
        bib_entries = []

        for index, row in df.iterrows():
            pdf_url = row['url']
            doi = row['doi']
            file_name = f"{row['release_rev_id']}"

            # Check if 'doi' and 'title' are both NULL
            if (pd.isna(row['doi']) and pd.isna(row['title'])) or (isinstance(row['url'], str) and ',' in row['url']):
                df.at[index, 'bib_generated'] = 'NO'
                continue

            bib_entry = f"@article{{{file_name},\n"

            # Check and replace NULL values for various entries
            bib_entry += f"  doi = {{{row['doi'] if pd.notna(row['doi']) else ''}}},\n"
            bib_entry += f"  url = {{{row['url']}}},\n"
            bib_entry += f"  month = {{{row['month'] if pd.notna(row['month']) else ''}}},\n"
            bib_entry += f"  year = {{{row['release_year'] if pd.notna(row['release_year']) else ''}}},\n"
            bib_entry += f"  publisher = {{{row['c_rev_publisher'] if pd.notna(row['c_rev_publisher']) else ''}}},\n"
            bib_entry += f"  volume = {{{row['volume'] if pd.notna(row['volume']) else ''}}},\n"
            bib_entry += f"  number = {{{row['number'] if pd.notna(row['number']) else ''}}},\n"
            bib_entry += f"  pages = {{{row['pages'] if pd.notna(row['pages']) else ''}}},\n"

            # Check if 'authors' is not null
            if pd.notna(row['authors']):
                bib_entry += f"  author = {{{row['authors']}}},\n"
            elif pd.notna(row['editors']):
                # If 'authors' is null and 'editors' is not null, add 'editor' entry
                bib_entry += f"  editor = {{{row['editors']}}},\n"
            else:
                # If both 'authors' and 'editors' are null, add 'author' with an empty string
                bib_entry += f"  author = {{}},\n"

            bib_entry += f"  title = {{{row['title'] if pd.notna(row['title']) else ''}}},\n"
            bib_entry += f"  journal = {{{row['journal'] if pd.notna(row['journal']) else ''}}},\n"
            bib_entry += "}\n"
            bib_entries.append(bib_entry)

            # Mark that a Bib-file was successfully generated
            df.at[index, 'bib_generated'] = 'YES'

        if bib_entries:
            # Combine all BibTeX entries into a single string
            bibtex_string = '\n'.join(bib_entries)

            # Check if 'doi' is null and use the same folder as the PDF
            if pd.isna(doi):
                folder_number = 0
                while True:
                    doi_folder = os.path.join(output_folder, f"10.xxxx{folder_number}")
                    if not os.path.exists(doi_folder):
                        os.makedirs(doi_folder)
                        break
                    folder_number += 1
            else:
                # doi_folder = os.path.join(output_folder, doi)
                # if not os.path.exists(doi_folder):
                #     os.makedirs(doi_folder)

                doi_folder = create_folders(output_folder, doi)

            # Save the BibTeX entries to a .bib file in the folder
            bib_file_path = os.path.join(doi_folder, f"{file_name}.bib")
            with open(bib_file_path, 'w', encoding='utf-8') as bib_file:
                bib_file.write(bibtex_string)

            print(f"BibTeX entries saved to {bib_file_path}")

    except Exception as e:
        print(f"Error: {e}")
        # Mark that a Bib-file could not be generated due to an error
        df.at[index, 'bib_generated'] = 'NO'

    return df


def processing_date(df):
    # Get the current timestamp
    processing_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Add the processing_date to the DataFrame
    df['processing_date'] = processing_date

    return df


# Define the main function
def process_and_store_data(df):
    try:
        # Run functions on the input DataFrame
        df = order_by_release_edit_date(df=df)
        df = download_pdfs_with_retry(df=df, output_folder=r'S:\Fatcat_papers')
        df = aggregate_dataframe(df=df)
        df = add_month_column(df=df)
        df = generate_bibtex_entries(df=df, output_folder=r'S:\Fatcat_papers')
        df = processing_date(df=df)

    except Exception as e:
        print(f"Error processing and storing data: {e}")
    return df


# Specify the full path to the text file
full_path_to_file = r'S:\processed_release_rev_ids.txt'


def connect_to_postgres(user, password, host, port, database, table_name, processed_tbl_name, sqlite_db_path, filter,
                        filter_values):
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)
        cursor = conn.cursor()

        # Create the "processed_papers" table if it doesn't exist
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {processed_tbl_name}  (
            id SERIAL PRIMARY KEY,
            release_rev_id UUID,
            doi TEXT,
            url TEXT,
            release_year bigint,
            release_date date,
            c_rev_publisher TEXT,
            rev_publisher TEXT,
            journal TEXT,
            volume TEXT,
            number TEXT,
            pages TEXT,
            authors TEXT,
            editors TEXT,
            title TEXT,
            release_edit_date timestamp with time zone,
            downloaded TEXT,
            status TEXT,
            month TEXT,
            bib_generated TEXT,
            processing_date timestamp with time zone
        );
        """
        cursor.execute(create_table_query)
        conn.commit()

        # Initialize the SQLite connection outside the loop
        conn_sqlite = sqlite3.connect(sqlite_db_path)

        # Get the table data based on the filter conditions
        print(f"Loading data from {table_name} where publisher is: {filter_values}")
        query = f"SELECT * FROM {table_name} WHERE {filter} in ('{filter_values}');"
        cursor.execute(query)
        rows = cursor.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cursor.description]

        # Create a Pandas DataFrame from the fetched rows
        table_data = pd.DataFrame(rows, columns=column_names)
        # print(table_data.head(5))

        # Get a list of distinct release_rev_id values
        release_rev_id_values = table_data['release_rev_id'].unique()

        # Load processed release_rev_id values from the text file
        processed_rev_ids = set()
        with open(full_path_to_file, "r") as file:
            for line in file:
                processed_rev_ids.add(line.strip())

        # Initialize a counter for the total number of processed papers
        processed_papers_count = 0

        # Loop through distinct release_rev_id values
        for release_rev_id in release_rev_id_values:
            print("-" * 80)

            # Check if release_rev_id is already in the processed set
            if release_rev_id in processed_rev_ids:
                print(f"Skipping release_rev_id {release_rev_id} as it is already processed and in the text file.")
                continue

            processed_papers_count += 1
            # Perform some processing for each iteration
            print(f"Iteration {processed_papers_count}: Processing release_rev_id: {release_rev_id}")

            # Filter data for the current release_rev_id
            current_data = table_data[table_data['release_rev_id'] == release_rev_id]

            # Process the data for the current release_rev_id
            df = process_and_store_data(df=current_data)
            print("Processed release_rev_id: ", release_rev_id)

            # Insert the processed data into the "processed_papers" table in SQLite
            df.to_sql(name=processed_tbl_name, con=conn_sqlite, if_exists='append', index=False)

            # Add the processed release_rev_id to the set and write it to the text file
            processed_rev_ids.add(release_rev_id)
            with open(full_path_to_file, "a") as file:
                file.write(release_rev_id + "\n")


        # Print the final number of iterations
        print(f"Total number of iterations: {processed_papers_count}")

        # Close the database connections
        cursor.close()
        conn.close()
        conn_sqlite.close()  # Close the SQLite connection after all processing is done

        return

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect to PostgreSQL with custom filter values")

    parser.add_argument("--filter_values", required=True, help="Filter values")

    args = parser.parse_args()

    connect_to_postgres(
        database="fatcat",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432",
        table_name="fatcat_bmt",
        processed_tbl_name="fatcat_processed_papers",
        sqlite_db_path="F:\\fatcat.db",
        filter="rev_publisher",
        filter_values=args.filter_values
    )

