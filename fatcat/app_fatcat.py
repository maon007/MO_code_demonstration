import sqlite3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to count the rows in the table
def count_rows_in_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute an SQL query to count the rows in the table
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")

    # Fetch the result
    count = cursor.fetchone()[0]
    conn.close()
    return count

# Function to count the number of PDFs that weren't successfully downloaded
def count_non_downloaded(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute an SQL query to count the rows in the table
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE downloaded = 'NO'")

    # Fetch the result
    count = cursor.fetchone()[0]
    conn.close()
    return count

# Function to count the number of PDFs that weren't successfully downloaded
def count_non_bib(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute an SQL query to count the rows in the table
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE bib_generated = 'NO'")

    # Fetch the result
    count = cursor.fetchone()[0]
    conn.close()
    return count

# Function to fetch and analyze data
def fetch_and_analyze_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT downloaded, bib_generated, release_year, rev_publisher, journal, month FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df

# Streamlit UI
st.title("Fatcat Table Statistics")

# Define the database file path
db_path = r'F:\fatcat.db'  # Use the 'r' prefix to treat it as a raw string

table_name = "fatcat_processed_papers"

# Process the database and count rows
try:
    count = count_rows_in_table(db_path, table_name)
    no_downloaded = count_non_downloaded(db_path, table_name)
    no_bib = count_non_bib(db_path, table_name)
    st.subheader(f"Total number of processed papers in {table_name}: {count}")
    st.caption(f"PDF wasn't downloaded for this no. of papers: {no_downloaded}")
    st.caption(f"Bib-file wasn't generated for this no. of papers: {no_bib}")



except Exception as e:
    st.write(f"Error: {e}")

# Fetch and analyze data
data = fetch_and_analyze_data(db_path, table_name)

# Create two subplots for the ratios
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot "downloaded" column ratio
downloaded_ratio = data['downloaded'].value_counts()
ax1.pie(downloaded_ratio, labels=downloaded_ratio.index, autopct='%1.1f%%', startangle=90)
ax1.set_title('Downloaded')

# Plot "bib_generated" column ratio
bib_generated_ratio = data['bib_generated'].value_counts()
ax2.pie(bib_generated_ratio, labels=bib_generated_ratio.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('Bib Generated')

# Display the pie chart
st.pyplot(fig)

# Additional statistics
# Create a layout with two columns for released years and months
col1, col2 = st.columns(2)

with col1:
    st.subheader("No of papers released each year")
    # Filter the data for release years greater than 1990
    filtered_data = data[data['release_year'] > 1990]
    # Group the data by 'release_year' and 'downloaded', and count the occurrences
    stacked_data_year = filtered_data.groupby(['release_year', 'downloaded']).size().unstack().fillna(0)
    # Switch the order of columns to have "Downloaded" first and "Not Downloaded" second
    st.bar_chart(stacked_data_year)

# Plot the top ten journals in the fourth column
with col2:
    st.subheader("No of papers released each month")
    # Exclude empty strings from release_month_counts
    filtered_data = filtered_data[filtered_data['month'] != '']
    stacked_data_month = filtered_data.groupby(['month', 'downloaded']).size().unstack().fillna(0)
    # st.bar_chart(stacked_data_month)
    st.bar_chart(stacked_data_month)

# Calculate the number of unique publishers
unique_publishers = data['rev_publisher'].nunique()

# Calculate the number of unique publishers
unique_journals = data['journal'].nunique()


# Create a layout with two columns for the top publishers and journals
col3, col4 = st.columns(2)

# Plot the first ten publishers with the highest values in the third column
with col3:
    st.subheader("Top 10 Publishers")
    st.write(f"Number of Unique Publishers: {unique_publishers}")
    top_publishers = data['rev_publisher'].value_counts().sort_values(ascending=False).head(10)
    st.bar_chart(top_publishers)

# Plot the top ten journals in the fourth column
with col4:
    st.subheader("Top 10 Journals")
    st.write(f"Number of Unique Journals: {unique_journals}")
    top_journals = data['journal'].value_counts().head(10)
    st.bar_chart(top_journals)
