# import libraries
import scispacy
import spacy
import pandas as pd
import os
import datetime
import sqlite3
import inflect
import torch
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity
import re
from datetime import datetime
import argparse

# Load the NER spacy models
ner_bi = spacy.load("en_ner_bionlp13cg_md")
ner_bc = spacy.load("en_ner_bc5cdr_md")
ner_jn = spacy.load("en_ner_jnlpba_md")
ner_list=[ner_bi, ner_bc, ner_jn]

# Load the Biobert tokenizer and model
biobert_tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1")
biobert_model = BertModel.from_pretrained("dmis-lab/biobert-large-cased-v1.1")


def extract_text_from_file(file_path):
    start_keywords = ['Abstract', 'a b s t r a c t',
                 'This file is available to download for the purposes of text mining, consistent with the principles of UK copyright law.']
    end_keywords = ['Introduction', 'INTRODUCTION', 'Methods', 'Objectives', 'Keywords', 'ABBREVIATIONS', 'Results',
               'Method', 'AIM']
    try:
        # Open and read the text file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        relevant_texts = []
        for start_keyword in start_keywords:
            # Find the start index using the start_keyword
            start_index = text.find(start_keyword)
            if start_index != -1:
                # Find the first occurrence of any of the end keywords after the start index
                end_index = len(text)
                for keyword in end_keywords:
                    current_index = text.find(keyword, start_index)
                    if current_index != -1:
                        end_index = min(end_index, current_index)

                # Extract the relevant text and add to the list
                relevant_text = text[start_index:end_index].strip()
                relevant_texts.append(relevant_text)

        return relevant_texts

    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return []
        
        
# Global variables for BioBERT tokenizer and model
def tokenize_text(text, biobert_tokenizer, biobert_model):
    inputs = biobert_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', 
                                           max_length=512, truncation=True)
    embeddings = biobert_model(**inputs).last_hidden_state[0]
    return inputs, embeddings


def extract_keywords_and_categories(text, ner_model):
    kwords = []
    categories = []
    # Process the text with the NER model
    docx = ner_model(text)
    # Print the named entities along with their labels and types
    for ent in docx.ents:
        kwords.append(ent.text)
        categories.append(ent.label_)
    return kwords, categories


def collect_keyword_data(kwords, categories):
    keyword_data = []
    for keyword_spacy, category in zip(kwords, categories):
        tokens_kws_bert = biobert_tokenizer.tokenize(keyword_spacy)
        tokens_ids_kws_bert = biobert_tokenizer.convert_tokens_to_ids(tokens_kws_bert)
        keyword_data.append((keyword_spacy, category, tokens_kws_bert, tokens_ids_kws_bert))
    return keyword_data


def calculate_keyword_embeddings(inputs, embeddings, keyword_data):
    keyword_embeddings = {}
    i = 0
    for keyword_spacy, category, tokens_kws_bert, tokens_ids_kws_bert in keyword_data:
        while i < len(inputs["input_ids"][0]):
            if inputs["input_ids"][0][i:i + len(tokens_ids_kws_bert)].tolist() == tokens_ids_kws_bert:
                start_pos = i
                end_pos = i + len(tokens_ids_kws_bert) - 1
                token_embeddings = embeddings[start_pos:end_pos + 1].mean(0)
                if keyword_spacy not in keyword_embeddings:
                    keyword_embeddings[keyword_spacy] = []
                keyword_embeddings[keyword_spacy].append(token_embeddings)
                i += len(tokens_ids_kws_bert)
                break
            else:
                i += 1
    for keyword, embedding_list in keyword_embeddings.items():
        keyword_embeddings[keyword] = torch.stack(embedding_list, 0).mean(0)
    return keyword_embeddings


# Global variable to keep track of the current group number
current_group_number = 0

def create_keyword_groups(keyword_embeddings, cosine_similarity_threshold):
    global current_group_number
    keyword_groups = []
    assigned_keywords = set()

    for keyword1, embedding1 in keyword_embeddings.items():
        if keyword1 in assigned_keywords:
            continue
        current_group_number += 1
        current_group = [(keyword1, 1.0, current_group_number)]  # Pairing the keyword with itself and a similarity of 1.0
        assigned_keywords.add(keyword1)

        for keyword2, embedding2 in keyword_embeddings.items():
            if keyword2 in assigned_keywords:
                continue
            similarity = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
            if similarity.item() > cosine_similarity_threshold:
                current_group.append((keyword2, similarity.item(), current_group_number))
                assigned_keywords.add(keyword2)

        keyword_groups.append(current_group)

    return keyword_groups


def prepare_data_for_dataframe(keyword_groups, keyword_data):
    data = []
    for group in keyword_groups:
        for keyword, similarity, group_number in group:
            data.append({
                'name': keyword,
                'group': f"Group {group_number}",
                'Cosine similarity': similarity,
                'category': next(item[1] for item in keyword_data if item[0] == keyword)
            })
    return data


def process_text_file(text, ner_list):
    all_dfs = []  # List to store DataFrames from each ner_list iteration
    cosine_similarity_threshold = 0.90

    inputs_and_embeddings = tokenize_text(text, biobert_tokenizer, biobert_model)
    inputs, embeddings = inputs_and_embeddings
    
    for ner_model in ner_list:
        kwords, categories = extract_keywords_and_categories(text, ner_model)
        keyword_data = collect_keyword_data(kwords, categories)
        keyword_embeddings = calculate_keyword_embeddings(inputs, embeddings, keyword_data)
        keyword_groups = create_keyword_groups(keyword_embeddings, cosine_similarity_threshold)
        data = prepare_data_for_dataframe(keyword_groups, keyword_data)
        df = pd.DataFrame(data)
        all_dfs.append(df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df


def pluralize_and_add_nlp_source(df):
    # Check if the DataFrame is empty
    if df.empty:
        return pd.DataFrame()

    # Initialize an empty list to store the updated data
    updated_data = []

    # Initialize the inflect engine
    p = inflect.engine()

    # Function to check if a word is plural or singular
    def is_plural(word):
        return p.singular_noun(word) is False

    # Function to create plurals from singulars and vice versa
    def create_plural_singular(word):
        try:
            return p.plural(word) if is_plural(word) else p.singular_noun(word)
        except:
            return word
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the original keyword, its group, and category
        original_keyword = row['name']
        group = row['group']
        category = row['category']

        # Get the singular and plural forms
        singular_form = create_plural_singular(original_keyword)
        plural_form = create_plural_singular(singular_form)

        # Add the original term to the DataFrame with 'BioBert' as the NLP source
        updated_data.append({
            'name': original_keyword,
            'group': group,
            'Cosine similarity': row['Cosine similarity'],
            'NLP source': 'BioBert',
            'category': category  # Add the category to the updated data
        })

        # Add the singular form to the DataFrame with 'Inflect library' as the NLP source
        updated_data.append({
            'name': singular_form,
            'group': group,
            'Cosine similarity': row['Cosine similarity'],
            'NLP source': 'Inflect library',
            'category': category  # Add the category to the updated data
        })

        # Add the plural form to the DataFrame with 'Inflect library' as the NLP source
        updated_data.append({
            'name': plural_form,
            'group': group,
            'Cosine similarity': row['Cosine similarity'],
            'NLP source': 'Inflect library',
            'category': category  # Add the category to the updated data
        })

    # Create the updated DataFrame
    updated_df = pd.DataFrame(updated_data)

    # Remove duplicates based on the 'name' column (case-insensitive)
    updated_df['name'] = updated_df['name'].str.lower()
    updated_df = updated_df.drop_duplicates(subset='name', keep='first')

    return updated_df


def drop_groups_with_special_characters(df):
    try:
        if 'name' not in df.columns or 'group' not in df.columns:
            return df
        # Check if any record in each group contains the '%' character, 'Fig' string, '\', '§', or ''
        groups_with_percent = df[df['name'].str.contains('%')]['group'].unique()
        groups_with_fig = df[df['name'].str.contains('Fig')]['group'].unique()
        groups_with_slash = df[df['name'].str.contains(r'/')]['group'].unique()
        groups_with_backslash = df[df['name'].str.contains('\n')]['group'].unique()
        groups_with_section_symbol = df[df['name'].str.contains('§')]['group'].unique()
        groups_with_special_character = df[df['name'].str.contains('')]['group'].unique()
        groups_with_special_characters = df[df['name'].str.contains(r'[±‘∼|_’)~(×“°”,]')]['group'].unique()
        groups_with_and = df[df['name'].str.contains(' and ')]['group'].unique()
        groups_with_single_letter = df[df['name'].str.len() == 1]['group'].unique()
        groups_with_only_numbers = df[df['name'].str.isnumeric()]['group'].unique()
        groups_with_brackets = df[df['name'].str.contains(r'\[|\]')]['group'].unique()
        groups_starts_with_hyphen = df[df['name'].str.startswith('-')]['group'].unique()

        
        # Combine the groups with '%' character, 'Fig' string, '\', '§', or ''
        groups_to_drop = (
            set(groups_with_percent) |
            set(groups_with_fig) |
            set(groups_with_slash) |
            set(groups_with_backslash) | 
            set(groups_with_section_symbol) |
            set(groups_with_special_character) |
            set(groups_with_special_characters) |
            set(groups_with_and) |
            set(groups_with_single_letter) |
            set(groups_with_only_numbers) |
            set(groups_with_brackets) |
            set(groups_starts_with_hyphen)
        )

        # Drop the groups that contain '%' character, 'Fig' string, '\', '§', ']', or ''
        df_filtered = df[~df['group'].isin(groups_to_drop)].copy()

        return df_filtered
    
    except KeyError as e:
        print(str(e))
        # Return an empty DataFrame if there's an error or no records to process
        return df_filtered


def drop_duplicates_by_name_category(df):
    # Drop duplicates based on 'name' and 'category' columns
    df_no_duplicates = df.drop_duplicates(subset=['name', 'category'], keep='first')
    return df_no_duplicates


def aggregate_final_df_by_name(df):
    # Group by 'name' column and aggregate columns with more than one record
    grouped_df = df.groupby('name')
#     aggregated_df = grouped_df.agg(lambda x: ', '.join(x) if len(x) > 1 else x.iloc[0])
    aggregated_df = grouped_df.agg(lambda x: tuple(x) if len(x) > 1 else x.iloc[0])
    
    # Keep only the first record for the 'group' column
    aggregated_df['group'] = grouped_df['group'].first()
    
    # Reset the index to bring 'name' back as a regular column
    aggregated_df.reset_index(inplace=True)
    
    return aggregated_df
    
    
def categorize_field(df):
    # Implementing fields
    def determine_field(category):
        if category in ['DISEASE', 'CANCER', 'ANATOMICAL_SYSTEM']:
            return 'MEDICINE'
        elif category in ['CHEMICAL', 'SIMPLE_CHEMICAL']:
            return 'CHEMISTRY'
        else:
            return 'BIOLOGY'

    df['field'] = df['category'].apply(determine_field)
    return df
    
    
def convert_all_columns_to_string(df):
    # Iterate through each column in the DataFrame
    for column in df.columns:
        if column == 'group':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        else:
            df[column] = df[column].astype(str)

    return df
    
    
# Function to create table in a database
def create_table(database_location, database_name, table_name):
    database_path = os.path.join(database_location,  database_name + '.db')
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Create the table
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            name                varchar(128) not null,
            groups              varchar(128) not null,
            category            varchar(128) not null,
            field               varchar(128) not null,
            file_name           varchar(128) not null,
            location            varchar(128) not null,
            date_of_insertion   varchar(128) not null,
            PRIMARY KEY (name, category)
        )
    ''')

    # Create an index on the 'name' and 'category' columns
    cursor.execute(f'''
        CREATE INDEX IF NOT EXISTS name_category_ind ON {table_name} (name, category)
    ''')
    
    conn.commit()
    conn.close()
    
    
def process_folder_and_subfolders(folder_path):
    # Initialize an empty list to store the results from each file
    all_results = []

    # Function to check if a file has the '.txt' extension
    def is_text_file(file_path):
        return file_path.lower().endswith('.txt')
    
    def extract_group_number(group_str):
        match = re.match(r'Group (\d+)', group_str)
        if match:
            return int(match.group(1))
        return 0
    
    def check_text_length(relevant_texts):
        for text in relevant_texts:
            # Tokenize the text to get the number of tokens
            tokens = biobert_tokenizer.tokenize(text)
            if len(tokens) > 512:
                return False
        return True

            
    def process_single_file(file_path):
        relevant_texts = extract_text_from_file(file_path)
        if relevant_texts:
            # Check the text length before proceeding with processing
            if not check_text_length(relevant_texts):
                print(f"Skipping file '{os.path.basename(file_path)}' due to exceeding 512 tokens.")
                return None

            # Concatenate relevant texts into a single string
            extracted_text = " ".join(relevant_texts)

            result = process_text_file(extracted_text, ner_list)
            result = pluralize_and_add_nlp_source(result)
            result = drop_groups_with_special_characters(result)

            # Add file name, file path, and time of analysis to the result DataFrame
            result['File Name'] = os.path.basename(file_path)
            result['File Path'] = file_path
            result['Analysis Time'] = datetime.now()
            if 'group' in result:
                result['group'] = result['group'].apply(extract_group_number)

            return result

        # If no relevant texts are found, return None for the result DataFrame 
#         return None
        return pd.DataFrame()
            

    # Iterate through each file in the folder and its subfolders
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if is_text_file(file_path):
                print('processing file:' , file)
                result = process_single_file(file_path)
                if result is not None:
                    all_results.append(result)

    # Combine all results into a single DataFrame
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Call the function to drop duplicates based on 'name' and 'category'
        final_df = drop_duplicates_by_name_category(df=final_df)
        # Assigning groups based on categories
        final_df = categorize_field(df=final_df)
        # Group by 'name' column and aggregate other columns as tuples
        final_df = aggregate_final_df_by_name(df=final_df)

       
        # Save the final DataFrame to Excel
        final_df.to_excel('kwords_detection_test.xlsx', index=False)
        print('The following df has been save do the Excel: ')
        print('Columns: ', final_df.columns)

        
        # Extracting specific columns
        final_df = final_df[['name', 'group', 'category', 'field', 'File Name', 'File Path', 'Analysis Time']]  
        final_df = convert_all_columns_to_string(df=final_df)
        
        return final_df
    else:
        print("No text files found in the specified folder and subfolders.")
        return None
    

def main(folder_path, database_location, database_name, table_name):
    # Connect to the database
    database_path = os.path.join(database_location, database_name + '.db')
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    # Save the results to the database
    final_result_df = process_folder_and_subfolders(folder_path)
    final_result_df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process folder and subfolders to save results in a database.")
    parser.add_argument("folder_path", help="Path to the folder containing the text files.")
    parser.add_argument("--ner_list", nargs="+", help="List of NER models to use.", default=[])
    parser.add_argument("--database_location", help="Path to the location where the database will be stored.")
    parser.add_argument("--database_name", help="Name of the database.")
    parser.add_argument("--table_name", help="Name of the table in the database.")
    args = parser.parse_args()

    folder_path = args.folder_path
    ner_list = args.ner_list
    database_location = args.database_location
    database_name = args.database_name
    table_name = args.table_name

    main(folder_path, ner_list, database_location, database_name, table_name)

