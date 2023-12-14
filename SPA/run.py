import os
import ntpath
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from joblib import Parallel, delayed
import src.model.bert_prediction as bp
import src.utils.keywords_detection as kd
import src.utils.result_preparation as rp
import src.utils.text_postprocessing as po
import src.utils.text_preprocessing as pr
import src.db.input_preparation as ip
from src.db.database_create import add_table_with_results, insert_data

load_dotenv()

model_path = os.getenv('MODEL_PATH')
documents_disk_path = os.getenv('DOCUMENTS_PATH')

model, tokenizer = bp.load_model_and_tokenizer(model_path, num_of_labels=4)

db_limit = 10
start_offset = int(os.getenv('START_DOCUMENT_OFFSET'))
max_offset = int(os.getenv('MAX_DOCUMENT_OFFSET'))

### constants ###
output_cols = ['doi', 'url', 'year', 'author', 'title', 'journal',
               '1st_keyword', 'general_term_x', 'displayed_term_x', '2nd_keyword',
               'general_term_y', 'displayed_term_y', 'sentences', 'category'
               # 'citations'
                ]

cols_to_merge = ['paper', '1st_keyword', '2nd_keyword', 'sentences', 'general_term_x', 'general_term_y',
                'displayed_term_x', 'normalized_term_x', 'normalized_term_y', 'category',
                'displayed_term_y',
                # 'citations'
                ]


def run_analyzer(conn, db_offset=0, db_limit=1, schema_name=''):
    warnings.filterwarnings("ignore")

    current_date = datetime.now().strftime("%Y-%m-%d")
    start_time = time.time()

    keywords = ip.sql2df(query=f"SELECT * from {schema_name}.keywords", db=conn)
    keywords = ip.clean_keywords(df=keywords, col='name')

    # query = """SELECT * FROM documents WHERE file_location NOT LIKE "%RECYCLE%" AND file_location LIKE "%oa_noncomm_txt\PMC005xxxxxx%";"""
    query = f"""SELECT * FROM {schema_name}.documents LIMIT {db_limit} OFFSET {db_offset} ;"""
    selection_bib = ip.sql2df(query=query, db=conn)
    selection_bib['file_loc'] = selection_bib['file_location'].str.replace('.bib', '.txt')

    selection_bib['file_loc'] = f'{documents_disk_path}' + selection_bib['file_loc']
    selection_bib['file_location'] = f'{documents_disk_path}' + selection_bib['file_location']
    
    if sys.platform == 'linux':
        selection_bib['file_loc'] = selection_bib['file_loc'].apply(lambda x: x.replace(ntpath.sep, os.sep))
        selection_bib['file_location'] = selection_bib['file_location'].apply(lambda x: x.replace(ntpath.sep, os.sep))
    elif 'win' in sys.platform:
        selection_bib['file_loc'] = selection_bib['file_loc'].apply(lambda x: x.replace(os.sep, ntpath.sep))
        selection_bib['file_location'] = selection_bib['file_location'].apply(lambda x: x.replace(os.sep, ntpath.sep))

    pr.remove_text(df=selection_bib,
                   stopwords_before=['Abstract', 'ABS T R AC T', 'ABSTRACT'],
                   stopwords_after=['References', 'REFERENCES',
                                    'Abbreviations', 'ABBREVIATIONS'])

    df_paper = kd.get_relations(df=selection_bib, kword_col=keywords['name'], conn=conn, schema_name=schema_name)

    df_paper = po.adding_norm_terms(df=df_paper, df_kwords=keywords)

    df_paper = df_paper[df_paper['normalized_term_x'] != df_paper['normalized_term_y']]

    df_paper['category'] = np.nan

    selection_bib = rp.extract_filename(df=selection_bib, input_col='file_location', output_col='doi_join')
    df_paper = rp.extract_filename(df=df_paper, input_col='paper', output_col='paper_name')

    result = rp.merge_relations_and_info(df=df_paper, bib_df=selection_bib,
                                         cols=output_cols, left_join='doi_join', right_join='paper_name')
    result = rp.sorting_data(df=result, cols=['year', 'title'], asc=[False, True])
    result['publisher'] = ' '

    end_time = time.time()
    elapsed_time = end_time - start_time

    # print("OFFSET: ", db_offset, ". Elapsed time in seconds: ", elapsed_time, ". Dataframe shape:", result.shape)
    return result


num_cores = os.cpu_count()
print('NUMBER OF CORES: ', num_cores)

offsets = [start_offset]
while max(offsets) <= max_offset: offsets.append(offsets[-1] + db_limit)

def run_analyzer_wrapped(db_offset, db_limit):
    conn = psycopg2.connect(host=os.getenv('POSTGRES_HOST'), dbname=os.getenv('POSTGRES_DB_NAME'),
                            user=os.getenv('POSTGRES_USER'), password=os.getenv('POSTGRES_PASSWORD'))
    return run_analyzer(conn, db_offset, db_limit, schema_name=os.getenv('POSTGRES_SCHEMA'))

def insert_data_wrapped(df):
    conn = psycopg2.connect(host=os.getenv('POSTGRES_HOST'), dbname=os.getenv('POSTGRES_DB_NAME'),
                            user=os.getenv('POSTGRES_USER'), password=os.getenv('POSTGRES_PASSWORD'))
    insert_data(df, conn, schema_name=os.getenv('POSTGRES_SCHEMA'))


cores_to_work = max(num_cores-3, 1)
while len(offsets) > 0:
    selected_offsets = offsets[:cores_to_work]
    del offsets[:cores_to_work]

    s_t = time.time()
    dfs = Parallel(n_jobs=cores_to_work)(delayed(run_analyzer_wrapped)(offset, db_limit) for offset in selected_offsets)
    concatenated_dfs = pd.concat(dfs)
    e_t = time.time()

    print('OFFSET DIFFERENCE: ', max(selected_offsets) - min(selected_offsets) + db_limit)
    print('PREPROCESSING TIME: ',  e_t-s_t)

    s_t = time.time()
    for i in range(len(dfs)):
        dfs[i]['input_data_bert'] = dfs[i][['sentences', '1st_keyword', '2nd_keyword']] \
            .apply(lambda x: '.'.join(x.astype(str)), axis=1)
        dfs[i] = bp.apply_BERT_model(df=dfs[i], input_col='input_data_bert', pred_col='category',
                                       model=model, tokenizer=tokenizer, batch_size=1024)
    e_t = time.time()
    print('MODEL PREDICTION TIME: ', e_t-s_t)

    s_t = time.time()
    Parallel(n_jobs=cores_to_work)(delayed(insert_data_wrapped)(df) for df in dfs)
    e_t = time.time()
    print('INSERTION TIME: ', e_t-s_t)
    print('MAX OFFSET:', max(selected_offsets))


# Parallel(n_jobs=)(delayed(run_analyzer_wrapped)(offset, db_limit) for offset in offsets)

# run_analyzer_wrapped(0, 100)
