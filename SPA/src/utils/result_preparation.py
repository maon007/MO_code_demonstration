import pandas as pd
import bibtexparser
import os


def bibtex2pandas(bib_path, bib_cols):
    """
    This function converts a bibtex file to a pandas DataFrame.
    Parameters:
    bib_path (str): The path of the bibtex file to be converted.
    bib_cols (list): A list of columns to be included in the DataFrame.
    returns: selection_bib (DataFrame): A DataFrame containing the selected columns from the bibtex file.
    """
    with open(bib_path, encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    df_bib = pd.DataFrame(bib_database.entries)
    selection_bib = df_bib[bib_cols]
    selection_bib["title"] = selection_bib["title"].str.replace("{", "").str.replace("}", "")
    return selection_bib


def swap_and_add(df):
    """
    Duplicate all the rows but swap the first and the second columns and keep the other columns.
    Parameters: df (pandas.DataFrame): The input dataframe.
    returns: pandas.DataFrame
    The combined dataframe with all the rows duplicated and the first and second columns swapped.
    """
    df2 = df.copy()
    df2['1st_keyword'] = df['2nd_keyword']
    df2['2nd_keyword'] = df['1st_keyword']

    df2['normalized_term_x'] = df['normalized_term_y']
    df2['normalized_term_y'] = df['normalized_term_x']

    df2['general_term_x'] = df['general_term_y']
    df2['general_term_y'] = df['general_term_x']

    df2['displayed_term_x'] = df['displayed_term_y']
    df2['displayed_term_y'] = df['displayed_term_x']

    df = df.append(df2, ignore_index=True)
    return df


def join_non_none(x):
    """Joins non-None values in a list with ';;' separator.
    returns: A string with non-None values joined by ';;' separator.
    """
    non_none_values = [str(val) for val in x if val is not None]
    return ';; '.join(non_none_values)


def merge_abbreviations(df, cols):
    """
   Merges the abbreviations found in the '1st_keyword' and '2nd_keyword' columns of a pandas DataFrame `df`,
   grouping and aggregating the data by the columns 'paper', 'Normalized_term_x' and 'Normalized_term_y'.
   returns: a new pandas DataFrame with the merged abbreviations and the grouped and aggregated data.
   """
    df_agg = df[cols].groupby(
        ['paper', 'normalized_term_x', 'normalized_term_y', 'displayed_term_x', 'displayed_term_y', ],
        as_index=False).agg({'1st_keyword': lambda x: ', '.join(sorted(set(x), key=len, reverse=True)),
                             '2nd_keyword': lambda x: ', '.join(sorted(set(x), key=len, reverse=True)),
                             'sentences': lambda x: (x.unique().tolist()),
                             'category': lambda x: (x.unique().tolist()),
                             'general_term_x': lambda x: (x.unique().tolist()),
                             'general_term_y': lambda x: (x.unique().tolist())})
    return df_agg


def extract_filename(df, input_col, output_col):
    """
    The function extracting filenames from filepaths
    Args: df, input_col, output_col:
    returns: pandas dataframe
    """
    df[output_col] = df[input_col].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    return df

def merge_relations_and_info(bib_df, df, cols, left_join, right_join):
    """
    This function merge the following dataframes:
    1) bib_df: df with the info about the article (metadata)
    2) df: df with the info about relations (data)
    - it also calculates the number of relations
    returns: pandas dataframe
    """
    merged_df = pd.merge(bib_df, df[['1st_keyword', 'general_term_x', 'displayed_term_x', '2nd_keyword', 'general_term_y',
                                  'displayed_term_y', 'paper', 'sentences', 'category',
                                  'paper_name']],
                      how='inner', left_on=left_join, right_on=right_join)
    result = merged_df[cols]
    result['bib_id'] = merged_df['id'].values.copy()
    return result


def sorting_data(df, cols, asc):
    """
    Sorting of the data
    returns: pandas dataframe
    """
    df = df.sort_values(by=cols, ascending=asc)
    df = df.reset_index(drop=True)
    return df


def df2csv(df, filename, separator):
    """
    Export the result to the csv with a predefined separator
    """
    df.to_csv(filename, index=False, sep=separator)
    return


def df2excel(df, file_name):
    """
    Export the result to the Excel
    """
    df.to_excel(file_name, engine='xlsxwriter')
    return
