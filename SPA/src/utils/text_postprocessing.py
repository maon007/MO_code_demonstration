import pandas as pd
import regex as re


def adding_norm_terms(df, df_kwords):
    """
    Add keyword index columns to a DataFrame based on a lookup DataFrame.
    The DataFrame containing the keyword index columns to add.
    returns: pandas.DataFrame
    """
    df_paper = pd.merge(pd.merge(df,
                                 df_kwords, left_on='1st_keyword', right_on='name', how='left'),
                                 df_kwords, left_on='2nd_keyword', right_on='name', how='left')
    return df_paper


def extract_author_kws(df, input_col, output_col):
    """
    Extracting sentences containing keywords mentioned by author
    returns: pandas.DataFrame
    """
    df[output_col] = df[input_col].apply(lambda x: x if ('Keywords' in x or
                                                              'keywords' in x or
                                                              'KEYWORDS' in x or
                                                              'Key Words' in x)
                                                              else '')
    return df


def text_cleaning(df, col):
    """
    Filters out rows from a pandas DataFrame `df` that contain certain keywords, citations, or strange characters.
    Specifically, the function removes any rows that contain the keywords 'Keywords', 'KEYWORDS', or 'Key Words' in the
    'sentences' column, as well as any rows that contain citations in square brackets in the 'sentences' column. Additionally,
    the function removes any rows that do not mention 'ABSTRACT' in the 'sentences' column. Finally, the function removes any
    strange characters such as 'ÂÃ¢' from the 'sentences' column.
    returns: pandas dataframe
    """
    df = df[~df[col].str.contains('Keywords|KEYWORDS|Key Words', na=False)]
    df[col] = df[col].apply(lambda x: x if x is None else x.replace('[ÂÃ¢]', ''))
    return df


def df_merge(df, col_name, col1, col2):
    """
    This function merge 2 columns with the sign '==>'
    return: pandas dataframe
    """
    df[col_name] = df[col1] + '==>' + df[col2]
    return df


def drop_suffix(df, col):
    """
    Remove suffix (.txt) from the name of a paper to be able to join it with metadata
    """
    df[col] = df[col].str.replace(r'.txt', '')
    return df

