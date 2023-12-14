import itertools
import pandas as pd


def excel2pandas(file_name):
    """
    Import the Excel to Pandas dataframe
    returns: pandas dataframe
    """
    k_words = pd.read_excel(file_name)
    return k_words


def clean_keywords(df, col):
    """
    Cleans the "keywords" column in a pandas dataframe by trimming whitespace from the strings,
    converting all keywords to lowercase, and removing any duplicates.
    returns: pandas.DataFrame: A new dataframe with the cleaned "keywords" column.
    """
    df[col] = df[col].str.strip()
    df[col] = df[col].str.lower()
    df = df.drop_duplicates(subset=col, keep='first')
    df = df.reset_index(drop=True)
    return df


def create_pairs_list(df, col):
    """
    Creates a list of lists with all possible pairwise combinations of the keywords in the "keywords" column
    of a cleaned pandas dataframe.
    returns: A list of lists, where each list contains two keywords.
    """
    pairs = list(itertools.combinations(df[col], 2))
    pairs_list = [[pair[0], pair[1]] for pair in pairs]
    return pairs_list


def keywords_tokens_prep(df, col):
    """
    Tokenizes keywords and returns a list of tuples where each keyword is an item of a tuple
    :param keywords: list of keywords
    :return: list of tuples of tokenized keywords
    """
    keywords_list = df[col].tolist()
    tokenized_kwords = [tuple(keyword.split()) for keyword in keywords_list]
    return tokenized_kwords





















