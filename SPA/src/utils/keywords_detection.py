import pandas as pd
import src.utils.text_preprocessing as pr
import re
import itertools


def get_relations(kword_col, df, conn=None, schema_name=''):
    """
    Extracts sentences containing at least 2 keywords from the predefined list of keywords.
    Args:
        articles (tuple): A tuple of two elements: a list of article filenames and the directory path where the articles are located.
        keywords (list): A list of keyword pairs to search for in the articles.
    returns: pandas DataFrame containing the paper, 1st_keyword, 2nd_keyword, and sentences where the keyword pairs appear.
    """
    curr = conn.cursor()
    col_list = set(kword_col.tolist())
    rows_to_append = []
    added_pairs = {}
    # Compile the regular expression pattern
    pattern = re.compile(r'\b(' + '|'.join(col_list) + r')\b')

    # for paper in articles[0]:
    for index, row in df.iterrows():
        # print('processing paper: ', paper)
        value = row['file_loc']
        bib_id = int(row['id'])
        # print('processing paper: ', value)
        # text = pr.read_files(file=paper, dir_path=articles[1])
        try:
            text = pr.read_files(file_location=value)
            for sentence in text:
                sentence_lower = sentence.lower()
                patterns = pattern.findall(sentence_lower)
                matches = [keyword[0] for keyword in patterns if keyword[0] in col_list]
                if len(matches) >= 2:
                    # If there are at least two matches, save all combinations of two keywords to the dataframe
                    for combo in itertools.combinations(matches, 2):
                        # Sort the pair of keywords to avoid duplicates and convert to tuple
                        pair = tuple(sorted(combo))
                        # Check if the same combination has already been added to the dataframe with the same sentence
                        if pair in added_pairs and sentence in added_pairs[pair]:
                            continue
                        if pair not in added_pairs:
                            added_pairs[pair] = set()
                        added_pairs[pair].add(sentence)
                        # Append the row to the dataframe
                        rows_to_append.append(
                            {'paper': value, '1st_keyword': pair[0], '2nd_keyword': pair[1], 'sentences': sentence,
                            'no_of_keywords': len(matches), 'bib_id': bib_id})
        except Exception as e:
            print(f"Error was detected: {e}")
            curr.execute('ROLLBACK')
            continue
    df = pd.DataFrame(rows_to_append, columns=['paper', '1st_keyword', '2nd_keyword', 'sentences', 'no_of_keywords', 'bib_id'])

    return df
