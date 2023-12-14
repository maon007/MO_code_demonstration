

def add_table_with_results(conn):

    # connect to an existing database connection and
    # create a new table to hold your data
    conn.execute("""CREATE TABLE IF NOT EXISTS relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doi	varchar(1024) not null,
        url	varchar(1024) null default '',	
        year int null default -1,
        publisher 	varchar(1024) null default '',	
        author		varchar(1024) null default '',
        title		varchar(1024) null default '',
        journal		varchar(2048) null default '',
        keyword1		varchar(255) not null,
        category_kw1         varchar(1024) null default '',
        general_term_1  varchar(1024) null default '',
        keyword2		varchar(255) not null,
        category_kw2         varchar(1024) null default '',
        general_term_2  varchar(1024) null default '',
        sentences		varchar(1048576) null default '',
        category	varchar(255) null default '',
        sentence_and_category	varchar(1048576) null default '',
        no_of_sentences varchar(1048576) null default '',
        source_of_papers varchar(1048576) null default ''
    );""")
    conn.commit()
    return conn


def insert_data(df, conn, schema_name=''):
    counter = 0
    curr = conn.cursor()

    sentences_unique = df.groupby(['sentences'])[['sentences', 'bib_id']].agg(lambda x: x.unique()[0])
    id_mapping = {}
    for _, row in sentences_unique.iterrows():
        try:
            curr.execute(f"""
                INSERT INTO {schema_name}.sentences (sentences, document_id)
                   VALUES (%s, %s) RETURNING id
                   """, (row['sentences'], row['bib_id']))

            sentence_id = curr.fetchone()[0]

            id_mapping[row['sentences']] = sentence_id
            # curr.execute('COMMIT')
        except Exception as e:
            print(f"Error was detected: {e}")
            continue

    conn.commit()

    for index, row in df.iterrows():
        try:
            curr.execute(f"""
                INSERT INTO {schema_name}.relationships (
                    id, document_id, sentence_id, keyword1, keyword2, category
                )
                VALUES (
                    DEFAULT, %s, %s, %s, %s, %s
                )
            """, (
                row['bib_id'], id_mapping[row['sentences']],
                row['1st_keyword'], row['2nd_keyword'],
                str(row['category'])
            ))
            counter += 1
        except Exception as e:
            print(f"Error was detected: {e}")
            continue

    # print("Data inserted successfully. In total {} rows".format(counter))
    conn.commit()
    conn.close()

def add_sentence_and_relationships_tables_for_thread(conn, number_of_thread=None):
    if number_of_thread is None:
        relationships, sentences = 'relationships', 'sentences'
    else:
        relationships, sentences = f'relationships_{number_of_thread}', f'sentences_{number_of_thread}'


    conn.execute(f"""CREATE TABLE IF NOT EXISTS {relationships} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        sentence_id INTEGER,
        keyword1		varchar(255) not null,
        category_kw1         varchar(255) null default '',
        general_term_1  varchar(255) null default '',
        keyword2		varchar(255) not null,
        category_kw2         varchar(255) null default '',
        general_term_2  varchar(255) null default '',
        category	varchar(255) null default ''
    );""")
    conn.commit()

    # create a new table to hold data
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {sentences} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentences		varchar(2056) null default '',
        document_id INTEGER
    );""")

    conn.commit()

    return relationships, sentences