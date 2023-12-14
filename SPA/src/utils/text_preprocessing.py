import os

def read_files(file_location):
    """
    Read files from the directory
    Open the file, read it, from each sentence create an item of a list
    return: content of the file
    """
    try:
        with open(fr"{file_location}", encoding="utf8", errors='ignore') as f:
            all_sentences = f.read()
            # split based on empty lines
            paragraphs = all_sentences.split('\n\n')
            all_sentences_list = []
            for paragraph in paragraphs:
                # split each paragraph based on dots
                sentences = paragraph.replace('\n', ' ').split('.')
                # append the resulting sentences to the list
                all_sentences_list.extend(sentences)
            return all_sentences_list
    except FileNotFoundError:
        print(f"File {file_location} not found. Skipping...")


def get_directory_content(dir_path):
    """
    Create a list, for every file directory, check if the specified path exists
    If so, store the result to the list.
    return: a list containg names of files from the specified folder and the path of the folder
    """
    files = []
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file)):
            files.append(file)
    return files, dir_path


def remove_text(df, stopwords_before, stopwords_after):
    """
    This function removes text from papers in a given directory before and/or after certain stopwords.
    Parameters:
    dir_path (str): The directory path of the papers you want to remove text from.
    stopwords_before (list): A list of stopwords that will be used to remove text before them.
    stopwords_after (list): A list of stopwords that will be used to remove text after them.
    return: truncated text
    """
    missing_files = set()
    for index, value in df['file_location'].items():
        if not os.path.exists(value):
            missing_files.add(value)
            print('couldnt find path:', value)
            continue
        with open(value, 'r+', encoding="latin-1") as file:
            # print('Removing unnecessary parts from:', value)
            text = file.read()
            for stopword in stopwords_before:
                stop_index = text.find(stopword)
                if stop_index == -1:
                    continue
                text = text[stop_index + len(stopword):]
            for stopword in stopwords_after:
                stop_index = text.find(stopword)
                if stop_index == -1:
                    continue
                text = text[:stop_index]
            file.seek(0)  # move file pointer to the beginning of the file
            file.write(text)
            file.truncate()  # truncate the file to the length of the new content
    if missing_files:
        with open("missing_txt_files.txt", "w") as missing_file:
            missing_file.write("\n".join(missing_files))


