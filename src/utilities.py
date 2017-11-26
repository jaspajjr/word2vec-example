import urllib.request
import zipfile
import tensorflow as tf
import collections.Counter as Counter
import os


def maybe_download(filename, url, expected_bytes):
    '''
    If the filename doesn't exist, then download it.
    '''
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify' + filename + '. Can you get to it in browser?')
    return filename


def read_data(filename):
    '''
    '''
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.name_list()[0])).split()
    return data


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []

    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dict


def collect_data(vocab_size=10000):
    '''
    Get the data
    '''
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', url, 31344016)
    vocabulary = read_data(filename)
    data, count, dictionary, reversed_dict = build_dataset(vocabulary,
                                                           vocab_size)
    return data, count, dictionary, reversed_dict
