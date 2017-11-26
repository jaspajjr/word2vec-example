import numpy as np
from keras.preprocessing import sequence
from utilities import collect_data


def main():
    vocab_size = 10000
    data, count, dictionary, reversed_dict = collect_data(vocab_size)
    window_size = 3
    vector_dim = 300
    epochs = 2000000
    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    sampling_table = sequence.make_sampling_table(vocab_size)
    couples, labels = sequence.skipgrams(
            data,
            vocab_size,
            window_size=window_size,
            sampling_table=sampling_table)
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype='int32')
    word_context = np.array(word_context, dtype='int32')

    print(couples[:10], labels[:10])


if __name__ == '__main__':
    main()
