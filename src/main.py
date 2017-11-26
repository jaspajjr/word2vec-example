from utilities import collect_data


def main():
    data, count, dictionary, reversed_dict = collect_data()
    print("Hello, World!")
    print(data[0:7])


if __name__ == '__main__':
    main()
