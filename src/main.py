from data import load_data, load_config

def main():
    config = load_config()
    data_path = config['data_path']
    data = load_data(data_path)

    # print headers
    print(data.head())

    # length
    print(len(data))

if __name__ == '__main__':
    main()
