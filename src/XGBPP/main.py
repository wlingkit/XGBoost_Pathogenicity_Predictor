import sys
import data_loader as data_loader

def main():
    df = data_loader.load_data('./data/variant_summary_50.txt')
    print(df)

if __name__ == "__main__":
    main()