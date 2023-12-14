import argparse
import pandas as pd
from synthcity.plugins import Plugins

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('model_name', type=str, help='Name of the synthetic data model to use. (e.g. adsgan, ctgan, tvae)')

    args = parser.parse_args()

    # Load data from the specified input file
    custom_data = pd.read_csv(args.input_file)

    print('Original data:')
    print(custom_data)

    # Load the specified synthetic model
    synthetic_model = Plugins().get(args.model_name)
    synthetic_model.fit(custom_data)

    # Generate synthetic data
    synthetic_data = synthetic_model.generate(100)
    synthetic_data.dataframe().to_csv(args.input_file.replace('.csv', '-synthetic.csv'), index=False)

if __name__ == '__main__':
    main()
