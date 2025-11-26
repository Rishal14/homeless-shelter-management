import pandas as pd

def extract_shelter_data(shelter_name, input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        shelter_df = df[df['shelter_name'] == shelter_name]
        
        if shelter_df.empty:
            print(f"No data found for shelter: {shelter_name}")
            return

        shelter_df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(shelter_df)} records for '{shelter_name}' to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    extract_shelter_data("Umeed Ki Kiran", "shelter_dataset_all_updated.csv", "single_shelter_data.csv")
