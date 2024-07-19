import os
import json

# Directory path
input_directory = 'fda_dataset/'

# Output file path
output_file = 'output.json'

try:
    # List to accumulate processed data
    all_processed_data = []
    
    # Iterate over files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):  # Check if the file is a JSON file
            filepath = os.path.join(input_directory, filename)
            
            with open(filepath, 'r') as file:
                data = json.load(file)
                result = data.get('results', [])
                
                for drug_data in result:
                    openfda_data = drug_data.get('openfda', {})
                    product_type = openfda_data.get('product_type', [])
                    
                    # Check if 'product_type' exists and is a non-empty list
                    if product_type and product_type[0] == "HUMAN OTC DRUG":
                        # Append drug_data to the list
                        all_processed_data.append(drug_data)
                    else:
                        continue

    # Write all_processed_data to the output JSON file
    with open(output_file, 'w') as outfile:
        json.dump(all_processed_data, outfile, indent=1)

except Exception as err:
    print(f"Error: {err}")


try:
    # List to accumulate processed data
    all_processed_data = []
    
    # Iterate over files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):  # Check if the file is a JSON file
            filepath = os.path.join(input_directory, filename)
            
            with open(filepath, 'r') as file:
                data = json.load(file)
                result = data.get('results', [])
                
                for drug_data in result:
                    openfda_data = drug_data.get('openfda', {})
                    product_type = openfda_data.get('product_type', [])
                    
                    # Check if 'product_type' exists and is a non-empty list
                    if product_type and product_type[0] == "HUMAN OTC DRUG":
                        # Append drug_data to the list
                        all_processed_data.append(drug_data)
                    else:
                        continue

    # Write all_processed_data to the output JSON file
    with open(output_file, 'w') as outfile:
        json.dump(all_processed_data, outfile, indent=1)

except Exception as err:
    print(f"Error: {err}")