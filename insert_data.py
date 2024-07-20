import json
import numpy as np
from sklearn.preprocessing import normalize
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the collection name and dimension
collection_name = 'medicine_data'
dim = 768  # Dimension of the vector field

# Function to create or drop a Milvus collection
def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=50, is_primary=True),
        FieldSchema(name="spl_product_data_elements", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="spl_unclassified_section", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="active_ingredient", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="active_ingredient_table", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="purpose", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="purpose_table", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="indications_and_usage", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="warnings", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="stop_use", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="do_not_use", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="when_using", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="keep_out_of_reach_of_children", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="dosage_and_administration", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="storage_and_handling", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="inactive_ingredient", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="package_label_principal_display_panel", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="set_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="effective_time", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="version", dtype=DataType.INT64),
        FieldSchema(name="application_number", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="brand_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="generic_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="manufacturer_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="product_ndc", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="product_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="route", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="substance_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="spl_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="spl_set_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="package_ndc", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="is_original_packager", dtype=DataType.BOOL),
        FieldSchema(name="unii", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)  # Vector field
    ]

    collection_schema = CollectionSchema(fields, description="Product Data Collection")
    collection = Collection(name=collection_name, schema=collection_schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
        "metric_type": "L2"  # Ensure the metric type is set correctly
    }

    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

# Function to load JSON data
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Function to prepare data for insertion
def prepare_data(json_data):
    ids = []
    spl_product_data_elements = []
    spl_unclassified_section = []
    active_ingredient = []
    active_ingredient_table = []
    purpose = []
    purpose_table = []
    indications_and_usage = []
    warnings = []
    stop_use = []
    do_not_use = []
    when_using = []
    keep_out_of_reach_of_children = []
    dosage_and_administration = []
    storage_and_handling = []
    inactive_ingredient = []
    package_label_principal_display_panel = []
    set_id = []
    effective_time = []
    version = []
    application_number = []
    brand_name = []
    generic_name = []
    manufacturer_name = []
    product_ndc = []
    product_type = []
    route = []
    substance_name = []
    spl_id = []
    spl_set_id = []
    package_ndc = []
    is_original_packager = []
    unii = []
    embeddings = []

    for entry in json_data:
        ids.append(entry['id'])
        spl_product_data_elements.append(entry.get('spl_product_data_elements', ''))
        spl_unclassified_section.append(entry.get('spl_unclassified_section', ''))
        active_ingredient.append(entry.get('active_ingredient', ''))
        active_ingredient_table.append(entry.get('active_ingredient_table', ''))
        purpose.append(entry.get('purpose', ''))
        purpose_table.append(entry.get('purpose_table', ''))
        indications_and_usage.append(entry.get('indications_and_usage', ''))
        warnings.append(entry.get('warnings', ''))
        stop_use.append(entry.get('stop_use', ''))
        do_not_use.append(entry.get('do_not_use', ''))
        when_using.append(entry.get('when_using', ''))
        keep_out_of_reach_of_children.append(entry.get('keep_out_of_reach_of_children', ''))
        dosage_and_administration.append(entry.get('dosage_and_administration', ''))
        storage_and_handling.append(entry.get('storage_and_handling', ''))
        inactive_ingredient.append(entry.get('inactive_ingredient', ''))
        package_label_principal_display_panel.append(entry.get('package_label_principal_display_panel', ''))
        set_id.append(entry.get('set_id', ''))
        effective_time.append(entry.get('effective_time', ''))
        version.append(int(entry.get('version', 0)))
        application_number.append(entry.get('openfda', {}).get('application_number', [''])[0])
        brand_name.append(entry.get('openfda', {}).get('brand_name', [''])[0])
        generic_name.append(entry.get('openfda', {}).get('generic_name', [''])[0])
        manufacturer_name.append(entry.get('openfda', {}).get('manufacturer_name', [''])[0])
        product_ndc.append(entry.get('openfda', {}).get('product_ndc', [''])[0])
        product_type.append(entry.get('openfda', {}).get('product_type', [''])[0])
        route.append(entry.get('openfda', {}).get('route', [''])[0])
        substance_name.append(entry.get('openfda', {}).get('substance_name', [''])[0])
        spl_id.append(entry.get('openfda', {}).get('spl_id', [''])[0])
        spl_set_id.append(entry.get('openfda', {}).get('spl_set_id', [''])[0])
        package_ndc.append(entry.get('openfda', {}).get('package_ndc', [''])[0])
        is_original_packager.append(entry.get('openfda', {}).get('is_original_packager', [False])[0])
        unii.append(entry.get('openfda', {}).get('unii', [''])[0])
        
        # Generate a random vector and normalize it
        vector = np.random.random((1, dim))  # Ensure correct shape for normalization
        normalized_vector = normalize(vector).flatten()  # Normalize and flatten to match required dimensions
        embeddings.append(normalized_vector.tolist())

    return [
        ids, spl_product_data_elements, spl_unclassified_section, active_ingredient, active_ingredient_table,
        purpose, purpose_table, indications_and_usage, warnings, stop_use, do_not_use, when_using,
        keep_out_of_reach_of_children, dosage_and_administration, storage_and_handling, inactive_ingredient,
        package_label_principal_display_panel, set_id, effective_time, version, application_number, brand_name,
        generic_name, manufacturer_name, product_ndc, product_type, route, substance_name, spl_id, spl_set_id,
        package_ndc, is_original_packager, unii, embeddings
    ]

def insert_data(filename):
    json_data = load_json(filename)
    data = prepare_data(json_data)
    
    
    collection = create_milvus_collection(collection_name, dim)
    
    # Insert data into collection
    collection.insert(data)

if __name__ == "__main__":
    insert_data('output.json')
