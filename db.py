from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the function to create a Milvus collection
def create_milvus_collection(collection_name, dim):
    # Check if the collection exists and drop if it does
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # Define fields with appropriate data types
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="effective_time", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="set_id", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="spl_id", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="brand_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="generic_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="manufacturer_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="product_ndc", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="product_type", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="route", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="substance_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="package_ndc", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="unii", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="spl_product_data_elements", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="spl_unclassified_section", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="active_ingredient", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="active_ingredient_table", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="purpose", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="purpose_table", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="indications_and_usage", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="warnings", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="stop_use", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="do_not_use", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="when_using", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="keep_out_of_reach_of_children", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="dosage_and_administration", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="storage_and_handling", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="inactive_ingredient", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="package_label_principal_display_panel", dtype=DataType.VARCHAR, max_length=255),
        
        # Vector field
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    # Create Collection Schema
    collection_schema = CollectionSchema(fields, description="Product Data Collection")

    # Create Collection
    collection = Collection(name=collection_name, schema=collection_schema)

    # index parameters
    index_params = {
        "index_type": "IVF_FLAT",
        "params": {
            "nlist": 128
        },
        "metric_type": "L2"
    }

    #  index on the vector field
    collection.create_index(field_name="embedding", index_params=index_params)

    return collection


collection_name = 'medicine_data'
collection = create_milvus_collection(collection_name, 768)

# Check if the collection exists
has = utility.has_collection(collection_name)
print(f"Does collection '{collection_name}' exist in Milvus: {has}")
