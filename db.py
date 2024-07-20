from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
connections.connect("default", host="localhost", port="19530")

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 8
print(fmt.format("start connecting to Milvus"))
# run program after starting up docker for milvus
def create_milvus_collection(collection_name, dim):
    # If the collection exists by name
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # Define fields
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
        # Add a vector field
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    # Create Collection Schema
    collection_schema = CollectionSchema(fields, description="Product Data Collection")

    # Create Collection
    collection = Collection(name=collection_name, schema=collection_schema)

    # Initialising index parameters
    index_params = {
        "index_type": "IVF_FLAT",
        "params": {
            "nlist": 128
        },
        "metric_type": "L2"  # Ensure the metric type is set correctly
    }

    # Create the index
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

collection_name = 'medicine_data'
collection = create_milvus_collection(collection_name, 768)

# Check if the collection exists
has = utility.has_collection(collection_name)
print(f"Does collection '{collection_name}' exist in Milvus: {has}")

collection = create_milvus_collection('medicine_data', 768)

