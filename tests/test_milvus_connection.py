from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "=== {:30} ==="
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")
print(fmt.format("Success"))
connections.disconnect("default")