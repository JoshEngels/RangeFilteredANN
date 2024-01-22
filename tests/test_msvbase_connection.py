import psycopg2

# Set the connection parameters
db_params = {
    'dbname': 'vectordb',
    'user': 'vectordb',
    'password': 'vectordb',
    'host': '172.17.0.2',  # Use the host machine's IP address
    'port': '5432',  # The default PostgreSQL port
}

# Establish the database connection
conn = psycopg2.connect(**db_params)
print("connected, success")