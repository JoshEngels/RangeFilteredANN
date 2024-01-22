import numpy as np


def write_small_example():
  data = np.array([[1, 2, 3], [4, 5, 6]])
  filter_values = np.array([0.1, 0.2])

  # Create an array to hold the formatted rows
  formatted_rows = []

  # Loop through the data and filter_values arrays to format each row
  for i, (row, filter_value) in enumerate(zip(data, filter_values), start=1):
      # Format the row with tab separation
      formatted_row = f"{i}\t{filter_value:.1f}\t{{{','.join(map(str, row))}}}"
      formatted_rows.append(formatted_row)

  # Join the formatted rows with newline characters to create the final TSV content
  tsv_content = '\n'.join(formatted_rows)

  # Save the TSV content to a file
  with open('experiments/index_cache/output.tsv', 'w') as tsv_file:
      tsv_file.write(tsv_content)

  print("TSV file 'output.tsv' has been created.")


# write_small_example()
import psycopg2


def drop_all_tables(cursor):
  # Get a list of all table names in the database
  cursor.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';")
  table_names = cursor.fetchall()

  # Generate and execute DROP TABLE statements for each table
  for table_name in table_names:
      table_name = table_name[0]  # Extract the table name from the result tuple
      drop_table_query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
      cursor.execute(drop_table_query)
      print(f"Dropped table: {table_name}")

  # Commit the transaction
  conn.commit()

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
print("connected")
# Create a cursor
cursor = conn.cursor()

# data must be in list, not numpy array
data = [[1, 2, 3], [4, 5, 6]]
filter_values = [0.1, 0.2]
ids = list(range(len(data)))

values_to_insert = [(id_value, filter_value, vector_value) for id_value, filter_value, vector_value in zip(ids, filter_values, data)]


drop_all_tables(cursor)

cursor.execute("create table t_table(id int, filter REAL, vector_1 REAL[3]);")
# cursor.execute("INSERT INTO my_table (data_column) VALUES (%s);", (psycopg2.Binary(binary_data),)
insert_query = "INSERT INTO t_table(id, filter, vector_1) VALUES (%s, %s, %s);"
cursor.executemany(insert_query, values_to_insert)

cursor.execute("SELECT * FROM t_table;")
rows = cursor.fetchall()
print(rows)
# Close the cursor and connection when done
cursor.close()
conn.close()
