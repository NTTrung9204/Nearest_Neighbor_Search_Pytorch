import mysql.connector
import time
from utils import query_random_k_vectors, query_all_vectors

# Kết nối tới cơ sở dữ liệu MySQL
connection = mysql.connector.connect(
    host="localhost",
    user="root",  # Tên người dùng MySQL
    password="123456",  # Mật khẩu MySQL
    database="numpy_array",  # Tên database
)

cursor = connection.cursor()

start_time = time.time()

# list_vectors = query_random_k_vectors(cursor, k=1000)
list_vectors = query_all_vectors(cursor, table_name="v6")

print(f"Số lượng vector: {len(list_vectors)}")
print(f"Thời gian để truy vấn: {time.time() - start_time:.4f} giây")

cursor.close()
connection.close()
