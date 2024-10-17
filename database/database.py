# database/database.py

import sqlite3
import os

DB_PATH = 'database/animals.db'

def create_connection(db_path=DB_PATH):
    """Tạo kết nối đến cơ sở dữ liệu SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        print(f"Kết nối thành công đến {db_path}")
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    """Tạo bảng trong cơ sở dữ liệu."""
    try:
        sql_create_table = """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor = conn.cursor()
        cursor.execute(sql_create_table)
        print("Tạo bảng 'detections' thành công.")
    except sqlite3.Error as e:
        print(e)

def insert_detection(conn, detection):
    """
    Chèn một bản ghi vào bảng 'detections'.

    Args:
        conn: Kết nối đến cơ sở dữ liệu.
        detection (tuple): (image_path, class_name, confidence)
    """
    sql_insert = """
    INSERT INTO detections (image_path, class_name, confidence)
    VALUES (?, ?, ?);
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql_insert, detection)
        conn.commit()
        print("Chèn bản ghi thành công.")
    except sqlite3.Error as e:
        print(e)

def query_detections(conn):
    """Truy vấn tất cả các bản ghi trong bảng 'detections'."""
    sql_query = "SELECT * FROM detections;"
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        return rows
    except sqlite3.Error as e:
        print(e)
        return []

def close_connection(conn):
    """Đóng kết nối đến cơ sở dữ liệu."""
    if conn:
        conn.close()
        print("Đã đóng kết nối đến cơ sở dữ liệu.")

if __name__ == '__main__':
    # Tạo kết nối và bảng khi chạy trực tiếp file này
    conn = create_connection()
    if conn is not None:
        create_table(conn)
        close_connection(conn)
