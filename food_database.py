# food_database.py
import os
import sqlite3

class FoodDatabase:
    def __init__(self, database_path):
        self.database_path = database_path
    
    def query_database(self, query):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute(query)
        query_values = [row[0] for row in cursor.fetchall()]
        conn.close()
        return query_values

    def fetch_food_names(self):
        return self.query_database("SELECT food_name FROM nutrition_expanded")

