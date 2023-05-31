import sqlite3
from sqlite3 import Error

def create_db_connection(db_file_location=r"./db.sqlite3"):
    """ Create a database connection to a SQLite database 
        If the DB file does not exist, this function will create it implicitly 
        :param db_file_location: location to the db file
        :return: Connection object or None"""
    db_conn = None
    try:
        db_conn = sqlite3.connect(db_file_location)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        return db_conn
    
def create_table(db_conn, sql_query):
    """ create a table from the create_table_sql statement
    :param db_conn: Connection to database
    :param sql_query: a CREATE TABLE statement
    :return: True if succesfully created new table, false if unsuccessful
    """
    try:
        cursor = db_conn.cursor()
        cursor.execute(sql_query)
    except Error as e:
        print(e)
        return False
    return True

def insert_row(db_conn, query, values):
    """ Inserts row into database 
    :param db_conn: Connection to database
    :param query: SQL query
    :param values: values to be inserted into database"""
    try:
        cursor = db_conn.cursor()
        cursor.execute(query, values)
        db_conn.commit()
    except Error as e:
        print(e)
        return False
    return True

def update(db_conn, query, values):
    """ Update row """
    cursor = db_conn.cursor()
    cursor.execute(query, values)
    return db_conn.commit()

def select_all(db_conn, query):
    """Select all rows from table provided in query"""
    cursor = db_conn.cursor()
    cursor.execute(query)

    rows = cursor.fetchall()
    return rows

def select_where(db_conn, query, values):
    """Select rows where condition is met
    :param db_conn: Cennection object to database
    :param query: sql query to execute
    :param values: tuple of values to include in the sql query
    """

    cursor = db_conn.cursor()
    cursor.execute(query, values)

    rows = cursor.fetchall()
    return rows

def delete_row(db_conn, query, values):
    """Run delete query
    :return:
    """
    cursor = db_conn.cursor()
    cursor.execute(query, values)
    db_conn.commit()