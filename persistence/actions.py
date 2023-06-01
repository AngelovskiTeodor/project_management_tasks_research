from . import db_operations
from . import queries

def get_db_connection(db_file_location=r"./persistence/db.sqlite3"):
    return db_operations.create_db_connection(db_file_location)

def create_generated_descriptions_table(db_conn):
    """ Creates a table for generated descriptions in database """
    return db_operations.create_table(db_conn,queries.CREATE_GENERATED_DESCRIPTIONS_TABLE)

def insert_description(db_conn, new_desc):
    """ Wrapper for inserting description into sqlite db
    :param db_conn: Connection to database
    :param desc: tuple with 3 values: id, title, description
    :return: True if succesfully created new table, false if unsuccessful """
    return insert(db_conn, queries.GENERATED_DESCRIPTIONS_TABLE_NAME, queries.GENERATED_DESCRIPTIONS_COLUMNS, new_desc)
    
def insert(db_conn, table, columns, values):
    """ Create request to insert object in database. All params are case-sensitive
    :param db_conn: Conection to database
    :param table: String of the name of the table
    :param columns: tuple of column names
    :param values: tuple of object values
    :return: True if succesfully created new table, false if unsuccessful  """
    query = queries.get_insert_query(table,columns)
    return db_operations.insert_row(db_conn, query, values)

def insert_title(db_conn, id, title):
    """Inserts title and id without description in database
    :param db_conn: Connection object to database
    :param id: id of the entry in provided raw csv data
    :param title: String of title
    """
    query = queries.get_insert_query(queries.GENERATED_DESCRIPTIONS_TABLE_NAME,"id,title")
    return db_operations.insert_row(db_conn, query, (id, title))

def update_description(db_conn, title, description):
    """ Adds/Changes the description for certain title 
    :param db_conn: Connection to database
    :param title: Title of task (string)
    :param description: new updated description (string)
    """
    return db_operations.update(db_conn, 
                                queries.UPDATE_DESCRIPTION, 
                                tuple((description,title)))

def get_titles(db_conn):
    """Selects all titles from database
    :param db_conn: Conection to database"""
    return db_operations.select_all(db_conn, queries.SELECT_TITLES)

def get_descriptions(db_conn, title=None):
    """Selects description matching the title
    If no title is provides, it will select all descriptions
    :param db_conn: Conection to database"""
    if title:
        return db_operations.select_where(db_conn, queries.SELECT_DESCRIPTION_BY_TITLE, (title,))
    return db_operations.select_all(db_conn, queries.SELECT_DESCRIPTIONS)

def get_all(db_conn, table=None):
    """Selects all data from table provided as parameter
    :param db_conn: Connection to database
    :param table: String of the table name in database. If None, than selects all data from generated_descriptions table
    """
    if table is None:
        table = queries.GENERATED_DESCRIPTIONS_TABLE_NAME
    query = queries.SELECT_ALL.format(table)
    return db_operations.select_all(db_conn, query)

def get_missing_descriptions_only(db_conn, limit=None):
    """Selects rows with missing descriptions
    :param db_conn: Connection object to database
    :param limit: Max number of rows from database
    """
    if limit:
        return db_operations.run_query(db_conn, queries.SELECT_MISSING_DESCRIPTIONS+" LIMIT "+str(limit))
    return db_operations.run_query(db_conn, queries.SELECT_MISSING_DESCRIPTIONS)


def delete_row_by_title(db_conn, title):
    """Delete row where title matches the parameter title
    :param db_conn: Conection to database
    :param title: title to be deleted from database"""
    return db_operations.delete_row(db_conn, queries.DELETE_ROW_BY_TITLE, (title,))
    
def count_total_entries(db_conn, table=queries.GENERATED_DESCRIPTIONS_TABLE_NAME):
    """Count total rows in table (including empty descriptions)
    :param db_conn: Database connection object
    :param table: String of table name
    """
    query = queries.COUNT_ALL.format(queries.GENERATED_DESCRIPTIONS_TABLE_NAME)
    total_entries_number = db_operations.count(db_conn, query)
    return total_entries_number

def count_descriptions(db_conn):
    """Count number of generated description in database
    :param db_conn: Connection object to database
    """
    query = queries.COUNT_NOT_NULL.format(queries.GENERATED_DESCRIPTIONS_TABLE_NAME, "description")
    total_descriptions_number = db_operations.count(db_conn,query)
    return total_descriptions_number

def count_missing_descriptions(db_conn):
    """Count number of titles without description
    :param db_conn: Connection object to database
    """
    query = queries.COUNT_NULL.format(queries.GENERATED_DESCRIPTIONS_TABLE_NAME, "description")
    total_missing_descriptions = db_operations.count(db_conn, query)
    return total_missing_descriptions
