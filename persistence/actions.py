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

def delete_row_by_title(db_conn, title):
    """Delete row where title matches the parameter title
    :param db_conn: Conection to database
    :param title: title to be deleted from database"""
    db_operations.delete_row(db_conn, queries.DELETE_ROW_BY_TITLE, (title,))
    