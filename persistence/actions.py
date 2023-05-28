import db_operations
import queries

def get_db_connection():
    return db_operations.create_db_connection()

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
    """ Adds/Changes the description for certain title """
    return db_operations.update(db_conn, 
                                queries.UPDATE_DESCRIPTION.format(queries.GENERATED_DESCRIPTIONS_TABLE_NAME), 
                                tuple(title,description))

def get_titles(db_conn):
    """Selects all titles from database
    :param db_conn: Conection to database"""
    return db_operations.select_all(db_conn, queries.SELECT_TITLES)

def get_descriptions(db_conn, title=None):
    """Selects description matching the title
    If no title is provides, it will select all descriptions
    :param db_conn: Conection to database"""
    if title:
        return db_operations.select_where(db_conn, queries.SELECT_DESCRIPTION_BY_TITLE, tuple(title))
    return db_operations.select_all(db_conn, queries.SELECT_DESCRIPTIONS)

def delete_row_by_title(db_conn, title):
    """Delete row where title matches the parameter title
    :param db_conn: Conection to database
    :param title: title to be deleted from database"""
    db_operations.delete_row(db_conn, queries.DELETE_ROW_BY_TITLE, tuple(title))
    