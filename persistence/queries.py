CREATE_GENERATED_DESCRIPTIONS_TABLE = \
    """CREATE TABLE IF NOT EXISTS generated_descriptions(
    id integer PRIMARY KEY,
    title text NOT NULL,
    description text
    )"""

GENERATED_DESCRIPTIONS_COLUMNS = "id,title,description"

GENERATED_DESCRIPTIONS_TABLE_NAME = "generated_descriptions"

SELECT_TITLES = "SELECT title FROM {}".format(GENERATED_DESCRIPTIONS_TABLE_NAME)

SELECT_DESCRIPTIONS = "SELECT description FROM {}".format(GENERATED_DESCRIPTIONS_TABLE_NAME)

SELECT_DESCRIPTION_BY_TITLE = "SELECT description FROM {} WHERE title = ?".format(GENERATED_DESCRIPTIONS_TABLE_NAME)

SELECT_ALL = "SELECT * FROM {}"

UPDATE_DESCRIPTION = "UPDATE {} SET description = ? WHERE title = ?".format(GENERATED_DESCRIPTIONS_TABLE_NAME)

DELETE_ROW_BY_TITLE = "DELETE FROM {} WHERE title = ?".format(GENERATED_DESCRIPTIONS_TABLE_NAME)

def get_insert_query(table, columns):
    return "INSERT INTO {}({}) VALUES(?,?,?)".format(table,columns)