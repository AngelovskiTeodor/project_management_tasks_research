from persistence import actions as db_actions
import pandas

def get_titles_with_missing_descrtiptions(db_conn, limit=None):
    """Return id and title of tasks with missing descriptions"""
    titles = db_actions.get_missing_descriptions_only(db_conn, limit)
    titles_dataframe = pandas.DataFrame(titles, columns = ['id', 'title', 'description'])
    return titles_dataframe

if __name__=="__main__":
    db = db_actions.get_db_connection()
    titles_df = get_titles_with_missing_descrtiptions(db)
    titles_df.info()
