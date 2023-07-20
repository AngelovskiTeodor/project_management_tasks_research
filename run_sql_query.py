from persistence.actions import get_db_connection as get_db
from persistence.actions import run_sql_query as run

def run_sql_query(query):
    """Runs SQL query"""
    db = get_db()
    result = run(db,sql_query)
    return result

if __name__=="__main__":
    print("Enter SQL Query to be executed: ", end='')
    sql_query = input()
    result = run_sql_query(sql_query)
    print(result)
