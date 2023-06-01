from persistence.actions import get_db_connection as get_db
from persistence.actions import run_sql_query as run

if __name__=="__main__":
    db = get_db()
    print("Enter SQL Query to be executed: ", end='')
    sql_query = input()
    result = run(db,sql_query)
    print(result)
