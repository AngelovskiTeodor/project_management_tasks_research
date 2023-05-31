from persistence import actions as db_actions
from data_utils import get_data_from_csv as get_data, get_rows_with_missing_values as get_empty_rows

def insert_titles_with_missing_descriptions(db_conn, dataframe):
    for index, row in dataframe.iterrows():
        db_actions.insert_title(db_conn, row.id, row.title)
        print(index)    # debugging

if __name__=="__main__":
    df = get_data()
    df.info()

    sub_df = get_empty_rows(df, "description")
    sub_df = sub_df[["id","title"]]
    sub_df.info()

    db = db_actions.get_db_connection()
    db_actions.create_generated_descriptions_table(db)

    insert_titles_with_missing_descriptions(db, sub_df)
    
    print("Number of titles in database: ", end='')
    print(db_actions.count_total_entries(db))

    print("Number of descriptions in database: ",end='')
    print(db_actions.count_descriptions(db))

    print("Number of missing descriptions in database: ", end='')
    print(db_actions.count_missing_descriptions(db))
