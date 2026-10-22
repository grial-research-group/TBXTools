def merge_databases(new_db_path, db_paths):
    '''
    Merges SQLite databases, where each extraction project is stored, into one database. This is especially useful with the BERT methodology. You can annotate data in different languages or domains and then merge them all into one database to train a unique model with.
    '''
    import sqlite3
    import shutil
    if len(db_paths) == 0:
        raise ValueError("The list of database paths is empty.")
        
    elif len(db_paths) == 1:
        raise ValueError(f"You need to pass 2 or more databases to merge")

    elif not isinstance(db_paths, list):
        raise ValueError(f"You need to pass a list of databases: ['db1.sqlite', 'db2.sqlite']")
    
    elif len(db_paths) > 1 and isinstance(db_paths, list):
        shutil.copy2(db_paths[0], new_db_path) # copies the first db into the new one
        print(f"Initialized {new_db_path} using {db_paths[0]}")
        conn = sqlite3.connect(new_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for db_path in db_paths[1:]: # continues merging dbs from the second one, since the first one was already copied
                print(f"Merging database: {db_path}...")
                
                try:
                    cursor.execute(f"ATTACH DATABASE '{db_path}' AS current_merge_db")

                    for table in tables:
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = [row[1] for row in cursor.fetchall() if row[5] == 0]
                        cols_string = ", ".join(columns)

                        try:
                            cursor.execute(f"INSERT INTO {table} ({cols_string}) SELECT {cols_string} FROM current_merge_db.{table}")

                        except sqlite3.OperationalError:
                            print(f"  -> Skipped table '{table}' (Not found in {db_path})")
                    
                    conn.commit()
                    
                except sqlite3.Error as e:
                    print(f"Failed to attach/process {db_path}: {e}")
                    
                finally:
                    cursor.execute("DETACH DATABASE current_merge_db")

            print(f"\nAll databases fully merged into {new_db_path}")

        except sqlite3.Error as e:
            print(f"A critical error occurred: {e}")

        finally:
            conn.close()