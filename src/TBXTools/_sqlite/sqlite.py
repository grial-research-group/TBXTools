import sqlite3
import os
from pathlib import Path

class SQLite:
    '''
    Manage SQLite functions.
    '''

    def __init__(self, project_name, corpus, stopwords=None, inner_stopwords=None, is_corpus_tagged=False, linguistic_patterns=None, evaluation_terms=None, external_terms=None, overwrite_project=False, lang=None, lang_code=None):
        from TBXTools._resources.resources import Resources

        self.cur = None
        self.MAX_INSERTS = 10000
        self.overwrite_project = overwrite_project
        self.project_name = None
        self.resources = Resources(lang=lang, lang_code=lang_code)

        self.TABLES_TO_LOAD_AT_START = ["corpus", "tagged_corpus", "stopwords", "inner_stopwords", "linguistic_patterns", "evaluation_terms", "external_terms"]
        
        self.descriptive_statistics_data = {}
        
        self.lang = lang
        self._lang_code = lang_code
        
        load_data = self.initialize_project(
            project_name=project_name, 
            overwrite_project=self.overwrite_project)
        
        if load_data:
            print("Loading data to database")

            self.load_data_to_tables(
                table_names=self.TABLES_TO_LOAD_AT_START, 
                corpus=corpus, 
                stopwords=stopwords, 
                inner_stopwords=inner_stopwords, 
                linguistic_patterns=linguistic_patterns,
                evaluation_terms=evaluation_terms,
                external_terms=external_terms,
                is_corpus_tagged=is_corpus_tagged, 
                lang=self._lang_code)

    def add_extension(self, project_name):
        '''Adds the extension .sqlite to the database file.'''
        return Path(project_name).with_suffix('.sqlite')

    def initialize_project(self, project_name, overwrite_project):
        '''Initialize the SQLite project, either opening an existing one or creating a new one.'''
        file_name = self.add_extension(project_name=project_name)
        self.project_name = file_name.stem

        if file_name.exists() and overwrite_project==False:
            self.open_project(project_name=project_name)

            return False

        elif file_name.exists() and overwrite_project==True:
            self.create_project(project_name=project_name, overwrite=True)  

            return True

        elif not file_name.exists():
            self.create_project(project_name=project_name)

            return True

        elif not file_name.exists() and overwrite_project==True:
            print("WARNING: overwrite_project is True, but database does not exist.", flush=True)
            self.create_project(project_name=project_name)

            return True

    def create_project(self,project_name,overwrite=False):
        '''Opens a project. If the project already exists, it raises an exception. To avoid the exception use overwrite=True. To open existing projects, use the open_project method.'''

        project_name = self.add_extension(project_name)
        print(f"\nCreating project: {project_name}" if not overwrite else f"\nOverwriting project: {project_name}")
        
        if os.path.isfile(project_name) and overwrite:
            os.remove(project_name)

        self.conn = sqlite3.connect(project_name)

        with self.conn:
            self.cur = self.conn.cursor()
            self.cur.execute("CREATE TABLE corpus(id INTEGER PRIMARY KEY AUTOINCREMENT, segment TEXT)")
            self.cur.execute("CREATE TABLE tokenized_corpus(id INTEGER PRIMARY KEY AUTOINCREMENT, tokenized_segment TEXT)")
            self.cur.execute("CREATE TABLE tokens (id INTEGER PRIMARY KEY AUTOINCREMENT, token TEXT, frequency INTEGER)")
            self.cur.execute("CREATE TABLE ngrams (id INTEGER PRIMARY KEY AUTOINCREMENT, ngram TEXT, n INTEGER, frequency INTEGER)")
            self.cur.execute("CREATE TABLE candidate_terms (id INTEGER PRIMARY KEY AUTOINCREMENT, candidate TEXT, n INTEGER, measure TEXT, value INTEGER)")
            self.cur.execute("CREATE TABLE external_terms (id INTEGER PRIMARY KEY AUTOINCREMENT, external_term TEXT)")
            self.cur.execute("CREATE TABLE tsr_terms (id INTEGER PRIMARY KEY AUTOINCREMENT, tsr_term TEXT)")
            self.cur.execute("CREATE TABLE stopwords (id INTEGER PRIMARY KEY AUTOINCREMENT, stopword TEXT)")
            self.cur.execute("CREATE TABLE inner_stopwords (id INTEGER PRIMARY KEY AUTOINCREMENT, inner_stopword TEXT)")
            self.cur.execute("CREATE TABLE exclusion_regexes (id INTEGER PRIMARY KEY AUTOINCREMENT, exclusion_regex TEXT)")
            self.cur.execute("CREATE TABLE linguistic_patterns (id INTEGER PRIMARY KEY AUTOINCREMENT, linguistic_pattern TEXT)")
            self.cur.execute("CREATE TABLE tagged_ngrams (id INTEGER PRIMARY KEY AUTOINCREMENT, tagged_ngram TEXT, n INTEGER, frequency INTEGER)")
            self.cur.execute("CREATE TABLE tagged_corpus(id INTEGER PRIMARY KEY AUTOINCREMENT, tagged_segment TEXT)")
            self.cur.execute("CREATE TABLE evaluation_terms(id INTEGER PRIMARY KEY AUTOINCREMENT, evaluation_term TEXT)")
            self.cur.execute("CREATE TABLE segment_labels(id INTEGER PRIMARY KEY AUTOINCREMENT, labels TEXT)")
            self.cur.execute("CREATE TABLE lemmatized_corpus(id INTEGER PRIMARY KEY AUTOINCREMENT, lemmatized_segment TEXT)")
            self.cur.execute("CREATE TABLE word_tokens(id INTEGER PRIMARY KEY AUTOINCREMENT, word_tokens TEXT)")

    def open_project(self,project_name):
        '''Opens an existing project.'''

        project_name = self.add_extension(project_name)
        print(f"Opening project: {project_name}", flush=True)

        self.conn = sqlite3.connect(project_name)
        self.cur = self.conn.cursor() 

    def read_corpus(self, corpus_generator, is_corpus_tagged=False, batch_size=1000):
        '''Reads a corpus in batches'''
        from .._processor.normalizer import Normalizer

        batch = []

        for segment in corpus_generator:
            clean_segment = Normalizer.normalize(segment)
            batch.append((clean_segment,))

            if len(batch) == batch_size:
                self.insert_segments(data=batch, tagged=is_corpus_tagged)

        if batch:
            self.insert_segments(data=batch, tagged=is_corpus_tagged)

    # LOAD METHODS
    def load_corpus(self, corpus, is_corpus_tagged=False):
        from typing import Generator
        
        if corpus:
            if isinstance(corpus, Generator):
                self.read_corpus(corpus_generator=corpus, is_corpus_tagged=is_corpus_tagged)

                print("Corpus loaded", flush=True)
            
            elif isinstance(corpus, list):
                if isinstance(corpus[0], Generator):
                    for corpus_gen in corpus:
                        self.read_corpus(corpus_generator=corpus_gen, is_corpus_tagged=is_corpus_tagged)

                    print(f"{len(corpus)} corpora loaded", flush=True)

                else:
                    self.read_corpus(corpus_generator=corpus)

                    print("Segments loaded", flush=True)
           
            else:
                raise ValueError("Corpus is not a Generator object")
            
    def load_stopwords(self, stopwords , encoding="utf-8"):
        '''Load the stopwords into the database.
        
        Args:
            stopwords: A stopwords list or file.
        '''
        data=[]

        if isinstance(stopwords, set) or isinstance(stopwords, list):
            data = [(word,) for word in sorted(stopwords)]
            print("Stopwords loaded")

        else:
            with open(stopwords, "r", encoding=encoding) as f:
                data = [(line.rstrip(),) for line in f]
            print("Stopwords loaded from file") 

        with self.conn:
            self.cur.executemany("INSERT INTO stopwords (stopword) VALUES (?)",data) 

    def load_inner_stopwords(self, inner_stopwords, encoding="utf-8"):
        '''Load the inner stopwords into the database.
        
        Args:
        
            stopwords: A inner stopwords list or file.
        '''
        data=[]

        if isinstance(inner_stopwords, set):
            data = [(word,) for word in sorted(inner_stopwords)]
            print("Inner stopwords loaded")

        else:
            with open(inner_stopwords, "r", encoding=encoding) as f:
                data = [(line.rstrip(),) for line in f]
            
            print("Inner stopwords loaded from file") 

        with self.conn:
            self.cur.executemany("INSERT INTO inner_stopwords (inner_stopword) VALUES (?)",data) 

    def load_linguistic_patterns(self, linguistic_patterns, encoding="utf-8"):
        data= []

        if linguistic_patterns:
            
            if isinstance(linguistic_patterns, list):
                data = [(linguistic_pattern,) for linguistic_pattern in linguistic_patterns]
                print("Linguistic patterns loaded")

            else:
                with open(linguistic_patterns, "r", encoding=encoding) as f:
                    first_line= f.readline()
                    if "frequency" in first_line.lower():

                        for line in f:
                            line = line.rstrip()
                            if line:
                                parts = line.split('\t')
                                data.append((parts[0],))
                    
                    else:
                        if first_line.strip():
                            data.append((first_line.strip().split('\t')[0],))
                            
                        for line in f:
                            line = line.strip()
                            if line:
                                data.append((line.split('\t')[0],))

                print("Linguistic patterns loaded from file")
                
            with self.conn:
                self.cur.executemany("INSERT INTO linguistic_patterns (linguistic_pattern) VALUES (?)",data) 

    def load_evaluation_terms(self, evaluation_terms, encoding='utf-8'):
        '''Loads the evaluation terms for the automatic learning of POS patterns'''

        data= []

        if not evaluation_terms:
            print("No evaluation terms to load into the database")
            return 
        
        if isinstance(evaluation_terms, list):
            data = [(evaluation_term,) for evaluation_term in evaluation_terms]
            print("Evaluation terms loaded")

        else: 
            with open(evaluation_terms, "r", encoding=encoding) as f:
                data = [(line.rstrip(),) for line in f]       
            print("Evaluation terms loaded from file")

        with self.conn:
            self.cur.executemany('INSERT INTO evaluation_terms (evaluation_term) VALUES (?)',data)
    
    def load_tsr_terms(self, tsr_terms,encoding="utf-8"):
        '''Loads the TSR terms from a text file (one term per line).'''
        data = []

        if not tsr_terms:
            print("TSR terms not found. Not applying TSR filtering.")
            return
        
        if isinstance(tsr_terms, list):
            data = [(tsr_term,) for tsr_term in tsr_terms]
            print("TSR terms loaded")

        else: 
            with open(tsr_terms, "r", encoding=encoding) as f:
                data = [(line.rstrip(),) for line in f]        
            print("TSR terms loaded from file")

        with self.conn:
            self.cur.executemany('INSERT INTO tsr_terms (tsr_term) VALUES (?)',data)

    def load_exclusion_regexes(self, exclusion_regexes, encoding='utf-8'):
        '''Loads the exclusion regular expressions for the source language.'''
        data=[]
        if exclusion_regexes:
            if isinstance(exclusion_regexes, list):
                data = [(regex,) for regex in exclusion_regexes]

                print("Exclusion regexes loaded")
            else:
                with open(exclusion_regexes, "r", encoding=encoding) as f:
                    data = [(line.rstrip(),) for line in f]
                print("Exclusion regexes loaded from file")

            with self.conn:
                self.cur.executemany('INSERT INTO exclusion_regexes (exclusion_regex) VALUES (?)',data)

    def load_external_terms(self, external_terms, encoding='utf-8'):
        data = []
        if external_terms:
            if isinstance(external_terms, list):
                data = [(external_term,) for external_term in external_terms]

                print("External terms loaded")
            else:
                with open(external_terms, "r", encoding=encoding) as f:
                    data = [(line.rstrip(),) for line in f]
                print("External terms loaded from file")
            
            data = set(data)
            with self.conn:
                self.cur.executemany('INSERT INTO external_terms (external_term) VALUES (?)', data)

    # INSERT METHODS
    def insert_segments(self, data, tagged=False, tokenized=False):
        with self.conn:
            if tagged:
                self.cur.executemany("INSERT INTO tagged_corpus (tagged_segment) VALUES (?)", data)
            elif tokenized:
                self.cur.executemany("INSERT INTO tokenized_corpus (tokenized_segment) VALUES (?)", data)
            else:
                self.cur.executemany("INSERT INTO corpus (segment) VALUES (?)", data)

    def insert_ngrams(self, data, tagged=False):
        '''Inserts Ngrams and Tagged Ngrams into the database.'''
        
        table = "tagged_ngrams" if tagged else "ngrams"
        column = "tagged_ngram" if tagged else "ngram"

        if not self.table_is_populated(table):
            query = f"INSERT INTO {table} ({column}, n, frequency) VALUES (?,?,?)"
            with self.conn:
                self.cur.executemany(query, data)

    # def insert_ngrams(self, data):
    #     '''Inserts ngrams into the database'''
    #     if not self.table_is_populated("ngrams"):
    #         data = data.items()

    #         for ngram, freq in data:
    #             with self.conn:
    #                 self.cur.execute("INSERT INTO ngrams (ngram, frequency) VALUES (?,?)", (ngram, freq))
    
    def insert_tokens(self, data):
        '''Inserts tokens into the database'''
        if not self.table_is_populated("tokens"):
            data = data.items()

            for token, freq in data:
                with self.conn:
                    self.cur.execute("INSERT INTO tokens (token, frequency) VALUES (?,?)", (token, freq))

    def insert_candidate_terms(self, data):
        '''Inserts candidate terms into the database'''
        if not self.table_is_populated("candidate_terms"):
            with self.conn:
                self.cur.executemany("INSERT INTO candidate_terms (candidate, n, measure, value) VALUES (?,?,?,?)", data)
            
    def insert_filtered_candidate_terms(self, data):
        with self.conn:
            self.cur.executemany("INSERT INTO filtered_candidate_terms (filtered_candidate, n, frequency, measure, value) VALUES (?,?,?,?,?)", data)

    def insert_linguistic_patterns(self, data):
        '''Insert linguistic candidates into the database.'''
        if not self.table_is_populated("linguistic_patterns"):
            with self.conn:
                self.cur.executemany("INSERT INTO linguistic_patterns (linguistic_pattern) VALUES (?)", data)

    def insert_segment_labels(self, data):
        '''Insert segment labels to train BERT models into the database.'''
        data= data.apply(lambda lst: [str(x) for x in lst])
        data = [(" ".join(labels),) for labels in data] # use when strings
        if not self.table_is_populated("segment_labels"):
            with self.conn:
                self.cur.executemany("INSERT INTO segment_labels (labels) VALUES (?)", data)

    def insert_lemmatized_corpus(self, data):
        '''Inserts lemmas into the database'''
        data = [" ".join(segment) for segment in data]

        data = [(segment,) for segment in data]
        if not self.table_is_populated("lemmatized_corpus"):
            with self.conn:
                self.cur.executemany("INSERT INTO lemmatized_corpus (lemmatized_segment) VALUES (?)", data)

    def insert_word_tokens(self, data):
        '''Inserts spacy word tokens into the database'''
        data = [" ".join(segment) for segment in data]

        data = [(segment,) for segment in data]
        if not self.table_is_populated("word_tokens"):
            with self.conn:
                self.cur.executemany("INSERT INTO word_tokens (word_tokens) VALUES (?)", data)

    # GET METHODS
    def get_segments(self, tagged=False, tokenized=False, to_list=False):
        '''Gets the segmented corpus as a list of segments from the database.'''
        with self.conn:
            if tagged:
                self.cur.execute("SELECT tagged_segment from tagged_corpus")
            
            elif tokenized:
                self.cur.execute("SELECT tokenized_segment from tokenized_corpus")
            
            else:
                self.cur.execute("SELECT segment from corpus")
                
            for row in self.cur:
                if to_list:
                    yield row[0].split()
                else:
                    yield row[0]
        
    def get_ngrams(self, tagged=False):
        '''Gets the list of Ngrams of tagged ngrams from the database'''
        data = []
        with self.conn:
            if tagged:
                self.cur.execute("SELECT tagged_ngram, n, frequency FROM tagged_ngrams ORDER BY frequency DESC")
            else:
                self.cur.execute("SELECT ngram, n, frequency FROM ngrams ORDER BY frequency DESC")

            for row in self.cur.fetchall():
                data.append(row)

        return data

    def get_candidate_terms(self):
        '''Gets the list of candidate terms from the database'''
        candidate_terms = []
        with self.conn:
            self.cur.execute("SELECT candidate, n, measure, value FROM candidate_terms ORDER BY value DESC")

            for candidates_row in self.cur.fetchall():
                candidate_terms.append(candidates_row)

        return candidate_terms

    def get(self, table):
        exception = {"exclusion_regexes" : "exclusion_regex"} #hay que encontrar una logica mejor, que podría ser darle el mismo nombre a tabla y columna
        if table in exception:
            column_name = exception[table]
        else:
            column_name = table.rstrip('s') 

        items = []
        with self.conn:
            self.cur.execute(f"SELECT {column_name} FROM {table}")
            for item in self.cur.fetchall():
                items.append(item[0])

        return items

    def new_get(self, column, table): # build a dict with table name and expression?
        items = []
        with self.conn:
            self.cur.execute(f"SELECT {column} FROM {table}")
            for item in self.cur.fetchall():
                items.append(item[0])

        return items
       
    def get_external_terms(self): 
        external_terms= []
        with self.conn:
            self.cur.execute("SELECT external_term FROM external_terms")

            for external_term in self.cur.fetchall():
                external_terms.append(external_term[0])

        return external_terms
    
    def get_lemmatized_corpus(self):
        lemmatized_corpus= []
        with self.conn:
            self.cur.execute("SELECT lemmatized_segment FROM lemmatized_corpus")

            for row in self.cur.fetchall():
                lemmatized_corpus.append(row[0])

        return lemmatized_corpus
    
    def get_word_tokens(self):
        word_tokens= []
        with self.conn:
            self.cur.execute("SELECT word_tokens FROM word_tokens")

            for word_tokens_row in self.cur.fetchall():
                word_tokens.append(word_tokens_row[0].split()) # as list for bert

        return word_tokens
    
    def get_segment_labels(self):
        labels = []
        with self.conn:
            self.cur.execute("SELECT labels FROM segment_labels")

            for labels_row in self.cur.fetchall():
                labels.append(labels_row[0].split())

        return labels
    
    def delete(self, table):
            with self.conn:
                self.cur.execute(f"DELETE FROM {table}")
                self.cur.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
  
    def delete_specific_candidate_term(self, candidates):
        with self.conn:
            for candidate in candidates:
                self.cur.execute("DELETE FROM candidate_terms WHERE candidate=?", (candidate,))

# ADD FUNCTIONS

# se pueden unificar
    def add_stopwords(self, stopwords_list):
        '''Add stopwords to the database that do not exist already.'''
        current_stopwords = self.get("stopwords")
        data = []
        for stopword in stopwords_list:
            if stopword in set(current_stopwords):
                continue
            else:
                data.append((stopword,))
        
        with self.conn:
            self.cur.executemany("INSERT INTO stopwords (stopword) VALUES (?)",data) 

    def add_inner_stopwords(self, inner_stopwords_list):
        '''Add inner stopwords to the database that do not exist already.'''
        current_inner_stopwords = self.get("inner_stopwords")
        data = []
        for inner_stopword in inner_stopwords_list:
            if inner_stopword in set(current_inner_stopwords):
                continue
            else:
                data.append((inner_stopword,))
        
        with self.conn:
            self.cur.executemany("INSERT INTO inner_stopwords (inner_stopword) VALUES (?)",data) 

# CHECK FUNCTIONS
    def table_is_populated(self, table_name):
        '''Checks if a table in the database contains data.'''
        with self.conn:
            self.cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = self.cur.fetchone()[0]
            if count > 0:
                return True
            else:
                return False
            
    def load_data_to_tables(self, table_names, corpus, is_corpus_tagged, stopwords, inner_stopwords, linguistic_patterns, evaluation_terms, external_terms, lang=None):

        loaders = {}

        if not stopwords:
            stopwords = self.resources.get_spacy_stopwords()
        loaders["stopwords"] = lambda: self.load_stopwords(stopwords=stopwords)

        if not inner_stopwords:
            inner_stopwords = self.resources.fetch_inner_stopwords()
        loaders["inner_stopwords"] = lambda: self.load_inner_stopwords(inner_stopwords=inner_stopwords)

        if is_corpus_tagged==False:
            loaders["corpus"] = lambda: self.load_corpus(corpus=corpus, is_corpus_tagged=False)

        elif is_corpus_tagged==True:
            loaders["tagged_corpus"] = lambda: self.load_corpus(corpus=corpus, is_corpus_tagged=True)

        if evaluation_terms:
            loaders["evaluation_terms"] = lambda: self.load_evaluation_terms(evaluation_terms=evaluation_terms)

        if linguistic_patterns:
            loaders["linguistic_patterns"] = lambda: self.load_linguistic_patterns(linguistic_patterns=linguistic_patterns)

        if external_terms:
            loaders["external_terms"] = lambda: self.load_external_terms(external_terms=external_terms)

        for table in table_names:
            loader = loaders.get(table)
            # if the table does not have data, it is loaded in
            if loader and not self.table_is_populated(table_name=table):
                loader()

    def calculate_descriptive_statistics(self):
        tables = ["corpus", "candidate_terms"]

        with self.conn:
            for table in tables:
                # counting segments and terms
                self.cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = self.cur.fetchone()[0]

                self.descriptive_statistics_data[table] = count

                # getting nmin and nmax and its top terms from the candidates ta ble
                if table == "candidate_terms":
                    self.cur.execute("SELECT MIN(n), MAX(n) FROM candidate_terms")

                    nmin, nmax = self.cur.fetchone()

                    for n in range(nmin, nmax+1): 
                        self.cur.execute("SELECT candidate,n,measure,value FROM candidate_terms WHERE n=? ORDER BY value DESC", (n,))
                        
                        self.descriptive_statistics_data[str(n)] = self.cur.fetchall()

                        self.descriptive_statistics_data["nrange"] = [nmin, nmax]