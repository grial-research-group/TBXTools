from .._sqlite.sqlite import SQLite
from .._results.results import Results
from .._processor.preprocessor import Preprocessor
from .._utils.utils import get_lang
from .._processor.file_parser import FileParser
import time


class Extractor:
    """
    Class to manage the monolingual terminology extraction pipeline. This class acts as the main controller, managing the interaction between the chosen extraction methodology, database storage, and processing components.

    Attributes:
        project_name (str): The unique name identifier for the current project. It determines the filename of the generated SQLite database.
        methodology (object): The extraction strategy instance (e.g., LinguisticExtractor).
        language (str): The language of the corpus text. Can be the name of the language or the ISO code (e.g., 'english' or 'en').
        corpus: The  corpus used as the source for terminology extraction.
        stopwords (list): Stopwords list. They are automatically chosen if none are passed. It accepts a file path or a list of strings.
        inner_stopwords (list): Inner stopwords list. Used to filter multiword terms. They are automatically chosen if the language is es, ca, en, or fr. It accepts a file path or a list of strings.
        overwrite_project (bool): If True, overwrites existing project data in the database.
    """

    def __init__(self, project_name, methodology, language, corpus=None, stopwords=None, inner_stopwords=None, overwrite_project=False):

        self.lang, self._lang_code = get_lang(language.lower())
        self._methodology = methodology
        self.parser = FileParser(src_lang=self._lang_code)
        _corpus = self.parser.parse(corpus=corpus)
        
        self._sqlite = SQLite(
            project_name=project_name,
            stopwords=stopwords,
            inner_stopwords=inner_stopwords,
            corpus=_corpus,
            is_corpus_tagged=getattr(
                self._methodology, 'is_corpus_tagged', False),
            linguistic_patterns=getattr(
                self._methodology, 'linguistic_patterns', None),
            evaluation_terms=getattr(
                self._methodology, 'evaluation_terms', None),
            overwrite_project=overwrite_project,
            lang_code=self._lang_code,
            lang=self.lang
        )

        self.stopwords = self._sqlite.get("stopwords")
        self.inner_stopwords = self._sqlite.get("inner_stopwords")

        self._methodology.extractor = self

# EXTRACTION FUNCTIONS
    def extract(self, timer=False, verbose=False) -> Results:
        '''
        Coordinates the extraction pipeline by fetching data from the database, calling the selected extraction methodology (linguistic or statistical), applying optional filtering/normalization procedures, and persisting the extracted candidates back to the SQLite database.

        Args:
            verbose (bool, optional): If True, enables detailed logging. Defaults to False.

        Returns:
            Results: An instance of the Results class.
        '''

        self._methodology.extractor = self

        # if we are not overwriting and the calculations have been done
        if self._sqlite.overwrite_project == False and self._sqlite.table_is_populated("candidate_terms"):
            print("Fetching data from database", flush=True)
            candidate_terms = self._sqlite.get_candidate_terms()
            ngrams = self._sqlite.get_ngrams()
            tokens = self._sqlite.get("tokens")
            tagged_ngrams = self._sqlite.get_ngrams(tagged=True)
            linguistic_patterns = self._sqlite.get("linguistic_patterns")

            results = Results(
                terms=candidate_terms,
                ngrams=ngrams if ngrams else None,
                tokens=tokens,
                tagged_ngrams=tagged_ngrams if tagged_ngrams else None,
                linguistic_patterns=linguistic_patterns if linguistic_patterns else None
            )

        else:
            print(f"\n{self._methodology.name} initialized", flush=True)
            print("Running term extraction", flush=True)

            # need to change when using yield in get segments
            segments = list(self._sqlite.get_segments(tagged=False))

            results = self._methodology.run(segments=segments, verbose=verbose)

            self._sqlite.insert_candidate_terms(results._terms)

        results._extractor = self
        results._methodology = self._methodology

        print("Term extraction finished", flush=True)

        if timer:
            end = time.time()
            length = end - self._methodology.start
            print(f"\nExtraction time: {length:.3f} seconds")

        return results

    def add_stopwords(self, stopwords_list):
        '''
        Adds standard stopwords to the project and updates the processor. Inserts the provided list of stopwords into the SQLite database and refreshes the internal processor's active stopword list.

        Args:
            stopwords_list (list[str]): A list of stopwords. 
        '''
        if isinstance(stopwords_list, list):
            self._sqlite.add_stopwords(stopwords_list=stopwords_list)
            self.stopwords = self._sqlite.get("stopwords")
            # updating the attribute of the class
            self._methodology.processor.stopwords = self.stopwords

    def add_inner_stopwords(self, inner_stopwords_list):
        '''
        Adds inner stopwords to the project and updates the processor. Inserts the provided list of inner stopwords into the SQLite database and refreshes the internal processor's active inner stopword list.

        Args:
            inner_stopwords_list (list[str]): A list of inner stopwords.
        '''
        if isinstance(inner_stopwords_list, list):
            self._sqlite.add_inner_stopwords(
                inner_stopwords_list=inner_stopwords_list)
            self._methodology.processor.inner_stopwords = self._sqlite.get(
                "inner_stopwords")
            self.inner_stopwords = self._sqlite.get("inner_stopwords")
