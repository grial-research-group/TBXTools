from TBXTools._results.bilingual import BilingualResults

class BilingualExtractor:
    """
    Class to manage the bilingual terminology extraction pipeline. 

    This class initializes two Extractor objects, source and target, and passes each argument respectively to each of them. Some arguments should be passed as tuples with the source on the left and target on the right, e.g. 'language=('en', 'ca') or 'corpus=("mental_health_en.txt", "mental_health_ca.txt")'. These arguments are: methodology, language, corpus stopwords, and inner_stopwords.

    Attributes:
        project_name (str): The unique name identifier for the current project. It determines the filename of the generated SQLite database.
        methodology (tuple of Methodology): The extraction strategy instance (e.g., LinguisticExtractor).
        language (tuple of str): The language of the corpus text. Can be the name of the language or the ISO code (e.g., 'english' or 'en').
        corpus (tuple of str or tuple of list): The  corpus used as the source for terminology extraction.
        stopwords (tuple of list): Stopwords list. They are automatically chosen if none are passed. It accepts a file path or a list of strings.
        inner_stopwords (tuple of list): Inner stopwords list. Used to filter multiword terms. They are automatically chosen if the language is es, ca, en, or fr. It accepts a file path or a list of strings.
        overwrite_project (bool): If True, overwrites existing project data in the database.
    """

    def __init__(self, project_name, methodology, language, corpus=None, stopwords=None, inner_stopwords=None, overwrite_project=False,):
        from .extractor import Extractor
        from .._processor.file_parser import FileParser
        from .._utils.utils import get_lang

        src_methodology, tgt_methodology = methodology
        src_stopwords, tgt_stopwords = stopwords if stopwords else None, None
        src_inner_stopwords, tgt_inner_stopwords = inner_stopwords if inner_stopwords else None, None
        src_language, tgt_language = language

        self.src_lang, self._src_lang_code = get_lang(src_language.lower())
        self.tgt_lang, self._tgt_lang_code = get_lang(tgt_language.lower())

        self.parser = FileParser(
            src_lang=self._src_lang_code,
            tgt_lang=self._tgt_lang_code,
        )

        src_corpus, tgt_corpus = self.parser.parse(corpus=corpus)
        
        self.src_extractor = Extractor(
            project_name=f"{project_name}-{self._src_lang_code}",
            methodology=src_methodology,
            corpus=src_corpus,
            stopwords=src_stopwords,
            inner_stopwords=src_inner_stopwords,
            language=self.src_lang,
            overwrite_project=overwrite_project
        )

        self.tgt_extractor = Extractor(
            project_name=f"{project_name}-{self._tgt_lang_code}",
            methodology=tgt_methodology,
            corpus=tgt_corpus,
            stopwords=tgt_stopwords,
            inner_stopwords=tgt_inner_stopwords,
            language=self.tgt_lang,
            overwrite_project=overwrite_project
        )

    def extract(self, verbose=False) -> BilingualResults:
        '''
        Coordinates the bilingual extraction pipeline by executing extraction independently on both source and target extractors and wrapping the combined outputs into a BilingualResults object.

        Args:
            verbose (bool, optional): If True, enables detailed logging. Defaults to False.

        Returns:
            BilingualResults: An instance containing both source and target extraction results.
        '''

        src_results = self.src_extractor.extract(verbose=verbose)

        tgt_results = self.tgt_extractor.extract(verbose=verbose)

        bilingual_results = BilingualResults(
            src_results=src_results,
            tgt_results=tgt_results,
            src_lang=self._src_lang_code,
            tgt_lang=self._tgt_lang_code
        )

        return bilingual_results
