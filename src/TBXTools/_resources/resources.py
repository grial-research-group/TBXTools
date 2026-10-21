from importlib import import_module
import spacy

class Resources:
    '''
    Resource class to manage internal resources of the tool.

    Args:
        lang_code: The language ISO code of the corpus language.
    '''
    def __init__(self, lang, lang_code):
        self.lang = lang
        self.lang_code = lang_code

    def fetch_inner_stopwords(self):
        try:
            module = import_module(f"TBXTools._resources.inner.{self.lang_code}")

            return module.INNER_STOPWORDS
        
        except ModuleNotFoundError:
            return set()
        
    def get_spacy_stopwords(self):
        '''
        Fetches the spaCy stop words for a given language.
        '''
        try:
            lang_class = spacy.util.get_lang_class(self.lang_code)
            
            return lang_class.Defaults.stop_words
            
        except ImportError:
            raise ValueError(f"Cannot import stopwords because language {self.lang} ({self.lang_code}) is not supported in spaCy. You must pass an external list of stopwords to Extractor using the 'stopwords' argument.")