
class LinguisticTagger:
    '''
    Manages linguistic POS-tagging
    
    Attributes:
        model_name (str): The name of the spaCy model to load (e.g.,'en_core_web_sm').
        nlp (spacy.Language or None): The loaded spaCy pipeline instance used for tagging.
    '''
    def __init__(self, nlp):
        self.nlp = nlp
    
    def tag_segment(self, segment: str) -> str:
        """
        Tags a single segment and returns it in the 'word|lemma|POS' format, separated by tabs.

        Args:
        segment (str): The input text string to be processed.

        Returns:
        str: A tab-separated string of tokens, or an empty string if input is invalid.
        """
        if not segment or not self.nlp:
            return ""
            
        doc = self.nlp(segment)
        data = []
        for token in doc:
            if not token.is_space:
                data.append(f"{token.text}|{token.lemma_}|{token.pos_}")
        
        return " ".join(data) if data else ""