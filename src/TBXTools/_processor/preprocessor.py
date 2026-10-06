import re

class Preprocessor():

    def __init__(self, methodology):
        import string
        from nltk.probability import FreqDist

        self.methodology = methodology
        self.nmin = getattr(self.methodology, 'nmin', None)
        self.nmax = getattr(self.methodology, 'nmax', None)

        self.tokens_freq_dist = FreqDist()
        self.ngrams_freq_dist = FreqDist()

        self.punctuation = [
            char for char in string.punctuation if char != ["'", "|"]]
        self.stopwords = None
        self.inner_stopwords = None

        self.model_name = None
        self.nlp = None

    def _set_filter_parameters(self):
        self.stopwords = self.methodology.extractor.stopwords
        self.inner_stopwords = self.methodology.extractor.inner_stopwords
        self.invalid_tokens = set(self.stopwords) | set(self.punctuation)

    def _calculate_tokens_freq_dist(self, tokenized_segment):
        self.tokens_freq_dist.update(tokenized_segment)

    def _calculate_ngrams_freq_dist(self, ngrams_list):
        self.ngrams_freq_dist.update(ngrams_list)

    def _extract_clean_ngrams_from_tagged(self):
        ngrams = []
        for ngram, freq in self.ngrams_freq_dist.most_common():
            ngram = ngram.split()
            words = []
            for element in ngram:
                element = element.split("|")

                words.append(element[0])

            words = " ".join(words)
            ngrams.append((words, freq))

        return ngrams
    
    def _compute_ngrams(self, tokenized_segment):
        from nltk.util import ngrams as calc_ngrams_nltk

        ngrams = []
        for n in range(self.nmin, self.nmax + 1):
            segment_ngrams = calc_ngrams_nltk(tokenized_segment, n)

            ngrams.extend(segment_ngrams)

        return ngrams

    def _filter_ngram(self, ngram):
        """
        Filters an ngram by checking for invalid stopwords and punctuation. It is rejected if it contains a stopword or punctation at its boundaries (first/last element) or an inner stopword or punctuation in its middle token(s) (between the first/last elements).

        Args: 
          ngram(str): The ngram to validate.

        Returns:
          str or None: The original term string if it passes all  filters, otherwise None.

        """
        split_ngram = ngram.lower().split()

        # stopwords and punctuation at boundaries
        if split_ngram[0] in self.invalid_tokens or split_ngram[-1] in self.invalid_tokens:
            return None

        # inner stopwords and punctuation
        for token in split_ngram[1:-1]:
            if token in self.inner_stopwords or token in self.punctuation:
                return None

        return ngram.strip()

    def _filter_tagged_ngram(self, tagged_ngram):
        """
        Filters a tagged ngram by checking for invalid stopwords and punctuation. It is rejected if it contains a stopword or punctation at its boundaries (first/last element) or an inner stopword or punctuation in its middle token(s) (between the first/last elements).

        Args: 
          ngram(str): The candidate term string to validate.

        Returns:
          str or None: The original term string if it passes all stopword filters, otherwise None.
        """
        tagged_ngram = (" ").join(tagged_ngram)

        split_ngram = tagged_ngram.lower().split()

        # Mental|Mental|PROPN 	 Disorders|Disorders|PROPN
        first_element = split_ngram[0].split("|")

        word = first_element[1] if len(first_element) > 1 else (
            first_element[0] if len(first_element) > 0 else "")

        if word and word in self.invalid_tokens:
            return None

        last_element = split_ngram[-1].split("|")
        last_word = last_element[1] if len(last_element) > 1 else (
            last_element[0] if len(last_element) > 0 else "")

        if last_word and last_word in self.invalid_tokens:
            return None

        for token in split_ngram[1:-1]:
            if token.split("|")[0] in self.inner_stopwords or token.split("|")[0] in self.punctuation:
                return None

        return tagged_ngram

    def _clean_ngram(self, ngram):
        """
        Cleans an ngram removing punctuation from the beginning and end of the string.
        """
        ngram = ngram.strip()

        cleaned = re.sub(r"^[,.\-:;\"¿]+|[,.\-:;\"?]+$", "", ngram)

        cleaned = cleaned.strip()
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1]

        # elif cleaned.startswith("("): # ¿¿??
        #     cleaned = cleaned.lstrip("(")

        cleaned = cleaned.strip()

        if cleaned.startswith("(") and not cleaned.endswith(")"):
            cleaned = cleaned[1:]

        if cleaned.count("(") != cleaned.count(")"):
            return None

        return cleaned

# LING METHODOLOGY
    def create_tagged_segments(self, segments, lang_code):
        """
        Pos tags a list of text segments.

        Args:
          segment (list of str): The input text string to be processed.

        Returns:
          tagged_segments (list of str): A list of POS tagged segments.
        """
        from .._utils.utils import get_spacy_model_from_code, load_spacy_model
        from ..methodology.linguistic.tagger import LinguisticTagger
        from tqdm import tqdm

        if lang_code and not self.model_name:
            self.model_name = get_spacy_model_from_code(lang_code)

            self.nlp = load_spacy_model(self.model_name)

        tagger = LinguisticTagger(self.nlp)

        tagged_segments = []
        for segment in tqdm(segments, total=len(segments), desc="Tagging segments"):

            single_tagged_segment = tagger.tag_segment(segment)

            if single_tagged_segment:
                tagged_segments.append(single_tagged_segment)

        return tagged_segments

    def translate_pattern(self, linguistic_patterns):
        """
        Translates a list of linguistic patterns into valid regular expressions. This method processes each pattern string (or the first element of a tuple), tokenizes it by whitespace, and converts specific custom syntax elements into regex equivalents.

        Args:
           linguistic_patterns (list of str or list of tuple): A list containing the linguistic patterns to be translated. If an element is a tuple, only the first string item is processed.

        Returns:
           list of str: A list of compiled regular expression strings."""

        translated_patterns= []

        for pattern_str in linguistic_patterns:
            if isinstance(pattern_str, tuple):
                pattern_str = pattern_str[0]

            aux = []
            for ptoken in pattern_str.split():
                auxtoken = []
                ptoken = ptoken.replace(".*", "[^\s]+") 
                for pelement in ptoken.split("|"):
                    if pelement == "#":
                        auxtoken.append("([^\s]+?)")                    
                    elif pelement == "":
                        auxtoken.append("[^\s]+?")
                    else:
                        if pelement.startswith("#"):
                            auxtoken.append("(" + pelement.replace("#", "") + ")")
                        else:
                            auxtoken.append(pelement)
                aux.append("\|".join(auxtoken))
            tp = "(" + " ".join(aux) + ")"
            
            translated_patterns.append(tp)

        return translated_patterns

    def _generate_linguistic_patterns(self, evaluation_terms, clean_ngrams, output_file, verbose=False):
        print("Linguistic patterns not found. Starting automatic pattern learning")
        from ..methodology.linguistic.patterns_learning import PatternsLearning
        from tqdm import tqdm
        ngrams_for_patterns = []
        pattern_learner = PatternsLearning()

        for term in tqdm(evaluation_terms, total=len(evaluation_terms), desc="Extracting n-grams for pattern learning"):
            for row in zip(clean_ngrams, list(self.ngrams_freq_dist.most_common())):

                clean_term = row[0][0]
                tagged_term = row[1][0]
                freq = row[1][1]

                if term == clean_term:
                    ngrams_for_patterns.append((tagged_term, freq))

        linguistic_patterns = pattern_learner.learn_linguistic_patterns(
            outputfile=output_file, 
            ngrams_for_patterns=ngrams_for_patterns, 
            verbose=verbose
        )

        if not linguistic_patterns:
            raise ValueError("Learning process produced no patterns. Please verify data.")

        linguistic_patterns = list(linguistic_patterns)

        translated_linguistic_patterns = self.translate_pattern(linguistic_patterns)

        return translated_linguistic_patterns
