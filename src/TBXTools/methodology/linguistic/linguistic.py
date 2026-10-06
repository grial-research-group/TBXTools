import re

class LinguisticMethodology():
    '''
    Manages linguistic terminology extraction.

    Attributes:
        name (str): The name of the methodology ("LinguisticMethodology).
        is_corpus_tagged (bool): If True, indicates that the input corpus is POS-tagged.
        linguistic_patterns (list of str | None): A list of linguistic patterns.
        evaluation_terms (list of str | None) : Reference terms from which we can extrapolate linguistic patterns if 'linguistic_patterns' is not provided.
        processor (Processor): An internal instance of the Processor class configured with 'nmin' and 'nmax' used to handle text preprocessing tasks.
    '''

    def __init__(self, nmin, nmax, min_freq=2, is_corpus_tagged=False, case_normalization=True, linguistic_patterns=None, evaluation_terms=None, patterns=None):
        from ..._processor.postprocessor import Postprocessor
        from ..._processor.preprocessor import Preprocessor

        self.name = "LinguisticMethodology"
        self.case_normalization = case_normalization
        # self.is_corpus_tagged = is_corpus_tagged
        self.linguistic_patterns = linguistic_patterns
        self.evaluation_terms = evaluation_terms
        self.nmin = nmin
        self.nmax = nmax
        self.min_freq = min_freq
        self.patterns = patterns

        self.extractor = None
        self.preprocessor = Preprocessor(methodology=self)
        self.postprocessor = Postprocessor()

    def run(self, segments, verbose=False):
        '''
        Extracts candidate terms from text segments using a linguistic methodology. This method coordinates the entire extraction process: it ensures that POS-tagged segments are available (generating them if missing), calculates n-grams, automatically learns POS patterns from evaluation terms if no patterns are provided, filters candidates using those patterns, and extracts the final terminology.

        Args:
            segments (list of str): A list of raw text segments to process.
            verbose (bool, optional): If True, enables detailed logging. Defaults to False.
        
        Returns:
            results: A Results object containing the candidate terms.
        '''
        from ..._results.results import Results
 
        evaluation_terms = self.extractor._sqlite.get("evaluation_terms")

        if not list(self.extractor._sqlite.get_segments(tagged=True)):
            tagged_segments = self.preprocessor.create_tagged_segments(
                segments=segments,
                lang_code=self.extractor._lang_code
            )

        self.preprocessor._set_filter_parameters()
        self._extract_tagged_ngrams(tagged_segments=tagged_segments)

        clean_ngrams = self.preprocessor._extract_clean_ngrams_from_tagged()

        if not self.extractor._sqlite.get("linguistic_patterns"):
            translated_linguistic_patterns = self.preprocessor._generate_linguistic_patterns(
                evaluation_terms=evaluation_terms,
                clean_ngrams=clean_ngrams,
                output_file=f"{self.extractor._sqlite.project_name}-ling-patterns.txt"
            )

        candidate_terms = self._extract_candidates(
            translated_patterns=translated_linguistic_patterns)

        if self.case_normalization:
                    candidate_terms = self.postprocessor.case_normalization(candidate_terms=candidate_terms, verbose=verbose)

        results = Results(terms=candidate_terms)

        # self.extractor._sqlite.insert_segments(tagged_segments, tagged=True)
        # self.extractor._sqlite.insert_ngrams(results._tagged_ngrams, tagged=True)
        # self.extractor._sqlite.insert_ngrams(results._ngrams)
        # self.extractor._sqlite.insert_linguistic_patterns(results._linguistic_patterns)

        return results

    def _extract_tagged_ngrams(self, tagged_segments):
        '''
        Extracts n-grams from tagged segments. It processes the text segments to generate tokens and n-grams, applies stopword filtering (both boundary and inner), and calculates their frequency.

        Args:
            segments: A list of text segments to process.
        '''
        ngrams = []

        for segment in tagged_segments:  # will need to change when using yield in get_segments
            tokenized_segment = segment.split()

            raw_ngrams = self.preprocessor._compute_ngrams(tokenized_segment)

            for raw_ngram in raw_ngrams:
                filtered_ngram = self.preprocessor._filter_tagged_ngram(raw_ngram)

                if filtered_ngram:
                    ngrams.append(filtered_ngram)

        self.preprocessor._calculate_ngrams_freq_dist(ngrams)

    def _extract_candidates(self, translated_patterns):
        '''
        Extract candidate terms from the extracted n-grams according to the learned linguistic patterns and the minimum frequency threshold.

        Returns:
            candidate_terms: A list of extracted candidate terms.
        '''
        candidate_terms = []         
        for ngram, freq in self.preprocessor.ngrams_freq_dist.most_common():
            n = len(ngram.split())

            if (freq >= self.min_freq
            and n >= self.preprocessor.nmin
            and n <= self.preprocessor.nmax):

                for pattern in translated_patterns:
                    processed_pattern = f"^{pattern}$"
                    match = re.fullmatch(processed_pattern, ngram)

                    if match:
                        candidate = " ".join(match.groups()[1:])
                        candidate_terms.append((candidate, n, "frequency", freq))
                        break

        return candidate_terms