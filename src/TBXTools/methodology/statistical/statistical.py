class StatisticalMethodology():
    '''
    Manages statistical terminology extraction.

    Attributes:
        name (str): The name of the methodology 
        _processor (Processor): An internal instance of the Processor class configured with 'nmin' and 'nmax' used to handle text preprocessing tasks.
        case_normalization: If True applies case_normalization to the candidate terms. Default to True.
        min_freq (int): The minimum frequency threshold. Only n-grams appearing at least "min_freq" times will be included in the output.
    '''

    def __init__(self, nmin, nmax, case_normalization=True, min_freq=2):
        from ..._processor.postprocessor import Postprocessor
        from ..._processor.preprocessor import Preprocessor

        self.name = "StatisticalMethodology"
        self.case_normalization = case_normalization
        self.min_freq = min_freq
        self.nmin = nmin
        self.nmax = nmax

        self.extractor = None
        self.preprocessor = Preprocessor(methodology=self)
        self.postprocessor = Postprocessor(methodology=self)

    def run(self, segments, verbose=False):
        from ..._results.results import Results
        '''
        Run the statistical extraction pipeline.

        Args:
            segments: A list of text segments to process.
        
        Returns:
            results: A Results object containing the extracted candidate terms.
        '''
        self.preprocessor._set_filter_parameters()
        self._extract_ngrams(segments)
        # self.extractor._sqlite.insert_ngrams(self.preprocessor.ngrams_freq_dist)
        candidate_terms = self._extract_candidates()

        if self.case_normalization:  # post? no me gusta aqui, quiza mejor en extract() o algo
            candidate_terms = self.postprocessor.case_normalization(
                candidate_terms=candidate_terms,
                verbose=verbose)

        results = Results(terms=candidate_terms)

        return results

    def _extract_ngrams(self, segments):
        '''
        Extracts n-grams from segments. It processes the text segments to generate tokens and n-grams, applies stopword filtering (both boundary and inner), and calculates their frequency.

        Args:
            segments: A list of text segments to process.
        '''
        ngrams = []

        for segment in segments:  # will need to change when using yield in get_segments
            tokenized_segment = segment.split()

            raw_ngrams = self.preprocessor._compute_ngrams(tokenized_segment)

            for raw_ngram in raw_ngrams:
                raw_ngram = " ".join(raw_ngram)
                filtered_ngram = self.preprocessor._filter_ngram(raw_ngram)

                if filtered_ngram:
                    clean_ngram = self.preprocessor._clean_ngram(                   filtered_ngram)

                    if clean_ngram:
                        ngrams.append(clean_ngram)

        self.preprocessor._calculate_ngrams_freq_dist(ngrams)

    def _extract_candidates(self):
        '''
        Helper function to extract candidate terms from the extracted n-grams according to the minimum frequency threshold.

        Returns:
            candidate_terms: A list of extracted candidate terms.
        '''
        candidate_terms = []

        for ngram, freq in self.preprocessor.ngrams_freq_dist.most_common():
            n = len(ngram.split())

            if (freq >= self.min_freq
                and n >= self.preprocessor.nmin
                and n <= self.preprocessor.nmax):

                candidate_terms.append((ngram, n, "frequency", freq))

        return candidate_terms
