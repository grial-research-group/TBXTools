class Results:
    '''
    Manages results returned by the different methodologies.

    Attributes:
        _terms: A list of extracted terms.
        _ngrams: A list of extracted Ngrams.
        _tokens: A list of extracted tokens.
        _tagged_ngrams: A list of tagged extracted Ngrams.
        _linguistic_patterns: A list of linguistic patterns.
        _methodology: Class to manage the methodology object.
    '''
    def __init__(self, terms=None, ngrams=None, tagged_ngrams=None, tokens=None, linguistic_patterns=None):
        self._terms = terms or []
        self._ngrams = ngrams or []
        self._tagged_ngrams= tagged_ngrams or []
        self._linguistic_patterns = linguistic_patterns or []
        self._tokens = tokens or []
        # add lang code for bilingualresults align

        self._methodology = None
        self._extractor = None

    def print_candidates(self, limit=20, n=None, verbose=False):
        '''
        Prints a list of the top candidate terms.

        Args:
            limit (int, optional): The number of terms accessed. Default is 20.
            n (int, optional): N-grams to print.
            verbose (int, optional): If True, prints out n and frequency of the candidate terms. Default is False.    
        '''
        terms = self._terms

        prefix = f"Top {limit} candidate terms" if limit is not None else "Candidate terms"
        n_suffix = f" (n = {n})" if n else ""
        print(f"\n{prefix}{n_suffix}:")
        
        if n is not None and not isinstance(n, int):
            raise ValueError("n must be an integer.")
        
        elif n and isinstance(n, int):
            filtered_terms = []
            for row in terms:
                term_n = row[1]
                if n == term_n:
                    filtered_terms.append((row[0], term_n, "frequency", row[3]))

            terms = filtered_terms
        
        if verbose:
            output_terms = []
            print(f"term, n, frequency")
            for row in terms:
                    out = f"{row[0]}, {row[1]}, {row[3]}"
                    output_terms.append(out)

        else:
            output_terms = [row[0] for row in terms]

        if limit:
            output_terms = output_terms[:limit]
        
        print("\n".join(output_terms))

    def tokens(self, limit=20):
        '''
        Gets the list of tokens

        Args:
            limit: The number of terms accessed. Default is 20.

        Return:
            list: a list of tokens.        
        '''
        tokens = [token[0] for token in self._tokens]
        # tokens = [(row[0], row[1], row[2]) for row in self._tokens]
        
        if limit == None:
            return tokens
        
        return tokens[:limit]
    
    def ngrams(self, limit=20): 
        '''
        Gets the list of Ngrams

        Args:
            limit: The number of Ngrams accessed. Default is 20.

        Return:
            list: a list of Ngrams.        
        '''
        ngrams = [ngram[0] for ngram in self._ngrams]

        if limit == None:
            return ngrams
        
        return ngrams[:limit]
    
    def tagged_ngrams(self, limit=20): 
        '''
        Gets the list of Tagged Ngrams
        
        Args:
            limit: The number of Tagged Ngrams accessed. Default is 20.
            
        Return:
            list: a list of Tagged Ngrams.
        '''
        tagged_ngrams= [tagged_ngram[0] for tagged_ngram in self._tagged_ngrams]

        if limit == None:
            return tagged_ngrams

        return tagged_ngrams[:limit]
    
    def nest_normalization(self, percent=10, verbose=False):
        '''
        Normalizes candidate term frequencies by accounting for nested subterms.
        
        It reduces the frequency of terms that appear inside longer candidate terms. A frequency compatibility interval (±percent%) is defined around each candidate term's frequency. The frequency of a nested term is only subtracted from the base term if it falls within this interval. Terms whose normalized frequency drops to 0 are removed from the final list.

        Args:
            percent: The frequency compatibility interval that is used to calculate if a term is nested inside another.
            verbose (bool, optional): Prints the process in the console. Defaults to False.
        '''
        candidate_terms = self._terms

        filtered_terms = self._methodology.postprocessor.nest_normalization(candidate_terms=candidate_terms, percent=percent, verbose=verbose)

        self._extractor._sqlite.delete("candidate_terms")
        self._extractor._sqlite.insert_candidate_terms(filtered_terms)
        self._terms = filtered_terms

    def lemmatization(self, verbose=False):
        '''
        Performs lemmatization of the terms.

        Args:
           verbose (bool, optional) : Prints the process in the console. Defaults to False.
        '''
        candidate_terms = self._terms

        filtered_terms = self._methodology.postprocessor.lemmatization(candidate_terms=candidate_terms, verbose=verbose)
        
        self._extractor._sqlite.delete("candidate_terms")
        self._extractor._sqlite.insert_candidate_terms(filtered_terms)
        self._terms = filtered_terms

    def tsr(self, tsr_terms=None, mode="strict", max_iterations=10000000000, debug=False):
        '''
        Filters the extracted candidate terms using Token Slot Recognition (TSR). The algorithm is based on the concept of terminological token, i.e., it filters out term candidates by taking into account their tokens.

        If mode is 'strict', a term candidate will be kept only if all the tokens are present in the corresponding position. If mode is 'flexible', a term candidate will be kept if any of the tokens is present in the corresponding position. If mode is 'combined', strict filtering is first used and is then followed by flexible filtering.

        Args:
            tsr_terms: The reference standard terms.
            mode (str, optional): Filtering mode ("strict", "flexible", "combined"). Defaults to "combined".
            max_iterations (int, optional): Loop ceiling for recursion. Defaults to 10000000000.
            debug (bool, optional): Prints the process in the console. Defaults to False.
        '''

        self._extractor._sqlite.load_tsr_terms(tsr_terms=tsr_terms)
        tsr_terms = self._extractor._sqlite.get("tsr_terms")
    
        if not tsr_terms:
            print("TSR terms not found. Not applying TSR filter")
            return

        candidate_terms = self._terms
        filtered_terms = self._methodology.postprocessor.apply_tsr_filter(tsr_terms=tsr_terms, candidate_terms=candidate_terms, mode=mode, max_iterations= max_iterations, debug=debug)
        
        self._terms = filtered_terms
        self._extractor._sqlite.delete("candidate_terms") 
        self._extractor._sqlite.insert_candidate_terms(self._terms)
        print(f"TSR filter completed. {len(self._terms)} candidates saved.")
            
    def regex_exclusion(self, regexes=None, mode="strict", verbose=False):
        '''
        Remove candidate terms that match regex expressions.

        Args:
            regexes: Regular expression patterns used to match and filter out unwanted terms.
            mode: Change the behaviour of the function, 'strict' removes full string matches, 'flexible' removes candidates that contain the regex. Defaults to 'strict'.
            verbose (bool, optional): Prints the process in the console. Default to False.
        '''
        
        self._extractor._sqlite.load_exclusion_regexes(exclusion_regexes=regexes)
        raw_regexes = self._extractor._sqlite.get("exclusion_regexes")

        if not raw_regexes:
            raise ValueError("Exclusion regexes not found. Not applying regex exclusion.")
        
        regexes = [(r,) for r in raw_regexes]
        
        candidate_terms = self._terms # get preprocessed terms instead 
        # return processed (self._terms = filtered_terms can stay the same i think)
        candidates_to_exclude = self._methodology.postprocessor.regex_exclusion(regexes=regexes, candidate_terms=candidate_terms, verbose=verbose, mode=mode)
        
        if candidates_to_exclude:
            self._extractor._sqlite.delete_specific_candidate_term(candidates=candidates_to_exclude)
            print(f"Excluded {len(candidates_to_exclude)} terms")
        else:
            print("No candidate terms excluded")

        filtered_terms = self._extractor._sqlite.get_candidate_terms()
        self._terms = filtered_terms

    def save_candidates(self, path, only_candidates=False, by=None, explode=False):
        '''
        Save the candidate terms to disk. The file is saved in the specified format. If no format is provided, it defaults to .txt.

        Supported formats: .txt, .csv, .xlsx, .jsonl

        Args:
            path: Path of the file to be saved.
            only_candidates (bool, optional): Exports only candidate terms, without n and frequencies.
            by (str, optional):  Available options "segment" and "id". If "segment", data is exported containing segment id, segment, and candidate terms extracted from it. If "id", the segment is omitted.
            explode (bool, optional): Exports each candidate in its own row, replicating id and/or segment. Only works if 'by' is being used.
        
        Raises:
            ValueError: If 'only_candidates' and 'by' are used at the same time.
        '''
        from pathlib import Path
        import pandas as pd

        path = Path(path)
        extension = path.suffix.lower()
        candidate_terms = self._extractor._sqlite.get_candidate_terms()

        if only_candidates and by == "segment" or only_candidates and by == "id":
            raise ValueError("only_candidates and by cannot be both True. Use one or the other.")
        
        elif by:
            if by not in ["segment", "id"]:
                raise ValueError("Unsupported format passed to 'by'. You need to pass 'segment' or 'id'.")
            
            segments = self._extractor._sqlite.get_segments()
            df_rows = []
            
            for id, segment in enumerate(segments):
                candidates = []
                for row in candidate_terms:
                    candidate = row[0]
                    if candidate in segment:
                        candidates.append(candidate)

                if candidates:
                    if by == "segment":
                        df_rows.append({"id": id, "segment": segment, "candidates" : candidates})
                    if by == "id":
                        df_rows.append({"id": id, "candidates" : candidates})

            output = pd.DataFrame(df_rows)
            
            if explode:
                output = output.explode(column="candidates")

        else:

            output = pd.DataFrame(candidate_terms, columns=['candidate', 'n', 'measure', 'value'])
            if only_candidates:
                output = output[['candidate']]
                       
        if not extension:
            extension = ".txt"
            path = path.with_suffix(extension)

        if extension == ".txt":
            output.to_csv(path, index=False, sep="\t")

        elif extension == ".csv":
            output.to_csv(path, index=False)
            
        elif extension == ".xlsx":
            output.to_excel(path, index=False)

        elif extension == ".jsonl":
            output.to_json(path, orient="records", lines=True)

        else:
            raise ValueError(f"Unsupported format '{extension}'. Supported formats: .txt, .csv, .xlsx, .jsonl")
        
        print(f"Candidate terms saved to disk ({path})", flush=True)

    def normalize_declension(self):
        from tqdm import tqdm
        candidate_terms = self._terms
        normalized_terms = []
        for row in tqdm(candidate_terms, desc="Normalizing declension of terms", total=len(candidate_terms)):
            term = row[0]

            split_term = term.split()
            first_token = split_term[0]

            if len(split_term) == 1:
                is_upper = term.isupper()

                if not is_upper:
                    term = self._methodology.postprocessor.lemmatize_term(term)
    
                normalized_terms.append((term, row[1], row[2], row[3]))

            elif len(split_term) > 1:
                
                term = self._methodology.postprocessor.lemmatize_term(term)
                normalized_terms.append((term, row[1], row[2], row[3]))
        
        self._extractor._sqlite.delete("candidate_terms")
        self._extractor._sqlite.insert_candidate_terms(normalized_terms)
        self._terms = normalized_terms

    def summary(self): #work in progress
        import textwrap
        self._extractor._sqlite.calculate_descriptive_statistics() # sqlite does the calculations and puts everything inside the attr descriptive_statistics_data (dict), which we access here
        data = self._extractor._sqlite.descriptive_statistics_data

        print()
        print(" SUMMARY ".center(60, "-"))
        print(f"Segments in corpus: {data['corpus']}")
        print(f"Candidate terms: {data['candidate_terms']}")

        print("-" * 60)
        for n in range(data["nrange"][0], data["nrange"][1]+1):
            num_ngrams = len(data.get(str(n), []))
            terms = [term[0] for term in data.get(str(n), [])[:10]] # top 10
            joined = ", ".join(terms)

            print(f"{n}-GRAMS  |  Total: {num_ngrams}")
            
            wrapped_terms = textwrap.fill(
                joined, 
                width=60, 
                initial_indent="  ", 
                subsequent_indent="  "
            )
            print(wrapped_terms)
            print()