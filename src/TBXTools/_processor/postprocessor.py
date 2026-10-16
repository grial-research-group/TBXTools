from .._helpers.helpers import get_spacy_model_from_code
from tqdm import tqdm
import re

class Postprocessor:

    '''Manages the text preprocessing pipeline for terminology extraction. This class
    provides methods for tokenizing text segments, applying lemmatization, case and nest normalizations, and filtering candidate terms using stopwords and regular expressions.

    Attributes:
        stopwords (list/set): A collection of standard words to filter out.
        inner_stopwords (list/set): A collection of words to filter out when found inside a term.
        nmin (int): The minimum number of words a candidate term can contain.
        nmax (int): The maximum number of words a candidate term can contain.
        lang_code (str): The ISO code for the language.
        model_name (str): The name of the spacy model used (e.g., "en_core_web_sm" or "ca_core_news_sm").
        nlp: The NLP pipeline or model used for text processing.
    '''

    def __init__(self):
        self.nmin = None
        self.nmax = None
        self.lang_code = None
        self.model_name = None
        self.nlp = None

    
    def case_normalization(self, candidate_terms, verbose=False):
        '''
        Performs case normalization on candidate terms while preserving acronyms. Converts candidate terms to lowercase except for uppercase tokens (e.g acronyms like 'ADH1B' or 'ADHD'). If a capitalized term exists as non-capitalized, the capitalized one will be deleted and the frequency of the non-capitalized one will be increased by the frequency of the capitalized.
                
        Args:
          candidate_terms: a list of tuple containing the candidate terms. 
          verbose: If True, enables detailed logging. Defaults to False.
                
        Returns:
          normalized_terms: A new list of tuple after applying case normalization.
        '''        
        print("Applying case normalization")

        freq_dict = {}

        for terms_row in candidate_terms:
            term = terms_row[0]
            freq = terms_row[3]

            tokens = term.split() #hacer split invece di tokenize
            normalized_tokens = []

            for token in tokens:
                
                if token.isupper():
                    normalized_tokens.append(token)
                else:
                    normalized_tokens.append(token.lower())

            normalized_term = " ".join(normalized_tokens)

            freq_dict[normalized_term] = freq_dict.get(normalized_term, 0) + freq

        normalized_terms = []
        for term, freq in freq_dict.items():
            n = len(term.split())

            row = (term, n, "frequency", freq)
            normalized_terms.append(row)

            if verbose:
                print(f"{term} -> {freq}")

        normalized_terms.sort(key=lambda row: row[3], reverse=True)

        return normalized_terms
    
    def lemmatization(self, candidate_terms, verbose=False):
        '''
        Performs lemmatization. Applies lemmatization to all candidate terms and merges duplicates by summing their frequencies.

        Args:
          candidate_terms: a list of tuple containing the candidate terms. 
          verbose: If True, enables detailed logging. Defaults to False.
        
        Returns:
          normalized_terms: A new list of tuple after applying lemmatization.
        '''
        
        from spacy.tokens import Doc 

        from .._helpers.helpers import load_spacy_model
        
        if self.lang_code and not self.model_name:

            self.model_name = get_spacy_model_from_code(self.lang_code)
            self.nlp = load_spacy_model(self.model_name)
            
        freq_dict = {}

        for terms_row in tqdm(candidate_terms, desc="Applying lemmatization", total=len(candidate_terms)):
            term = terms_row[0].strip()
            freq = terms_row[3]

            words = term.split() #is this necesary? Doc probably splits already, if we split then I THINK words like l'estat dont get tokenized and lemmatized like spaCy does

            doc = Doc(self.nlp.vocab, words=words)

            for name, proc in self.nlp.pipeline:
                doc = proc(doc)

            lemmatized_term = " ".join([token.lemma_.strip() for token in doc])

            freq_dict[lemmatized_term] = freq_dict.get(lemmatized_term, 0) + freq

        normalized_terms = []
        for lemma, freq in freq_dict.items():

            n = len(lemma.split())
            row = (lemma, n, "frequency", freq)

            normalized_terms.append(row)

            if verbose:
                print(f"{lemma} -> {freq}")

        normalized_terms.sort(key=lambda row: row[3], reverse=True)

        return normalized_terms
        
    def nest_normalization(self, candidate_terms, percent=10, verbose=False):
        """
        Normalizes candidate term frequencies by accounting for nested subterms. Reduces the frequency of terms that appear inside longer candidate terms. A frequency compatibility interval (±percent%) is defined around each candidate term's frequency. The frequency of a nested term is only subtracted from the base term if it falls within this interval. Terms whose normalized frequency drops to 0 are removed from the final list.

        Args:
          candidate_terms: a list of tuple containing the candidate terms.
          percent (int or float, optional): The percentage threshold defining the compatibility interval around each frequency. Defaults to 10.
          verbose: If True, enables detailed logging. Defaults to False.
        
        Returns:
          updated_terms: A new list of tuple after applying nest normalization.
        """
        updated_terms = []
        terms_by_n = {}

        for row in candidate_terms:
            candidate_term = row[0]
            candidate_term_n = row[1]
            candidate_term_freq = row[3]
            if candidate_term_n not in terms_by_n:
                terms_by_n[candidate_term_n] = []

            terms_by_n[candidate_term_n].append((candidate_term, candidate_term_freq))
   
        for row in tqdm(candidate_terms, desc="Applying nested frequency normalization", total=len(candidate_terms)):
            candidate_term = row[0]
            candidate_term_n = row[1]
            candidate_term_freq = row[3]

            nested_frequency = 0

            # frequency bounds (POST control)
            fmax = candidate_term_freq + candidate_term_freq * (percent / 100)
            fmin = candidate_term_freq - candidate_term_freq * (percent / 100)
            
            # compare the current term only with longer terms
            for longer_n, longer_terms in terms_by_n.items():
                if longer_n <= candidate_term_n:
                    continue

                for longer_term, longer_term_freq in longer_terms:
                    if candidate_term == longer_term: #skip identical terms
                        continue

                    #split terms into tokens
                    candidate_tokens = candidate_term.split()
                    longer_tokens = longer_term.split()

                    # search for an exact token sequence match
                    for i in range(len(longer_tokens) - len(candidate_tokens) + 1):
                        window = longer_tokens[i:i + len(candidate_tokens)]

                        if window == candidate_tokens:
                                if fmin <= longer_term_freq <= fmax:
                                    nested_frequency += longer_term_freq 
                                    break
                            
            # compute the normalized frequency
            normalized_freq = max(candidate_term_freq - nested_frequency, 0)
       
            updated_row = (candidate_term, candidate_term_n, "frequency", normalized_freq)

            # remove terms whose normalized frequency becomes 0
            if normalized_freq > 0:
                updated_terms.append(updated_row)

            elif verbose:
                print(
                f"Removed '{candidate_term}' "
                f"because normalized frequency became 0"
                )
        updated_terms.sort(key=lambda row: row[3], reverse=True)

        return updated_terms
    
    def regex_exclusion(self, regexes, candidate_terms, mode, verbose=False):
        '''
        Remove candidate terms that match regex expressions. It takes data in tuples as rows and outputs a list of candidate terms to exclude.

        Args:
          regexes: regular expression patterns used to match and filter out unwanted terms.
          candidate_terms: a list of tuple containing the candidate terms.
          mode: 'strict' removes full string matches, 'flexible' removes candidates that contain the regex. Defaults to 'strict'. 
          verbose: If True, enables detailed logging. Defaults to False.
        
        Returns:
          candidates_to_exclude: a list of candidate terms to exclude.
        '''
        
        import re
         
        candidates_to_exclude = []
        for candidate_row in tqdm(candidate_terms, desc="Applying regex exclusion", total=len(candidate_terms)):
            candidate = candidate_row[0]
            candidate_n = candidate_row[1]
            candidate_n = int(candidate_n)

            for regex_expression in regexes:
                regex_expression = regex_expression[0]
                regex_n = len(regex_expression.split())
                
                if mode == "strict":
                    match = re.fullmatch(regex_expression, candidate)
                    is_valid_match = bool(match and regex_n == candidate_n)

                if mode == "flexible":
                    match = re.search(regex_expression, candidate)
                    is_valid_match = bool(match)

                if is_valid_match:
                    candidates_to_exclude.append(candidate)

                    if verbose:
                        print(f"'{candidate}' removed by: {regex_expression}")

                    break

        return candidates_to_exclude
    
    def ngram_calculation(self, segments, is_corpus_tagged=False, minfreq=2):
        '''
        Calculates ngrmas and their frequencies from a list of text segments. If the corpus is tagged, it extracts both the raw text ngrams and the tagged ngrams to use when applying the linguistic extraction.

        Args:
          segments (list) : A list of text segments to process.
          is_corpus_tagged (bool) : If True the corpus is tagged. If False the corpus is not tagged. Defaults to False
          min_freq(int) : The minimum frequency threshold. Only ngrams appearing at least "minfreq" times will be included in the output.

        Returns:
          ngrams_output (list of tuple) : A list of tuples containing the ngram strings with their lenght and frequency.
          tagged_ngrams_output (list of tuple) : A list of tuples containing the original tagged n-grams with their length and frequency. This list remains empty if `is_corpus_tagged` is False.       
        '''

        ngrams_output = []
        tagged_ngrams_output = []

        for ngram, freq in self.ngrams_freq_dist.most_common(): 
            if freq >= minfreq:
                if is_corpus_tagged:

                    candidate_words = [ngt.split("|")[0] for ngt in ngram]
                    clean_ngram = " ".join(candidate_words)

                    ngrams_output.append((clean_ngram, len(ngram), freq))

                    tagged_ngrams_output.append((" ".join(ngram), len(ngram), freq))

                else:
                    ngrams_output.append((" ".join(ngram), len(ngram), freq))

        return ngrams_output, tagged_ngrams_output
    
    def apply_tsr_filter(self, tsr_terms, candidate_terms, mode="strict", max_iterations=10000000000, debug=False): 
        '''
        Filters the extracted candidate terms using the TSR (Token Slot Recognition) method. The algorithm is based on the concept of terminological token to filter out term candidates. It reads the terminological tokens from a list of terms (tsr_terms) and stores them taking into account their position in the terminological unit (first, middle, last). The TSR method filters term candidates by taking into account their tokens. To do so, 3 filtering variants are designed: strict, flexible and combined. In strict TSR filtering, a term candidate will be kept only if all the tokens are present in the corresponding position. In flexible TSR filtering, a term candidate will be kept if any of the tokens is present in the corresponding position. In combined TSR filtering, strict filtering is first used and is then followed by flexible filtering. In flexible and combined mode the algorithm performs the filtering process recursively, that is, by enlarging the list of terminological tokens with the new selected term candidates.
        
            Args:
                tsr_terms: The reference standard terms.
                candidate_terms (list of list): Candidates terms.
                mode (str, optional): Filtering mode ("strict", "flexible", "combined"). Defaults to "combined".
                max_iterations (int, optional): Loop ceiling for recursion. Defaults to 10000000000.
                debug (bool, optional): Defaults to False.
    
            Returns:
                updated_terms(list of list): Final candidate terms that passed the tsr filter structured as [term, n, freq, measure, value].
        '''
        component = {}  
        firstcomponent = {}
        middlecomponent = {}
        lastcomponent = {}

        if debug:
            print(f"\n==================================================================")
            print(f"--- [DEBUG] START apply_tsr_filter | Type: '{mode}' ---")
            print(f"==================================================================")
            print(f"[DEBUG] Extracting components from {len(tsr_terms)} reference tsr_terms")
        
        #from tsr terms list to the 4 dictionaries      
        for tsr_term in tsr_terms:
            tsr_ngrams = tsr_term.split() 
            if len(tsr_ngrams) == 1:
                firstcomponent[tsr_ngrams[0].lower()] = 1 
                lastcomponent[tsr_ngrams[0].lower()] = 1
            if len(tsr_ngrams) >= 2: 
                firstcomponent[tsr_ngrams[0].lower()] = 1
                lastcomponent[tsr_ngrams[-1].lower()] = 1
                component[tsr_ngrams[0].lower()] = 1
                component[tsr_ngrams[-1].lower()] = 1
                if len(tsr_ngrams) >= 3:
                    for i in range(1, len(tsr_ngrams) - 1):
                        middlecomponent[tsr_ngrams[i].lower()] = 1
                        component[tsr_ngrams[i].lower()] = 1

        if debug:
            print(f"\n[DEBUG] Initial TSR Vocabulary:")
            print(f"  • First ({len(firstcomponent)}): {sorted(list(firstcomponent.keys()))}")
            print(f"  • Middle ({len(middlecomponent)}): {sorted(list(middlecomponent.keys()))}")
            print(f"  • Last ({len(lastcomponent)}): {sorted(list(lastcomponent.keys()))}")

        new = True  #flag used to control the loop- initialized True to ensure the loop runs at least once
        newcandidates = {} #candidate-frequency
        hashmeasure = {} #to store the measurement modes for each accepted candidate ("tsr")
        iterations = 0 #how many times the loop executes

        while new: #the loop keeps running as long as new is True
            iterations += 1 
            new = False #Immediately resets the new flag to False at the beginning of the round. If the code later finds and accepts a new candidate term, it will set this back to True to trigger another iteration. If no new terms are found, the loop will exit
            accepted_in_this_iteration = 0

            words_added_to_first = []
            words_added_to_last = []
            words_added_to_middle = []
            
            if debug:
                print(f"------------ ITERATION {iterations} ------------")

            for term in candidate_terms:
                candidate = term[0]
                n = term[1]
                frequency = term[3]
                measure = "frequency" 
                
                rcamps = candidate.split()
                truesfalses = [] 
                match_details = [] 
                
                first_n = str(rcamps[0]).lower()
                last_n = str(rcamps[-1]).lower()
                
                if first_n in firstcomponent: 
                    truesfalses.append(True)
                    match_details.append(f"FIRST: '{first_n}' [OK]")
                else:
                    truesfalses.append(False)
                    match_details.append(f"FIRST: '{first_n}' [FAIL]")
                
                if last_n in lastcomponent: 
                    truesfalses.append(True)
                    match_details.append(f"LAST: '{last_n}' [OK]")
                else:
                    truesfalses.append(False)
                    match_details.append(f"LAST: '{last_n}' [FAIL]")

                if n > 2:
                    middle_c = True
                    mid_details = []
                    for i in range(1, n - 1):
                        mid_n = str(rcamps[i]).lower()
                        if mid_n in middlecomponent:
                            mid_details.append(f"'{mid_n}' [OK]")
                        else:
                            middle_c = False
                            mid_details.append(f"'{mid_n}' [FAIL]")
                    
                    truesfalses.append(middle_c)
                    match_details.append(f"MIDDLE: {', '.join(mid_details)}")

                is_accepted = False

                if mode == "strict":
                    if False not in truesfalses:
                        if candidate not in newcandidates: 
                            newcandidates[candidate] = frequency
                            hashmeasure[candidate] = measure
                            new = True 
                            is_accepted = True
                            
                            w_first_low, w_last_low = rcamps[0].lower(), rcamps[-1].lower()
                            if w_first_low not in firstcomponent: words_added_to_first.append(w_first_low)
                            if w_last_low not in lastcomponent: words_added_to_last.append(w_last_low)
                            
                            firstcomponent[w_first_low] = 1 
                            lastcomponent[w_last_low] = 1

                elif mode == "flexible": 
                    if True in truesfalses:
                        if candidate not in newcandidates:
                            newcandidates[candidate] = frequency
                            hashmeasure[candidate] = measure
                            new = True
                            is_accepted = True
                            
                            w_first_low, w_last_low = rcamps[0].lower(), rcamps[-1].lower()
                            if w_first_low not in firstcomponent: words_added_to_first.append(w_first_low)
                            if w_last_low not in lastcomponent: words_added_to_last.append(w_last_low)
                            
                            firstcomponent[w_first_low] = 1
                            lastcomponent[w_last_low] = 1
                            component[w_first_low] = 1
                            component[w_last_low] = 1
                            
                            if n > 2:
                                for i in range(1, n - 1):
                                    w_mid = rcamps[i].lower()
                                    if w_mid not in middlecomponent: words_added_to_middle.append(w_mid)
                                    middlecomponent[w_mid] = 1
                                    component[w_mid] = 1

                elif mode == "combined":
                    is_strict_round = (iterations == 1)
                    condition_met = (False not in truesfalses) if is_strict_round else (True in truesfalses)

                    if condition_met:
                        if candidate not in newcandidates:
                            newcandidates[candidate] = frequency
                            hashmeasure[candidate] = measure
                            new = True
                            is_accepted = True
                            
                            w_first_low, w_last_low = rcamps[0].lower(), rcamps[-1].lower()
                            if w_first_low not in firstcomponent: words_added_to_first.append(w_first_low)
                            if w_last_low not in lastcomponent: words_added_to_last.append(w_last_low)
                            
                            firstcomponent[w_first_low] = 1
                            lastcomponent[w_last_low] = 1
                            
                            if n > 2:
                                for i in range(1, n - 1):
                                    w_mid = rcamps[i].lower()
                                    if w_mid not in middlecomponent: words_added_to_middle.append(w_mid)
                                    middlecomponent[w_mid] = 1
                                    component[w_mid] = 1
                            
                            if not is_strict_round:
                                component[w_first_low] = 1
                                component[w_last_low] = 1

                if is_accepted:
                    accepted_in_this_iteration += 1
                    if debug:
                        print(f"  [ACCEPTED] Candidate: '{candidate}'")
                        print(f"  Match Detail: {' | '.join(match_details)}")
                else:
                    if debug and candidate not in newcandidates:
                        print(f"  [REJECTED]  Candidate: '{candidate}'")
                        print(f"  Match detail: {' | '.join(match_details)}")

            if debug: 
                print(f"\n--> ITERATION {iterations} SUMMARY:")
                print(f"    • Candidates accepted in this iteration: {accepted_in_this_iteration}")
                print(f"    • Newly learned tokens: First={words_added_to_first} | Middle={words_added_to_middle} | Last={words_added_to_last}")
                print(f"    • Continue to next iteration?  -> {new}\n")
                            
            if iterations >= max_iterations:
                if debug: print(f"[DEBUG] Maximum iteration limit reached ({max_iterations}). Stopping.")
                break

        updated_terms = [] 
        for new_candidate in newcandidates:
            term = new_candidate
            n = len(new_candidate.split())
            freqtotal = newcandidates[new_candidate]
            measure = hashmeasure[new_candidate]
            updated_terms.append((term, n, measure, freqtotal))

        updated_terms.sort(key=lambda row: row[3], reverse=True)

        if debug:
            print(f"==================================================================")
            print(f"--- [DEBUG] TSR FILTER COMPLETED ---")
            print(f"Final accepted candidates ({len(updated_terms)}):")
            for idx, (t, n, m, f) in enumerate(updated_terms, 1):
                print(f"  {idx:2d}. '{t}' (n={n}, freq={f})")
            print(f"==================================================================\n")

        return updated_terms