import operator

class PatternsLearning:
    '''
    Manages automatic learning of POS patterns to perform linguistic extraction.
    '''
    def __init__(self): 
        pass
         
    def learn_linguistic_patterns(self, outputfile, ngrams_for_patterns, showfrequencies=True, encoding="utf-8", verbose=False, representativity=100):
        '''
        Automatically extracts linguistic patterns from a collection of pre-filtered, POS-tagged n-grams. It sorts the patterns from most to least frequent and writes them to an external text file, using a percentage threshold (representativity) to discard rare patterns.

        Args:

        outputfile (str): The file path where the learned linguistic patterns will be saved.
        ngrams_for_patterns (list of tuples): A collection of rows fetched from the database, where each tuple contains (tagged_ngram_string, n_size, frequency).
        showfrequencies (bool, optional): If True, displays the pattern's frequency into the output file. Defaults to True.
        encoding (str, optional): Defaults to "utf-8".
        verbose (bool, optional): Defaults to False.
        representativity (int, optional): A setting (0 to 100) that decides how many patterns to save based on their frequency. It sorts patterns from most to least common and stops saving once your chosen percentage of the total data is reached (setting it to 100 saves everything).

        Returns:
        
        learntpatterns (dict): A dictionary of learned patterns where keys are the generated rule strings (e.g., "|#|NOUN #||ADJ") and values are their corresponding frequencies.
        '''
        from tqdm import tqdm
        learntpatterns = {} # the key is the pattern and the value is its frequency
        acufreq = 0         # to accumulate the frequencies

        #('health|health|NOUN professionals|professional|NOUN', 7)- this is filtered tagged ngrams
        if ngrams_for_patterns:
            for tagged_ngram in tqdm(ngrams_for_patterns, total=len(ngrams_for_patterns), desc="Learning patterns"):
                if not tagged_ngram:
                    continue                
                try:    
                    tagged_ngram_string = tagged_ngram[0]
                    tagged_components = tagged_ngram_string.split() #['mental|mental|ADJ', 'disorders|disorder|NOUN']
                    n = len(tagged_components)
                    frequency = tagged_ngram[1]
                    
                    if len(tagged_components) != n:
                        n = len(tagged_components)

                    candidate_words = []
                    for component in tagged_components:
                        parts = component.split("|")
                        word = parts[0] if len(parts) > 0 else component
                        candidate_words.append(word)

                    clean_components = candidate_words #['mental', 'disorders']
                    patternbrut = []

                    for position in range(0, n):
                        comp_parts = tagged_components[position].split("|")
                        
                        clean_text = comp_parts[0] if len(comp_parts) > 0 else ""
                        clean_lemma = comp_parts[1] if len(comp_parts) > 1 else clean_text
                        clean_tag = comp_parts[2] if len(comp_parts) > 2 else "UNK" #fallback tag 

                        patternpart = ""
                        if position < len(clean_components):
                            if clean_components[position] == clean_lemma:
                                patternpart = "|#|" + clean_tag
                            elif clean_components[position] == clean_text:
                                patternpart = "#||" + clean_tag
                            else:
                                patternpart = "#||" + clean_tag  # Fallback default
                        
                        patternbrut.append(patternpart)

                    pattern = " ".join(patternbrut)

                    if pattern in learntpatterns:
                        learntpatterns[pattern] += frequency
                    else:
                        learntpatterns[pattern] = frequency
                    
                    acufreq += frequency

                except IndexError as ie:
                    print(f" Skipped invalid n-gram: {tagged_ngram}")
                    print(f" Error detail: {ie}")
                    continue

        sorted_patterns = sorted(learntpatterns.items(), key=operator.itemgetter(1), reverse=True)
        acufreq2 = 0

        with open(outputfile, "w", encoding=encoding) as f:
            if showfrequencies:
                f.write("term\tfrequency\n")
            else:
                f.write("term\n")

            for pattern, score in sorted_patterns:

                if acufreq > 0:
                    percent = 100 * acufreq2 / acufreq

                    if percent > representativity:
                        break
                acufreq2 += score  
    
                if showfrequencies:
                    output = pattern + "\t" + str(score)
                else:
                    output = pattern

                f.write(output + "\n")
                if verbose:
                    print(output)

            print(f"Linguistic patterns saved in {outputfile}")
        
        return sorted_patterns