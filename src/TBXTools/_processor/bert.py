import pandas as pd
from tqdm import tqdm
from .._helpers.helpers import get_spacy_model_from_code, load_spacy_model
import string

class BertProcessor():

    def __init__(self, methodology):
        self.methodology = methodology
        self.model_name = None
        self.stopwords = None
        self.inner_stopwords = None
        self.lang_code = None
        self.punctuation = string.punctuation

        self.model = None # maybe a Model class with model.tokenizer model.data_collator etc.
        self.tokenizer = None
        self.data_collator = None
        self.trainer = None
        self.labeling_scheme = None
        self._labeling_scheme_list = None
        self._label2id = None
        self._id2label = None

    def _load_model(self):
        from transformers import AutoModelForTokenClassification
        import torch
        device = torch.device("cuda")

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self._labeling_scheme_list),
            id2label=self._id2label,
            label2id=self._label2id).to(device)
        
    def _load_tokenizer_and_data_collator(self):
        from transformers import AutoTokenizer, DataCollatorForTokenClassification

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, max_length=512, force_download=False, do_lower_case=False, use_fast=True)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def _load_trainer(self):
        from transformers import Trainer
        self.trainer = Trainer(model=self.model, data_collator=self.data_collator)

    def _model_init(self): #unify with _load_model?
        from transformers import AutoModelForTokenClassification

        print(f"Loading model {self.model_name}", flush=True)
        return AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=len(self._labeling_scheme_list), id2label=self._id2label, label2id=self._label2id)
    
    def _set_filter_parameters(self):
        self.stopwords = self.methodology.extractor.stopwords
        self.inner_stopwords = self.methodology.extractor.inner_stopwords

    def choose_labels(self, labeling_scheme):
        if not labeling_scheme or labeling_scheme.upper() == "BIO":
            self.labeling_scheme = "BIO"
            self._labeling_scheme_list = ['O', 'B', 'I']
            
        elif labeling_scheme.upper() == "BILOU":
            self.labeling_scheme = "BILOU"
            self._labeling_scheme_list = ['O', 'B', 'I', 'L', 'U']

        else:
            raise RuntimeError(f"{self.labeling_scheme.upper()} labels not supported. Current labeling schemes supported: BIO, BILOU.")

        self._label2id = {l: i for i, l in enumerate(self._labeling_scheme_list)}
        self._id2label = {i: l for l, i in self._label2id.items()}
            
    def preprocess_train(self, df, expand_labels=False):
        encoding = self._encode(
            df["word_tokens"].tolist(),
            is_split_into_words=True) # its tokenized, so True
        
        tokenized_segments = self._tokenize_segments(encoding) # with bert, bert encoding output

        df["tokens"] = tokenized_segments
        df["input_ids"] = encoding["input_ids"]
        df["attention_mask"] = encoding["attention_mask"]

        # tokens_FD = self._calculate_tokens_FD(df["tokens"])

        df["labels"] = self._align_labels(encoding, df["labels"].tolist(), is_split_into_words=True, expand_labels=expand_labels)

        df["labels"] = self._transform_labels_into_ints(df["labels"])

        # return tokens_FD, df
        return df
    
    def annotate(self, segments, external_terms, spacy_model_size="sm"):
        '''
        Runs the whole annotation process.
        '''
        spacy_model = get_spacy_model_from_code(self.lang_code, size=spacy_model_size)
        nlp = load_spacy_model(spacy_model)

        word_tokens, lemmas = self._lemmatize_segments(nlp, segments) # with spacy

        df = pd.DataFrame({
            "text": segments,
            "word_tokens": word_tokens, 
            "lemmas": lemmas})
        
        tagged_terms = self._lemmatize_and_annotate_terms(nlp, external_terms=external_terms)
        df["labels"] = self._annotate_corpus(df["lemmas"], tokenized_terms=tagged_terms)

        return df

    def preprocess_test(self, segments, lang_code):
        spacy_model = get_spacy_model_from_code(lang_code)
        nlp = load_spacy_model(spacy_model)
        word_tokens = []

        # can use just the tokenizer, not the whole model
        for segment in tqdm(segments, desc="Tokenizing segments", total=len(segments)):

            doc = nlp(segment)
            tokens = [t.text for t in doc]
            word_tokens.append(tokens)

        encoding = self._encode(word_tokens, is_split_into_words=True)
        word_ids_list = [encoding.word_ids(i) for i in range(len(word_tokens))]

        df = pd.DataFrame({
            "tokens": word_tokens,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "word_ids": word_ids_list
            })

        return df
            
    def _encode(self, segments, is_split_into_words=False):
        encoding = self.tokenizer(
            segments,
            is_split_into_words=is_split_into_words, # True
            truncation=True,
            padding="max_length",
            max_length=512,
            return_offsets_mapping=False)

        return encoding

    def clean_punctuation(self, word):
        return word.strip(self.punctuation).lower()
    
    def strip_stopwords(self, term):
        # we can maybe use filter_ngram from the preprocessor
        term = term.strip(string.punctuation + " ")
        words = term.split()

        while words and self.clean_punctuation(words[0]) in self.stopwords:
            words.pop(0)

        while words and self.clean_punctuation(words[-1]) in self.stopwords:
            words.pop()

        return " ".join(words).strip(string.punctuation + " ")

    def process_predictions(self, predicted_candidates):
        # we can maybe use clean_ngram from the preprocessor
        '''Light postprocessing that removes stopwords at the beginning or end of the candidate term. Terms that are solely quotation marks ("") are also removed.
        
        The function takes a list of lists of terms.'''
        import re
        clean_terms = []
        ignore = [",", ".", "-"]

        for list_of_terms in predicted_candidates:
            # print(list_of_terms)
            cleaned = []
            for term in list_of_terms:
                term = term.strip()

                if not term:
                    continue
            
                if all(char in ignore for char in term):
                    continue

                if term in self.stopwords or term == "—":
                    continue 
            
                else:
                    term = self.strip_stopwords(term)

                    term = re.sub(r"\s*-\s*", "-", term)
                    term = re.sub(r"\(\s+", "(", term)
                    term = re.sub(r"\s+\)", ")", term)
                    term = re.sub(r"\[\s+", "[", term)
                    term = re.sub(r"\s+\]", "]", term)
                    term = re.sub(r" '([a-zA-Z])", r"'\1", term) #fix apostrophe
                    
                    if term == "":
                        continue
                    else:
                        cleaned.append(term)

            clean_terms.append(cleaned)

        return clean_terms
    
    def _annotate_corpus(self, tokenized_segments, tokenized_terms):
        terms_by_len = self._order_terms_by_len(tokenized_terms)
        labels = []

        for tokenized_segment in tqdm(tokenized_segments, 
                            desc=f"Annotating segments with {self.labeling_scheme} labels", total=len(tokenized_segments)):

            # unaligned labels fix
            segment = [token for token in tokenized_segment if token not in ["[CLS]", "[SEP]", "[PAD]"]]

            segment_labels = ["O"] * len(segment)

            for length, terms in terms_by_len.items():
            # for length, terms in terms_by_len:

                # if terms length is longer than n tokens in segment skip
                if length > len(segment):
                    continue

                max_start = len(segment) - length + 1

                for start_idx in range(max_start):

                    # if already labelled
                    if segment_labels[start_idx] != "O":
                        continue

                    for term in terms:
                        term_tokens = term["lemmas"]
                        # if first element is not the same as the first token of term, skip
                        if segment[start_idx] != term_tokens[0]:
                            continue
                        
                        # better than slicing like previously
                        match = True
                        for num in range(length):
                            if segment[start_idx + num] != term_tokens[num].lower():
                                match = False
                                break

                        if match:
                            segment_labels[start_idx:start_idx + length] = term["labels"]

            labels.append(segment_labels)

        return labels

    def _order_terms_by_len(self, tokenized_terms):
        from collections import defaultdict
        terms_by_len = defaultdict(list)

        for term in tokenized_terms:
            length = len(term["lemmas"])
            terms_by_len[length].append(term)

        return terms_by_len
    
    def _transform_labels_into_ints(self, labels):
        labels_ints = []
        for sequence in labels:
            label_ids = []

            for label in sequence:
                if label == -100: # PAD, CLS, SEP
                    label_ids.append(-100)
                else:
                    label_ids.append(self._label2id[label])

            labels_ints.append(label_ids)

        return labels_ints

    def lemmatize_term(self, term): #used in normalize_declension
        import spacy
        import re

        nlp = spacy.load(get_spacy_model_from_code(self.lang_code))
        doc = nlp(term)
        result = []

        for i, token in enumerate(doc):
            if token.pos_ == "NOUN" and i > 0 and doc[i-1].pos_ == "NOUN":
                result.append(token.text.lower())

            else:
                result.append(token.lemma_.lower())

        result = " ".join(result)
        # temporary postprocessing
        result = re.sub(r"\s*-\s*", "-", result)

        return result
        
    def _lemmatize_and_annotate_terms(self, nlp, external_terms):
        term_lemmas = {}

        for term in tqdm(external_terms, desc="Lemmatizing and annotating external terms", total=len(external_terms)):

            doc = nlp(term)
            lemmas = [t.lemma_.lower() for t in doc]

            # fix in order to stop tagging things like IS (International System)
            if len(lemmas) == 1 and lemmas[0] in self.stopwords:
                continue

            term_lemmas[term] = lemmas

        annotated_terms = []
        for term, lemmas in term_lemmas.items(): # dont need tqdm as this is instant
            
            num_of_lemmas = len(lemmas) #same logic as in tokenize and tag terms func

            # BILOU is WIP
            if self.labeling_scheme == "BILOU":
                if num_of_lemmas == 1:
                    labels = ["U"]
                else:
                    labels = ["B"] + ["I"] * (num_of_lemmas - 2) + ["L"]
                    
            elif self.labeling_scheme == "BIO":
                labels = ["B"] + ["I"] * (num_of_lemmas - 1)

            else:
                raise ValueError(f"Invalid labeling scheme detected: '{self.labeling_scheme}'")
            
            annotated_terms.append({
                "lemmas": lemmas,
                "labels": labels
            })
            
        annotated_terms.sort(key=lambda x: len(x["lemmas"]), reverse=True) # sorting by length in descending order

        return annotated_terms
            
    def _lemmatize_segments(self, nlp, segments):
        # spacy tokenizer/lemmatizer
        tokens_list = []
        lemmas_list = []
        print()
        for segment in tqdm(segments, desc="Tokenizing and lemmatizing segments", total=len(segments)):

            doc = nlp(segment)
            tokens = [t.text for t in doc]
            tokens_list.append(tokens)
            lemmas = [t.lemma_.lower() for t in doc]
            lemmas_list.append(lemmas)

        return tokens_list, lemmas_list
    
    def _tokenize_segments(self, encoding_data):
        # bert tokenizer
        tokenized_segments = []
        for ids in encoding_data["input_ids"]:
            tokenized_segments.append(self.tokenizer.convert_ids_to_tokens(ids))

        return tokenized_segments
    
    def _calculate_tokens_FD(self, tokenized_segments):
        import nltk

        tokensFD = nltk.probability.FreqDist()
        tokensFD.update(tokenized_segments)
        tokens_output = list(tokensFD.most_common())

        return tokens_output

    def _align_labels(self, encoding, unaligned_labels, is_split_into_words=False, expand_labels=False):
        aligned_labels = []

        if is_split_into_words: # if input is pre-tokenized # always should be

            for i, labels in enumerate(unaligned_labels):
                word_ids = encoding.word_ids(batch_index=i)
                aligned_segment = []
                previous_word_id = None
                
                for word_id in word_ids:
                    if word_id is None:
                        aligned_segment.append(-100)

                    elif word_id != previous_word_id:
                        aligned_segment.append(labels[word_id])
                    else:

                        label = labels[word_id]
                        
                        if not expand_labels and label.startswith(("B", "L", "U")):
                            aligned_segment.append(label.replace("B", "I", 1)) #for bio tags
                            # aligned_segment.append("I" + label[1:]) # for bilou

                        else: # expanding labels
                            aligned_segment.append(label)
                            
                    previous_word_id = word_id
                        
                aligned_labels.append(aligned_segment)

        else:

            for i, labels in enumerate(unaligned_labels):
                word_ids = encoding.word_ids(batch_index=i)
                
                aligned_segment = []
                label_idx = 0
                
                for word_id in word_ids:
                    if word_id is None:
                        aligned_segment.append(-100)
                    else:
                        if label_idx < len(labels):
                            aligned_segment.append(labels[label_idx])
                            label_idx += 1
                        else:
                            aligned_segment.append(-100)
                            
                aligned_labels.append(aligned_segment)
            
        return aligned_labels

    def _bio_to_terms(self, tokens, labels):
        # use with spacy tokenization
        terms = []
        current = []
        # O B I
        # 0 1 2
        for token, label in zip(tokens, labels):
            if label == 1: # B
                if current:
                    terms.append(" ".join(current))
                current = [token]

            elif label == 2 and current:
                current.append(token)

            # elif label == 2:  # I
            #     if current:
            #         current.append(token)
            #     else: # if there is no B token, i.e. I is alone, treat as B?
            #         current = [token]
            
            else: # 0
                if current:
                    terms.append(" ".join(current))
                    current = []

        if current:
            terms.append(" ".join(current))

        return terms
    
    def _flatten_list(self, list_of_lists): #needed for bio_tag_bert

        output = [token for sublist in list_of_lists for token in sublist]
        return output