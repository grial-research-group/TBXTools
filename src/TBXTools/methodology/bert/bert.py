from ..base import BaseMethodology
from ..._results.results import Results
from ..._processor.bert import BertProcessor
from collections import Counter

class BertMethodology(BaseMethodology):
    '''
    Manages terminology extraction with a BERT model.

    Attributes:
        model (str): Fine-tuned model for terminology extraction using labels.
        labeling_scheme (str): The labels used in the fine-tuning of the model. Defaults to "BIO".
    '''

    def __init__(self, model, labeling_scheme="BIO", is_corpus_annotated=False):
        from transformers import logging
        logging.set_verbosity_error()

        self.name = "BertMethodology"
        self.model_name = model
        self.is_corpus_annotated = is_corpus_annotated
        
        self.processor = BertProcessor(methodology=self)
        self.extractor = None
        self.processor.choose_labels(labeling_scheme=labeling_scheme)

    def run(self, segments, verbose=False):
        '''
        Extracts candidate terms using a previously fine-tuned BERT model on annotated data to predict terms.

        Args:
            segments: A list of segments to process.
        
        Returns:
            Results: An object containing the candidate terms and dataframe.
        '''

        from datasets import Dataset
        import numpy as np
        print(f'\nInitializing model:  {self.model_name}', flush=True)
        self.processor.model_name = self.model_name
        
        self.processor._set_filter_parameters()
        self.processor._load_model()
        self.processor._load_trainer()
        self.processor._load_tokenizer_and_data_collator()
            
        dataframe = self.processor.preprocess_test(segments=segments, lang_code=self.extractor._lang_code)
        eval_data = Dataset.from_pandas(dataframe)

        print("\nPredicting terms")
        trainer = self.processor.trainer
        prediction_logits, _ , _ = trainer.predict(eval_data)
        predictions = np.argmax(prediction_logits, axis=2)
        predicted_terms = []
                
        for i in range(len(eval_data)):
            tokens = eval_data[i]['tokens']
            predicted_ids = predictions[i]
            word_ids = eval_data[i]['word_ids']
            
            aligned_labels = []
            previous_word_idx = None
            
            for word_idx, pred_id in zip(word_ids, predicted_ids):
                if word_idx is None:
                    continue

                elif word_idx != previous_word_idx:
                    aligned_labels.append(pred_id)
                
                else:
                    pass
                
                previous_word_idx = word_idx

            reconstructed = self.processor._bio_to_terms(
                tokens=tokens,
                labels=aligned_labels)

            predicted_terms.append(reconstructed)

        clean_terms = self.processor.process_predictions(predicted_terms)

        dataframe['predicted_terms'] = clean_terms

        #i dont remember why im doing this, but keep for now, otherwise it wont work
        clean_terms = dataframe['predicted_terms'].tolist()
        clean_terms = self.processor._flatten_list(clean_terms)

        # output for tbxtools, calculating count of each term
        candidate_terms = []
        term_counts = Counter(clean_terms)
        term_counts = dict(sorted(term_counts.items(), key=lambda item: item[1], reverse=True))
        for term, count in term_counts.items():
            if term:
                n = len(term.split(" "))
                candidate_terms.append((term, n, "count", count))
        # tokenized_segments = []

        results = Results(terms=candidate_terms, df=dataframe)

        # self.extractor._sqlite.insert_segments(data=tokenized_segments, tagged=False, tokenized=True)
        # old line, im not inserting tokens rn, need to check
        # self.extractor._sqlite.insert_tokens(data=results._tokens)

        return results