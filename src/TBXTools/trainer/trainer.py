from .._processor.bert import BertProcessor
from .._sqlite.sqlite import SQLite
from .._results.metrics import Metrics
from .._processor.file_parser import FileParser
from .._helpers.helpers import get_lang

class BertTrainer:
    '''
    Class to handle data to fine-tune a BERT-based model for automatic terminology extraction.

    Attributes:
        project_name (str): The unique name identifier for the current project, which also determines the filename of the generated SQLite database.
        corpus: The text corpus used as the training data.
        external_terms: File path to the terms that will be used in the data annotation process.
        labeling_scheme (str, optional): The labeling scheme to annotate the data or train the model with. Defaults to BIO.
    '''

    def __init__(self, project_name, language, corpus=None, external_terms=None, stopwords=None, inner_stopwords=None, overwrite_project=False, seed=123, labeling_scheme="BIO"):
        from transformers import logging
        logging.set_verbosity_error()
        self.lang, self._lang_code = get_lang(language.lower())

        self.name = "BertTrainer"
        self._processor = BertProcessor()
        self._processor.choose_labels(labeling_scheme=labeling_scheme)
        self._processor.lang_code = self._lang_code
        self._metrics = Metrics()
        self._seed = seed
        
        parser = FileParser(src_lang=self._lang_code)
        _corpus = parser.parse(corpus=corpus)

        self._sqlite = SQLite(
            project_name=project_name, 
            stopwords=stopwords,
            inner_stopwords=inner_stopwords, 
            corpus=_corpus,
            external_terms=external_terms, 
            overwrite_project=overwrite_project, 
            lang_code=self._lang_code,
            lang=self.lang, 
        )

        self.stopwords = self._sqlite.get("stopwords")
        self.inner_stopwords = self._sqlite.get("inner_stopwords")
        
    def train(self, model, save_as=None, split=False, expand_labels=False, sample=None, only_segments_with_terms=True, lr=5e-05, batch_size=16, epochs=3, weight_decay=0.01, gradient_accumulation_steps=1, warmup_ratio=0.1):
        '''
        Fine-tunes the chosen model for automatic terminology extraction.

        Args:
            model_name (str, optional): The BERT-based model to be fine-tuned.
            sample (int, optional): Number of sentences to randomly sample out of the annotated data.
            save_as (str, optional): Path of the model to save to disk.
            split (bool, optional): If True, it splits the data in train (0.8) and test (0.2).
            expand_labels (bool, optional): If True, assigns B labels to all the subwords of the first word and I to the remaining words/subwords, instead of assigning B only to the first subword token of the term. Defaults to False.
            lr (int, optional): Learning rate of the model. Defaults to 5e-05.
            batch_size (int, optional): Defaults to 16.
            epochs (int, optional): Defaults to 3.
            weight_decay (int, optional): Defaults to 0.01.
        '''
        from transformers import Trainer, set_seed
        from datasets import Dataset
        from transformers import TrainingArguments, EarlyStoppingCallback
        set_seed(self._seed)
        print("Running training", flush=True)
        print(f'\nInitializing model:  {model}', flush=True)
        self._processor.model_name = model
        self._processor._load_model()
        self._processor._load_tokenizer_and_data_collator()
        
        model_ = self._processor.model
        data_collator = self._processor.data_collator
        
        df = self._fetch_data_from_db(
            sample=sample, 
            only_segments_with_terms=only_segments_with_terms,
        )
        
        df = self._processor.preprocess_train(
            df=df, 
            expand_labels=expand_labels,
        )

        if not split:
            train_data = Dataset.from_pandas(df)

            training_args = TrainingArguments(
            eval_strategy="no",
            logging_strategy="no",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            seed=self._seed,
            data_seed=self._seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio
            )

            trainer = Trainer(
                model=model_,
                args=training_args,
                train_dataset=train_data,
                data_collator=data_collator # needed to pad the sentences
            )
        
        elif split:
            from sklearn.model_selection import train_test_split
            print("\nSplitting training data into train (0.85) and val (0.15)")

            train_df, val_df = train_test_split(df, test_size=0.15, random_state=self._seed)
            # train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=self._seed)
            print(f"Train segments: {len(train_df)} - Val segments: {len(val_df)}")

            train_data = Dataset.from_pandas(train_df)
            val_data = Dataset.from_pandas(val_df)

            model_folder_name = self._processor.model_name.replace("/", "-")
            model_output_dir = f"./trainer_output/{model_folder_name}"

            training_args = TrainingArguments(
                output_dir=model_output_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                logging_strategy="epoch",
                save_total_limit=3,
                metric_for_best_model="f1",
                greater_is_better=True,
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=weight_decay,
                seed=self._seed,
                data_seed=self._seed,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_ratio=warmup_ratio)

            self._metrics.eval_data= val_data
            self._metrics.processor = self._processor
            trainer = Trainer(
                model=model_,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data,
                compute_metrics=self._metrics.compute_metrics,
                data_collator=data_collator,
                # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                )

        print('Fine-tuning model', flush=True)
        print(f"Parameters: lr = {lr}, batch size = {batch_size}, weight decay = {weight_decay}, warmup ratio = {warmup_ratio}, epochs = {epochs}")
        trainer.train()

        # test_results = trainer.evaluate(test_data)
        # print("\nResults on the test data")
        # print(test_results)

        if save_as:
            trainer.save_model(f'{save_as}')
            print(f"Model saved as '{save_as}'")

    def annotate(self, sample=None):
        '''
        Generate training data to fine-tune a model. 
        
        Automatically annotate a corpus using an external list of terms. The resulting annotated data get saved in the database.

        Attributes:
            sample (int, optional): Number of sentences to randomly sample out of the corpus. Useful for testing purposes.
        '''
        print("Running annotation", flush=True)

        if not self._sqlite.table_is_populated("corpus") or not self._sqlite.table_is_populated("external_terms"):
            raise RuntimeError("Corpus or external terms not found. They need to be passed as arguments to BertTrainer.")
        
        if self._sqlite.table_is_populated("word_tokens") and self._sqlite.table_is_populated("segment_labels") and self._sqlite.overwrite_project==False:
            raise RuntimeError("Annotation cancelled. Word tokens and labels found in database. You may run 'train()' to use the existing data or use 'overwrite_project=True' to overwrite the existing data in the database.")
        
        self._processor.stopwords = self.stopwords
        self._processor.inner_stopwords = self.inner_stopwords

        segments = list(self._sqlite.get_segments())
        external_terms = set(self._sqlite.get_external_terms())

        if sample and isinstance(sample, int):
            import random
            random.seed(self._seed)
            print(f"Sampling {sample} random sentences")
            segments = random.sample(segments, sample)

        df = self._processor.annotate(segments=segments, external_terms=external_terms)

        # self._sqlite.delete("corpus")
        # self._sqlite.insert_segments(data=df["text"].tolist(), tagged=False, tokenized=False) 

        self._sqlite.insert_word_tokens(data=df["word_tokens"].tolist())
        self._sqlite.insert_lemmatized_corpus(data=df["lemmas"].tolist())
        self._sqlite.insert_segment_labels(data=df["labels"]) # unaligned

        # df.to_csv("inspect-df.csv", index=False)

        # tokens_FD = self._processor._calculate_tokens_FD(df["word_tokens"])

        # self._sqlite.insert_segments(data=df["tokens"].tolist(), tagged=False, tokenized=True, in_list_of_lists=True)

        # self._sqlite.insert_tokens(data=tokens_FD)

        with_terms = df["labels"].apply(lambda seg_labels: any(label in ["B", "I"] for label in seg_labels))
        df_with_terms = df[with_terms]
        df_without_terms = df[~with_terms]

        print(f"\n{len(df)} total segments annotated and saved to database.")
        print(f"{len(df_with_terms)} segments with terms.")
        print(f"{len(df_without_terms)} segments without terms.")

    def load_annotated(self, data):
        # should take a json or csv (cuidado con strings dentro de listas) 
        # segment? labels (check db)
        pass
    
    def _fetch_data_from_db(self, sample=None, only_segments_with_terms=True):
        import pandas as pd
        print("\nFetching data from database", flush=True)

        segments = self._sqlite.get_segments()
        word_tokens = self._sqlite.get_word_tokens()
        segment_labels = self._sqlite.get_segment_labels()  

        if segment_labels and word_tokens:
            data = {
                "segment": pd.Series(segments),
                "word_tokens": pd.Series(word_tokens),
                "labels": pd.Series(segment_labels)
                }
            
            dataframe = pd.DataFrame(data=data)

            if only_segments_with_terms: # True by default
                with_terms = dataframe["labels"].apply(lambda seg_labels: any(label in ["B", "I"] for label in seg_labels))
            
                dataframe = dataframe[with_terms]

            if isinstance(sample, int): 
                print(f"Sampling {sample} random sentences")
                dataframe = dataframe.sample(n=sample, random_state=123)

            return dataframe
        else:
            raise RuntimeError("Annotated data not found in database. Run 'annotate()' before 'train()'.")   

    # this func needs an update
    def hp_tuning(self, models, sample=None, lr_range=(1e-5, 5e-5), epoch_range=(3, 6), batch_sizes=(8, 16, 32), weight_decay_range=(0.0, 0.05), warmup_ratio_range=(0.0, 0.2), n_trials=15):
        from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
        from sklearn.model_selection import train_test_split
        from datasets import Dataset
        import tempfile

        import warnings
        warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")

        print("\nRunning hyperparameter tuning on the following models:", flush=True)

        if isinstance(models, str):
            models = [models]

        for model_name in models:
            print(model_name)

        for model_name in models:
            self._processor = BertProcessor()
            self._processor.model_name = model_name
            self._processor._load_tokenizer_and_data_collator()

            dataframe = self._fetch_data_from_db(sample=sample)
            dataframe = self._processor.preprocess_train(df=dataframe, expand_labels=False)
            train_df, eval_df = train_test_split(dataframe, test_size=0.2, random_state=self._seed)
            train_data = Dataset.from_pandas(train_df)
            eval_data = Dataset.from_pandas(eval_df)

            def hp_space(trial):
                lr = trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)
                num_train_epochs = trial.suggest_int("num_train_epochs", epoch_range[0], epoch_range[1])
                batch_size = trial.suggest_categorical("per_device_train_batch_size", batch_sizes)
                weight_decay = trial.suggest_float("weight_decay", weight_decay_range[0], weight_decay_range[1])
                warmup_ratio = trial.suggest_float("warmup_ratio", warmup_ratio_range[0], warmup_ratio_range[1])

                print(f"\n   Trial: {trial.number} \nParameters: lr = {lr}, epochs = {num_train_epochs}, batch_size = {batch_size}, weight_decay = {weight_decay}, warmup_ratio = {warmup_ratio}")
                return {
                    "learning_rate": lr,
                    "num_train_epochs": num_train_epochs,
                    "per_device_train_batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "warmup_ratio": warmup_ratio
                    }

            # model_folder_name = model_name.replace("/", "-")
            # model_output_dir = f"./trainer_output/{model_folder_name}"

            # add gradient accumulation steps and warmup ratio
            with tempfile.TemporaryDirectory() as temp_dir: # to delete the temp dir
                training_args = TrainingArguments(
                    output_dir=temp_dir,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model="f1",
                    save_total_limit=1,
                    learning_rate=self.lr,
                    per_device_train_batch_size=self.batch_size,
                    num_train_epochs=self.epochs,
                    weight_decay=self.weight_decay,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    warmup_ratio=self.warmup_ratio,
                    seed=self._seed,
                    data_seed=self._seed)
                
                self._metrics.eval_data= eval_data
                self._metrics.processor = self._processor
                
                trainer = Trainer(
                    model_init=self._processor._model_init,
                    args=training_args,
                    train_dataset=train_data,
                    eval_dataset=eval_data,
                    compute_metrics=self._metrics.compute_metrics,
                    data_collator=self._processor.data_collator,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                    )

                print(f"\nFine-tuning {self._processor.model_name}")
                best_run = trainer.hyperparameter_search(
                    direction="maximize",
                    backend="optuna",
                    hp_space=hp_space,
                    n_trials=n_trials)
                
            #     results.append({
            #         "model": model_name,
            #         "best_score": best_run.objective,
            #         "learning_rate": best_run.hyperparameters["learning_rate"],
            #         "epochs": best_run.hyperparameters["num_train_epochs"],
            #         "batch_size": best_run.hyperparameters["per_device_train_batch_size"],
            #         "weight_decay": best_run.hyperparameters["weight_decay"],
            #         "warmup_ratio": best_run.hyperparameters["warmup_ratio"]
            #     })
                
            # results = pd.DataFrame(results)
            # if not output_file:
            #     output_file = "hp-tuning-results"

            # results.to_csv(f"{output_file}.csv", index=False)