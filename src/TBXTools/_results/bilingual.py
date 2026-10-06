from .._processor.aligner import Aligner
from .._utils.utils import get_lang

class BilingualResults:
    
    def __init__(self, src_results, tgt_results, src_lang=None, tgt_lang=None):
        
        self.src = src_results
        self.tgt = tgt_results
        
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
                   
        self.aligner= Aligner()
        self.aligned_terms = None
    
    def src_terms(self):
        return [row[0] for row in self.src._terms]
    
    def tgt_terms(self):
        return [row[0] for row in self.tgt._terms]
    
    
    def align(self, model_name="sentence-transformers/LaBSE", threshold=0.75, synonym=False, score=True):  
        
        self.synonym = synonym
    
        src_list = self.src_terms()
        tgt_list = self.tgt_terms()
        
        self.aligned_terms = self.aligner.align_words(
            src_list=src_list,
            tgt_list=tgt_list,
            model=model_name,
            threshold=threshold,
            synonym=synonym,
            score=score
        )

        return self.aligned_terms
            
    def save_aligned_candidates(self, path, reverse=False):
        '''
        Save the aligned terms to disk. The file is saved in the specified format. If no format is provided, it defaults to .txt.

        Supported formats: .txt, .csv, .xlsx, .json, .jsonl

        Args:
            path: Path of the file to be saved.
            reverse: (bool, optional): If True, swaps the column order and entries.
            (target language first, source language second). Defaults to False.
        
        Raises:
            ValueError: If the destination file extension is not supported.
        '''
                
        from pathlib import Path
        import pandas as pd

        path = Path(path)
        extension = path.suffix.lower()
        
        candidate_terms = self.aligned_terms or []
        if not candidate_terms:
            print("No aligned candidate terms to save.", flush=True)
            return
        
        synonym = self.synonym

        # REVERSE
        if reverse:
            col_1_name = f"{self.tgt_lang}_tgt" if self.src_lang == self.tgt_lang else self.tgt_lang
            col_2_name = f"{self.src_lang}_src" if self.src_lang == self.tgt_lang else self.src_lang

            formatted_terms = [
                (row[1], row[0], row[2]) if len(row) >= 3 else (row[1], row[0])
                for row in candidate_terms
            ]
        else:
            col_1_name = f"{self.src_lang}_src" if self.src_lang == self.tgt_lang else self.src_lang
            col_2_name = f"{self.tgt_lang}_tgt" if self.src_lang == self.tgt_lang else self.tgt_lang

            formatted_terms = candidate_terms

        sample_row = formatted_terms[0] if formatted_terms else ()
        cols = [col_1_name, col_2_name]
        if len(sample_row) >= 3:
            cols.append("score")

        output = pd.DataFrame(formatted_terms, columns=cols[: len(sample_row)])

        if synonym and not output.empty:
            agg_dict = {col_2_name: list}
            if "score" in output.columns:
                agg_dict["score"] = list
            output = output.groupby(col_1_name, as_index=False).agg(agg_dict)
            
        # Exports
        if not extension:
            extension = ".txt"
            path = path.with_suffix(extension)

        if extension in [".txt", ".csv", ".xlsx"]:
            output_export = output.copy()
            if synonym and not output_export.empty:
                output_export[col_2_name] = output_export[col_2_name].apply(
                    lambda x: ", ".join(map(str, x))
                )
                if "score" in output_export.columns:
                    output_export["score"] = output_export["score"].apply(
                        lambda x: ", ".join(map(str, x))
                    )

            if extension == ".txt":
                output_export.to_csv(path, index=False, sep="\t")
            elif extension == ".csv":
                output_export.to_csv(path, index=False)
            elif extension == ".xlsx":
                output_export.to_excel(path, index=False)

        elif extension == ".json":
            output.to_json(path, orient="records", indent=4, force_ascii=False)
        elif extension == ".jsonl":
            output.to_json(path, orient="records", lines=True, force_ascii=False)
        else:
            raise ValueError(f"Unsupported format '{extension}'.")

        print(f"Aligned candidate terms saved to disk ({path})", flush=True)