from importlib.resources import files

class JSONLDataset:
    '''
    Class to load external ATE datasets for evaluation.
    '''
    def __init__(self, dataset_name, split):
        valid_splits = ["train", "test"]
        
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
        
        self.dataset_name = dataset_name
        self.split = split

        self.df = None
        self.golden_labels = None
        self.corpus = None

        self._path = files("TBXTools.datasets.data") / f"{dataset_name}_{self.split}.jsonl"

    def to_pandas(self):
        import pandas as pd
        return pd.read_json(self._path, orient='records', lines=True)

def _data_to_class(dataset):
    '''
    Helper function that populates the JSONLDataset class with the chosen dataset.
    '''
    df = dataset.to_pandas()
    dataset.df = df
    dataset.golden_labels = df['terms'].to_dict()
    dataset.corpus = df['segment'].to_list()

    return dataset

def load_detech26(split="train"):
    '''
    Load the DETECH2026 task A dataset for automatic terminology extraction.

    Split takes 'train' or 'test'.
    '''
    dataset = JSONLDataset(dataset_name="detech26", split=split)

    dataset = _data_to_class(dataset=dataset)
    
    return dataset

# def load_tutorial(split="test"):
#     '''
#     Load the data used in the tutorial.
#     '''
#     dataset = JSONLDataset(dataset_name="tutorial-wiki", split=split)

#     dataset = data_to_class(dataset=dataset)

#     return dataset