class _SplitCorpus():

    def __init__(self, train, test):

        self.train = train
        self.test = test

    def save(self, split=None):

        if split == "train":
            # save self.train
            pass

def train_test_split(corpus, stream=False, train_size=0.85, test_size=0.15, seed=123):

    if train_size + test_size != 1:
        raise ValueError("Split sizes do not sum up to 1")

    import random

    if stream:
        def _split_generator(target_split):
            
            with open(corpus, "r", encoding="utf-8") as f: # falta normalizar etc.
                for i, line in enumerate(f):

                    random_num = random.Random(f"{seed}_{i}")
                    roll = random_num.random()

                    if roll < train_size:
                        split = "train"
                    else:
                        split = "test"

                    if split == target_split:
                        yield line

        # print("Streaming enabled. Corpus split sizes will be approximate", flush=True)

        return _SplitCorpus(
            train=_split_generator("train"),
            test=_split_generator("test"),
        )

    else:
        random.seed(seed)
        segments = []
        
        with open(corpus, "r", encoding="utf-8") as f:
            for line in f:
                segments.append(line)

        test_segs = random.sample(segments, round(len(segments) * test_size))
        train_segs = list(set(segments) - set(test_segs))

        # print(f"Corpus split results: train - {len(train_segs)} | test - {len(test_segs)}\n", flush=True)

        return _SplitCorpus(
            train=train_segs,
            test=test_segs,
        )