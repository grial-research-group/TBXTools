class _SplitCorpus():

    def __init__(self, train, val, test):

        self.train = train
        self.val = val
        self.test = test

    def save(self, split=None):

        if split == "train":
            # save self.train
            pass

def train_val_test_split(corpus, train_size=0.7, val_size=0.15, test_size=0.15, seed=123):

    if train_size + val_size + test_size != 1:
        raise ValueError("Split sizes do not sum up to 1")

    import random
    random.seed(seed)
    segments = []
    
    with open(corpus, "r", encoding="utf-8") as f: # falta normalizar etc.
        for line in f:
            segments.append(line)

    test_segs = random.sample(segments, round(len(segments) * test_size))
    rest = list(set(segments) - set(test_segs))
    val_segs = random.sample(rest, round(len(segments) * val_size)) 
    train_segs = list(set(rest) - set(val_segs))

    print(f"Split results: train - {len(train_segs)} | val - {len(val_segs)} | test - {len(test_segs)}\n", flush=True)

    return _SplitCorpus(
        train=train_segs,
        val=val_segs,
        test=test_segs,
    )