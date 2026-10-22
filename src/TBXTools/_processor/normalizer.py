class Normalizer():

    def normalize(segment):
        """
        Normalizes a text segment. 

        Args: 
          segment (str): A text segment to be normalized.

        Returns: 
          str: A normalized string.
        """
        import unicodedata
        import re

        punct_table = str.maketrans(
            "-−–‐“”„`´’‘",
            "----\"\"\"''''")
        double_quotes = re.compile(r",,|''|‘’|’’")

        normalized = unicodedata.normalize("NFKC", segment)
        normalized = normalized.replace("\xa0", " ")
        normalized = double_quotes.sub('"', normalized)
        normalized = normalized.translate(punct_table)
        normalized = normalized.strip()

        return normalized