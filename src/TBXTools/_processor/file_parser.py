from xml.etree import ElementTree as etree
from TBXTools._utils.utils import get_lang
from pathlib import Path
from typing import Generator

class FileParser:

    def __init__(self, src_lang=None, tgt_lang=None):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.corpus = None

    def parse(self, corpus):
        if corpus:

            if isinstance(corpus, Generator):  # for bilingual extraction
                return corpus

            # parsing a list of corpora or a list of segments
            elif isinstance(corpus, list):
                return self._parse_list(corpus_list=corpus)                 

            # parsing moses or 2 separate txt files
            elif isinstance(corpus, tuple) and len(corpus) == 2:
                src_file, tgt_file = corpus

                src_corpus_gen = (src for src in self._parse_txt(src_file))
                tgt_corpus_gen = (tgt for tgt in self._parse_txt(tgt_file))

                return (src_corpus_gen, tgt_corpus_gen)

            else:  # parsing bilingual
                ext = Path(corpus).suffix.lower()
                if ext in [".tab", ".tsv"]:  # parsing tsv
                    src_corpus_gen = (src for src, tgt in self._parse_tsv(corpus))
                    tgt_corpus_gen = (tgt for src, tgt in self._parse_tsv(corpus))

                    return (src_corpus_gen, tgt_corpus_gen)

                elif ext == ".tmx":  # parsing tmx
                    src_corpus_gen = (src for src, tgt in self._parse_tmx(corpus))
                    tgt_corpus_gen = (tgt for src, tgt in self._parse_tmx(corpus))

                    return (src_corpus_gen, tgt_corpus_gen)

                elif ext == ".txt":  # parsing monolingual txt
                    monolingual_corpus_gen = (seg for seg in self._parse_txt(corpus))

                    return monolingual_corpus_gen

                else:
                    raise ValueError(
                        f"Unsupported file format: {ext}. Supported formats: moses, txt, tsv and tmx")

    def _parse_tmx(self, file_path):
        xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"

        for event, elem in etree.iterparse(str(file_path), events=("end",)):
            if elem.tag == "tu":
                s_text, t_text = None, None
                for tuv in elem.findall("tuv"):
                    l_attr = (tuv.attrib.get(xml_lang) or tuv.attrib.get(
                        "lang") or "").lower().strip()
                    seg = tuv.find("seg")
                    if seg is not None and seg.text and seg.text.strip():
                        if l_attr.startswith(self.src_lang):
                            s_text = seg.text.strip()
                        elif l_attr.startswith(self.tgt_lang):
                            t_text = seg.text.strip()

                if s_text and t_text:

                    yield s_text, t_text

                elem.clear()

    def _parse_tsv(self, file_path, encoding="utf-8"):
        src_list, tgt_list = [], []
        with open(str(file_path), "r", encoding=encoding, errors="ignore") as cf:
            for linia in cf:
                linia = linia.rstrip("\r\n").strip()
                if not linia:
                    continue

                camps = linia.split("\t")
                if len(camps) >= 2:

                    yield camps[0].strip(), camps[1].strip()

    def _parse_txt(self, file_path, encoding="utf-8"):

        with open(str(file_path), "r", encoding=encoding, errors="ignore") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:

                    yield clean_line

    
    def _parse_list(self, corpus_list):

        if corpus_list:
            if isinstance(corpus_list, list):
                if Path(corpus_list[0]).is_file():
                    is_file = True
                else:
                    is_file = False

                if is_file:
                    corpora_generators = []
                    for file in corpus_list:
                        corpora_generators.append(self.parse(corpus=file))

                    return corpora_generators

                else:
                    return corpus_list