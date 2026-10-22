def get_lang(language):
    '''
    Takes a language name, two letter code or three letter code to return the name and the code.

    Returns:
        language_name: The name of the language in English.
        language_code: The ISO 639-1 language code (e.g. 'en').
    '''
    import pycountry
    if len(language) == 2: # if language is 2 letter iso code
        language_pyc = pycountry.languages.get(alpha_2=language)
        language_code = language_pyc.alpha_2
        language_name = language_pyc.name

    elif len(language) == 3:
        language_pyc = pycountry.languages.get(alpha_3=language)

        if language_pyc:
            language_code = language_pyc.alpha_2 if hasattr(language_pyc, 'alpha_2') else language_pyc.alpha_3
            language_name = language_pyc.name

    elif len(language) > 3: # if language is the name of the language
        for lang in pycountry.languages:
            if lang.name.lower() == language.lower():
                language_pyc = pycountry.languages.get(name=language)
                language_code = language_pyc.alpha_2
                language_name = language_pyc.name

    return language_name, language_code

def get_spacy_model_from_code(lang_code, size="sm"):
        """
        Takes a language code and returns the correct spaCy model name.

        Args:
          lang_code: the ISO language code

        Returns:
          The corresponding spaCy model name.
        """
        sizes = ["sm", "md", "lg", "trf"]

        if size not in sizes:
            raise ValueError(f"Incorrect size. Supported sizes from smallest to biggest: sm, md, lg, trf")
        
        if not lang_code:
            return None
        
        if lang_code == "en":
            return f"en_core_web_{size}"
        
        return f"{lang_code}_core_news_{size}"

def load_spacy_model(model_name): 
        """
        Checks for the spaCy model, downloading it via subprocess if missing, then loads it into self.nlp.

        Args:
            model_name (str): The name of the spaCy model to be loaded (e.g.,'en_core_web_sm').
        
        Raises:
            SystemExit: If the model download fails, the program terminates with an
            exit code of 1.
        """
        import spacy
        import subprocess
        import sys

        if not spacy.util.is_package(model_name):
            print(f"Downloading and installing spaCy model: {model_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                print(f"\nModel {model_name} downloaded successfully.")

                nlp = spacy.load(model_name)

            except Exception as e:
                print(f"Error downloading model '{model_name}': {e}")
                sys.exit(1)
        else:
            nlp = spacy.load(model_name)
        
        return nlp