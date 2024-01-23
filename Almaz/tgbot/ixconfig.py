class ixconfig_class:
    url_promt = "https://raw.githubusercontent.com/terrainternship/GPT_labsud/main/Galina/FLSE_promt"
    url_bd = 'https://github.com/terrainternship/GPT_labsud/raw/main/federallab_bd_index_v2.zip'
    url_question = 'https://github.com/terrainternship/GPT_labsud/raw/main/federallab_bd_question.zip'
    openai_model = "gpt-3.5-turbo-1106"
    # index_path = ""
    # OPENAI_API_KEY = ""
    dialog_depth = 3
    refiner_prompt = """Given the following user query and conversation log, formulate a question that would be 
    the most relevant to provide the user with an answer from a knowledge base.
    \n\nCONVERSATION LOG: \n{conversation}
    \n\nQuery: {query}\n\n
    Answer in Russian.\n\nRefined Query:"""
    stream = False
    testing = False
    printusage = False
    askrating = True
    maxquota = 10000 ## Квота в токенах на 10 часов

ixconfig = ixconfig_class()
