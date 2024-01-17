class GptModel:
    def __init__(self, name, input_price, output_price, max_tokens_limit, max_tokens_for_answer):
        self.name = name
        self.input_price = input_price
        self.output_price = output_price
        self.max_tokens_limit = max_tokens_limit
        self.max_tokens_for_answer = max_tokens_for_answer
        self.max_tokens_for_request = max_tokens_limit - max_tokens_for_answer
