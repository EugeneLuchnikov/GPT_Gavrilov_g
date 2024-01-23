from gptindex.ixapp import ixapp
from openai import AsyncOpenAI

def test():
    client: AsyncOpenAI()
    config = ixapp.config
    stream = config.openai_model
    full_response = ""
    if config.testing:
        full_response = "Ответ системы"
    else:
        responses = await client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": ixapp.promt},
                {"role": "user",
                 "content": f"Документ с информацией для ответа пользователю:\n {content}.\nВопрос клиента: {refined_query}"}
            ],
            temperature=0,
            stream=stream,
        )
        if stream:
            async for response in responses:
                delta = (response.choices[0].delta.content or "")
                if len(delta) > 0:
                    full_response += delta
                    await self.edit_text(full_response + "▌", wait=True)
        else:
            full_response = responses.choices[0].message.content