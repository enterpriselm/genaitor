def clean_genaitor_response(response):
    response = response['content']
    return response.replace("Sure, here's your prompt:\n\n",'').replace('\n\nPrompt:','')