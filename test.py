from anthropic import AnthropicFoundry

endpoint = "https://cgft-code-tools.services.ai.azure.com/anthropic/"
deployment_name = "claude-sonnet-4-5"
api_key = "9EwSK6FKkRu0NLK9NykHihHcfhK1VHYFbQOo51kHTD6mnqfcwZJ4JQQJ99CAACHYHv6XJ3w3AAAAACOGneG3"

client = AnthropicFoundry(
    api_key=api_key,
    base_url=endpoint
)

message = client.messages.create(
    model=deployment_name,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=1024,
)

print(message.content)