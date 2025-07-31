# How to Use Function Calling with Azure OpenAI in Azure AI Foundry Models

The latest versions of gpt-35-turbo and gpt-4 are fine-tuned to work with functions and are able to determine when and how a function should be called. When one or more functions are included in your request, the model determines if any should be called based on the prompt context. When the model determines a function should be called, it responds with a JSON object containing the function arguments.

The models formulate API calls and structure data outputs based on the functions you specify. While the models can generate these calls, it's up to you to execute them, ensuring you maintain control.

## Working with Functions: 3-Step Process

At a high level, working with functions involves these three steps:

1. Call the chat completions API with your functions and the user's input
2. Use the model's response to call your API or function
3. Call the chat completions API again, including the function response to get a final response

> **Important**
> The `functions` and `function_call` parameters have been deprecated with the release of the [`2023-12-01-preview`](https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2023-12-01-preview/inference.json) API version.
> - The replacement for `functions` is the [`tools`](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference#chat-completions) parameter
> - The replacement for `function_call` is the [`tool_choice`](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference#chat-completions) parameter

## Function Calling Support

### Models Supporting Parallel Function Calling
- `gpt-35-turbo` (`1106`)
- `gpt-35-turbo` (`0125`)
- `gpt-4` (`1106-Preview`)
- `gpt-4` (`0125-Preview`)
- `gpt-4` (`vision-preview`)
- `gpt-4` (`2024-04-09`)
- `gpt-4o` (`2024-05-13`)
- `gpt-4o` (`2024-08-06`)
- `gpt-4o` (`2024-11-20`)
- `gpt-4o-mini` (`2024-07-18`)
- `gpt-4.1` (`2025-04-14`)
- `gpt-4.1-mini` (`2025-04-14`)

> Support for parallel function calling was first added in API version [`2023-12-01-preview`](https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2023-12-01-preview/inference.json)

### Models Supporting Basic Function Calling with Tools
- All models that support parallel function calling
- `codex-mini` (`2025-05-16`)
- `o3-pro` (`2025-06-10`)
- `o4-mini` (`2025-04-16`)
- `o3` (`2025-04-16`)
- `gpt-4.1-nano` (`2025-04-14`)
- `o3-mini` (`2025-01-31`)
- `o1` (`2024-12-17`)

> **Note**
> The `tool_choice` parameter is now supported with `o3-mini` and `o1`. For more information on supported parameters with o-series models, see the [reasoning models guide](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning).

> **Important**
> Tool/function descriptions are currently limited to 1024 characters with Azure OpenAI. This article will be updated if this limit changes.

## Basic Function Calling Example

This example demonstrates a simple function call to check the time in three hardcoded locations using a single tool/function.

```python
import os
import json
from openai import AzureOpenAI
from datetime import datetime
from zoneinfo import ZoneInfo

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-02-01-preview"
)

# Define the deployment you want to use for your chat completions API calls
deployment_name = "<YOUR_DEPLOYMENT_NAME_HERE>"

# Simplified timezone data
TIMEZONE_DATA = {
    "tokyo": "Asia/Tokyo",
    "san francisco": "America/Los_Angeles",
    "paris": "Europe/Paris"
}

def get_current_time(location):
    """Get the current time for a given location"""
    print(f"get_current_time called with location: {location}")
    location_lower = location.lower()

    for key, timezone in TIMEZONE_DATA.items():
        if key in location_lower:
            print(f"Timezone found for {key}")
            current_time = datetime.now(ZoneInfo(timezone)).strftime("%I:%M %p")
            return json.dumps({
                "location": location,
                "current_time": current_time
            })

    print(f"No timezone data found for {location_lower}")
    return json.dumps({"location": location, "current_time": "unknown"})

def run_conversation():
    # Initial user message
    messages = [{"role": "user", "content": "What's the current time in San Francisco"}] # Single function call
    #messages = [{"role": "user", "content": "What's the current time in San Francisco, Tokyo, and Paris?"}] # Parallel function call with a single tool/function defined

    # Define the function for the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                },
            }
        }
    ]

    # First API call: Ask the model to use the function
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Process the model's response
    response_message = response.choices[0].message
    messages.append(response_message)

    print("Model's response:")
    print(response_message)

    # Handle function calls
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "get_current_time":
                function_args = json.loads(tool_call.function.arguments)
                print(f"Function arguments: {function_args}")
                time_response = get_current_time(
                    location=function_args.get("location")
                )
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "get_current_time",
                    "content": time_response,
                })
    else:
        print("No tool calls were made by the model.")

    # Second API call: Get the final response from the model
    final_response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
    )

    return final_response.choices[0].message.content

# Run the conversation and print the result
print(run_conversation())
```

**Output:**

```
Model's response:
ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_pOsKdUlqvdyttYB67MOj434b', function=Function(arguments='{"location":"San Francisco"}', name='get_current_time'), type='function')])
Function arguments: {'location': 'San Francisco'}
get_current_time called with location: San Francisco
Timezone found for san francisco
The current time in San Francisco is 09:24 AM.
```

## Parallel Function Calling

To convert the basic example to parallel function calling, modify the messages array to request times for multiple locations:

```python
#messages = [{"role": "user", "content": "What's the current time in San Francisco"}] # Single function call
messages = [{"role": "user", "content": "What's the current time in San Francisco, Tokyo, and Paris?"}] # Parallel function call with a single tool/function defined
```

**Output:**

```
Model's response:
ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_IjcAVz9JOv5BXwUx1jd076C1', function=Function(arguments='{"location": "San Francisco"}', name='get_current_time'), type='function'), ChatCompletionMessageToolCall(id='call_XIPQYTCtKIaNCCPTdvwjkaSN', function=Function(arguments='{"location": "Tokyo"}', name='get_current_time'), type='function'), ChatCompletionMessageToolCall(id='call_OHIB5aJzO8HGqanmsdzfytvp', function=Function(arguments='{"location": "Paris"}', name='get_current_time'), type='function')])
Function arguments: {'location': 'San Francisco'}
get_current_time called with location: San Francisco
Timezone found for san francisco
Function arguments: {'location': 'Tokyo'}
get_current_time called with location: Tokyo
Timezone found for tokyo
Function arguments: {'location': 'Paris'}
get_current_time called with location: Paris
Timezone found for paris
As of now, the current times are:

- **San Francisco:** 11:15 AM
- **Tokyo:** 03:15 AM (next day)
- **Paris:** 08:15 PM
```

Parallel function calls allow multiple function calls to be executed simultaneously, improving performance by reducing the number of API calls needed.

Each function call in the `tool_calls` array has a unique `id`. To respond to these calls, add new messages to the conversation with the results, each referencing the corresponding `tool_call_id`.

> The `tool_choice` parameter options:
> - `"auto"` (default): Model decides whether to call a function
> - Specific function name: Force a particular function call
> - `"none"`: Prevent function calls, generate user-facing message

## Parallel Function Calling with Multiple Functions

This example demonstrates function calling with two different tools/functions defined.

```python
import os
import json
from openai import AzureOpenAI
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-02-01-preview"
)

# Provide the model deployment name you want to use for this example
deployment_name = "YOUR_DEPLOYMENT_NAME_HERE"

# Simplified weather data
WEATHER_DATA = {
    "tokyo": {"temperature": "10", "unit": "celsius"},
    "san francisco": {"temperature": "72", "unit": "fahrenheit"},
    "paris": {"temperature": "22", "unit": "celsius"}
}

# Simplified timezone data
TIMEZONE_DATA = {
    "tokyo": "Asia/Tokyo",
    "san francisco": "America/Los_Angeles",
    "paris": "Europe/Paris"
}

def get_current_weather(location, unit=None):
    """Get the current weather for a given location"""
    location_lower = location.lower()
    print(f"get_current_weather called with location: {location}, unit: {unit}")

    for key in WEATHER_DATA:
        if key in location_lower:
            print(f"Weather data found for {key}")
            weather = WEATHER_DATA[key]
            return json.dumps({
                "location": location,
                "temperature": weather["temperature"],
                "unit": unit if unit else weather["unit"]
            })

    print(f"No weather data found for {location_lower}")
    return json.dumps({"location": location, "temperature": "unknown"})

def get_current_time(location):
    """Get the current time for a given location"""
    print(f"get_current_time called with location: {location}")
    location_lower = location.lower()

    for key, timezone in TIMEZONE_DATA.items():
        if key in location_lower:
            print(f"Timezone found for {key}")
            current_time = datetime.now(ZoneInfo(timezone)).strftime("%I:%M %p")
            return json.dumps({
                "location": location,
                "current_time": current_time
            })

    print(f"No timezone data found for {location_lower}")
    return json.dumps({"location": location, "current_time": "unknown"})

def run_conversation():
    # Initial user message
    messages = [{"role": "user", "content": "What's the weather and current time in San Francisco, Tokyo, and Paris?"}]

    # Define the functions for the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                },
            }
        }
    ]

    # First API call: Ask the model to use the functions
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Process the model's response
    response_message = response.choices[0].message
    messages.append(response_message)

    print("Model's response:")
    print(response_message)

    # Handle function calls
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            print(f"Function call: {function_name}")
            print(f"Function arguments: {function_args}")

            if function_name == "get_current_weather":
                function_response = get_current_weather(
                    location=function_args.get("location"),
                    unit=function_args.get("unit")
                )
            elif function_name == "get_current_time":
                function_response = get_current_time(
                    location=function_args.get("location")
                )
            else:
                function_response = json.dumps({"error": "Unknown function"})

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })
    else:
        print("No tool calls were made by the model.")

    # Second API call: Get the final response from the model
    final_response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
    )

    return final_response.choices[0].message.content

# Run the conversation and print the result
print(run_conversation())
```

**Output:**

```
Model's response:
ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_djHAeQP0DFEVZ2qptrO0CYC4', function=Function(arguments='{"location": "San Francisco", "unit": "celsius"}', name='get_current_weather'), type='function'), ChatCompletionMessageToolCall(id='call_q2f1HPKKUUj81yUa3ITLOZFs', function=Function(arguments='{"location": "Tokyo", "unit": "celsius"}', name='get_current_weather'), type='function'), ChatCompletionMessageToolCall(id='call_6TEY5Imtr17PaB4UhWDaPxiX', function=Function(arguments='{"location": "Paris", "unit": "celsius"}', name='get_current_weather'), type='function'), ChatCompletionMessageToolCall(id='call_vpzJ3jElpKZXA9abdbVMoauu', function=Function(arguments='{"location": "San Francisco"}', name='get_current_time'), type='function'), ChatCompletionMessageToolCall(id='call_1ag0MCIsEjlwbpAqIXJbZcQj', function=Function(arguments='{"location": "Tokyo"}', name='get_current_time'), type='function'), ChatCompletionMessageToolCall(id='call_ukOu3kfYOZR8lpxGRpdkhhdD', function=Function(arguments='{"location": "Paris"}', name='get_current_time'), type='function')])
Function call: get_current_weather
Function arguments: {'location': 'San Francisco', 'unit': 'celsius'}
get_current_weather called with location: San Francisco, unit: celsius
Weather data found for san francisco
Function call: get_current_weather
Function arguments: {'location': 'Tokyo', 'unit': 'celsius'}
get_current_weather called with location: Tokyo, unit: celsius
Weather data found for tokyo
Function call: get_current_weather
Function arguments: {'location': 'Paris', 'unit': 'celsius'}
get_current_weather called with location: Paris, unit: celsius
Weather data found for paris
Function call: get_current_time
Function arguments: {'location': 'San Francisco'}
get_current_time called with location: San Francisco
Timezone found for san francisco
Function call: get_current_time
Function arguments: {'location': 'Tokyo'}
get_current_time called with location: Tokyo
Timezone found for tokyo
Function call: get_current_time
Function arguments: {'location': 'Paris'}
get_current_time called with location: Paris
Timezone found for paris
Here's the current information for the three cities:

### San Francisco
- **Time:** 09:13 AM
- **Weather:** 72°C (quite warm!)

### Tokyo
- **Time:** 01:13 AM (next day)
- **Weather:** 10°C

### Paris
- **Time:** 06:13 PM
- **Weather:** 22°C

Is there anything else you need?
```

> **Important**
> JSON responses might not always be valid, so add error handling logic to your code. For some use cases, you may need to use fine-tuning to improve [function calling performance](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning-functions).

## Prompt Engineering with Functions

When you define a function in your request, the details are injected into the system message using syntax the model has been trained on. Functions consume tokens in your prompt, so you can apply prompt engineering techniques to optimize function calling performance. The model uses the full prompt context (function definitions, system message, and user messages) to determine if a function should be called.

### Improving Quality and Reliability

If the model isn't calling your functions as expected, try these approaches:

#### 1. Provide More Detailed Function Definitions

Include meaningful descriptions for functions and parameters:

```json
"location": {
    "type": "string",
    "description": "The location of the hotel. The location should include the city and the state's abbreviation (i.e. Seattle, WA or Miami, FL)"
}
```

#### 2. Provide Context in System Messages

Use system messages to guide the model's behavior:

```json
{"role": "system", "content": "You're an AI assistant designed to help users search for hotels. When a user asks for help finding a hotel, you should call the search_hotels function."}
```

#### 3. Instruct the Model to Ask Clarifying Questions

Prevent assumptions by instructing the model to seek clarification:

```json
{"role": "system", "content": "Don't make assumptions about what values to use with functions. Ask for clarification if a user request is ambiguous."}
```

### Reducing Errors

To reduce function call errors:

- If the model generates calls for functions you didn't provide, include this in your system message:
  `"Only use the functions you have been provided with."`

## Using Function Calling Responsibly

Function calling presents potential risks when integrating language models with other systems. Follow these best practices:

- **Validate Function Calls**: Verify all parameters, the function being called, and ensure alignment with intended actions
- **Use Trusted Data and Tools**: Only use verified data sources to prevent malicious instruction injection
- **Follow Principle of Least Privilege**: Grant minimal necessary access for functions to perform their jobs
- **Consider Real-World Impact**: Be aware of consequences for actions like code execution, database updates, or notifications
- **Implement User Confirmation Steps**: Require user confirmation before executing actions, especially for critical functions

For more information, see the [Overview of Responsible AI practices for Azure OpenAI models](https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/openai/overview).

## Next Steps

- [Learn more about Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/overview)
- For more function calling examples, check the [Azure OpenAI Samples GitHub repository](https://aka.ms/oai/functions-samples)
- Get started with GPT-35-Turbo using [the GPT-35-Turbo quickstart](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/chatgpt-quickstart)
