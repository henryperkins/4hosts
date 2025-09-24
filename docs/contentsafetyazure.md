---
title: "Quickstart: Groundedness detection (preview) - Azure AI services"
source: "https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-groundedness?tabs=python&pivots=programming-language-rest"
author:
  - "[[PatrickFarley]]"
published:
created: 2025-09-23
description: "Learn how to detect whether the text responses of large language models (LLMs) are grounded in the source materials provided by the users."
tags:
  - "clippings"
---
## Quickstart: Use Groundedness detection (preview)

This guide shows you how to use the groundedness detection API. This feature automatically detects and corrects text that goes against the provided source documents, ensuring that the generated content is aligned with factual or intended references. Below, we explore several common scenarios to help you understand how and when to apply these features to achieve the best outcomes.

## Prerequisites

- An Azure account. If you don't have one, you can [create one for free](https://azure.microsoft.com/pricing/purchase-options/azure-account?icid=ai-services).
- An [Azure AI resource](https://ms.portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub/%7E/AIServices).

## Setup

Follow these steps to use the Content Safety **try it out** page:

1. Go to [Azure AI Foundry](https://ai.azure.com/?cid=learnDocs) and navigate to your project/hub. Then select the **Guardrails + controls** tab on the left nav and select the **Try it out** tab.
2. On the **Try it out** page, you can experiment with various Guardrails & controls features such as text and image content, using adjustable thresholds to filter for inappropriate or harmful content.

![Screenshot of the try it out page for Guardrails & controls.](https://learn.microsoft.com/en-us/azure/ai-foundry/media/content-safety/try-it-out.png)

Screenshot of the try it out page for Guardrails & controls.

The Groundedness detection panel lets you detect whether the text responses of large language models (LLMs) are grounded in the source materials provided by the users.

1. Select the **Groundedness detection** panel.
2. Select a sample content set on the page, or input your own for testing.
3. Optionally, enable the reasoning feature and select your Azure OpenAI resource from the dropdown.
4. Select **Run test**. The service returns the groundedness detection result.

For more information, see the [Groundedness detection conceptual guide](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness).

## Prerequisites

- An Azure subscription - [Create one for free](https://azure.microsoft.com/free/cognitive-services/)
- Once you have your Azure subscription, [create a Content Safety resource](https://aka.ms/acs-create "Create a Content Safety resource") in the Azure portal to get your key and endpoint. Enter a unique name for your resource, select your subscription, and select a resource group, [supported region](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview#region-availability), and supported pricing tier. Then select **Create**.
- The resource takes a few minutes to deploy. After it does, go to the new resource. In the left pane, under **Resource Management**, select **API Keys and Endpoints**. Copy one of the subscription key values and endpoint to a temporary location for later use.
- (Optional) If you want to use the *reasoning* feature, create an Azure OpenAI in Azure AI Foundry Models resource with a GPT model deployed.
- [cURL](https://curl.haxx.se/) or [Python](https://www.python.org/downloads/) installed.

## Authentication

For enhanced security, you need to use Managed Identity (MI) to manage access to your resources, for more details, please refer to [Security](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview#security).

In the simple case without the *reasoning* feature, the Groundedness detection API classifies the ungroundedness of the submitted content as `true` or `false`.

- [cURL](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/?tabs=python&pivots=programming-language-rest#tabpanel_1_curl)
- [Python](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/?tabs=python&pivots=programming-language-rest#tabpanel_1_python)

Create a new Python file named *quickstart.py*. Open the new file in your preferred editor or IDE.

1. Replace the contents of *quickstart.py* with the following code. Enter your endpoint URL and key in the appropriate fields. Optionally, replace the `"query"` or `"text"` fields in the body with your own text you'd like to analyze.
	```python
	import http.client
	import json
	endpoint = "<your_custom_subdomain>.cognitiveservices.azure.com"
	conn = http.client.HTTPSConnection(endpoint)
	payload = json.dumps({
	  "domain": "Generic",
	  "task": "QnA",
	  "qna": {
	    "query": "How much does she currently get paid per hour at the bank?"
	  },
	  "text": "12/hour",
	  "groundingSources": [
	    "I'm 21 years old and I need to make a decision about the next two years of my life. Within a week. I currently work for a bank that requires strict sales goals to meet. IF they aren't met three times (three months) you're canned. They pay me 10/hour and it's not unheard of to get a raise in 6ish months. The issue is, **I'm not a salesperson**. That's not my personality. I'm amazing at customer service, I have the most positive customer service \"reports\" done about me in the short time I've worked here. A coworker asked \"do you ask for people to fill these out? you have a ton\". That being said, I have a job opportunity at Chase Bank as a part time teller. What makes this decision so hard is that at my current job, I get 40 hours and Chase could only offer me 20 hours/week. Drive time to my current job is also 21 miles **one way** while Chase is literally 1.8 miles from my house, allowing me to go home for lunch. I do have an apartment and an awesome roommate that I know wont be late on his portion of rent, so paying bills with 20hours a week isn't the issue. It's the spending money and being broke all the time.\n\nI previously worked at Wal-Mart and took home just about 400 dollars every other week. So I know i can survive on this income. I just don't know whether I should go for Chase as I could definitely see myself having a career there. I'm a math major likely going to become an actuary, so Chase could provide excellent opportunities for me **eventually**."
	  ],
	  "reasoning": false
	})
	headers = {
	  'Ocp-Apim-Subscription-Key': '<your_subscription_key>',
	  'Content-Type': 'application/json'
	}
	conn.request("POST", "/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview", payload, headers)
	res = conn.getresponse()
	data = res.read()
	print(data.decode("utf-8"))
	```
2. Run the application with the `python` command:
	```
	python quickstart.py
	```
	Wait a few moments to get the response.

To test a summarization task instead of a question answering (QnA) task, use the following sample JSON body:

```json
{
  "domain": "Medical",
  "task": "Summarization",
  "text": "Ms Johnson has been in the hospital after experiencing a stroke.",
  "groundingSources": [
    "Our patient, Ms. Johnson, presented with persistent fatigue, unexplained weight loss, and frequent night sweats. After a series of tests, she was diagnosed with Hodgkin’s lymphoma, a type of cancer that affects the lymphatic system. The diagnosis was confirmed through a lymph node biopsy revealing the presence of Reed-Sternberg cells, a characteristic of this disease. She was further staged using PET-CT scans. Her treatment plan includes chemotherapy and possibly radiation therapy, depending on her response to treatment. The medical team remains optimistic about her prognosis given the high cure rate of Hodgkin’s lymphoma."
  ],
  "reasoning": false
}
```

The following fields must be included in the URL:

| Name | Required | Description | Type |
| --- | --- | --- | --- |
| **API Version** | Required | This is the API version to be used. The current version is: api-version=2024-09-15-preview. Example: `<endpoint>/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview` | String |

The parameters in the request body are defined in this table:

| Name | Description | Type |
| --- | --- | --- |
| **domain** | (Optional) `MEDICAL` or `GENERIC`. Default value: `GENERIC`. | Enum |
| **task** | (Optional) Type of task: `QnA`, `Summarization`. Default value: `Summarization`. | Enum |
| **qna** | (Optional) Holds QnA data when the task type is `QnA`. | String |
| \- `query` | (Optional) This represents the question in a QnA task. Character limit: 7,500. | String |
| **text** | (Required) The LLM output text to be checked. Character limit: 7,500. | String |
| **groundingSources** | (Required) Uses an array of grounding sources to validate AI-generated text. See [Input requirements](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview#input-requirements) for limits. | String array |
| **reasoning** | (Optional) Specifies whether to use the reasoning feature. The default value is `false`. If `true`, you need to bring your own Azure OpenAI GPT-4o (versions 0513, 0806) to provide an explanation. Be careful: using reasoning increases the processing time. | Boolean |

After you submit your request, you'll receive a JSON response reflecting the Groundedness analysis performed. Here’s what a typical output looks like:

```json
{
  "ungroundedDetected": true,
  "ungroundedPercentage": 1,
  "ungroundedDetails": [
    {
      "text": "12/hour."
    }
  ]
}
```

The JSON objects in the output are defined here:

| Name | Description | Type |
| --- | --- | --- |
| **ungroundedDetected** | Indicates whether the text exhibits ungroundedness. | Boolean |
| **ungroundedPercentage** | Specifies the proportion of the text identified as ungrounded, expressed as a number between 0 and 1, where 0 indicates no ungrounded content and 1 indicates entirely ungrounded content. This is not a confidence level. | Float |
| **ungroundedDetails** | Provides insights into ungrounded content with specific examples and percentages. | Array |
| \- **`text`** | The specific text that is ungrounded. | String |

The Groundedness detection API provides the option to include *reasoning* in the API response. With reasoning enabled, the response includes a `"reasoning"` field that details specific instances and explanations for any detected ungroundedness.

In order to use your Azure OpenAI GPT-4o (versions 0513, 0806) resource to enable the reasoning feature, use Managed Identity to allow your Content Safety resource to access the Azure OpenAI resource:

1. Enable Managed Identity for Azure AI Content Safety.
	Navigate to your Azure AI Content Safety instance in the Azure portal. Find the **Identity** section under the **Settings** category. Enable the system-assigned managed identity. This action grants your Azure AI Content Safety instance an identity that can be recognized and used within Azure for accessing other resources.
	![Screenshot of a Content Safety identity resource in the Azure portal.](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/media/content-safety-identity.png)
	Screenshot of a Content Safety identity resource in the Azure portal.
2. Assign role to Managed Identity.
	Navigate to your Azure OpenAI instance, select **Add role assignment** to start the process of assigning an Azure OpenAI role to the Azure AI Content Safety identity.
	![Screenshot of adding role assignment in Azure portal.](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/media/add-role-assignment.png)
	Screenshot of adding role assignment in Azure portal.
	Choose the **User** or **Contributor** role.
	![Screenshot of the Azure portal with the Contributor and User roles displayed in a list.](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/media/assigned-roles-simple.png)
	Screenshot of the Azure portal with the Contributor and User roles displayed in a list.

In your request to the Groundedness detection API, set the `"reasoning"` body parameter to `true`, and provide the other needed parameters:

```json
{
  "domain": "Medical",
  "task": "Summarization",
  "text": "The patient name is Kevin.",
  "groundingSources": [
    "The patient name is Jane."
  ],
  "reasoning": true,
  "llmResource": {
    "resourceType": "AzureOpenAI",
    "azureOpenAIEndpoint": "<your_OpenAI_endpoint>",
    "azureOpenAIDeploymentName": "<your_deployment_name>"
  }
}
```

- [cURL](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/?tabs=python&pivots=programming-language-rest#tabpanel_2_curl)
- [Python](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/?tabs=python&pivots=programming-language-rest#tabpanel_2_python)

Create a new Python file named *quickstart.py*. Open the new file in your preferred editor or IDE.

1. Replace the contents of *quickstart.py* with the following code. Enter your endpoint URL and key in the appropriate fields. Also enter your Azure OpenAI endpoint and deployment name. Optionally, replace the `"query"` or `"text"` fields in the body with your own text you'd like to analyze.
	```python
	import http.client
	import json
	conn = http.client.HTTPSConnection("<endpoint>/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview")
	payload = json.dumps({
	  "domain": "Generic",
	  "task": "QnA",
	  "qna": {
	    "query": "How much does she currently get paid per hour at the bank?"
	  },
	  "text": "12/hour",
	  "groundingSources": [
	    "I'm 21 years old and I need to make a decision about the next two years of my life. Within a week. I currently work for a bank that requires strict sales goals to meet. IF they aren't met three times (three months) you're canned. They pay me 10/hour and it's not unheard of to get a raise in 6ish months. The issue is, **I'm not a salesperson**. That's not my personality. I'm amazing at customer service, I have the most positive customer service \"reports\" done about me in the short time I've worked here. A coworker asked \"do you ask for people to fill these out? you have a ton\". That being said, I have a job opportunity at Chase Bank as a part time teller. What makes this decision so hard is that at my current job, I get 40 hours and Chase could only offer me 20 hours/week. Drive time to my current job is also 21 miles **one way** while Chase is literally 1.8 miles from my house, allowing me to go home for lunch. I do have an apartment and an awesome roommate that I know wont be late on his portion of rent, so paying bills with 20hours a week isn't the issue. It's the spending money and being broke all the time.\n\nI previously worked at Wal-Mart and took home just about 400 dollars every other week. So I know i can survive on this income. I just don't know whether I should go for Chase as I could definitely see myself having a career there. I'm a math major likely going to become an actuary, so Chase could provide excellent opportunities for me **eventually**."
	  ],
	  "reasoning": True
	  "llmResource": {
	   "resourceType": "AzureOpenAI",
	   "azureOpenAIEndpoint": "<your_OpenAI_endpoint>",
	   "azureOpenAIDeploymentName": "<your_deployment_name>"
	  }
	})
	headers = {
	  'Ocp-Apim-Subscription-Key': '<your_subscription_key>',
	  'Content-Type': 'application/json'
	}
	conn.request("POST", "/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview", payload, headers)
	res = conn.getresponse()
	data = res.read()
	print(data.decode("utf-8"))
	```
2. Run the application with the `python` command:
	```
	python quickstart.py
	```
	Wait a few moments to get the response.

The parameters in the request body are defined in this table:

| Name | Description | Type |
| --- | --- | --- |
| **domain** | (Optional) `MEDICAL` or `GENERIC`. Default value: `GENERIC`. | Enum |
| **task** | (Optional) Type of task: `QnA`, `Summarization`. Default value: `Summarization`. | Enum |
| **qna** | (Optional) Holds QnA data when the task type is `QnA`. | String |
| \- `query` | (Optional) This represents the question in a QnA task. Character limit: 7,500. | String |
| **text** | (Required) The LLM output text to be checked. Character limit: 7,500. | String |
| **groundingSources** | (Required) Uses an array of grounding sources to validate AI-generated text. See [Input requirements](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview#input-requirements) for limits, | String array |
| **reasoning** | (Optional) Set to `true`, the service uses Azure OpenAI resources to provide an explanation. Be careful: using reasoning increases the processing time and incurs extra fees. | Boolean |
| **llmResource** | (Required) If you want to use your own Azure OpenAI GPT-4o (versions 0513, 0806) resource to enable reasoning, add this field and include the subfields for the resources used. | String |
| \- `resourceType ` | Specifies the type of resource being used. Currently it only allows `AzureOpenAI`. We only support Azure OpenAI GPT-4o (versions 0513, 0806) resources and do not support other models. | Enum |
| \- `azureOpenAIEndpoint ` | Your endpoint URL for Azure OpenAI service. | String |
| \- `azureOpenAIDeploymentName` | The name of the specific model deployment to use. | String |

After you submit your request, you'll receive a JSON response reflecting the Groundedness analysis performed. Here’s what a typical output looks like:

```json
{
  "ungroundedDetected": true,
  "ungroundedPercentage": 1,
  "ungroundedDetails": [
    {
      "text": "12/hour.",
      "offset": {
        "utf8": 0,
        "utf16": 0,
        "codePoint": 0
      },
      "length": {
        "utf8": 8,
        "utf16": 8,
        "codePoint": 8
      },
      "reason": "None. The premise mentions a pay of \"10/hour\" but does not mention \"12/hour.\" It's neutral. "
    }
  ]
}
```

The JSON objects in the output are defined here:

| Name | Description | Type |
| --- | --- | --- |
| **ungroundedDetected** | Indicates whether the text exhibits ungroundedness. | Boolean |
| **ungroundedPercentage** | Specifies the proportion of the text identified as ungrounded, expressed as a number between 0 and 1, where 0 indicates no ungrounded content and 1 indicates entirely ungrounded content. This is not a confidence level. | Float |
| **ungroundedDetails** | Provides insights into ungrounded content with specific examples and percentages. | Array |
| \- **`text`** | The specific text that is ungrounded. | String |
| \- **`offset`** | An object describing the position of the ungrounded text in various encoding. | String |
| \- `offset > utf8` | The offset position of the ungrounded text in UTF-8 encoding. | Integer |
| \- `offset > utf16` | The offset position of the ungrounded text in UTF-16 encoding. | Integer |
| \- `offset > codePoint` | The offset position of the ungrounded text in terms of Unicode code points. | Integer |
| \- **`length`** | An object describing the length of the ungrounded text in various encoding. (utf8, utf16, codePoint), similar to the offset. | Object |
| \- `length > utf8` | The length of the ungrounded text in UTF-8 encoding. | Integer |
| \- `length > utf16` | The length of the ungrounded text in UTF-16 encoding. | Integer |
| \- `length > codePoint` | The length of the ungrounded text in terms of Unicode code points. | Integer |
| \- **`reason`** | Offers explanations for detected ungroundedness. | String |

The groundedness detection API includes a correction feature that automatically corrects any detected ungroundedness in the text based on the provided grounding sources. When the correction feature is enabled, the response includes a `"correction Text"` field that presents the corrected text aligned with the grounding sources.

To use your Azure OpenAI GPT-4o (versions 0513, 0806) resource for enabling the correction feature, use Managed Identity to allow your Content Safety resource to access the Azure OpenAI resource. Follow the steps in the [earlier section](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/?tabs=python&pivots=programming-language-rest#connect-your-own-gpt-deployment) to set up the Managed Identity.

In your request to the groundedness detection API, set the `"correction"` body parameter to `true`, and provide the other necessary parameters:

```json
{
  "domain": "Medical",
  "task": "Summarization",
  "text": "The patient name is Kevin.",
  "groundingSources": [
    "The patient name is Jane."
  ],
  "correction": true,
  "llmResource": {
    "resourceType": "AzureOpenAI",
    "azureOpenAIEndpoint": "<your_OpenAI_endpoint>",
    "azureOpenAIDeploymentName": "<your_deployment_name>"
  }
}
```

- [cURL](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/?tabs=python&pivots=programming-language-rest#tabpanel_3_curl)
- [Python](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/?tabs=python&pivots=programming-language-rest#tabpanel_3_python)

Create a Python script named quickstart.py and include the following code. Update the endpoint URL and key as appropriate:

```python
conn = http.client.HTTPSConnection("<endpoint>/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview")
payload = json.dumps({
  "domain": "Generic",
  "task": "Summarization",
  "text": "The patient name is Kevin.",
  "groundingSources": [
    "The patient name is Jane."
  ],
  "correction": True,
  "llmResource": {
   "resourceType": "AzureOpenAI",
   "azureOpenAIEndpoint": "<your_OpenAI_endpoint>",
   "azureOpenAIDeploymentName": "<your_deployment_name>"
  }
})
headers = {
  'Ocp-Apim-Subscription-Key': '<your_subscription_key>',
  'Content-Type': 'application/json'
}
conn.request("POST", "/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))
```

The parameters in the request body are defined in this table:

| Name | Description | Type |
| --- | --- | --- |
| **domain** | (Optional) `MEDICAL` or `GENERIC`. Default value: `GENERIC`. | Enum |
| **task** | (Optional) Type of task: `QnA`, `Summarization`. Default value: `Summarization`. | Enum |
| **qna** | (Optional) Holds QnA data when the task type is `QnA`. | String |
| \- `query` | (Optional) This represents the question in a QnA task. Character limit: 7,500. | String |
| **text** | (Required) The LLM output text to be checked. Character limit: 7,500. | String |
| **groundingSources** | (Required) Uses an array of grounding sources to validate AI-generated text. See [Input requirements](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview#input-requirements) for limits. | String Array |
| **correction** | (Optional) Set to `true`, the service uses Azure OpenAI resources to provide the corrected text, ensuring consistency with the grounding sources. Be careful: using correction increases the processing time and incurs extra fees. | Boolean |
| **llmResource** | (Required) If you want to use your own Azure OpenAI GPT-4o (versions 0513, 0806) resource to enable reasoning, add this field and include the subfields for the resources used. | String |
| \- `resourceType ` | Specifies the type of resource being used. Currently it only allows `AzureOpenAI`. We only support Azure OpenAI GPT-4o (versions 0513, 0806) resources and do not support other models. | Enum |
| \- `azureOpenAIEndpoint ` | Your endpoint URL for Azure OpenAI service. | String |
| \- `azureOpenAIDeploymentName` | The name of the specific model deployment to use. | String |

The response includes a `"correction Text"` field containing the corrected text, ensuring consistency with the provided grounding sources.

The correction feature detects that `Kevin` is ungrounded because it conflicts with the grounding source `Jane`. The API returns the corrected text: `"The patient name is Jane."`

```json
{
  "ungroundedDetected": true,
  "ungroundedPercentage": 1,
  "ungroundedDetails": [
    {
      "text": "The patient name is Kevin"
    }
  ],
  "correction Text": "The patient name is Jane"
}
```

The JSON objects in the output are defined here:

| Name | Description | Type |
| --- | --- | --- |
| **ungroundedDetected** | Indicates if ungrounded content was detected. | Boolean |
| **ungroundedPercentage** | The proportion of ungrounded content in the text. This is not a confidence level. | Float |
| **ungroundedDetails** | Details of ungrounded content, including specific text segments. | Array |
| \- **`text`** | The specific text that is ungrounded. | String |
| \- **`offset`** | An object describing the position of the ungrounded text in various encoding. | String |
| \- `offset > utf8` | The offset position of the ungrounded text in UTF-8 encoding. | Integer |
| \- `offset > utf16` | The offset position of the ungrounded text in UTF-16 encoding. | Integer |
| \- **`length`** | An object describing the length of the ungrounded text in various encoding. (utf8, utf16, codePoint), similar to the offset. | Object |
| \- `length > utf8` | The length of the ungrounded text in UTF-8 encoding. | Integer |
| \- `length > utf16` | The length of the ungrounded text in UTF-16 encoding. | Integer |
| \- `length > codePoint` | The length of the ungrounded text in terms of Unicode code points. | Integer |
| \- **`correction Text`** | The corrected text, ensuring consistency with the grounding sources. | String |

If you want to clean up and remove an Azure AI services subscription, you can delete the resource or resource group. Deleting the resource group also deletes any other resources associated with it.

- [Azure portal](https://learn.microsoft.com/en-us/azure/ai-services/multi-service-resource?pivots=azportal#clean-up-resources)
- [Azure CLI](https://learn.microsoft.com/en-us/azure/ai-services/multi-service-resource?pivots=azcli#clean-up-resources)

- [Groundedness detection concepts](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness)
- Combine Groundedness detection with other LLM safety features like [Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak).

---

## Additional resources

Training

Module

[Implement generative AI guardrails with Azure AI Content Safety - Training](https://learn.microsoft.com/en-us/training/modules/moderate-content-detect-harm-azure-ai-content-safety/?source=recommendations)

Learn how to implement generative AI guardrails with Azure AI Content Safety.
