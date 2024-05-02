# Week 5: Thursday

In today's assignment, we'll be creating an Open Source LLM-powered LangChain RAG Application.

There are 2 main sections to this assignment:

## Build 🏗️

### Build Task 1: Deploy LLM and Embedding Model to SageMaker Endpoint Through Hugging Face Inference Endpoints

#### LLM Endpoint

Select "Inference Endpoint" from the "Solutions" button in Hugging Face:

![image](https://i.imgur.com/6KC9TCD.png)

Create a "+ New Endpoint" from the Inference Endpoints dashboard.

![image](https://i.imgur.com/G6Bq9KC.png)

Select the `NousResearch/Llama-2-7b-chat-hf` model repository and name your endpoint. Select N. Virginia as your region (`us-east-1`). Give your endpoint an appropriate name.

Select the following settings for your `Advanced Configuration`.

![image](https://i.imgur.com/c0HQ7g1.png)

Create a `Protected` endpoint.

![image](https://i.imgur.com/Ak8kchZ.png)

If you were successful, you should see the following screen:

![image](https://i.imgur.com/IBYG3wm.png)

#### Embedding Model Endpoint
We'll be using `Snowflake/snowflake-arctic-embed-m` for our embedding model again!

The process is the same as the LLM - but we'll make a few specific tweaks:

Let's make sure our settings reflect the following screenshot:

![image](https://i.imgur.com/UYu9ktb.png)

> #### NOTE: PLEASE SHUTDOWN YOUR INSTANCES WHEN YOU HAVE COMPLETED THE ASSIGNMENT TO PREVENT UNESSECARY CHARGES.

### Build Task 2: Create RAG Pipeline with LangChain & LangSmith

We'll work through this week's notebook after setting up our endpoints!

The notebook will be broken into the following parts:

- 🤝 Breakout Room #1:
  1. Set up Hugging Face Inference Endpoints
  2. Install required libraries
  3. Set Environment Variables
  4. Testing our Hugging Face Inference Endpoint
  5. Creating LangChain components powered by the endpoints
  6. Retrieving data from Arxiv
  7. Creating a simple RAG pipeline with [LangChain v0.1.0](https://blog.langchain.dev/langchain-v0-1-0/)
  
- 🤝 Breakout Room #2:
  1. Set-up LangSmith
  2. Creating a LangSmith dataset
  3. Creating a custom evaluator
  4. Initializing our evaluator config
  5. Evaluating our RAG pipeline
 
The Colab link is provided [here](https://colab.research.google.com/drive/1CVHGdSDFhfeyGl20ZK-f52lst3xqKVCf?usp=sharing)

### Terminating Your Resources

Please head to the settings of each endpoint and select `Delete Endpoint`. You will need to type the name of the endpoint to delete the resources.

### Deliverables

- Completed Notebook
- Screenshot of endpoint usage

Example Screen Shot:

![image](https://i.imgur.com/qfbcVpS.png)

## Ship 🚢

Create a Hugging Face Space powered by a SageMaker Endpoint!

### Deliverables

- A short Loom of the space, and a 1min. walkthrough of the application in full

## Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped a open-source LLM-powered Retrieval Augmented Generation Application with LangChain! 🎉🤖

🔍 Three Key Takeaways:
1️⃣ 
2️⃣ 
3️⃣ 

Let's continue pushing the boundaries of what's possible in the world of AI and question-answering. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#LangChain #QuestionAnswering #RetrievalAugmented #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```

> #### NOTE: PLEASE SHUTDOWN YOUR INSTANCES WHEN YOU HAVE COMPLETED THE ASSIGNMENT TO PREVENT UNESSECARY CHARGES.
