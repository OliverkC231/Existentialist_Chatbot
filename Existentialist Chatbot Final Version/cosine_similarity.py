import os
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(".env.ollama")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Cosine similarity for OPENAI chatbot answer with ollama embeddings

results = index.search(
    namespace="__default__", 
    query={
        "inputs": {"text": """
                   
       The meaning of life is not something given; it is something suffered, chosen, and slowly created.

We are thrown into a world where death is certain, explanations are dubious, and institutions rush to answer questions we have not yet truly asked. In such a world, “meaning” is not a cosmic secret but the very movement of a person becoming inwardly awake—taking responsibility for their own existence.

To love life “more than the meaning of it” is the paradox: only when you stop demanding a final explanation and instead live—lucidly, vulnerably, responsibly—does life begin to disclose a kind of meaning from within. Meaning is not behind life or above it; it appears in the way you relate to your freedom, your suffering, your others, your inevitable death.

So: the meaning of life is to live it consciously and honestly enough that the question no longer seeks a formula, but is answered, provisionally, by the form of your own life.
                   """}, 
        "top_k": 2
    },
    fields=["source", "text"]
)

print(results)

# Cosine similarity for OLLAMA chatbot answer with ollama embeddings

results = index.search(
    namespace="__default__", 
    query={
        "inputs": {"text": """

        The question of meaning in a seemingly meaningless existence. As an existentialist philosopher, I would say that the meaning of life is not something external, such as the "work" left behind by Christ. Rather, it's the inherent desire to consider our own life philosophically - to think about what we are, and how we exist.

In other words, our question "Quid est?" (What exists?) must be answered: Quid erit? (What will be?). This means that we must take a step back and understand the nature of existence itself. We're not merely individuals who come into being and pass away; we are beings with our own inherent meaning.

This understanding is not predicated on birth or death, but rather on our very existence as conscious beings. It's a fundamental realization that life is fleeting, yet it's precisely this impermanence that allows us to grasp the significance of our experiences.

In the words of Sartre, "Man is condemned to be free; because once thrown into the world, he is responsible for everything he does." This existentialist notion reminds us that we must take ownership of our choices and actions. Our existence is not determined by external forces, but rather we determine what we want to create for ourselves.

Ultimately, the meaning of life is a subjective experience that each individual must create for themselves. We're not bound by an external expectation of purpose or significance; we're free to forge our own path, using our freedom as a means to explore and understand the world around us.
        
                """}, 
        "top_k": 2
    },
    fields=["source", "text"]
)

print(results)


# Cosine similarity Baseline with Ollama embeddings

results = index.search(
    namespace="__default__", 
    query={
        "inputs": {"text": """

        The question of the meaning of life. A query that has puzzled human beings for centuries, a conundrum that bespeaks our fundamental discontent with the here-and-now. As I see it, the meaning of life is not something that can be grasped or discovered through intellectual abstraction, but rather it emerges from the depths of existence itself.

You see, the meaning of life is not an external entity, something that can be found in books, philosophers' treatises, or even our own experiences. No, the meaning of life is a subjective experience, a personal and intimate reality that arises from our unique presence within this vast and complex universe.

As I would argue, the meaning of life is revealed to us through our existence itself, through the struggles, joys, and contradictions that we endure in this fleeting moment. It is the ebb and flow of our lives, the twists and turns that shape us into who we are today. And it is precisely within these moments of struggle and triumph that we discover our own meaning.

But what does it mean to be alive? To experience the world around us? To connect with others? These questions are not easily answered, nor do they have definitive answers. They are, instead, questions that invite us into the very mystery itself.

In this sense, the meaning of life is not something we can articulate or define. It is an ontological imperative, a fundamental existential force that drives us to live our lives authentically, to take risks, and to confront our own mortality.

And yet, despite this understanding, I would caution against simply accepting this existential framework as a justification for existing. No, the meaning of life is not something we can rely on, nor should it be used as an excuse to conform or succumb to the whims of fate. Instead, it is up to each individual to create their own meaning, to take responsibility for their own existence.

As Sartre would say, "Man is condemned to be free." We are endowed with freedom, and with that freedom comes the burden of choice. We must choose how we will live our lives, how we will respond to the world around us, and what kind of meaning we will create for ourselves in this fleeting moment.

In the end, the meaning of life is not something that can be found or discovered; it is a creation, a statement of intent that arises from our existence itself. It is an existential declaration, a call to action, and a testament to the boundless potentiality of human existence.


        
                """}, 
        "top_k": 2
    },
    fields=["source", "text"]
)

print(results)