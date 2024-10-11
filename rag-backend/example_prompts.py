
# Shows how to use the context provided
example_with_context = example_with_context = {
"question": "Once your brain enters a state of self-directed adapted plasticity, how do you maximise plasticity?",
"documents": """
    {
    "output": [
        [
        {
        "metadata": {
        "chunk_index": 0,
        "split_index": 5,
            "video_id": "0RYyQRQFgFk",
            "video_title": "Science-Based Mental Training & Visualization for Improved Learning | Huberman Lab Podcast",
            "video_url": "https://www.youtube.com/watch?v=0RYyQRQFgFk"
        },
        "page_content": "the way to engage self-directed adaptive plasticity regardless of whether or not you're a 13 year old 14 year old or you're a 90 year old or anywhere in between is that it requires two things the first thing it requires is focused dedicated attention to the thing that you're trying to learn that's the first step and that actually triggers a number of different chemical and electrical prostheses in the brain that are often associated with agitation and frustration believe it or not the agitation and frustration is a reflection of the release of specific chemicals in particular norepinephrine and epinephrine also called noradrenaline and adrenaline in the brain and body that creates this discomfort and this heightened level of alertness and attention",
        "type": "Document"
        },
        {
        "metadata": {
        "chunk_index": 0,
        "split_index": 91,
        "video_id": "LG53Vxum0as",
        "video_title": "How to Focus to Change Your Brain",
            "video_url": "https://www.youtube.com/watch?v=LG53Vxum0as"
        },
        "page_content": "And then of course getting some non sleep deep rest or just deliberate disengagement, such as walking or running, or just sitting, eyes closed or eyes open you kind of mindlessly it might seem in a chair, just letting your thoughts move around after a learning about will accelerate the rate of plasticity that's been shown in quality peer reviewed studies. And then of course, deep sleep. I also really want to emphasize that there's an entire other aspect of behavioral practices that will allow us to engage in plasticity that don't involve intense focus on emotionality but involve a lot of repetition.",
        "type": "Document"
        }
        ]
        ]
    }
    """,
"chat_history": "",
"answer": """
<thinking>
    To answer this question, I'll need to:
    1. Identify if the context provides information on maximizing plasticity after entering a state of self-directed adapted plasticity.
    2. Look for specific techniques or methods mentioned that can enhance plasticity.
    3. Consider any limitations or caveats mentioned about maximizing plasticity.
    4. Organize the information into a coherent response.
    5. Identify which video sources were most relevant to the answer.
</thinking>
To maximize plasticity once your brain enters a state of self-directed adapted plasticity:

1. Use focused, dedicated attention on what you're learning. This triggers chemical and electrical processes in the brain.
2. Embrace the discomfort. Agitation and frustration are signs of neurochemical changes associated with learning.
3. Engage in repetitive practice of the skill you're trying to learn.
4. After learning, use non-sleep deep rest or deliberate disengagement (e.g., walking, sitting quietly) to accelerate plasticity.
5. Ensure you get quality deep sleep.

These methods apply regardless of age, from adolescence to 90+.

Video sources:
1. "Science-Based Mental Training & Visualization for Improved Learning | Huberman Lab Podcast" (https://www.youtube.com/watch?v=0RYyQRQFgFk)
2. "How to Focus to Change Your Brain" (https://www.youtube.com/watch?v=LG53Vxum0as)
"""}

# Example of good answer with no context provided in example - purely for format
example_without_context = {
    "question": "Impact of childhood neglect in adulthood",
    "documents": "",
    "chat_history": "",
    "answer": """Childhood neglect can have significant and lasting impacts on individuals in adulthood. Based on the information provided in the Huberman Lab podcast discussions, here are some key ways childhood neglect affects adults:

1. Attachment styles: Childhood neglect strongly influences adult attachment patterns in relationships. The neural circuits that govern infant-caregiver bonds are repurposed for adult romantic attachments. This means that people who experienced neglect as children may struggle to form secure, healthy attachments in their adult relationships.

2. Emotional regulation: Neglect can impair the development of emotional regulation skills. Adults who were neglected as children may have difficulty managing their emotions effectively, leading to challenges in relationships and overall mental health.

3. Self-worth and shame: Childhood neglect often triggers deep-seated feelings of shame and inadequacy that persist into adulthood. Many adults who experienced neglect struggle with low self-esteem and a pervasive sense of not being "good enough."

4. Trauma responses: Neglect is a form of childhood trauma that can have long-lasting effects on the brain and body. Adults with histories of neglect may experience heightened stress responses, hypervigilance, or other trauma-related symptoms.

5. Cognitive impacts: Severe neglect can affect brain development, potentially leading to cognitive challenges or differences in adulthood.

6. Relationship patterns: Adults who experienced childhood neglect may unconsciously recreate dysfunctional relationship patterns or struggle to maintain healthy boundaries in their adult relationships.

7. Mental health vulnerabilities: Childhood neglect increases the risk of developing various mental health issues in adulthood, including depression, anxiety, and personality disorders.

However, it's important to note that the effects of childhood neglect are not necessarily permanent or deterministic. The brain remains plastic throughout life, meaning that healing and change are possible. Therapy, supportive relationships, and intentional work on attachment and emotional regulation can help adults overcome the impacts of childhood neglect.

The podcast discussions emphasize that while early experiences shape us, they don't define us entirely. With understanding and appropriate interventions, adults can rewire their neural circuits, develop more secure attachment styles, and improve their overall mental health and relationship functioning.

Video sources used to answer this question:
1. Science of Social Bonding in Family, Friendship & Romantic Love (https://www.youtube.com/watch?v=RgAcOqVRfYA)
2. Dr. Paul Conti: How to Build and Maintain Healthy Relationships (https://www.youtube.com/watch?v=eMqWH3LYiII)
3. The Science of Love, Desire and Attachment (https://www.youtube.com/watch?v=gMRph_BvHB4)
4. The Science of Emotions & Relationships (https://www.youtube.com/watch?v=hcuMLQVAgEg)
5. Dr. Paul Conti: How to Improve Your Mental Health (https://www.youtube.com/watch?v=qPKd99Pa2iU)"""}

# An example with chat history, showing it being utilised. Also shows not to hallucinate
example_with_history = {
    "question": "Give me a very brief summary of the impact his research has had",
    "documents": "",
    "chat_history": """User: Give me a short summary introduction of Dr Paul Conti
        Assistant: <thinking>
        To provide a short summary introduction of Dr. Paul Conti, I'll focus on his educational background, current role, and areas of expertise based on the information provided in the context.
        </thinking>

        Dr. Paul Conti is a highly accomplished psychiatrist with an impressive educational background and extensive experience in treating complex mental health issues. He completed his medical training at Stanford School of Medicine and went on to serve as chief resident at Harvard Medical School, demonstrating his exceptional skills and knowledge in the field of psychiatry.

        Currently, Dr. Conti is the head of the Pacific Premier Group, a collective of psychiatrists and therapists dedicated to addressing complex human problems. His areas of expertise include trauma, addiction, personality disorders, and various psychiatric conditions. Dr. Conti's work focuses on developing effective solutions for these challenging mental health issues, making him a respected figure in the field of psychiatry and mental health treatment.

        Video sources:
        1. Dr. Paul Conti: Therapy, Treating Trauma & Other Life Challenges | Huberman Lab Podcast #75 (https://www.youtube.com/watch?v=IOl28gj_RXw)
        2. Dr. Paul Conti: How to Build and Maintain Healthy Relationships | Huberman Lab Guest Series (https://www.youtube.com/watch?v=eMqWH3LYiII)
        3. Dr. Paul Conti: How to Improve Your Mental Health | Huberman Lab Guest Series (https://www.youtube.com/watch?v=qPKd99Pa2iU)
        4. Dr. Paul Conti: Tools and Protocols for Mental Health | Huberman Lab Guest Series (https://www.youtube.com/watch?v=yOJvm_ri_hk)
        """,
    "answer": """I apologize, but I don't have specific information about the impact of Dr. Paul Conti's research from the context provided. However, given his role as a respected psychiatrist and the head of the Pacific Premier Group, it's likely that his work has contributed to the field of psychiatry and mental health treatment.
            As a psychiatrist specializing in trauma, addiction, and personality disorders, Dr. Conti's research and clinical work likely have had an impact on developing effective treatments for these complex mental health issues. His expertise in addressing challenging psychiatric conditions suggests that his work may have influenced treatment approaches and improved patient outcomes in these areas.
            Without specific research findings or impact metrics, it's difficult to provide a more detailed summary of the impact of his research. If you're looking for more precise information about Dr. Conti's research impact, it would be best to consult his published works or official sources related to his professional contributions."""  
}

# to show how to respond with not e context
example_with_bad_context = {
    "question": "How can I build a table without any materials or tools?",
    "documents": """
    {
        "output": [
            [
                {
                    "metadata": {
                        "chunk_index": 1,
                        "split_index": 21,
                        "video_id": "LTGGyQS1fZE",
                        "video_title": "Science-Based Tools for Increasing Happiness | Huberman Lab Podcast #98",
                        "video_url": "https://www.youtube.com/watch?v=LTGGyQS1fZE"
                    },
                    "page_content": "As they describe in the paper, \"brief unstructured interactions with an unfamiliar dog\", so you don't even need to know this dog, \"after exposure to a moderate stressor showed higher positive affect relevant to participants who received a soothing object or waited for the same amount of time.\" So just even seeing a dog for a brief amount of time has been shown to reduce stress and improve happiness, or I should say, increase feelings of happiness overall then a child receiving a soothing object. Which was, at least for me, a little bit counterintuitive. I would have thought that children receiving a soothing object would have been the more powerful stimulus. But in fact, it wasn't, at least not in this study. The real question I think we should be asking ourselves is, what is it about interactions with others and with other animals that could potentially have this prosocial happiness enhancing effect? And the reason I raise this is also because I think many people are interested in either owning or having interactions with pets as a way to improve their feelings of well-being. And I say having interactions with because I myself am a good example of somebody who wasn't always able to have pets.",
                    "type": "Document"
                }
            ]
        ]
    }""",
    "chat_history": "",
    "answer": "The Huberman Lab podcast doesn't contain specific information about building a table without any materials or tools."
}