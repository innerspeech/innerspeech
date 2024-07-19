from openai import OpenAI
import openai
import os
import editdistance

def rescore_with_perplexity(nbests):
        
    # Get API key from environment variable
    YOUR_API_KEY = os.getenv("PERPLEXITY_API_KEY")

    client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

    messages = [
        {
            "role": "system",
            "content": (
                """Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions. Choose the transcription that is most accurate, ensuring it is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond with the chosen transcription with all lower case, with no punctuncations, and without any explanatory text."""
            ),
        },
        {
            "role": "user",
            "content": (
                """Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions. Choose the transcription that is most accurate, ensuring it is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond with the chosen transcription with all lower case, with no punctuncations, and without any explanatory text."""
                "\n".join(nbests)
            ),
        },
    ]
        
    # chat completion without streaming
    response = client.chat.completions.create(
        model="mixtral-8x7b-instruct",
        messages=messages,
    )

    full_results = response.choices[0].message.content
    
    if '\n' in full_results:
        results = full_results.split("\n")
        return results[0]
    else:
        return full_results
    
def rescore_with_openai(nbests):
        
    # Get API key from environment variable
    YOUR_API_KEY = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=YOUR_API_KEY)

    messages = [
        {
            "role": "system",
            "content": (
                """Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions. Choose the transcription that is most accurate, ensuring it is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond with the chosen transcription with all lower case, with no punctuncations, and without any explanatory text."""
            ),
        },
        {
            "role": "user",
            "content": (
                "\n".join(nbests)
            ),
        },
    ]
        
    # chat completion without streaming
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    full_results = response.choices[0].message.content
    
    if '\n' in full_results:
        results = full_results.split("\n")
        return results[0]
    else:
        return full_results

def rescore_with_openai_instruct(nbests):
        
    # Get API key from environment variable
    YOUR_API_KEY = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=YOUR_API_KEY)

    BASE_PROMPT = """Q: Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions. Choose the transcription that is most accurate, ensuring it is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond with the chosen transcription with all lower case, with no punctuncations, and without any explanatory text."""
    prompt = BASE_PROMPT + "\n".join(nbests)

    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=1,
        max_tokens=500,
        n=1,
        stop=None,
        presence_penalty=0,
        frequency_penalty=0.1,
        )   


    full_results = response.choices[0].text
    
    full_results = full_results.strip()

    return full_results

    # if '\n' in full_results:
    #     results = full_results.split("\n")
    #     return results[0]
    # else:
    #     return full_results


import glob
import time

decodedSeq_files = glob.glob('./results/decodedSeq*0.1*.txt')
print(decodedSeq_files)

# create a list of list of the decoded sequences
full_decodedSeqs = []
for file_name in decodedSeq_files:
    with open(file_name, 'r') as f:
        preds = f.readlines()
        preds = [x.strip() for x in preds]
        full_decodedSeqs.append(preds)

for idx in range(len(full_decodedSeqs[0])):
    print(idx, "th sentence")
    decodedSeqs_for_rescore = []
    for decodedSeqs in full_decodedSeqs:
        decodedSeqs_for_rescore.append(decodedSeqs[idx])
    
    # rescored_decodedSeqs = rescore_with_perplexity(decodedSeqs_for_rescore)
    rescored_decodedSeqs = rescore_with_openai(decodedSeqs_for_rescore)
    time.sleep(0.25)
    
    # print(decodedSeqs_for_rescore)
    # print(rescored_decodedSeqs)
    # check the editdistance of the rescored version with the original, if too different, then keep the original
    if editdistance.eval(decodedSeqs_for_rescore[2], rescored_decodedSeqs) > 10:
        rescored_decodedSeqs = decodedSeqs_for_rescore[2]
    print(rescored_decodedSeqs)


# # rescore the decoded sequences with perplexity
# rescored_decodedSeqs = []
# for decodedSeqs in full_decodedSeqs:
#     rescored_decodedSeqs.append(rescore_with_perplexity(decodedSeqs))

# # result = rescore_with_perplexity([["hello wod", "hello worl", "hello world"]])

# # print(result)