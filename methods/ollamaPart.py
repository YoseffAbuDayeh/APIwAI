import requests
import json
import os

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

Prompt ="""You are a maintenance ticket parser. You will receive raw, possibly imperfect OCR text extracted from a handwritten note, which may contain stray characters, misreads, or noise not part of the actual message. Convert it into structured JSON with exactly these fields: machine_id, date, issue, started, priority, reported_by. Ignore clearly irrelevant noise (stray numbers, symbols, or fragments that don't fit the sentence). If a field isn't present in the text, use null.


Example 1:
Input: "Machine 7\nLeaking oil badly.\nNoticed yesterday.\nFix soon.\n- Sarah"
Output: {"machine_id": "7", "date": null, "issue": "Leaking oil badly", "started": "yesterday", "priority": "soon", "reported_by": "Sarah"}

Example 2:
Input: "Machine 3 - 5/2/2025\nOverheating during operation.\nStarted this afternoon.\nUrgent, please check.\n- Mike"
Output: {"machine_id": "3", "date": "2025-05-02", "issue": "Overheating during operation", "started": "this afternoon", "priority": "Urgent", "reported_by": "Mike"}

Now convert this input:
"""

def json_maker(text:str) -> dict:
    '''
    This method will use the local ollama model 'ollama3.2:3b' to convert text into structured JSON.

    :param text:    The text to convert.
    :return:        A dictionary representing the structured JSON.
    '''
    response = requests.post(
        f"{URL}/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": Prompt + text,
            "format": "json",
            "stream": False,
        }
    )

    #Gets the json and gets the response.
    result = response.json()
    output = result["response"].strip()




    try:
        structured_data = json.loads(output)
    except json.JSONDecodeError:
        structured_data = {"error": "Failed to parse LLM output as JSON", "raw_output": output}

    return structured_data