import os
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(api_key="")  # Replace with your key

def get_sum_from_openai(a, b):
    prompt = f"Add the two numbers and return only the result: {a} + {b}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    result = response.choices[0].message.content.strip()
    return result

if __name__ == "__main__":
    num1 = int(input("Enter first number: "))
    num2 = int(input("Enter second number: "))

    result = get_sum_from_openai(num1, num2)
    print(f"Result from GPT-4o: {result}")
