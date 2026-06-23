import truststore
truststore.inject_into_ssl()

import os
import time
import PyPDF2
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

with open("Karthik Resume GenAI.pdf", "rb") as pdf_file:
    reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = "".join(page.extract_text() or "" for page in reader.pages)

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=(
            "You are a helpful assistant. Answer questions about the "
            f"following document:\n\n{pdf_text}"
        )
    ),
)


def ask(chat, question, retries=4):
    for attempt in range(retries):
        try:
            return chat.send_message(question)
        except errors.ServerError:
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                print(f"(Model busy, retrying in {wait}s...)")
                time.sleep(wait)
            else:
                raise


print("Ask questions about the document. Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", "q", ""):
        print("Goodbye!")
        break
    try:
        response = ask(chat, question)
        print(f"\nAI: {response.text}\n")
    except errors.ServerError:
        print("\nAI: Server is still overloaded, please try again.\n")
    except errors.ClientError as e:
        print(f"\nError: {e}\n")