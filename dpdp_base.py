## gemini api key -  AIzaSyA4csVVtoDNmqqO0ZMFGtC_Qf5G0p--_So
# https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyA4csVVtoDNmqqO0ZMFGtC_Qf5G0p--_So
# project number: 1060832116545
#Code to retrieve API key from Gemini and convert pdf to text and summarize the contents in it

import os
import PyPDF2
import google.generativeai as genai

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA4csVVtoDNmqqO0ZMFGtC_Qf5G0p--_So"  # Replace with your API key

# Configure the Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Path to your PDF file
pdf_path = r"C:\Users\kishore l\gdg\dpdp.pdf"
text_file_path = r"C:\Users\kishore l\gdg\extracted_text.txt"  # Path to save the text file

def extract_text_from_pdf(pdf_path, max_pages=2):
    """Extract text from the first few pages of a PDF."""
    text = ""
    num_extracted_pages = 0
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        # Extract text from the first max_pages pages
        for i, page in enumerate(pdf_reader.pages):
            if i >= max_pages:
                break
            text += page.extract_text()
            num_extracted_pages += 1
    return text, num_extracted_pages, total_pages

# Extract text from the first two pages of the PDF
extracted_text, num_extracted_pages, total_pages = extract_text_from_pdf(pdf_path, max_pages=2)

# Save the extracted text to a text file
with open(text_file_path, "w", encoding="utf-8") as text_file:
    text_file.write(extracted_text)

print(f"Extracted text saved to: {text_file_path}")
print(f"Number of pages extracted: {num_extracted_pages}/{total_pages}")

# (Optional) Send the extracted text to the Gemini API for further analysis
generation_config = {
    "temperature": 0.7,
    "max_output_tokens": 512,  # Limit the response size
    "top_p": 0.9,
    "top_k": 40,
}

try:
    # Create a generative model instance
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",  # You can adjust this to another valid model
        generation_config=generation_config,
    )

    # Send the extracted text to Gemini for further analysis or summarization
    response = model.start_chat(history=[]).send_message(
        f"Analyze the following text:\n\n{extracted_text}"
    )

    print("\nResponse from Gemini API:\n", response.text)

except Exception as e:
    print("Error:", str(e))
