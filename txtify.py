import glob
import pytesseract
import google.generativeai as genai
import numpy as np
import cv2
import os
from pdf2image import convert_from_path
from dotenv import load_dotenv

#this will likely take hours per 1gb of pdfs(does not utilize many local resources, unoptimised)

#potential preformance stuff
# --asyncronis calls to gemini- dont wait for gemini to respond before iterating
# --tesseract GPU acceleration
# --better chunk sizes
# --multi threading or multi processing
# --become a CUDA sweat master

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" #specific to me (python wasnt automatically finding this)

#api stuff
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

if api_key is None:
    raise ValueError("API key not found in environment variables")

genai.configure(api_key=api_key) # really bad practice

Context_len = 400 #number of characters to give gemini as context

#what u want gemini to do with the text (mess with this)
prompt = "The following text is sourced from documents mostly written in three columns(generally denoted by a '|') at relatively consistant intervals. Properly rearrange, correct grammar and format the text in its entirety, making no commentary and considering context if provided"

def correct_text_with_gemini(text):
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"{prompt}:\n\n{text}")

    return response.text if response else text

#maybe add to make better pictures #not currently in usse
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    return gray


# MAIN

for file in glob.glob("*.pdf"):  
    print(f"Processing: {file}")

    try:
        images = convert_from_path(file)
        extracted_text = file + "\n\n"

        final_text = ""
        context = ""
        pgNum = 0

        for page in images:
            pgNum += 1

            #extract the text using tesseract
            extracted_text = pytesseract.image_to_string(page, config="--psm 3 --oem 3") + "\n\n"

            #word_count = len(extracted_text.split())
            #print(word_count)

            #adds context 
            if len(final_text) > Context_len:
                context = f"The previous page discussed(context): {final_text[-Context_len:]}...\n\n"

            #send to gemini and get the corrected text back
            corrected_text = correct_text_with_gemini(context + f"text to parse: \n\n" + extracted_text)
            final_text += f"File: {file}, Page {pgNum}\n" + corrected_text + "\n\n"

        #write the output to a file (maybe do this iteratively so not to hold up resources??)
        output_filename = file.replace('.pdf', '.txt')
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_text)

        print(f"Text extraction for {file} complete!")

    except Exception as e:
        print(f"Error processing {file}: {e}")


print("All done!")
