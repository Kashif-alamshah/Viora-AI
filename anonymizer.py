import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import fitz

analyzer   = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def read_pdf(path: str) -> str:
    doc  = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def anonymize_text(text: str) -> str:
    # If message contains any image path, skip anonymization entirely
    if re.search(r'[A-Za-z]:\\.*?\.(jpg|jpeg|png|bmp|webp)', text, re.IGNORECASE):
        return text

    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text

def process_input(user_input: str) -> str:
    stripped = user_input.strip().strip('"').strip("'")
    if stripped.lower().endswith(".pdf"):
        print("PDF detected, extracting text...")
        text = read_pdf(stripped)
        print("Anonymizing PHI...")
        return anonymize_text(text)
    
    return anonymize_text(user_input)