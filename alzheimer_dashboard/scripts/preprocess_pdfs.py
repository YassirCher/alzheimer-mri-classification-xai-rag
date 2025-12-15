import pdfplumber
import re
import os
from pathlib import Path
from typing import List

class PDFPreprocessor:
    """Clean and preprocess PDFs for optimal RAG performance"""
    
    def __init__(self, pdf_dir='data/pdfs', output_dir='data/articles'):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common pattern: "Page 1 of 10")
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Remove headers/footers (PMC articles common patterns)
        text = re.sub(r'PMC\d+', '', text)
        text = re.sub(r'PMID:\s*\d+', '', text)
        
        # Remove URLs (keep DOIs)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Fix common PDF extraction issues
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('–', '-')
        text = text.replace('—', '-')
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def extract_metadata(self, pdf_path: Path) -> dict:
        """Extract metadata from PDF"""
        metadata = {
            'filename': pdf_path.stem,
            'source': pdf_path.name
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if pdf.metadata:
                    metadata.update({
                        'title': pdf.metadata.get('Title', ''),
                        'author': pdf.metadata.get('Author', ''),
                        'subject': pdf.metadata.get('Subject', '')
                    })
        except Exception as e:
            print(f"Error extracting metadata: {e}")
        
        return metadata
    
    def extract_text_from_pdf(self, pdf_path: Path) -> tuple[str, dict]:
        """Extract and clean text from PDF"""
        print(f"Processing: {pdf_path.name}")
        
        full_text = ""
        metadata = self.extract_metadata(pdf_path)
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n\n"
            
            # Clean the extracted text
            cleaned_text = self.clean_text(full_text)
            
            print(f"  ✓ Extracted {len(cleaned_text)} characters")
            return cleaned_text, metadata
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return "", metadata
    
    def add_document_structure(self, text: str, metadata: dict) -> str:
        """Add structure to document for better chunking"""
        structured_text = f"""
TITLE: {metadata.get('title', metadata.get('filename', 'Unknown'))}
SOURCE: {metadata.get('source', 'Unknown')}
AUTHOR: {metadata.get('author', 'Unknown')}

===== DOCUMENT CONTENT =====

{text}
"""
        return structured_text
    
    def process_all_pdfs(self):
        """Process all PDFs in directory"""
        pdf_files = list(Path(self.pdf_dir).glob('*.pdf'))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files\n")
        
        for pdf_file in pdf_files:
            # Extract text
            text, metadata = self.extract_text_from_pdf(pdf_file)
            
            if not text:
                continue
            
            # Add structure
            structured_text = self.add_document_structure(text, metadata)
            
            # Save to output
            output_file = Path(self.output_dir) / f"{pdf_file.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(structured_text)
            
            print(f"  ✓ Saved to: {output_file.name}\n")
        
        print(f"✅ Processing complete! {len(pdf_files)} files processed.")

if __name__ == '__main__':
    # Create directories
    os.makedirs('data/pdfs', exist_ok=True)
    os.makedirs('data/articles', exist_ok=True)
    
    # Process PDFs
    preprocessor = PDFPreprocessor()
    preprocessor.process_all_pdfs()