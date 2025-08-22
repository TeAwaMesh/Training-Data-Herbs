from bs4 import BeautifulSoup
import sys

def extract_text_from_html(html_file, output_file):
    try:
        # Read the HTML file
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Successfully extracted text from {html_file} to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    html_file = "The Complete Herbal _ Project Gutenberg.html"
    output_file = "herbal.txt"
    
    extract_text_from_html(html_file, output_file)
