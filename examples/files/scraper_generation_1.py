import requests
from bs4 import BeautifulSoup

def scrape_paper_links(html_content):
    """
    Scrapes all paper links from the given HTML content.

    Args:
        html_content (str): The HTML content to parse.

    Returns:
        list: A list of paper links found in the HTML.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    paper_links = []

    # This selector might need adjustment based on the actual HTML structure
    # Example: Find all 'a' tags with 'href' attribute inside a specific div
    for link in soup.find_all('a', href=True):
        paper_links.append(link['href'])

    return paper_links

def save_links_to_file(links, filename="examples/files/paper_links.txt"):
    """
    Saves the scraped links to a file.

    Args:
        links (list): A list of links to save.
        filename (str): The name of the file to save the links to.
    """
    try:
        with open(filename, 'w') as f:
            for link in links:
                if link.startswith('http'):
                    f.write(link + '\n')
        print(f"Links saved to {filename}")
    except Exception as e:
        print(f"Error saving links to file: {e}")

def main():
    """
    Main function to execute the scraping and saving process.
    """
    html = requests.get('https://scholar.google.com/scholar?hl=pt-BR&as_sdt=0%2C5&q=physics+informed+neural+networks&btnG=&oq=physics+informed+').content
    links = scrape_paper_links(html)
    save_links_to_file(links)