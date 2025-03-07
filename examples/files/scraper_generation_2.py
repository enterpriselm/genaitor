
"""
This Python code provides a structured representation of the HTML snippet from a Google Scholar search results page,
focusing on the citation information for a specific paper on "Physics Informed Deep Learning".
It extracts and organizes key details such as the paper's title, authors, publication details,
and links to related resources.

The code doesn't execute any web scraping or HTML parsing directly. Instead, it models the data
that would be obtained from parsing the HTML using a library like BeautifulSoup.
This approach allows for a clear and documented representation of the data structure
without requiring live access to the internet or relying on specific HTML parsing implementations.

Example Usage:

# After parsing the HTML with BeautifulSoup (or similar) and extracting the relevant data:
paper_data = {
    "title": "Physics informed deep learning for function approximation & PDE discovery",
    "authors": "M Raissi, P Perdikaris, GE Karniadakis",
    "journal": "Journal of computational physics, 2019 - Elsevier",
    "abstract": "… approximate solutions to partial differential equations (PDEs). We present two distinct deep learning … \n… that we refer to as physics informed neural networks (PINNs). The first type is physics-…",
    "pdf_link": "https://arxiv.org/pdf/1711.10566",
    "citation_link": "/citations?user=z46i-8EAAAAJ&hl=pt-BR&oe=ASCII&oi=sra",
    "cited_by_link": "/scholar?cites=13665969991236718147&as_sdt=2005&sciodt=0,5&hl=pt-BR&oe=ASCII",
    "related_articles_link": "/scholar?q=related:Vb3_10dIKzYJ:scholar.google.com/&scioq=physics+informed+neural+networks&hl=pt-BR&oe=ASCII&as_sdt=0,5",
    "all_versions_link": "/scholar?cluster=13665969991236718147&hl=pt-BR&oe=ASCII&as_sdt=0,5"
}

# Create a Paper object to represent the data:
paper = Paper(paper_data)

# Access the extracted information:
print(f"Title: {paper.title}")
print(f"Authors: {paper.authors}")
print(f"Journal: {paper.journal}")
print(f"PDF Link: {paper.pdf_link}")
print(f"Cited By Link: {paper.cited_by_link}")
"""


class Paper:
    """
    Represents a research paper entry from Google Scholar search results.
    """

    def __init__(self, data):
        """
        Initializes a Paper object with data extracted from the HTML.

        Args:
            data (dict): A dictionary containing the paper's information.
        """
        self.title = data.get("title", "")
        self.authors = data.get("authors", "")
        self.journal = data.get("journal", "")
        self.abstract = data.get("abstract", "")
        self.pdf_link = data.get("pdf_link", "")
        self.citation_link = data.get("citation_link", "")
        self.cited_by_link = data.get("cited_by_link", "")
        self.related_articles_link = data.get("related_articles_link", "")
        self.all_versions_link = data.get("all_versions_link", "")

    def __str__(self):
        """
        Returns a string representation of the Paper object.
        """
        return f"Title: {self.title}\nAuthors: {self.authors}\nJournal: {self.journal}"


if __name__ == "__main__":
    # Example Usage (simulated data)
    paper_data = {
        "title": "Physics informed deep learning for function approximation & PDE discovery",
        "authors": "M Raissi, P Perdikaris, GE Karniadakis",
        "journal": "Journal of computational physics, 2019 - Elsevier",
        "abstract": "… approximate solutions to partial differential equations (PDEs). We present two distinct deep learning … \n… that we refer to as physics informed neural networks (PINNs). The first type is physics-…",
        "pdf_link": "https://arxiv.org/pdf/1711.10566",
        "citation_link": "/citations?user=z46i-8EAAAAJ&hl=pt-BR&oe=ASCII&oi=sra",
        "cited_by_link": "/scholar?cites=13665969991236718147&as_sdt=2005&sciodt=0,5&hl=pt-BR&oe=ASCII",
        "related_articles_link": "/scholar?q=related:Vb3_10dIKzYJ:scholar.google.com/&scioq=physics+informed+neural+networks&hl=pt-BR&oe=ASCII&as_sdt=0,5",
        "all_versions_link": "/scholar?cluster=13665969991236718147&hl=pt-BR&oe=ASCII&as_sdt=0,5",
    }

    paper = Paper(paper_data)

    print(paper)  # Print the string representation of the paper
    print(f"PDF Link: {paper.pdf_link}")  # Access and print the PDF link
    print(f"Cited By Link: {paper.cited_by_link}")  # Access and print the "Cited By" link
