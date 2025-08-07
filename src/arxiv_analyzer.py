import os
import feedparser
import openai
from scholarly import scholarly
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_arxiv_papers(category='cond-mat.str-el', max_results=10):
    """Fetches papers from the arXiv API."""
    url = f'http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}'
    feed = feedparser.parse(url)
    return feed.entries

def get_author_info(author_name):
    """Gets author information from Google Scholar."""
    try:
        search_query = scholarly.search_author(author_name)
        author = next(search_query)
        return {
            'h_index': author.get('hindex', 'N/A'),
            'recent_publications': [pub['bib']['title'] for pub in author.get('publications', [])[:5]]
        }
    except StopIteration:
        return {
            'h_index': 'N/A',
            'recent_publications': []
        }

def get_topic_prevalence(topic):
    """Gets the prevalence of a topic using web search."""
    # This is a placeholder. A real implementation would use a search engine API.
    # For this example, we'll simulate a search.
    print(f"Searching for topic: {topic}")
    return "High" # Simulated prevalence

def analyze_paper_impact(paper):
    """Analyzes the potential impact of a paper."""
    title = paper.title
    authors = [author.name for author in paper.authors]
    summary = paper.summary

    author_info = [get_author_info(author) for author in authors]

    # A simple way to get topics is to use the summary.
    # A more advanced approach would be to use NLP to extract keywords.
    topics = summary.split('.')[0] # Use the first sentence as a proxy for topics
    topic_prevalence = get_topic_prevalence(topics)

    prompt = f"""
    Analyze the potential impact of the following research paper.
    
    Title: {title}
    Authors: {', '.join(authors)}
    Author Information: {author_info}
    Topics: {topics}
    Topic Prevalence: {topic_prevalence}
    Summary: {summary}

    Based on the authors' h-index, their recent publications, and the prevalence of the research topic,
    provide a score from 1 to 10 for the potential impact of this paper, where 1 is low impact and 10 is high impact.
    Also, provide a brief summary of the paper and a justification for the score.

    Format your response as:
    Score: [score]
    Summary: [summary]
    Justification: [justification]
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert research analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error analyzing paper: {e}"

def main(m=5):
    """Main function to get and analyze papers."""
    print("Fetching papers from arXiv...")
    papers = get_arxiv_papers()
    
    if not papers:
        print("No papers found.")
        return

    analyzed_papers = []
    for paper in papers:
        print(f"Analyzing: {paper.title}")
        analysis = analyze_paper_impact(paper)
        analyzed_papers.append({'paper': paper, 'analysis': analysis})

    # Sort papers by score (assuming the score is at the beginning of the analysis string)
    def get_score(item):
        try:
            score_line = item['analysis'].split('\n')[0]
            score = int(score_line.split(': ')[1])
            return score
        except (IndexError, ValueError):
            return 0

    analyzed_papers.sort(key=get_score, reverse=True)

    print(f"\n--- Top {m} Potentially Impactful Papers ---")
    for item in analyzed_papers[:m]:
        print(f"\nTitle: {item['paper'].title}")
        print(f"Authors: {', '.join(author.name for author in item['paper'].authors)}")
        print(f"Published: {item['paper'].published}")
        print(f"Link: {item['paper'].link}")
        print("\n--- Analysis ---")
        print(item['analysis'])
        print("------------------")

if __name__ == "__main__":
    main()
