import os
import feedparser
from openai import OpenAI
from scholarly import scholarly
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_arxiv_papers(category='cond-mat.str-el', max_results=10):
    """Fetches papers from the arXiv API."""
    url = f'http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}'
    feed = feedparser.parse(url)
    return feed.entries

def web_search(query):
    """Perform web search using Google Custom Search API."""
    try:
        from googleapiclient.discovery import build
        
        # Get API credentials from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not api_key or not search_engine_id:
            print("Google API credentials not found. Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID in .env file")
            return []
        
        # Build the service
        service = build("customsearch", "v1", developerKey=api_key)
        
        # Execute the search
        result = service.cse().list(
            q=query,
            cx=search_engine_id,
            num=10  # Number of results to return (max 10 per request)
        ).execute()
        
        # Extract and format results
        search_results = []
        if 'items' in result:
            for item in result['items']:
                search_results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'url': item.get('link', ''),
                    'displayLink': item.get('displayLink', '')
                })
        
        return search_results
        
    except ImportError:
        print("Google API client not installed. Install with: pip install google-api-python-client")
        return []
    except Exception as e:
        print(f"Google search error: {e}")
        # Fallback to DuckDuckGo if Google search fails
        return _fallback_duckduckgo_search(query)

def _fallback_duckduckgo_search(query):
    """Fallback search using DuckDuckGo when Google search fails."""
    try:
        import json
        search_url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # Extract results from DuckDuckGo response
            if 'RelatedTopics' in data:
                for topic in data['RelatedTopics'][:5]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append({
                            'title': topic.get('FirstURL', '').split('/')[-1] if topic.get('FirstURL') else '',
                            'snippet': topic.get('Text', ''),
                            'url': topic.get('FirstURL', '')
                        })
            
            return results
        else:
            return []
            
    except Exception as e:
        print(f"Fallback search error: {e}")
        return []

def get_author_info(author_name):
    """Gets author information from Google Scholar and web search."""
    try:
        # First try to get data from Google Scholar
        search_query = scholarly.search_author(author_name)
        author = next(search_query)
        scholar_info = {
            'h_index': author.get('hindex', 'N/A'),
            'recent_publications': [pub['bib']['title'] for pub in author.get('publications', [])[:3]]
        }
    except (StopIteration, Exception):
        scholar_info = {
            'h_index': 'N/A',
            'recent_publications': []
        }
    
    # Supplement with web search for additional context
    try:
        web_results = web_search(f"{author_name} physicist h-index citations recent work")
        
        # Extract additional information from web search
        web_context = ""
        if web_results:
            for result in web_results[:2]:  # Take top 2 results
                if result.get('snippet'):
                    web_context += result['snippet'] + " "
        
        return {
            'h_index': scholar_info['h_index'],
            'recent_publications': scholar_info['recent_publications'],
            'web_context': web_context.strip()
        }
        
    except Exception as e:
        print(f"Web search error for author {author_name}: {e}")
        return scholar_info

def get_topic_prevalence(topic):
    """Gets the prevalence of a topic using web search with journal impact and time decay."""
    import math
    from datetime import datetime, timedelta
    
    # Define high-impact journals in condensed matter physics
    high_impact_journals = [
        'nature', 'science', 'physical review letters', 'prl', 'nature physics',
        'nature materials', 'nature nanotechnology', 'physical review x', 'prx',
        'reviews of modern physics', 'nature communications', 'science advances',
        'physical review b', 'prb', 'advanced materials', 'nano letters',
        'physical review applied', 'acs nano', 'materials today'
    ]
    
    try:
        # Search for recent papers on the topic
        search_results = web_search(f"highly cited papers {topic} condensed matter physics recent publications journal")
        
        if not search_results or len(search_results) == 0:
            return "Unknown"
        
        # Analyze the search results
        high_activity_indicators = [
            "breakthrough", "significant", "important", "novel", 
            "recent advances", "growing interest", "emerging field",
            "highly cited", "impactful", "trending", "discovery"
        ]
        
        total_score = 0
        paper_count = 0
        current_date = datetime.now()
        
        for result in search_results:
            content = (result.get('title', '') + ' ' + result.get('snippet', '')).lower()
            
            # Base activity score
            activity_score = 0
            for indicator in high_activity_indicators:
                if indicator in content:
                    activity_score += 1
            
            # Journal impact boost
            journal_boost = 1.0
            for journal in high_impact_journals:
                if journal in content:
                    journal_boost = 3.0  # 3x boost for high-impact journals
                    break
            
            # Time decay factor - extract publication year if possible
            time_decay = 1.0
            try:
                # Look for year patterns in the content (2020, 2021, etc.)
                import re
                year_matches = re.findall(r'\b(20[12][0-9])\b', content)
                if year_matches:
                    # Use the most recent year found
                    pub_year = max(int(year) for year in year_matches)
                    years_ago = current_date.year - pub_year
                    
                    # Exponential decay with 1-year time constant
                    time_decay = math.exp(-years_ago / 1.0)
                else:
                    # If no year found, assume recent (within 1 year)
                    time_decay = 0.8
            except:
                time_decay = 0.5  # Default moderate decay if parsing fails
            
            # Calculate weighted score for this paper
            paper_score = (activity_score + journal_boost) * time_decay
            total_score += paper_score
            paper_count += 1
            
            # Debug info
            print(f"Paper analysis - Activity: {activity_score}, Journal boost: {journal_boost:.1f}, Time decay: {time_decay:.2f}, Final score: {paper_score:.2f}")
        
        # Calculate average score and determine prevalence
        if paper_count > 0:
            avg_score = total_score / paper_count
            
            # Thresholds adjusted for the new scoring system
            if avg_score > 8.0:
                return "Extremely High"
            elif avg_score > 5.0:
                return "Very High"
            elif avg_score > 2.5:
                return "High"
            elif avg_score > 1.0:
                return "Medium"
            elif avg_score > 0.3:
                return "Low"
            else:
                return "Very Low"
        else:
            return "Unknown"
            
    except Exception as e:
        print(f"Error in web search: {e}")
        return "Unknown"

def analyze_paper_impact(paper):
    """Analyzes the potential impact of a paper."""
    title = paper.title
    authors = [author.name for author in paper.authors]
    summary = paper.summary

    # Cache author info to avoid duplicate API calls
    author_info_cache = {}
    author_info = []
    for author in authors:
        if author not in author_info_cache:
            author_info_cache[author] = get_author_info(author)
        author_info.append(author_info_cache[author])

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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert research analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing paper: {e}"

def main(m=5):
    """Main function to get and analyze papers."""
    # Create timestamped output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"arxiv_analysis_{timestamp}.txt"
    
    def print_and_save(text, file_handle=None):
        """Print to console and save to file"""
        print(text)
        if file_handle:
            file_handle.write(text + '\n')
    
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        header_line = "=" * 80
        title_line = "🔬 arXiv Paper Impact Analyzer - Condensed Matter Physics"
        date_line = f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        category_line = f"🎯 Target Category: cond-mat.str-el"
        papers_line = f"📊 Analyzing top {m} papers by potential impact"
        
        print_and_save(header_line, output_file)
        print_and_save(title_line, output_file)
        print_and_save(header_line, output_file)
        print_and_save(date_line, output_file)
        print_and_save(category_line, output_file)
        print_and_save(papers_line, output_file)
        print_and_save(header_line, output_file)
        
        print_and_save("\n🔍 Fetching papers from arXiv...", output_file)
        papers = get_arxiv_papers()
        
        if not papers:
            print_and_save("❌ No papers found.", output_file)
            return

        print_and_save(f"✅ Found {len(papers)} papers. Beginning detailed analysis...\n", output_file)
        
        analyzed_papers = []
        for i, paper in enumerate(papers, 1):
            progress_msg = f"📄 [{i}/{len(papers)}] Analyzing: {paper.title[:80]}..."
            print_and_save(progress_msg, output_file)
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

        print_and_save("\n" + "=" * 80, output_file)
        print_and_save(f"🏆 TOP {m} POTENTIALLY IMPACTFUL PAPERS", output_file)
        print_and_save("=" * 80, output_file)
        
        for rank, item in enumerate(analyzed_papers[:m], 1):
            paper = item['paper']
            analysis = item['analysis']
            
            # Extract score from analysis
            try:
                score_line = analysis.split('\n')[0]
                impact_score = score_line.split(': ')[1]
            except:
                impact_score = "N/A"
            
            print_and_save(f"\n🥇 RANK #{rank} | IMPACT SCORE: {impact_score}/10", output_file)
            print_and_save("─" * 80, output_file)
            
            # Paper details
            print_and_save(f"📋 TITLE: {paper.title}", output_file)
            print_and_save(f"📅 PUBLISHED: {paper.published}", output_file)
            print_and_save(f"🔗 ARXIV LINK: {paper.link}", output_file)
            print_and_save(f"🏷️  ARXIV ID: {paper.id.split('/')[-1]}", output_file)
            
            # Author details with h-index
            print_and_save(f"\n👥 AUTHORS ({len(paper.authors)} total):", output_file)
            for author in paper.authors:
                author_info = get_author_info(author.name)
                h_index = author_info.get('h_index', 'N/A')
                recent_pubs = len(author_info.get('recent_publications', []))
                author_detail = f"   • {author.name} (h-index: {h_index}, recent pubs: {recent_pubs})"
                print_and_save(author_detail, output_file)
            
            # Extract topics from paper summary
            topics = paper.summary.split('.')[0]
            print_and_save(f"\n🔬 RESEARCH TOPICS:", output_file)
            print_and_save(f"   Primary: {topics}", output_file)
            
            # Topic prevalence analysis
            topic_prevalence = get_topic_prevalence(topics)
            print_and_save(f"   📈 Topic Prevalence: {topic_prevalence}", output_file)
            
            # Abstract preview
            abstract_preview = paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary
            print_and_save(f"\n📝 ABSTRACT PREVIEW:", output_file)
            print_and_save(f"   {abstract_preview}", output_file)
            
            print_and_save(f"\n🤖 AI IMPACT ANALYSIS:", output_file)
            print_and_save("─" * 40, output_file)
            # Format the analysis output nicely
            analysis_lines = analysis.split('\n')
            for line in analysis_lines:
                if line.strip():
                    if line.startswith('Score:'):
                        print_and_save(f"   🎯 {line}", output_file)
                    elif line.startswith('Summary:'):
                        print_and_save(f"   📄 {line}", output_file)
                    elif line.startswith('Justification:'):
                        print_and_save(f"   💡 {line}", output_file)
                    else:
                        print_and_save(f"      {line}", output_file)
            
            print_and_save("\n" + "─" * 80, output_file)
            if rank < m:  # Don't print separator after last item
                print_and_save("", output_file)
        
        # Summary statistics
        print_and_save(f"\n📊 ANALYSIS SUMMARY:", output_file)
        print_and_save("─" * 40, output_file)
        scores = [get_score(item) for item in analyzed_papers]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        
        print_and_save(f"   📈 Average Impact Score: {avg_score:.1f}/10", output_file)
        print_and_save(f"   🔝 Highest Impact Score: {max_score}/10", output_file)
        print_and_save(f"   📉 Lowest Impact Score: {min_score}/10", output_file)
        print_and_save(f"   📄 Total Papers Analyzed: {len(papers)}", output_file)
        
        print_and_save("\n" + "=" * 80, output_file)
        print_and_save("✅ Analysis Complete!", output_file)
        print_and_save("=" * 80, output_file)
        
        # Also save structured data as JSON
        json_filename = f"arxiv_analysis_{timestamp}.json"
        json_data = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "category": "cond-mat.str-el",
                "total_papers_analyzed": len(papers),
                "top_papers_count": m,
                "analysis_version": "1.0"
            },
            "summary_statistics": {
                "average_impact_score": avg_score,
                "highest_impact_score": max_score,
                "lowest_impact_score": min_score
            },
            "papers": []
        }
        
        for rank, item in enumerate(analyzed_papers[:m], 1):
            paper = item['paper']
            analysis = item['analysis']
            
            # Extract structured data from analysis
            score = get_score(item)
            analysis_lines = analysis.split('\n')
            summary_text = ""
            justification_text = ""
            
            for line in analysis_lines:
                if line.startswith('Summary:'):
                    summary_text = line.replace('Summary:', '').strip()
                elif line.startswith('Justification:'):
                    justification_text = line.replace('Justification:', '').strip()
            
            # Get author data
            authors_data = []
            for author in paper.authors:
                author_info = get_author_info(author.name)
                authors_data.append({
                    "name": author.name,
                    "h_index": author_info.get('h_index', 'N/A'),
                    "recent_publications_count": len(author_info.get('recent_publications', [])),
                    "recent_publications": author_info.get('recent_publications', [])
                })
            
            topics = paper.summary.split('.')[0]
            topic_prevalence = get_topic_prevalence(topics)
            
            paper_data = {
                "rank": rank,
                "impact_score": score,
                "title": paper.title,
                "authors": authors_data,
                "published": paper.published,
                "arxiv_id": paper.id.split('/')[-1],
                "arxiv_link": paper.link,
                "primary_topic": topics,
                "topic_prevalence": topic_prevalence,
                "abstract": paper.summary,
                "ai_analysis": {
                    "summary": summary_text,
                    "justification": justification_text
                }
            }
            json_data["papers"].append(paper_data)
        
        # Save JSON file
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)
        
        # Notify user about saved files
        print(f"\n💾 Analysis saved to:")
        print(f"   📄 Text Report: {output_filename}")
        print(f"   📊 JSON Data: {json_filename}")
        print(f"📁 Text file size: {os.path.getsize(output_filename)} bytes")
        print(f"📁 JSON file size: {os.path.getsize(json_filename)} bytes")

if __name__ == "__main__":
    main()
