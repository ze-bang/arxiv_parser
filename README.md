# arXiv Paper Impact Analyzer

A Python tool that fetches daily condensed matter physics papers from arXiv, analyzes their potential impact using author metrics and topic prevalence, and provides summaries of the most promising papers.

## Features

- **Automated Paper Fetching**: Downloads the latest papers from arXiv's `cond-mat.str-el` category
- **Author Analysis**: Uses Google Scholar to gather author h-index and publication history
- **Topic Prevalence Analysis**: Uses Google Custom Search to assess research topic popularity with:
  - High-impact journal detection (3x boost for Nature, Science, PRL, etc.)
  - Exponential time decay (1-year time constant)
- **AI-Powered Summarization**: Uses OpenAI GPT to generate impact scores and paper summaries
- **Smart Ranking**: Ranks papers by potential impact and displays top candidates

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. API Keys Setup

You'll need to set up API keys for:

#### OpenAI API
1. Go to [OpenAI API](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add it to your `.env` file

#### Google Custom Search API
1. **Create a Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one

2. **Enable Custom Search API**:
   - Go to APIs & Services > Library
   - Search for "Custom Search API"
   - Click and enable it

3. **Get API Key**:
   - Go to APIs & Services > Credentials
   - Click "Create Credentials" > "API Key"
   - Copy the API key

4. **Create Custom Search Engine**:
   - Go to [Google Custom Search](https://cse.google.com/)
   - Click "New search engine"
   - Add `*.arxiv.org`, `*.nature.com`, `*.science.org` as sites to search
   - Click "Create"
   - Get the "Search engine ID" from the setup page

5. **Update .env file**:
```bash
OPENAI_API_KEY="your_openai_api_key_here"
GOOGLE_API_KEY="your_google_api_key_here"
GOOGLE_SEARCH_ENGINE_ID="your_search_engine_id_here"
```

### 3. Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the analyzer
python src/arxiv_analyzer.py
```

## How It Works

1. **Paper Fetching**: Connects to arXiv API to get latest `cond-mat.str-el` submissions
2. **Author Analysis**: For each author, queries Google Scholar for h-index and recent publications
3. **Topic Analysis**: Uses Google Custom Search to find related papers and assess topic prevalence
4. **Impact Scoring**: Combines author metrics, topic prevalence, journal impact factor, and recency
5. **AI Summary**: Uses GPT to generate impact scores (1-10) and paper summaries
6. **Ranking**: Sorts papers by impact score and displays top results

## Configuration

- **Number of papers to analyze**: Modify `max_results` in `get_arxiv_papers()`
- **Number of top papers to display**: Change the `m` parameter in `main(m=5)`
- **Journal impact list**: Update `high_impact_journals` in `get_topic_prevalence()`
- **Time decay factor**: Modify the exponential decay formula in `get_topic_prevalence()`

## API Limits

- **Google Custom Search**: 100 free searches per day
- **OpenAI**: Pay-per-use based on your plan
- **Google Scholar**: No official API, uses `scholarly` library (may have rate limits)

## Troubleshooting

- **Google API errors**: Check that Custom Search API is enabled and your quotas
- **Scholar timeouts**: The `scholarly` library may hit rate limits; the script will continue with available data
- **OpenAI errors**: Verify your API key and account has sufficient credits
