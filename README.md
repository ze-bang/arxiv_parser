# ArXiv Paper Impact Analyzer

This system fetches daily submissions from arXiv's cond-mat.str-el (Condensed Matter - Strongly Correlated Electrons) category and generates impact metrics using LLM analysis with web search capabilities. It helps researchers quickly identify papers with high potential impact by analyzing author reputation, abstract content, topic novelty, and field activity.

## Features

- **Daily arXiv Monitoring**: Automatically fetches new papers from the condensed matter strongly correlated electrons category
- **Author Reputation Analysis**: Evaluates authors based on citation counts, h-index, and institutional affiliation
- **Content Analysis**: Uses LLM to assess abstract novelty and potential impact
- **Research Context Search**: Identifies field activity and competing work
- **Impact Scoring**: Combines multiple factors into a comprehensive impact score
- **Visualization**: Generates visual reports and trend analysis
- **Email Notifications**: Sends alerts for high-impact papers
- **REST API**: Access analysis through a simple web API

## Setup

### Basic Installation

1. Clone this repository
2. Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and add your API keys:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### Advanced Configuration

- **API Keys**: For optimal performance, add the following optional API keys to your `.env` file:
  - `SEMANTIC_SCHOLAR_API_KEY`: For accurate author citation data
  - `SERPAPI_API_KEY`: For better web search results
  
- **Email Notifications**: Configure email settings in `.env` to receive notifications about high-impact papers:
  - `SMTP_SERVER`, `SMTP_PORT`: Your email provider's SMTP settings
  - `EMAIL_FROM`, `EMAIL_PASSWORD`: Your email credentials
  - `EMAIL_RECIPIENTS`: Comma-separated list of recipients
  - `NOTIFY_THRESHOLD`: Minimum impact score to trigger notifications

## Usage

### Command Line Interface

The system provides a flexible command-line interface:

#### Basic Analysis

```bash
# Analyze today's papers
python main.py run

# Analyze papers for a specific date
python main.py run --date 2025-06-01

# Analyze papers from a date range
python main.py run --start-date 2025-06-01 --end-date 2025-06-07
```

#### Scheduled Mode

```bash
# Run daily at 9:00 AM
python main.py schedule

# Customize scheduled run
python main.py schedule --time 08:30 --no-notify --keep-days 60 --limit 10
```

#### Visualization

```bash
# Generate visualization for a results file
python main.py visualize --file results/daily_papers_2025-06-05.json

# Generate trend analysis for the past 7 days
python main.py visualize --trends
```

#### API Server

```bash
# Start API server on default port (5000)
python main.py api

# Specify custom port
python main.py api --port 8080
```

#### Testing

```bash
# Run in test mode with mock data
python main.py test
```

### API Endpoints

When running the API server, the following endpoints are available:

- **POST /analyze**: Analyze papers for a specific date
  - Parameters:
    - `date`: (optional) Date in format YYYY-MM-DD
    - `limit`: (optional) Maximum number of papers to analyze

## Configuration

You can customize the system's behavior by editing these files:

- **`.env`**: API keys and environment-specific settings
- **`config.py`**: Scoring weights, thresholds, and default behavior

### Scheduled Execution

For regular monitoring, set up a cron job to run the analyzer daily:

```bash
# Add to crontab
0 9 * * * /path/to/arxiv_parser/run_daily.sh
```

## Output

Results are saved as JSON files in the `results` directory:
- `daily_papers_YYYY-MM-DD.json`: Daily analysis results
- `daily_papers_YYYY-MM-DD_report.png`: Visualization report
- `trend_analysis_YYYY-MM-DD.png`: Trend analysis across multiple days

## Example Impact Analysis

The system evaluates papers based on multiple factors:

- **Author Score** (30%): Based on citations, h-index, and institutional reputation
- **Abstract Score** (40%): Analysis of abstract content and research methodology
- **Novelty Score** (20%): Presence of novel techniques or discoveries
- **Citation Potential** (10%): LLM assessment of future citation potential

Each paper receives an impact score from 0-100, with higher scores indicating greater potential impact.

## License

MIT License
# arxiv_parser
