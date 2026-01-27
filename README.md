# MarketPulse AI

**An intelligent financial news aggregation and analysis platform powered by AI**

MarketPulse AI is a comprehensive web application that aggregates real-time financial news from MoneyControl, performs sentiment analysis, provides AI-powered insights, and delivers personalized market intelligence to help users make informed investment decisions.

---
## 🌟 Problem Statement
- Investors waste time visiting multiple platforms and reading lengthy financial news to understand market impact.
- MarketPulse AI automatically analyzes financial news using ML to classify sentiment as positive or negative in real time.
- It presents market prices, news sentiment, and overall market direction on a single, easy-to-read dashboard for faster decisions.

---
## 🌟 Key Features

### 📰 Real-Time News Aggregation
- Automated scraping of latest financial news from MoneyControl across 15+ categories
- Coverage of Business, Markets, Economy, IPOs, Startups, Banking, and more
- Smart caching with 5-minute refresh intervals for optimal performance
- Parallel scraping for fast data retrieval

### 🤖 AI-Powered Analysis
- **Sentiment Analysis**: FinBERT-based sentiment scoring for every article (positive/negative/neutral)
- **AI Summarization**: Local T5 model (Flan-T5) for privacy-focused article summaries
- **Intelligent Chatbot**: Local Llama 3 (via Ollama) financial assistant for market queries
- **Trend Detection**: Identifies trending news based on engagement metrics

### 📊 Market Data Integration
- Real-time stock prices and market indices
- Historical stock data with customizable time periods
- Stock search across 1000+ NSE-listed companies
- Interactive charts and visualizations using Recharts

### 👤 Personalized Experience
- User authentication and profile management
- Custom watchlists for tracking favorite stocks
- **Smart "For Me" Feed**: Strictly filters news for user's watchlist stocks using advanced regex and alias matching
- Category-based news filtering (Economy, Stocks, Commodities, etc.)

### 🔍 Advanced Search & Filtering
- Global search across headlines, descriptions, and categories
- Multi-category filtering
- Stock-specific news filtering
- Time-based filters (trending, weekly, etc.)
- Pagination for efficient browsing

---

## 🏗️ Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **Database**: SQLite with SQLAlchemy ORM
- **AI/ML**: 
  - **Llama 3 / 3.2**: Local LLM for chatbot (via Ollama)
  - **Flan-T5**: Local Transformer model for summarization
  - **FinBERT**: Sentiment analysis
- **Web Scraping**: BeautifulSoup4, Requests
- **Market Data**: yfinance for real-time stock data
- **Authentication**: Passlib with bcrypt for secure password hashing

### Frontend
- **Framework**: React 19 with Vite
- **Routing**: React Router DOM
- **Styling**: Tailwind CSS 4
- **Charts**: Recharts for data visualization
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **State Management**: React hooks

### Development Tools
- **Package Manager**: npm (frontend), pip (backend)
- **Code Quality**: ESLint for linting
- **Environment**: python-dotenv for configuration

---

## 📁 Project Structure

```
MarketPulseAI-main/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── scraper.py              # News scraping engine
│   ├── sentiment.py            # FinBERT sentiment analysis
│   ├── chatbot.py              # Llama 3 (Ollama) chatbot integration
│   ├── summarizer.py           # T5 summarization logic
│   ├── market_data.py          # Stock data fetching
│   ├── database.py             # Database models and operations
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment variables
│   ├── venv/                   # Python virtual environment
│   └── moneycontrol_news.json  # Cached news data
│
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── pages/              # Page components
│   │   ├── App.jsx             # Main application component
│   │   └── main.jsx            # Application entry point
│   ├── public/                 # Static assets
│   ├── package.json            # Node dependencies
│   ├── vite.config.js          # Vite configuration
│   └── tailwind.config.js      # Tailwind CSS configuration
│
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20+
- **Ollama**: Must be installed and running locally
  - Pull Llama 3 model: `ollama pull llama3.2`

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the backend server**
   ```bash
   python main.py
   ```
   Server will start at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Run the development server**
   ```bash
   npm run dev
   ```
   Application will open at `http://localhost:5173`

---

## 📡 API Endpoints

### News
- `GET /news` - Fetch paginated news with filtering options
- `POST /news/view` - Increment article view count
- `POST /news/summary` - Get AI-generated summary for an article

### Market Data
- `GET /market` - Get current market indices
- `GET /stock/{symbol}` - Get detailed stock information
- `GET /stock/{symbol}/history` - Get historical stock data
- `GET /search-stocks` - Search for stocks by name or symbol

### User Management
- `POST /auth/signup` - Create new user account
- `POST /auth/login` - User authentication
- `GET /watchlist/{email}` - Get user's watchlist
- `POST /watchlist` - Add stock to watchlist
- `DELETE /watchlist/{email}/{symbol}` - Remove from watchlist

### AI Features
- `POST /chat` - Chat with AI financial assistant

---

## 🎯 Use Cases

1. **Individual Investors**: Stay updated with latest market news and sentiment analysis
2. **Day Traders**: Quick access to trending news and real-time stock data
3. **Financial Analysts**: AI-powered summaries for rapid information processing
4. **Portfolio Managers**: Track watchlist stocks with personalized news feeds
5. **Students & Researchers**: Learn about market trends with AI assistance

---

## 🔐 Security Features

- Secure password hashing with bcrypt
- SQL injection prevention through SQLAlchemy ORM
- CORS configuration for secure cross-origin requests
- Environment-based API key management
- Input validation and sanitization

---

## 🎨 Features in Detail

### News Scraping Engine
- **Multi-threaded scraping**: Parallel processing of 15+ news categories
- **Smart caching**: Reduces server load with intelligent cache management
- **Deep metadata extraction**: Extracts images, timestamps, and full article content
- **Duplicate detection**: Prevents duplicate articles in the database

### Sentiment Analysis
- **FinBERT model**: Specialized financial sentiment analysis
- **Confidence scores**: Provides sentiment probability scores
- **Real-time processing**: Analyzes sentiment during article scraping
- **Cached results**: Stores sentiment for faster subsequent access

### AI Chatbot
- **Context-aware**: Understands financial terminology and market context
- **Multi-turn conversations**: Maintains conversation history
- **Market insights**: Provides analysis based on latest news and data
- **Natural language**: User-friendly conversational interface

---

## 📈 Performance Optimizations

- Background news scraping to avoid blocking API requests
- In-memory caching for frequently accessed data
- Lazy loading of AI models on server startup
- Efficient database queries with proper indexing
- Pagination for large datasets
- Concurrent processing for parallel operations

---

## 🛠️ Development

### Running Tests
```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm run test
```

### Building for Production
```bash
# Frontend
cd frontend
npm run build

# Backend (use production ASGI server)
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 🐛 Known Issues & Limitations

- News scraping limited to MoneyControl (can be extended to other sources)
- Sentiment analysis requires GPU for optimal performance
- API rate limits on free tier of Gemini AI
- Historical data limited by yfinance API capabilities

---

## 🔮 Future Enhancements

- [ ] Multi-source news aggregation (Bloomberg, Reuters, etc.)
- [ ] Advanced portfolio tracking and analytics
- [ ] Email/SMS alerts for watchlist stocks
- [ ] Mobile application (React Native)
- [ ] Social sentiment analysis from Twitter/Reddit
- [ ] Technical analysis indicators
- [ ] Backtesting capabilities
- [ ] Multi-language support

---

## 📝 License

This project is for educational and personal use. Please ensure compliance with MoneyControl's terms of service when scraping their content.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

## 👨‍💻 Author

**Ayushi**

---

## 🙏 Acknowledgments

- MoneyControl for financial news data
- Google Gemini AI for natural language processing
- FinBERT team for sentiment analysis model
- Yahoo Finance for market data API
- Open source community for amazing tools and libraries

---

## 📞 Support

For questions or support, please open an issue in the repository.

---

**Built with ❤️ for the financial community**
