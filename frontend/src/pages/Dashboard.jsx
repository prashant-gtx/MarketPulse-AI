import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import { Newspaper, TrendingUp, MessageSquare, Send, X, ExternalLink, ArrowUpRight, ArrowDownRight, Minus, Activity, Linkedin, LogOut, Sun, Moon, Bell, Settings, Bookmark, Star as StarIcon, Search, Plus, RefreshCw, Eye, Flame, ChevronDown, Menu, Briefcase, ArrowRight, BarChart3 } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import ThemeToggle from '../components/ThemeToggle';
import NewsSummaryModal from '../components/NewsSummaryModal';


const API_BASE = "http://localhost:8000";

const NSE_STOCKS = [
    { name: "Reliance Industries Ltd", symbol: "RELIANCE" },
    { name: "Religare Enterprises Ltd", symbol: "RELIGARE" },
    { name: "Reliance Infrastructure Ltd", symbol: "RELINFRA" },
    { name: "HDFC Bank Ltd", symbol: "HDFCBANK" },
    { name: "Infosys Ltd", symbol: "INFY" },
    { name: "Tata Consultancy Services Ltd", symbol: "TCS" },
    { name: "ICICI Bank Ltd", symbol: "ICICIBANK" },
    { name: "Axis Bank Ltd", symbol: "AXISBANK" },
    { name: "Bharti Airtel Ltd", symbol: "BHARTIARTL" },
    { name: "State Bank of India", symbol: "SBIN" },
    { name: "Bajaj Finance Ltd", symbol: "BAJFINANCE" },
    { name: "Larsen & Toubro Ltd", symbol: "LT" },
    { name: "Maruti Suzuki India Ltd", symbol: "MARUTI" },
    { name: "Kotak Mahindra Bank Ltd", symbol: "KOTAKBANK" },
    { name: "Asian Paints Ltd", symbol: "ASIANPAINT" },
];

function Dashboard({ onLogout, isAuthenticated, userName }) {
    const navigate = useNavigate();
    const [news, setNews] = useState([]);
    const [marketData, setMarketData] = useState(null);
    const [chatQuery, setChatQuery] = useState("");
    const [chatHistory, setChatHistory] = useState([]);
    const [loadingChat, setLoadingChat] = useState(false);
    const [isChatOpen, setIsChatOpen] = useState(false);
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [isWatchlistOpen, setIsWatchlistOpen] = useState(false);
    const [isViewModalOpen, setIsViewModalOpen] = useState(false);
    const [addingStock, setAddingStock] = useState(null);


    // News Summary State
    const [summaryModalOpen, setSummaryModalOpen] = useState(false);
    const [selectedArticle, setSelectedArticle] = useState(null);
    const [articleSummary, setArticleSummary] = useState("");
    const [loadingSummary, setLoadingSummary] = useState(false);
    const [loadingNews, setLoadingNews] = useState(false);



    // Settings & Profile States
    const [isEditingName, setIsEditingName] = useState(false);
    const [tempName, setTempName] = useState(userName);
    const userDob = localStorage.getItem('userDob') || '';
    const [marketAlertsEnabled, setMarketAlertsEnabled] = useState(localStorage.getItem('marketAlerts') === 'true');

    // Stock Search States
    const [stockQuery, setStockQuery] = useState("");
    const [stockResults, setStockResults] = useState([]);
    const [selectedStocks, setSelectedStocks] = useState([]);
    const userEmail = localStorage.getItem('userEmail');

    const [watchlist, setWatchlist] = useState(() => {
        try {
            const saved = localStorage.getItem('newsWatchlist');
            return saved ? JSON.parse(saved) : [];
        } catch (e) {
            console.error("Error parsing newsWatchlist:", e);
            return [];
        }
    });

    // Pagination & Filter State
    const [currentPage, setCurrentPage] = useState(1);
    const [itemsPerPage] = useState(24);
    const [totalPages, setTotalPages] = useState(1);
    const [dateFilter, setDateFilter] = useState('all'); // all, for_me, watchlist, week, ...
    const [newsSearchQuery, setNewsSearchQuery] = useState(""); // Global news search state
    const [isSearchExpanded, setIsSearchExpanded] = useState(false);
    const [isTimeDropdownOpen, setIsTimeDropdownOpen] = useState(false);
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const chatEndRef = useRef(null);

    // Debounced Search Effect
    useEffect(() => {
        const delayDebounceFn = setTimeout(() => {
            fetchNews(1, newsSearchQuery);
        }, 500); // Reduced to 500ms

        return () => clearTimeout(delayDebounceFn);
    }, [newsSearchQuery]);

    // Auto-refresh logic (Polling)
    useEffect(() => {
        fetchMarketData();

        const interval = setInterval(() => {
            // Only refresh news if on first page and no active search
            if (currentPage === 1 && !newsSearchQuery) {
                fetchNews(1, "", true);
            }
            fetchMarketData();
        }, 5000);

        return () => clearInterval(interval);
    }, [currentPage, newsSearchQuery, dateFilter, selectedStocks, isAuthenticated]);

    useEffect(() => {
        if (isAuthenticated && userEmail) {
            fetchWatchlist();
        }
    }, [isAuthenticated, userEmail]);

    const fetchWatchlist = async () => {
        try {
            const res = await axios.get(`${API_BASE}/watchlist/${userEmail}`);
            setSelectedStocks(res.data);
        } catch (err) {
            console.error("Error fetching watchlist:", err);
        }
    };

    useEffect(() => {
        localStorage.setItem('marketAlerts', marketAlertsEnabled);
    }, [marketAlertsEnabled]);

    useEffect(() => {
        localStorage.setItem('newsWatchlist', JSON.stringify(watchlist));
    }, [watchlist]);

    useEffect(() => {
        setTempName(userName);
    }, [userName]);

    useEffect(() => {
        fetchNews(1, newsSearchQuery);
    }, [dateFilter, selectedStocks]);

    // Robust Google Translate Initialization
    useEffect(() => {
        const loadTranslate = () => {
            if (!window.googleTranslateElementInit) {
                window.googleTranslateElementInit = () => {
                    if (document.getElementById('google_translate_element')) {
                        new window.google.translate.TranslateElement({ pageLanguage: 'en' }, 'google_translate_element');
                    }
                };
            }

            const existingScript = document.getElementById('google-translate-script');
            if (!existingScript) {
                const script = document.createElement('script');
                script.id = 'google-translate-script';
                script.type = 'text/javascript';
                script.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
                script.async = true;
                document.body.appendChild(script);
            } else if (window.google && window.google.translate) {
                // If script exists and google is available, re-init manually
                window.googleTranslateElementInit();
            }
        };

        // Delay slightly to ensure component mount is fully settled
        const timer = setTimeout(loadTranslate, 1000);
        return () => clearTimeout(timer);
    }, []);



    const handleStockSearch = async (query) => {
        setStockQuery(query);
        if (query.trim().length > 0) {
            try {
                const res = await axios.get(`${API_BASE}/search-stocks?q=${query}`);
                setStockResults(res.data);
            } catch (err) {
                console.error("Error searching stocks:", err);
            }
        } else {
            setStockResults([]);
        }
    };

    const addStockToWatchlist = async (stock) => {
        if (!userEmail) {
            alert("Session expired. Please log in again.");
            return;
        }
        if (!selectedStocks.find(s => s.symbol === stock.symbol)) {
            setAddingStock(stock.symbol);
            try {
                const response = await axios.post(`${API_BASE}/watchlist`, {
                    email: userEmail,
                    symbol: stock.symbol,
                    name: stock.name
                });
                console.log("Added to watchlist:", response.data);
                setSelectedStocks([...selectedStocks, stock]);
            } catch (err) {
                console.error("Error adding to watchlist:", err);
                const msg = err.response?.data?.detail || "Failed to add to watchlist. Please try again.";
                alert(msg);
            } finally {
                setAddingStock(null);
            }
        }
        setStockQuery("");
        setStockResults([]);
    };

    const removeStockFromWatchlist = async (symbol) => {
        try {
            await axios.delete(`${API_BASE}/watchlist/${userEmail}/${symbol}`);
            setSelectedStocks(selectedStocks.filter(s => s.symbol !== symbol));
        } catch (err) {
            console.error("Error removing from watchlist:", err);
        }
    };

    const handleSaveName = () => {
        localStorage.setItem('userName', tempName);
        setIsEditingName(false);
        window.location.reload();
    };

    useEffect(() => {
        if (isChatOpen && chatEndRef.current) {
            chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [chatHistory, isChatOpen]);


    const fetchNews = async (page = 1, query = "", isUpdate = false) => {
        if (!isUpdate) setLoadingNews(true);
        try {
            const params = {
                page: page,
                limit: itemsPerPage,
                q: query
            };

            // Enhanced: Send personalization to backend
            // Enhanced: Send personalization to backend
            if (dateFilter === 'for_me' && isAuthenticated) {
                // User Requirement: "If i have not added any stocks in the watchlist then it should be blank"
                // strict check: if no stocks, don't fetch, just set empty.
                if (!selectedStocks || selectedStocks.length === 0) {
                    setNews([]);
                    setTotalPages(1);
                    setLoadingNews(false);
                    return;
                }

                // STRICT 'For Me' Logic (User Request):
                // 1. Only show articles related to stocks in watchlist.
                // 2. Ignore category preferences to ensure we don't miss news about the stock.
                const stockSymbols = selectedStocks.map(s => s.symbol).filter(Boolean).join(',');
                if (stockSymbols) {
                    params.stocks = stockSymbols;
                }
            } else {
                // For General Feed (All Time, Trending, Time filters):
                // User Requirement: "custom feed should be linked to the user only. if i logout ... default all categories"

                if (isAuthenticated && dateFilter !== 'trending') {
                    const userEmail = localStorage.getItem('userEmail');
                    // Store/Retrieve per user to prevent leaking prefs between users on same device
                    const storageKey = userEmail ? `userCategories_${userEmail}` : 'userCategories';

                    const userCategories = JSON.parse(localStorage.getItem(storageKey) || '[]');
                    if (userCategories.length > 0) {
                        params.categories = userCategories.join(',');
                    }
                }
                // If not authenticated, params.categories remains undefined -> Backend returns ALL news.

                if (dateFilter !== 'all' && dateFilter !== 'trending') {
                    params.filter_type = dateFilter;
                } else if (dateFilter === 'trending') {
                    params.filter_type = 'trending';
                }
            }

            const res = await axios.get(`${API_BASE}/news`, { params });

            if (res.data.items) {
                setNews(res.data.items);
                setTotalPages(res.data.pages);
                if (page) setCurrentPage(page);
            } else {
                setNews(res.data);
            }

        } catch (err) {
            console.error("Error fetching news:", err);
        } finally {
            if (!isUpdate) setLoadingNews(false);
        }
    };

    const fetchMarketData = async () => {
        try {
            const res = await axios.get(`${API_BASE}/market`);
            setMarketData(res.data);
        } catch (err) {
            console.error("Error fetching market data:", err);
        }
    };

    const handleChatSubmit = async (e) => {
        e.preventDefault();
        if (!chatQuery.trim()) return;

        const query = chatQuery;
        setChatQuery("");

        // Add User Message
        setChatHistory(prev => [...prev, { role: 'user', content: query }]);
        setLoadingChat(true);

        try {
            const res = await axios.post(`${API_BASE}/chat`, { query });
            setChatHistory(prev => [...prev, { role: 'ai', content: res.data.response }]);
        } catch (err) {
            setChatHistory(prev => [...prev, { role: 'ai', content: "Error communicating with AI." }]);
        } finally {
            setLoadingChat(false);
        }
    };

    // Simplified: Backend now handles filtering for accuracy across the whole dataset.
    const getFilteredNews = () => {
        if (!Array.isArray(news)) return [];
        return news;
    };

    // Pagination Logic moved to Backend mostly, but we still have client side filters?
    // If we use backend pagination, we can't easily filter by "watchlist" on client unless backend supports it.
    // The requirement says "Only the latest news... at the top. Older news... Saved permanently... Pagination".
    // It doesn't explicitly say "All filters must work on the entire database".

    // For now, allow getFilteredNews to filter the *current page* of news.
    // Or, simpler: Just show what backend returns and trust backend sorting (timestamp desc).

    const filteredNews = getFilteredNews();
    // const totalPages = Math.max(1, Math.ceil(filteredNews.length / itemsPerPage)); // Old logic

    // Ensure current page is within bounds
    // const safeCurrentPage = Math.min(currentPage, totalPages);
    // const indexOfLastItem = safeCurrentPage * itemsPerPage;
    // const indexOfFirstItem = Math.max(0, indexOfLastItem - itemsPerPage);
    // const currentNews = filteredNews.slice(indexOfFirstItem, indexOfLastItem);

    // With backend pagination, 'news' is already the chunk.
    const currentNews = filteredNews; // Assuming backend filtering not yet implemented, this might only filter the current page.

    const paginate = (pageNumber) => {
        setCurrentPage(pageNumber);
        fetchNews(pageNumber, newsSearchQuery);
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };


    // Helpers
    const handleImageError = (e) => {
        e.target.onerror = null;
        e.target.src = '/default-news.png';
    };

    const handleNewsClick = async (e, item) => {
        e.preventDefault();

        // 1. Open Modal immediately with article info
        setSelectedArticle(item);
        setSummaryModalOpen(true);
        setArticleSummary(""); // Clear previous summary
        setLoadingSummary(true);

        // 2. Track view count
        try {
            await axios.post(`${API_BASE}/news/view`, { link: item.link });
            setNews(prev => prev.map(n => n.link === item.link ? { ...n, views: (n.views || 0) + 1 } : n));
        } catch (e) {
            console.error("Failed to update view count", e);
        }

        // 3. Fetch Summary
        try {
            const res = await axios.post(`${API_BASE}/news/summary`, { link: item.link });
            setArticleSummary(res.data.summary);

            // Sync sentiment logic removed to maintain consistency (User Request)
            // The sentiment will remains as originally calculated from the headline.
        } catch (err) {
            console.error("Error fetching summary:", err);
            setArticleSummary("Failed to generate summary. Please try reading the full article.");
        } finally {
            setLoadingSummary(false);
        }
    };

    const getSentimentStyle = (sentiment) => {
        switch (sentiment?.toLowerCase()) {
            case 'positive': return 'border-2 border-emerald-500 hover:border-emerald-600';
            case 'negative': return 'border-2 border-rose-500 hover:border-rose-600';
            default: return 'border border-slate-200 dark:border-white/10 hover:shadow-xl dark:hover:shadow-blue-900/20';
        }
    };

    const getSentimentIcon = (sentiment) => {
        switch (sentiment?.toLowerCase()) {
            case 'positive': return <ArrowUpRight className="w-3 h-3" />;
            case 'negative': return <ArrowDownRight className="w-3 h-3" />;
            default: return <Minus className="w-3 h-3" />;
        }
    };

    const formatDate = (isoString) => {
        if (!isoString) return '';
        // If string contains a comma, assume it is already formatted by backend (e.g. "27 Jan 2026, 09:30 AM")
        if (isoString.includes(',')) return isoString;

        const date = new Date(isoString);
        if (isNaN(date.getTime())) return isoString; // Fallback if not valid date
        return new Intl.DateTimeFormat('en-IN', {
            day: 'numeric', month: 'short', year: 'numeric',
            hour: '2-digit', minute: '2-digit'
        }).format(date);
    };

    return (
        <div className="h-screen w-full bg-slate-50 dark:bg-[#0a0c10] text-slate-900 dark:text-white font-sans selection:bg-blue-500 selection:text-white flex flex-col overflow-hidden relative transition-colors duration-300">

            {/* Background Glow Effects */}
            <div className="absolute top-0 left-0 w-[500px] h-[500px] bg-blue-600/5 dark:bg-blue-600/10 rounded-full blur-[120px] pointer-events-none" />
            <div className="absolute bottom-0 right-0 w-[500px] h-[500px] bg-purple-600/5 dark:bg-purple-600/10 rounded-full blur-[120px] pointer-events-none" />

            {/* Top Running Ticker (Marquee) */}
            <div className="bg-blue-600/10 dark:bg-blue-600/10 border-b border-slate-200 dark:border-white/5 overflow-hidden py-1.5 backdrop-blur-md relative z-50">
                <div className="flex animate-marquee whitespace-nowrap w-max">
                    {[...Array(2)].map((_, i) => (
                        <div key={i} className="flex">
                            {[...Array(6)].map((_, j) => (
                                <div key={j} className="flex">
                                    {marketData?.commodities && Object.entries(marketData.commodities).map(([key, data]) => (
                                        <div key={key} className="flex items-center gap-2 mr-12">
                                            <span className="text-blue-600 dark:text-blue-300 font-bold text-[10px] uppercase tracking-wider">{key}</span>
                                            <span className={`text-[10px] font-mono font-bold ${data.change >= 0 ? 'text-emerald-500 dark:text-emerald-400' : 'text-rose-500 dark:text-rose-400'}`}>
                                                {data.price.toLocaleString()}
                                                <span className="ml-1 opacity-75">
                                                    {data.change >= 0 ? '▲' : '▼'} {Math.abs(data.percent_change)}%
                                                </span>
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            ))}
                        </div>
                    ))}
                </div>
            </div>

            {/* Navbar / Header - Full Width */}
            <header className="shrink-0 z-40 sticky top-0">
                <div className="w-full">
                    <div className="bg-white/40 dark:bg-gray-900/60 backdrop-blur-xl border-b border-slate-200 dark:border-white/10 px-2 py-2 flex flex-col md:flex-row justify-between items-center gap-4 relative overflow-hidden group">
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-500/5 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 pointer-events-none" />

                        {/* Logo */}
                        <div className="flex items-center gap-4">
                            <img
                                src="/logo.png"
                                alt="MarketPulse AI Logo"
                                className="h-16 md:h-20 w-auto object-contain rounded-2xl shadow-lg transition-all duration-300 hover:scale-105 active:scale-95 dark:brightness-110 brightness-95 contrast-110"
                            />
                        </div>

                        {/* Market Ticker Pill */}
                        <div className="bg-slate-900/90 dark:bg-black/80 backdrop-blur-xl border border-slate-200 dark:border-white/10 rounded-2xl flex items-center shadow-2xl relative overflow-hidden flex-wrap md:flex-nowrap">
                            {marketData?.indices ? (
                                <>
                                    <div className="px-5 py-4 border-r border-slate-700/50 flex items-center gap-3 shrink-0">
                                        <span className="relative flex h-2.5 w-2.5">
                                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                                        </span>
                                        <span className="text-[11px] font-black text-emerald-400 tracking-[0.15em]">LIVE</span>
                                    </div>

                                    {['Sensex', 'Nifty 50'].map((key, idx) => {
                                        const data = marketData.indices[key];
                                        if (!data) return null; // specific index might be missing in partial load

                                        const isPositive = data.change >= 0;
                                        const colorClass = isPositive ? 'text-emerald-400' : 'text-rose-400';
                                        const strokeColor = isPositive ? '#10b981' : '#f43f5e';

                                        const points = data.history || [];
                                        const min = points.length > 0 ? Math.min(...points) : 0;
                                        const max = points.length > 1 ? Math.max(...points) : (points[0] || 100) + 10;
                                        const range = max - min || 1;
                                        const width = 100;
                                        const height = 28;

                                        // Safety check: if only 1 point, create a flat line
                                        const strokePoints = points.length > 1 ? points : (points.length === 1 ? [points[0], points[0]] : [100, 100]);

                                        const pathD = strokePoints.map((p, i) => {
                                            const x = (i / (strokePoints.length - 1)) * width;
                                            const y = height - ((p - min) / range) * height;
                                            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                                        }).join(' ');

                                        return (
                                            <div key={key} className={`flex items-center gap-6 px-6 py-4 ${idx !== 0 ? 'border-l border-slate-700/50' : ''}`}>
                                                <div className="flex flex-col min-w-[140px]">
                                                    <span className="text-slate-500 dark:text-gray-400 text-[10px] font-bold uppercase tracking-[0.2em] mb-1.5">{key}</span>
                                                    <div className="flex items-center gap-3">
                                                        <span className="text-lg font-black text-white tracking-tight">
                                                            {data.price.toLocaleString()}
                                                        </span>
                                                        <div className={`flex items-center gap-2 text-[11px] font-bold ${colorClass}`}>
                                                            <span>{data.change > 0 ? '+' : ''}{data.change.toLocaleString()}</span>
                                                            <span className="opacity-70">({data.percent_change > 0 ? '+' : ''}{data.percent_change}%)</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="w-[100px] h-[30px] flex items-center ml-2">
                                                    <svg width="100" height="28" className="overflow-visible">
                                                        <path
                                                            d={pathD}
                                                            fill="none"
                                                            stroke={strokeColor}
                                                            strokeWidth="3"
                                                            strokeLinecap="round"
                                                            strokeLinejoin="round"
                                                            style={{ filter: `drop-shadow(0 0 8px ${strokeColor}44)` }}
                                                        />
                                                    </svg>
                                                </div>
                                            </div>
                                        )
                                    })}
                                </>
                            ) : (
                                <div className="px-8 py-5 text-slate-400 dark:text-gray-500 text-xs font-bold tracking-widest flex items-center gap-3">
                                    <RefreshCw className="w-4 h-4 animate-spin text-blue-500" />
                                    CONNECTING TO MARKET DATA...
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1 w-full px-2 py-2 overflow-hidden flex flex-col relative z-10 text-slate-900 dark:text-white">



                {/* Filters and Actions Row */}
                <div className="mb-4 flex flex-row flex-nowrap justify-between items-center gap-4 pb-1">
                    <div className="flex flex-nowrap gap-2 p-1 bg-slate-200/50 dark:bg-black/20 rounded-xl border border-slate-200 dark:border-white/5 w-fit shrink-0 relative">
                        {/* Time Filter Dropdown */}
                        <div className="relative">
                            <button
                                onClick={(e) => { e.stopPropagation(); setIsTimeDropdownOpen(!isTimeDropdownOpen); }}
                                className="px-4 h-10 rounded-xl text-xs font-semibold whitespace-nowrap transition-all bg-blue-600 text-white shadow-lg shadow-blue-500/20 flex items-center gap-2 min-w-[120px] justify-between"
                            >
                                <span>
                                    {[
                                        { id: 'all', label: 'All Time' },
                                        { id: 'week', label: 'This Week' },
                                        { id: 'prev_week', label: 'Last Week' },
                                        { id: 'month', label: 'This Month' },
                                        { id: 'year', label: 'This Year' },
                                    ].find(f => f.id === dateFilter)?.label || 'All Time'}
                                </span>
                                <ChevronDown className={`h-3 w-3 transition-transform duration-200 ${isTimeDropdownOpen ? 'rotate-180' : ''}`} />
                            </button>

                            {isTimeDropdownOpen && (
                                <>
                                    <div className="fixed inset-0 z-[60]" onClick={() => setIsTimeDropdownOpen(false)} />
                                    <div className="absolute top-full left-0 mt-2 w-40 bg-white dark:bg-gray-800 border border-slate-200 dark:border-white/10 rounded-xl shadow-2xl z-[70] overflow-hidden py-1.5 animate-in fade-in zoom-in duration-200">
                                        {[
                                            { id: 'all', label: 'All Time' },
                                            { id: 'week', label: 'This Week' },
                                            { id: 'prev_week', label: 'Last Week' },
                                            { id: 'month', label: 'This Month' },
                                            { id: 'year', label: 'This Year' },
                                        ].map(filter => (
                                            <button
                                                key={filter.id}
                                                onClick={() => {
                                                    setDateFilter(filter.id);
                                                    setCurrentPage(1);
                                                    setIsTimeDropdownOpen(false);
                                                }}
                                                className={`w-full text-left px-4 py-2 text-xs font-semibold whitespace-nowrap transition-all ${dateFilter === filter.id
                                                    ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
                                                    : 'text-slate-600 dark:text-gray-400 hover:bg-slate-50 dark:hover:bg-white/5'
                                                    }`}
                                            >
                                                {filter.label}
                                            </button>
                                        ))}
                                    </div>
                                </>
                            )}
                        </div>

                        {/* Special Filters */}
                        {[
                            ...(isAuthenticated ? [
                                { id: 'for_me', label: 'For Me' }
                            ] : []),
                            { id: 'trending', label: 'Trending' },
                        ].map(filter => (
                            <button
                                key={filter.id}
                                onClick={() => {
                                    if (dateFilter === filter.id) {
                                        setDateFilter('all');
                                    } else {
                                        setDateFilter(filter.id);
                                    }
                                    setCurrentPage(1);
                                }}
                                className={`px-4 h-10 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${dateFilter === filter.id
                                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
                                    : 'text-slate-600 dark:text-gray-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-300 dark:hover:bg-white/5'
                                    }`}
                            >
                                {filter.label}
                            </button>
                        ))}
                    </div>

                    <div className="flex items-center gap-3 shrink-0">
                        {/* Expandable Search */}
                        <div className={`relative flex items-center transition-all duration-300 ${isSearchExpanded ? 'w-64' : 'w-10'}`}>
                            <div className={`absolute inset-y-0 left-0 flex items-center justify-center w-10 h-10 cursor-pointer text-slate-400 hover:text-blue-500 transition-colors z-10 ${isSearchExpanded ? 'pointer-events-none' : ''}`}
                                onClick={() => { setIsSearchExpanded(true); }}
                            >
                                <Search className="h-4 w-4" />
                            </div>

                            <input
                                type="text"
                                value={newsSearchQuery}
                                onChange={(e) => { setNewsSearchQuery(e.target.value); setCurrentPage(1); }}
                                onBlur={() => { if (!newsSearchQuery) setIsSearchExpanded(false); }}
                                placeholder={isSearchExpanded ? "Search news..." : ""}
                                className={`block w-full h-10 bg-slate-200/50 dark:bg-black/20 border border-slate-200 dark:border-white/5 rounded-xl text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all text-xs font-medium pl-10 pr-4 ${isSearchExpanded ? 'opacity-100' : 'opacity-0 cursor-pointer'}`}
                                onClick={() => setIsSearchExpanded(true)}
                            />
                            {isSearchExpanded && (
                                <button
                                    className="absolute inset-y-0 right-0 pr-3 flex items-center cursor-pointer text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                                    onClick={() => { setIsSearchExpanded(false); setNewsSearchQuery(""); }}
                                >
                                    <X className="h-3 w-3" />
                                </button>
                            )}
                        </div>

                        <ThemeToggle />
                        {isAuthenticated ? (
                            <div className="flex items-center gap-3">
                                <button
                                    onClick={() => {
                                        setStockQuery("");
                                        setStockResults([]);
                                        setIsViewModalOpen(true);
                                    }}
                                    className="px-4 h-10 bg-blue-600 hover:bg-blue-700 text-white rounded-xl flex items-center gap-2 transition-all duration-300 font-bold shadow-lg shadow-blue-500/25 dark:shadow-blue-900/40 hover:-translate-y-0.5 active:translate-y-0 whitespace-nowrap"
                                >
                                    <BarChart3 className="h-4 w-4" />
                                    <span>View Stock Details</span>
                                </button>

                                <button
                                    onClick={() => setIsWatchlistOpen(true)}
                                    className="px-6 h-10 bg-blue-600 hover:bg-blue-700 text-white rounded-xl flex items-center gap-3 transition-all duration-300 font-bold shadow-lg shadow-blue-500/25 dark:shadow-blue-900/40 hover:-translate-y-0.5 active:translate-y-0 group whitespace-nowrap"
                                >
                                    <Plus className="h-5 w-5 group-hover:rotate-90 transition-transform duration-500" />
                                    <span>Add Watchlist</span>
                                </button>

                                {/* Hamburger Menu */}
                                <div className="relative">
                                    <button
                                        onClick={() => setIsMenuOpen(!isMenuOpen)}
                                        className="w-10 h-10 flex items-center justify-center bg-white dark:bg-black/20 border border-slate-200 dark:border-white/10 rounded-xl text-slate-600 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 hover:bg-slate-50 dark:hover:bg-white/5 transition-all duration-300 shadow-sm"
                                    >
                                        <Menu className="h-6 w-6" />
                                    </button>

                                    <div className={`absolute top-full right-0 mt-3 w-64 bg-white dark:bg-gray-900 border border-slate-200 dark:border-white/10 rounded-2xl shadow-2xl z-[70] overflow-hidden py-2 transition-all duration-300 origin-top-right ${isMenuOpen ? 'opacity-100 scale-100 translate-y-0 visible' : 'opacity-0 scale-95 -translate-y-2 invisible pointer-events-none'}`}>
                                        {/* User Info */}
                                        <div className="px-5 py-4 border-b border-slate-100 dark:border-white/5 flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-blue-600 to-blue-400 flex items-center justify-center text-white font-bold text-lg shadow-lg">
                                                {userName?.charAt(0).toUpperCase()}
                                            </div>
                                            <div className="flex flex-col overflow-hidden">
                                                <span className="text-sm font-bold text-slate-900 dark:text-white truncate">Hello, {userName}</span>
                                                <span className="text-[10px] text-slate-500 dark:text-gray-400 uppercase tracking-widest font-bold">Pro Account</span>
                                            </div>
                                        </div>

                                        {/* Menu Items */}
                                        <div className="p-2 space-y-1">
                                            <div className="px-3 py-2 flex items-center justify-between hover:bg-slate-50 dark:hover:bg-white/5 rounded-xl transition-colors group">
                                                <div className="flex items-center gap-3">
                                                    {localStorage.getItem('theme') === 'dark' ? <Moon className="h-4 w-4 text-purple-400" /> : <Sun className="h-4 w-4 text-orange-400" />}
                                                    <span className="text-sm font-semibold text-slate-600 dark:text-gray-400">Theme</span>
                                                </div>
                                                <ThemeToggle />
                                            </div>

                                            <button
                                                onClick={() => { navigate('/categories'); setIsMenuOpen(false); }}
                                                className="w-full px-3 py-2.5 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-white/5 rounded-xl transition-colors group text-slate-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400"
                                            >
                                                <Briefcase className="h-4 w-4 transition-transform group-hover:scale-110" />
                                                <span className="text-sm font-semibold">Customize Feed</span>
                                            </button>

                                            <button
                                                onClick={() => { setIsSettingsOpen(true); setIsMenuOpen(false); }}
                                                className="w-full px-3 py-2.5 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-white/5 rounded-xl transition-colors group text-slate-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400"
                                            >
                                                <Settings className="h-4 w-4 transition-transform group-hover:rotate-45" />
                                                <span className="text-sm font-semibold">Settings</span>
                                            </button>

                                            <div className="h-px bg-slate-100 dark:border-white/5 mx-3 my-2" />

                                            {/* Google Translate Wrapper */}
                                            <div className="px-3 py-2">
                                                <div id="google_translate_element" className="w-full"></div>
                                            </div>

                                            <div className="h-px bg-slate-100 dark:border-white/5 mx-3 my-2" />

                                            <button
                                                onClick={() => { onLogout(); setIsMenuOpen(false); }}
                                                className="w-full px-3 py-2.5 flex items-center gap-3 hover:bg-rose-50 dark:hover:bg-rose-900/20 rounded-xl transition-colors group text-rose-500"
                                            >
                                                <LogOut className="h-4 w-4" />
                                                <span className="text-sm font-semibold">Logout</span>
                                            </button>
                                        </div>
                                    </div>
                                    {isMenuOpen && <div className="fixed inset-0 z-[60]" onClick={() => setIsMenuOpen(false)} />}
                                </div>
                            </div>
                        ) : (
                            <div className="flex items-center gap-2">
                                <Link
                                    to="/login"
                                    className="px-6 h-10 bg-blue-600 hover:bg-blue-700 text-white rounded-xl flex items-center justify-center text-xs font-bold transition-all shadow-lg shadow-blue-500/25 active:scale-95"
                                >
                                    Login
                                </Link>
                                <Link
                                    to="/signup"
                                    className="px-6 h-10 flex items-center justify-center rounded-xl bg-blue-600 text-white hover:bg-blue-700 transition-all shadow-lg shadow-blue-500/20 text-xs font-bold"
                                >
                                    Sign Up
                                </Link>
                            </div>
                        )}
                    </div>
                </div>

                {/* Sentiment Legend */}
                <div className="mb-4 bg-white/60 dark:bg-gray-900/40 backdrop-blur-xl border border-slate-200 dark:border-white/10 rounded-xl p-3 shadow-sm">
                    <div className="flex flex-wrap items-center gap-4">
                        <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-slate-600 dark:text-gray-400 uppercase tracking-wider">
                                Sentiment Analysis:
                            </span>
                        </div>

                        <div className="flex flex-wrap items-center gap-4">
                            {/* Positive */}
                            <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-50 dark:bg-emerald-900/20 border-2 border-emerald-500 rounded-lg">
                                <ArrowUpRight className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                                <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-300">Positive</span>
                                <span className="text-[10px] text-emerald-600/70 dark:text-emerald-400/70">Bullish News</span>
                            </div>

                            {/* Neutral */}
                            <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 dark:bg-gray-800/40 border border-slate-300 dark:border-white/20 rounded-lg">
                                <Minus className="w-3.5 h-3.5 text-slate-600 dark:text-gray-400" />
                                <span className="text-xs font-semibold text-slate-700 dark:text-gray-300">Neutral</span>
                                <span className="text-[10px] text-slate-600/70 dark:text-gray-400/70">Informational</span>
                            </div>

                            {/* Negative */}
                            <div className="flex items-center gap-2 px-3 py-1.5 bg-rose-50 dark:bg-rose-900/20 border-2 border-rose-500 rounded-lg">
                                <ArrowDownRight className="w-3.5 h-3.5 text-rose-600 dark:text-rose-400" />
                                <span className="text-xs font-semibold text-rose-700 dark:text-rose-300">Negative</span>
                                <span className="text-[10px] text-rose-600/70 dark:text-rose-400/70">Bearish News</span>
                            </div>
                        </div>


                    </div>
                </div>

                {/* Today at a Glance Section */}
                {
                    isAuthenticated && (
                        <div className="mt-4 mb-2">
                            <div className="bg-white/40 dark:bg-gray-900/40 backdrop-blur-xl border border-slate-200 dark:border-white/10 rounded-2xl p-4 flex items-center justify-between shadow-sm">
                                <div className="flex items-center gap-6">
                                    <span className="text-xs font-bold text-slate-500 dark:text-gray-500 uppercase tracking-widest">Today at a Glance</span>
                                    <div className="h-4 w-[1px] bg-slate-200 dark:bg-white/10"></div>
                                    <div className="flex items-center gap-2 text-sm font-semibold">
                                        <span className="text-slate-500 dark:text-gray-400">Market Mood:</span>
                                        {marketData?.indices ? (
                                            (() => {
                                                const indices = Object.values(marketData.indices);
                                                const positiveCount = indices.filter(d => d.change >= 0).length;
                                                const isBullish = positiveCount >= indices.length / 2;
                                                return (
                                                    <span className={`flex items-center gap-1.5 ${isBullish ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                                                        <span className={`w-2 h-2 rounded-full ${isBullish ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]'}`}></span>
                                                        {isBullish ? 'Bullish' : 'Bearish'}
                                                    </span>
                                                );
                                            })()
                                        ) : (
                                            <span className="text-slate-400 animate-pulse text-xs italic">Loading...</span>
                                        )}
                                    </div>
                                    {marketData?.indices && Object.entries(marketData.indices).map(([key, data]) => (
                                        <div key={key} className="flex items-center gap-2 text-sm font-semibold">
                                            <span className="text-slate-500 dark:text-gray-400 uppercase text-[10px] tracking-tighter">{key}:</span>
                                            <span className={data.change >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}>
                                                {data.change >= 0 ? '▲' : '▼'} {data.price.toLocaleString()}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )
                }


                {/* News Grid */}
                <div className="flex-1 overflow-y-auto min-h-0 pr-2 pb-6 -mr-2 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-gray-800 scrollbar-track-transparent flex flex-col">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 mb-8">
                        {/* Loading State Overlay */}
                        {(loadingNews && news.length > 0) && (
                            <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/10 dark:bg-black/10 backdrop-blur-[1px] pointer-events-none">
                                <div className="bg-white dark:bg-gray-900 px-4 py-2 rounded-full shadow-lg border border-slate-200 dark:border-white/10 flex items-center gap-2">
                                    <RefreshCw className="w-4 h-4 animate-spin text-blue-500" />
                                    <span className="text-xs font-bold">Updating...</span>
                                </div>
                            </div>
                        )}

                        {(news.length === 0 || loadingNews && news.length === 0) && Array.from({ length: 9 }).map((_, i) => (
                            <div key={i} className="h-64 bg-slate-200 dark:bg-gray-800/40 rounded-2xl animate-pulse border border-slate-200 dark:border-white/5"></div>
                        ))}

                        {currentNews.map((item, idx) => {
                            if (!item) return null;
                            return (
                                <div key={idx} className={`group bg-white dark:bg-gray-900/40 backdrop-blur-md rounded-xl p-3 hover:bg-slate-50 dark:hover:bg-gray-800/60 transition-all duration-300 hover:-translate-y-1 flex flex-col h-full relative overflow-hidden ${getSentimentStyle(item.sentiment)}`}>
                                    <div className="relative h-32 rounded-lg overflow-hidden mb-3 shrink-0 bg-slate-100 dark:bg-gray-800">
                                        <img
                                            src={item.image_url || '/default-news.png'}
                                            onError={handleImageError}
                                            alt="News"
                                            className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110 opacity-90 group-hover:opacity-100"
                                        />



                                        <div className="absolute inset-0 bg-gradient-to-t from-slate-900/20 dark:from-gray-900/80 via-transparent to-transparent opacity-60" />

                                        {item.views > 15 && (
                                            <div className="absolute top-3 left-3 bg-orange-500 text-white text-[9px] font-black px-2 py-0.5 rounded-full tracking-tighter shadow-xl flex items-center gap-1 animate-pulse">
                                                <Flame className="w-3 h-3 fill-current" /> TRENDING
                                            </div>
                                        )}

                                        {isAuthenticated && dateFilter === 'for_me' && (
                                            <div className="absolute bottom-3 left-3 bg-blue-600 text-white text-[9px] font-black px-2 py-0.5 rounded tracking-tighter shadow-xl">
                                                FOR YOU
                                            </div>
                                        )}


                                    </div>

                                    <div className="px-2 pb-2 flex-1 flex flex-col">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-slate-500 dark:text-gray-500 text-[10px] font-mono flex items-center gap-3">
                                                <span className="flex items-center gap-1">
                                                    TIMELINE <span className="w-1 h-1 bg-slate-400 dark:bg-gray-600 rounded-full" /> {formatDate(item.timestamp)}
                                                </span>
                                                <span className="flex items-center gap-1" title="Views">
                                                    <Eye className="w-3 h-3" /> {item.views || 0}
                                                </span>
                                            </span>
                                            {isAuthenticated && (
                                                <button
                                                    onClick={() => {
                                                        if (watchlist.includes(item.headline)) {
                                                            setWatchlist(watchlist.filter(id => id !== item.headline));
                                                        } else {
                                                            setWatchlist([...watchlist, item.headline]);
                                                        }
                                                    }}
                                                    className={`transition-colors ${watchlist.includes(item.headline) ? 'text-blue-600 dark:text-blue-400' : 'text-slate-400 hover:text-blue-500 dark:text-gray-500 dark:hover:text-blue-400'}`}
                                                >
                                                    <Bookmark className={`w-4 h-4 ${watchlist.includes(item.headline) ? 'fill-current' : ''}`} />
                                                </button>
                                            )}
                                        </div>

                                        <h3 className="font-semibold text-base leading-snug text-slate-800 dark:text-gray-100 group-hover:text-blue-600 dark:group-hover:text-blue-100 transition-colors mb-3 line-clamp-3">
                                            {item.headline}
                                        </h3>

                                        <div className="mt-auto flex justify-end">
                                            <a
                                                href={item.link}
                                                onClick={(e) => handleNewsClick(e, item)}
                                                className="flex items-center gap-1 text-xs font-bold text-blue-600 dark:text-blue-400 group-hover:text-blue-700 dark:group-hover:text-blue-300 transition-colors pointer-events-auto cursor-pointer"
                                            >
                                                READ MORE <ExternalLink className="w-3 h-3" />
                                            </a>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Pagination */}
                    {
                        totalPages > 1 && (
                            <div className="flex justify-center pb-20 pt-4 gap-2">
                                <button
                                    onClick={() => paginate(currentPage - 1)}
                                    disabled={currentPage === 1}
                                    className="px-4 py-2 bg-white dark:bg-gray-800 border border-slate-200 dark:border-white/10 rounded-lg text-xs font-bold text-slate-500 dark:text-gray-400 hover:bg-slate-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                                >
                                    Previous
                                </button>

                                <div className="flex gap-1">
                                    {(() => {
                                        const pages = [];
                                        if (totalPages <= 7) {
                                            for (let i = 1; i <= totalPages; i++) pages.push(i);
                                        } else {
                                            if (currentPage <= 4) {
                                                for (let i = 1; i <= 5; i++) pages.push(i);
                                                pages.push('...');
                                                pages.push(totalPages);
                                            } else if (currentPage >= totalPages - 3) {
                                                pages.push(1);
                                                pages.push('...');
                                                for (let i = totalPages - 4; i <= totalPages; i++) pages.push(i);
                                            } else {
                                                pages.push(1);
                                                pages.push('...');
                                                for (let i = currentPage - 1; i <= currentPage + 1; i++) pages.push(i);
                                                pages.push('...');
                                                pages.push(totalPages);
                                            }
                                        }

                                        return pages.map((page, i) => (
                                            <button
                                                key={i}
                                                onClick={() => typeof page === 'number' && paginate(page)}
                                                disabled={typeof page !== 'number'}
                                                className={`w-8 h-8 flex items-center justify-center rounded-lg text-xs font-bold transition-all border ${page === currentPage
                                                    ? 'bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-500/20'
                                                    : typeof page === 'number'
                                                        ? 'bg-white dark:bg-gray-800 border-slate-200 dark:border-white/10 text-slate-500 dark:text-gray-400 hover:bg-slate-50 dark:hover:bg-gray-700 hover:text-slate-900 dark:hover:text-white'
                                                        : 'border-transparent text-slate-400 dark:text-gray-500 cursor-default'
                                                    }`}
                                            >
                                                {page}
                                            </button>
                                        ));
                                    })()}
                                </div>

                                <button
                                    onClick={() => paginate(currentPage + 1)}
                                    disabled={currentPage === totalPages}
                                    className="px-4 py-2 bg-white dark:bg-gray-800 border border-slate-200 dark:border-white/10 rounded-lg text-xs font-bold text-slate-500 dark:text-gray-400 hover:bg-slate-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                                >
                                    Next
                                </button>
                            </div>
                        )
                    }


                    {/* Footer */}
                    <footer className="mt-8 border-t border-slate-200 dark:border-white/5 pt-8 pb-4">
                        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
                            <div className="text-center md:text-left">
                                <h4 className="font-display font-bold text-lg text-slate-900 dark:text-white mb-1">MarketPulse AI</h4>
                                <p className="text-xs text-slate-500 dark:text-gray-500">© 2026 MarketPulse AI. All rights reserved.</p>
                            </div>

                            <div className="flex flex-col items-center md:items-end gap-3">
                                <span className="text-[10px] font-bold uppercase tracking-widest text-blue-600 dark:text-blue-400">Developed By</span>
                                <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                                    {[
                                        { name: "Prashant Gupta", linkedin: "https://www.linkedin.com/in/prashant-gtx/" },
                                        { name: "Aniruddha Dawkhare", linkedin: "https://www.linkedin.com/in/aniruddha-dawkhare-38a86928a/" },
                                        { name: "Yashodhan Agashe", linkedin: "https://www.linkedin.com/in/yashodhan-agashe-7969b1289/" },
                                        { name: "Ayushi Punde", linkedin: "https://www.linkedin.com/in/ayushi-punde-9520a0366/" },
                                        { name: "Rujali Nagbhidkar", linkedin: "https://www.linkedin.com/in/rujali-nagbhidkar/" },
                                        { name: "Priyanka Mankar", linkedin: "https://www.linkedin.com/in/priyanka-mankar-28b251379/" },
                                    ].map((member, i) => (
                                        <a
                                            key={i}
                                            href={member.linkedin}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="flex items-center gap-2 bg-slate-100 dark:bg-gray-800/50 hover:bg-blue-500/5 dark:hover:bg-blue-600/20 border border-slate-200 dark:border-white/5 hover:border-blue-500/30 px-3 py-1.5 rounded-lg transition-all group"
                                        >
                                            <Linkedin className="w-3 h-3 text-slate-400 dark:text-gray-400 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors" />
                                            <span className="text-[10px] font-medium text-slate-600 dark:text-gray-300 group-hover:text-blue-900 dark:group-hover:text-blue-100">{member.name}</span>
                                        </a>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </footer>
                </div>
            </main >

            {/* Settings Modal */}
            {
                isSettingsOpen && (
                    <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
                        <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-md" onClick={() => setIsSettingsOpen(false)}></div>
                        <div className="relative w-full max-w-lg bg-white dark:bg-[#0F1115] border border-slate-200 dark:border-white/10 rounded-3xl shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
                            <div className="p-6 border-b border-slate-200 dark:border-white/10 flex justify-between items-center rounded-t-3xl bg-white dark:bg-[#0F1115]">
                                <h3 className="text-xl font-bold text-slate-900 dark:text-white">Settings</h3>
                                <button onClick={() => setIsSettingsOpen(false)} className="p-2 hover:bg-slate-100 dark:hover:bg-white/5 rounded-full transition-colors">
                                    <X className="w-5 h-5 text-slate-500" />
                                </button>
                            </div>
                            <div className="p-6 space-y-6">
                                <div className="space-y-4">
                                    <label className="text-xs font-bold uppercase tracking-widest text-slate-500">Account Preferences</label>
                                    <div className="p-4 bg-slate-50 dark:bg-white/5 rounded-2xl border border-slate-200 dark:border-white/5 space-y-3">
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <p className="text-sm font-bold text-slate-900 dark:text-gray-100">Display Name</p>
                                                {isEditingName ? (
                                                    <input
                                                        type="text"
                                                        value={tempName}
                                                        onChange={(e) => setTempName(e.target.value)}
                                                        className="mt-1 bg-white dark:bg-black/40 border border-slate-200 dark:border-white/10 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 w-full"
                                                        autoFocus
                                                    />
                                                ) : (
                                                    <p className="text-xs text-slate-500 dark:text-gray-500">{userName || 'User'}</p>
                                                )}
                                            </div>
                                            <div className="flex gap-2">
                                                {isEditingName ? (
                                                    <>
                                                        <button onClick={handleSaveName} className="text-xs font-bold text-emerald-600 hover:text-emerald-500">Save</button>
                                                        <button onClick={() => { setIsEditingName(false); setTempName(userName); }} className="text-xs font-bold text-slate-500 hover:text-slate-400">Cancel</button>
                                                    </>
                                                ) : (
                                                    <button onClick={() => setIsEditingName(true)} className="text-xs font-bold text-blue-600 hover:text-blue-500">Edit</button>
                                                )}
                                            </div>
                                        </div>

                                        <div className="flex items-center justify-between pt-3 border-t border-slate-200 dark:border-white/5">
                                            <div>
                                                <p className="text-sm font-bold text-slate-900 dark:text-gray-100">Date of Birth</p>
                                                <p className="text-xs text-slate-500 dark:text-gray-500">{userDob || 'Not set'}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div className="p-6 bg-slate-50 dark:bg-white/5 border-t border-slate-200 dark:border-white/10 flex justify-end">
                                <button onClick={() => setIsSettingsOpen(false)} className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl shadow-lg transition-all active:scale-95">
                                    Done
                                </button>
                            </div>
                        </div>
                    </div>
                )
            }

            {/* View Stock Modal */}
            {
                isViewModalOpen && (
                    <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
                        <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-md" onClick={() => setIsViewModalOpen(false)}></div>
                        <div className="relative w-full max-w-lg bg-white dark:bg-[#0F1115] border border-slate-200 dark:border-white/10 rounded-3xl shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
                            <div className="p-6 border-b border-slate-200 dark:border-white/10 flex justify-between items-center rounded-t-3xl bg-white dark:bg-[#0F1115]">
                                <h3 className="text-xl font-bold text-slate-900 dark:text-white">View Stock Details</h3>
                                <button onClick={() => setIsViewModalOpen(false)} className="p-2 hover:bg-slate-100 dark:hover:bg-white/5 rounded-full transition-colors">
                                    <X className="w-5 h-5 text-slate-500" />
                                </button>
                            </div>

                            <div className="p-6">
                                <div className="relative group mb-4">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <Search className="h-4 w-4 text-slate-400 group-focus-within:text-blue-500 transition-colors" />
                                    </div>
                                    <input
                                        type="text"
                                        value={stockQuery}
                                        onChange={(e) => handleStockSearch(e.target.value)}
                                        placeholder="Search company (e.g. Tata, Infosys)..."
                                        className="block w-full pl-10 pr-4 py-3 bg-slate-100 dark:bg-black/20 border border-slate-200 dark:border-white/5 rounded-2xl text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all text-sm"
                                        autoFocus
                                    />
                                </div>

                                {stockResults.length > 0 && (
                                    <div className="bg-white dark:bg-[#1A1D23] border border-slate-200 dark:border-white/10 rounded-2xl shadow-sm overflow-hidden max-h-[50vh] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-200 dark:scrollbar-thumb-gray-800">
                                        {stockResults.map((stock, i) => (
                                            <div
                                                key={i}
                                                onClick={() => {
                                                    navigate(`/stock/${stock.symbol}`);
                                                    setIsViewModalOpen(false);
                                                }}
                                                className="p-4 hover:bg-slate-50 dark:hover:bg-white/5 cursor-pointer border-b last:border-0 border-slate-100 dark:border-white/5 transition-colors group flex justify-between items-center"
                                            >
                                                <div className="flex-1">
                                                    <p className="font-bold text-sm text-slate-900 dark:text-gray-100 group-hover:text-blue-600 transition-colors">{stock.name}</p>
                                                    <p className="text-[10px] text-slate-500 uppercase tracking-tighter">NSE: {stock.symbol}</p>
                                                </div>
                                                <div className="p-2 bg-slate-100 dark:bg-white/5 rounded-lg">
                                                    <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-blue-600 transition-colors" />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {!stockQuery && selectedStocks.length > 0 && (
                                    <div className="mt-4">
                                        <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">Your Watchlist</h4>
                                        <div className="space-y-2 max-h-[40vh] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-200 dark:scrollbar-thumb-gray-800 pr-1">
                                            {selectedStocks.map((stock, i) => (
                                                <div
                                                    key={i}
                                                    onClick={() => {
                                                        navigate(`/stock/${stock.symbol}`);
                                                        setIsViewModalOpen(false);
                                                    }}
                                                    className="p-3 bg-slate-50 dark:bg-white/5 border border-slate-100 dark:border-white/5 rounded-xl hover:border-blue-500/50 hover:shadow-md cursor-pointer transition-all group flex justify-between items-center"
                                                >
                                                    <div>
                                                        <p className="font-bold text-sm text-slate-900 dark:text-gray-100 group-hover:text-blue-600 transition-colors">{stock.name}</p>
                                                        <p className="text-[10px] text-slate-500 uppercase">{stock.symbol}</p>
                                                    </div>
                                                    <ArrowRight className="w-4 h-4 text-slate-300 group-hover:text-blue-500 transition-colors" />
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {stockQuery && stockResults.length === 0 && (
                                    <div className="text-center py-8 text-slate-400 text-sm">
                                        No stocks found matching "{stockQuery}"
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )
            }

            {/* Watchlist Modal */}
            {
                isWatchlistOpen && (
                    <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
                        <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-md" onClick={() => setIsWatchlistOpen(false)}></div>
                        <div className="relative w-full max-w-lg bg-white dark:bg-[#0F1115] border border-slate-200 dark:border-white/10 rounded-3xl shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
                            <div className="p-6 border-b border-slate-200 dark:border-white/10 flex justify-between items-center rounded-t-3xl bg-white dark:bg-[#0F1115]">
                                <h3 className="text-xl font-bold text-slate-900 dark:text-white">Watchlist</h3>
                                <button onClick={() => setIsWatchlistOpen(false)} className="p-2 hover:bg-slate-100 dark:hover:bg-white/5 rounded-full transition-colors">
                                    <X className="w-5 h-5 text-slate-500" />
                                </button>
                            </div>

                            {/* Search Section - Fixed at top, z-index high to overlap list */}
                            <div className="p-6 pb-0 relative z-[220]">
                                <div className="relative group">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <Search className="h-4 w-4 text-slate-400 group-focus-within:text-blue-500 transition-colors" />
                                    </div>
                                    <input
                                        type="text"
                                        value={stockQuery}
                                        onChange={(e) => handleStockSearch(e.target.value)}
                                        placeholder="Search company (e.g. Reliance, HDFC, Infosys)"
                                        className="block w-full pl-10 pr-4 py-3 bg-slate-100 dark:bg-black/20 border border-slate-200 dark:border-white/5 rounded-2xl text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all text-sm"
                                    />
                                </div>

                                {stockResults.length > 0 && (
                                    <div className="absolute left-6 right-6 z-[1000] mt-2 bg-white dark:bg-[#1A1D23] border border-slate-200 dark:border-white/10 rounded-2xl shadow-2xl overflow-y-auto max-h-[60vh] animate-in fade-in slide-in-from-top-2 duration-200 scrollbar-thin scrollbar-thumb-slate-200 dark:scrollbar-thumb-gray-800">
                                        {stockResults.map((stock, i) => (
                                            <div
                                                key={i}
                                                className="p-4 hover:bg-slate-50 dark:hover:bg-white/5 border-b last:border-0 border-slate-100 dark:border-white/5 transition-colors group flex justify-between items-center bg-white dark:bg-[#1A1D23]"
                                            >
                                                <div className="flex-1 pointer-events-none">
                                                    <p className="font-bold text-sm text-slate-900 dark:text-gray-100 group-hover:text-blue-600 transition-colors">{stock.name}</p>
                                                    <p className="text-[10px] text-slate-500 uppercase tracking-tighter">NSE: {stock.symbol}</p>
                                                </div>
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        addStockToWatchlist(stock);
                                                    }}
                                                    disabled={addingStock === stock.symbol}
                                                    className="ml-2 p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 text-white rounded-lg transition-colors shadow-lg shadow-blue-500/30 flex items-center justify-center min-w-[32px] min-h-[32px]"
                                                >
                                                    {addingStock === stock.symbol ? (
                                                        <RefreshCw className="w-4 h-4 animate-spin" />
                                                    ) : (
                                                        <Plus className="w-4 h-4" />
                                                    )}
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {/* Scrollable List Area */}
                            <div className="p-6 space-y-6 max-h-[60vh] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-200 dark:scrollbar-thumb-gray-800">
                                <div className="space-y-4">
                                    <div className="flex flex-col gap-3">
                                        {selectedStocks.map((stock, i) => (
                                            <div key={i} className="flex items-center justify-between p-3 bg-slate-50 dark:bg-white/5 border border-slate-200 dark:border-white/5 rounded-xl animate-in zoom-in duration-200">
                                                <div className="flex flex-col">
                                                    <span className="text-sm font-bold text-slate-900 dark:text-white">{stock.name}</span>
                                                    <span className="text-[10px] text-slate-500 uppercase font-mono">{stock.symbol}</span>
                                                </div>

                                                <div className="flex items-center gap-2">
                                                    <button
                                                        onClick={() => removeStockFromWatchlist(stock.symbol)}
                                                        className="p-1.5 hover:bg-rose-500/10 text-slate-400 hover:text-rose-500 rounded-lg transition-colors"
                                                        title="Remove from Watchlist"
                                                    >
                                                        <X className="w-4 h-4" />
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                        {selectedStocks.length === 0 && (
                                            <p className="text-xs text-slate-400 dark:text-gray-600 italic py-2">No stocks tracked yet</p>
                                        )}
                                    </div>
                                </div>
                            </div>
                            <div className="p-6 bg-slate-50 dark:bg-white/5 border-t border-slate-200 dark:border-white/10 flex justify-end">
                                <button onClick={() => setIsWatchlistOpen(false)} className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl shadow-lg transition-all active:scale-95">
                                    Done
                                </button>
                            </div>
                        </div>
                    </div>
                )
            }

            {/* News Summary Modal */}
            <NewsSummaryModal
                isOpen={summaryModalOpen}
                onClose={() => setSummaryModalOpen(false)}
                article={selectedArticle}
                summary={articleSummary}
                isLoading={loadingSummary}
            />

            {/* Floating Chat Interface */}
            <div className="fixed bottom-8 right-8 z-[100] flex flex-col items-end pointer-events-none">
                <div className={`
          mb-4 w-[400px] md:w-[550px] h-[600px] md:h-[750px] max-h-[80vh] bg-white/95 dark:bg-[#0F1115]/95 backdrop-blur-xl border border-slate-200 dark:border-white/10 rounded-3xl shadow-2xl flex flex-col overflow-hidden transition-all duration-300 origin-bottom-right pointer-events-auto
          ${isChatOpen ? 'opacity-100 scale-100 translate-y-0' : 'opacity-0 scale-95 translate-y-10 pointer-events-none'}
        `}>
                    <div className="flex items-center justify-between p-4 bg-gradient-to-r from-blue-500/5 to-purple-500/5 dark:from-blue-900/30 dark:to-purple-900/30 border-b border-slate-200 dark:border-white/10 shrink-0">
                        <div className="flex items-center gap-3">
                            <div className="relative">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/20">
                                    <MessageSquare className="w-4 h-4 text-white" />
                                </div>
                                <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 bg-emerald-500 border-2 border-white dark:border-[#0F1115] rounded-full"></div>
                            </div>
                            <div>
                                <h3 className="font-bold text-sm text-slate-800 dark:text-gray-100">MarketPulse Assistant</h3>
                                <p className="text-[10px] text-blue-600 dark:text-blue-300/80">Powered by Ollama (Llama 3.2)</p>
                            </div>
                        </div>
                        <button onClick={() => setChatHistory([])} className="p-2 hover:bg-slate-100 dark:hover:bg-white/5 rounded-full transition-colors text-slate-400 dark:text-gray-400 hover:text-blue-500 hover:bg-blue-50 dark:hover:text-blue-400" title="Clear Chat">
                            <RefreshCw className="w-4 h-4" />
                        </button>
                        <button onClick={() => setIsChatOpen(false)} className="p-2 hover:bg-slate-100 dark:hover:bg-white/5 rounded-full transition-colors text-slate-400 dark:text-gray-400 hover:text-slate-900 dark:hover:text-white">
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-200 dark:scrollbar-thumb-gray-800 scrollbar-track-transparent bg-gradient-to-b from-transparent to-slate-50 dark:to-black/20">
                        {
                            chatHistory.length === 0 && (
                                <div className="flex flex-col items-center justify-center h-full text-center p-6 space-y-4">
                                    <div className="w-16 h-16 bg-blue-500/10 rounded-2xl flex items-center justify-center rotate-3 border border-blue-500/20">
                                        <TrendingUp className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                                    </div>
                                    <div>
                                        <h4 className="text-slate-800 dark:text-gray-200 font-medium mb-1">How can I help you?</h4>
                                        <p className="text-xs text-slate-500 dark:text-gray-500 mb-4">Ask me anything about the Indian Stock Market.</p>
                                    </div>
                                </div>
                            )
                        }

                        {
                            chatHistory.map((msg, idx) => (
                                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                                    <div className={`
                  max-w-[85%] p-3.5 rounded-2xl text-xs leading-relaxed shadow-sm
                  ${msg.role === 'user'
                                            ? 'bg-blue-600 text-white rounded-tr-sm shadow-blue-500/10'
                                            : 'bg-slate-100 dark:bg-gray-800/80 border border-slate-200 dark:border-white/5 text-slate-800 dark:text-gray-200 rounded-tl-sm backdrop-blur-sm'}
                `}>
                                        {/* Use ReactMarkdown with prose class for formatting */}
                                        <div className="prose dark:prose-invert prose-sm max-w-none text-xs marker:text-blue-500">
                                            <ReactMarkdown
                                                components={{
                                                    strong: ({ node, ...props }) => <span className="font-bold text-slate-900 dark:text-white" {...props} />,
                                                    ul: ({ node, ...props }) => <ul className="list-disc pl-4 space-y-1 my-2" {...props} />,
                                                    li: ({ node, ...props }) => <li className="pl-1" {...props} />,
                                                    p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />
                                                }}
                                            >
                                                {msg.content}
                                            </ReactMarkdown>
                                        </div>
                                    </div>
                                </div>
                            ))
                        }

                        {
                            loadingChat && (
                                <div className="flex justify-start">
                                    <div className="bg-slate-100 dark:bg-gray-800/50 rounded-2xl p-3 flex gap-1.5 items-center">
                                        <div className="w-1.5 h-1.5 bg-blue-500 dark:bg-blue-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                                        <div className="w-1.5 h-1.5 bg-blue-500 dark:bg-blue-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                                        <div className="w-1.5 h-1.5 bg-blue-500 dark:bg-blue-400 rounded-full animate-bounce"></div>
                                    </div>
                                </div>
                            )
                        }
                        <div ref={chatEndRef} />
                    </div>

                    <div className="p-4 bg-white dark:bg-[#0F1115] border-t border-slate-200 dark:border-white/5 shrink-0">
                        <form onSubmit={handleChatSubmit} className="relative group">
                            <input
                                type="text"
                                value={chatQuery}
                                onChange={(e) => setChatQuery(e.target.value)}
                                placeholder="Type your message..."
                                className="w-full bg-slate-50 dark:bg-gray-900 border border-slate-200 dark:border-gray-800 rounded-xl pl-4 pr-12 py-3 text-xs text-slate-800 dark:text-white placeholder-slate-400 dark:placeholder-gray-500 focus:outline-none focus:border-blue-500/50 transition-all"
                            />
                            <button
                                type="submit"
                                disabled={loadingChat || !chatQuery.trim()}
                                className="absolute right-2 top-2 p-1.5 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg text-white disabled:opacity-50 hover:shadow-lg hover:shadow-blue-500/20 transition-all hover:scale-105 active:scale-95"
                            >
                                <Send className="w-4 h-4" />
                            </button>
                        </form>
                    </div>
                </div>

                <button
                    onClick={() => setIsChatOpen(!isChatOpen)}
                    className={`
            pointer-events-auto p-0 w-16 h-16 rounded-full shadow-2xl flex items-center justify-center transition-all duration-300 hover:scale-110 active:scale-95 border border-white/10 relative overflow-hidden group
            ${isChatOpen ? 'bg-slate-800 dark:bg-gray-800 rotate-90' : 'bg-transparent'}
            `}
                >
                    {!isChatOpen && (
                        <div className="absolute inset-0 bg-gradient-to-br from-blue-600 to-purple-600 opacity-100 group-hover:opacity-90 transition-opacity" />
                    )}

                    <div className="relative z-10 text-white">
                        {isChatOpen ? <X className="w-7 h-7" /> : <MessageSquare className="w-7 h-7" />}
                    </div>

                    {!isChatOpen && <div className="absolute inset-0 bg-blue-500 blur-xl opacity-30 dark:opacity-50 -z-10" />}
                </button>
            </div>
        </div >
    );
}

export default Dashboard;
