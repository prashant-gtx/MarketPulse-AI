from sqlalchemy.orm import Session
from database import User, SentNotification, NewsAnalytics
from scraper import get_latest_news_raw 
from email_service import EmailService
from datetime import datetime
import logging
import re

class NotificationManager:
    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
        self.email_service = EmailService()

    def check_and_notify(self):
        print("Running Notification Check...")
        db = self.db_session_factory()
        try:
            users = db.query(User).all()
            if not users:
                return

            # Get latest news
            news_items = get_latest_news_raw()
            if not news_items:
                print("No news to check.")
                return
            
            for user in users:
                watchlist_items = user.watchlist
                if not watchlist_items:
                    continue
                
                watchlist_symbols = {item.symbol.lower() for item in watchlist_items}
                watchlist_names = {item.name.lower() for item in watchlist_items}
                
                matches = []
                
                for news in news_items:
                    link = news.get("link")
                    if not link: continue
                    
                    # Check if already sent
                    # Optimization: In a high-scale system, query all sent links for user once
                    sent_record = db.query(SentNotification).filter(
                        SentNotification.user_id == user.id,
                        SentNotification.article_link == link
                    ).first()
                    
                    if sent_record:
                        continue
                        
                    headline = news.get("headline", "").lower()
                    full_content = news.get("full_content", "").lower()
                    
                    # Search text
                    search_text = f"{headline} {full_content}"
                    
                    # Basic Match Logic
                    # Look for symbol or name in text
                    # Use regex word boundaries for symbols to avoid false positives (e.g. "IT" in "IT sector")
                    
                    is_match = False
                    
                    # Check Symbols
                    for sym in watchlist_symbols:
                        # Escape special chars in symbol
                        pattern = r'\b' + re.escape(sym) + r'\b'
                        if re.search(pattern, search_text):
                            is_match = True
                            break
                    
                    # Check Names if no symbol match
                    if not is_match:
                         for name in watchlist_names:
                            if len(name) < 4: continue # Skip short names to avoid noise
                            if name in search_text:
                                is_match = True
                                break
                    
                    if is_match:
                        matches.append(news)

                if matches:
                    self.send_notification(user, matches, db)
                    
        except Exception as e:
            print(f"Error in Notification Job: {e}")
        finally:
            db.close()

    def send_notification(self, user, articles, db):
        if not articles:
            return

        subject = f"MarketPulse AI: {len(articles)} new updates on your watchlist"
        
        # Simple HTML Email Template
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #2c3e50;">MarketPulse AI: Watchlist Update</h2>
            <p>Hi {user.name},</p>
            <p>We found new articles related to stocks in your watchlist:</p>
            <ul style="padding-left: 20px;">
        """
        
        sent_entries = []
        
        for article in articles:
            title = article.get("headline", "No Title")
            link = article.get("link", "#")
            
            # Longer summary: prefer description, fallback to first 250 chars of content
            raw_desc = article.get("description") or ""
            if len(raw_desc) < 50: # If description is too short/empty, use content
                raw_desc = article.get("full_content", "")
            
            snippet = raw_desc[:280] + "..." if len(raw_desc) > 280 else raw_desc
            
            # Metadata
            timestamp = article.get("timestamp", "Just now")
            sentiment = article.get("sentiment", "Neutral").capitalize()
            sentiment = article.get("sentiment", "Neutral").capitalize()
            
            # Color code sentiment
            sent_color = "#7f8c8d" # Gray
            if sentiment.lower() == "positive": sent_color = "#27ae60" # Green
            elif sentiment.lower() == "negative": sent_color = "#c0392b" # Red
            
            body += f"""
                <li style="margin-bottom: 25px; border-bottom: 1px solid #eee; padding-bottom: 15px;">
                    <div style="font-size: 16px; margin-bottom: 5px;">
                        <strong><a href='{link}' style="color: #2980b9; text-decoration: none;">{title}</a></strong>
                    </div>
                    
                    <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 8px;">
                        📅 {timestamp} &nbsp;|&nbsp; 
                        Sentiment: <span style="color: {sent_color}; font-weight: bold;">{sentiment}</span>
                    </div>
                    
                    <div style="color: #555; font-size: 14px;">
                        {snippet}
                    </div>
                </li>
            """
            
            sent_entries.append(SentNotification(
                user_id=user.id,
                article_link=link,
                sent_at=datetime.utcnow()
            ))
            
        body += """
            </ul>
            <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;"/>
            <p style="font-size: 12px; color: #888;">
                You are receiving this because you have stocks in your MarketPulse watchlist.<br/>
                MarketPulse AI © 2026
            </p>
        </body>
        </html>
        """
        
        # Send Email
        if self.email_service.send_email(user.email, subject, body):
            try:
                # Save sent notifications to DB
                db.bulk_save_objects(sent_entries)
                db.commit()
                print(f"Sent notification to {user.email} for {len(articles)} articles.")
            except Exception as e:
                print(f"Error saving sent notifications for {user.email}: {e}")
                db.rollback()


