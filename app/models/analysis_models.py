from sqlalchemy import Column, String, Float, Integer, Text, DateTime, Numeric
from app.db import Base

class StockSectorAnalysis(Base):
    __tablename__ = 'stock_sector_analysis'

    sector = Column(String, primary_key=True)
    investment_count = Column(Integer)
    investment_types = Column(String)
    avg_return_pct = Column(Float)
    avg_price = Column(Float)
    volatility = Column(Float)
    top_performers = Column(String)  # Store as string instead of JSON
    momentum_score = Column(Float)

class ComprehensiveSectorAnalysis(Base):
    __tablename__ = 'comprehensive_sector_analysis'

    sector = Column(String, primary_key=True)
    investment_count = Column(Integer)
    investment_types = Column(String)
    avg_return_pct = Column(Float)
    avg_price = Column(Float)
    volatility = Column(Float)
    top_performers = Column(String)  # Store as string instead of JSON
    momentum_score = Column(Float)

class EnhancedSectorScores(Base):
    __tablename__ = 'enhanced_sector_scores'

    sector = Column(String, primary_key=True)
    asset_type = Column(String)
    avg_return_percent = Column(Float)  # Maps to avg_return_%
    avg_volatility_percent = Column(Float)  # Maps to avg_volatility_%
    avg_score = Column(Float)
    investments_analyzed = Column(Integer)
    top_performer = Column(String)

class MutualFund(Base):
    __tablename__ = 'mutual_funds'

    id = Column(Integer, primary_key=True)
    fund_name = Column(Text)
    fund_house = Column(Text)
    category = Column(Text)
    nav = Column(Numeric)
    returns_1y = Column(Numeric)
    returns_3y = Column(Numeric)
    risk_level = Column(Text)
    last_updated = Column(DateTime)

class GoldInvestment(Base):
    __tablename__ = 'gold_investments'

    id = Column(Integer, primary_key=True)
    type = Column(Text)
    issuer = Column(Text)
    price_per_gram = Column(Numeric)
    purity = Column(Text)
    returns_1y = Column(Numeric)
    last_updated = Column(DateTime)