import React, { useState } from 'react';
import { TrendingUp, Target, BarChart3, Coins, Building2, Award, ArrowRight, CheckCircle, ChevronDown } from 'lucide-react';
import axios from 'axios';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

interface InvestmentOption {
  id: string;
  name: string;
  icon: React.ComponentType<any>;
  description: string;
}

interface SectorOption {
  id: string;
  name: string;
}

interface DetailedInvestment {
  allocation_percent: number;
  allocation_amount: number;
  top_picks: any[];
  average_expected_return: number;
}

interface PredictionResult {
  recommendedAllocation: {
    stocks: number;
    mutualFunds: number;
    gold: number;
  };
  expectedReturn: number;
  riskScore: number;
  recommendations: string[];
  detailedInvestments?: {
    stocks?: DetailedInvestment;
    mutualFunds?: DetailedInvestment;
    gold?: DetailedInvestment;
  };
  backtestResults?: any;
  sectorAnalysis?: any;
}

interface ComprehensiveRecommendation {
  status: string;
  message: string;
  recommendations: {
    stocks: SectorRecommendation[];
    mutual_funds: SectorRecommendation[];
    gold: SectorRecommendation[];
  };
  timestamp: string;
}

interface SectorRecommendation {
  sector: string;
  predicted_return: number;
  ml_score: number;
  investment_count: number;
  avg_price: number;
  commentary: string;
  stocks?: StockRecommendation[];
  funds?: FundRecommendation[];
  gold_investments?: GoldRecommendation[];
}

interface StockRecommendation {
  symbol: string;
  company_name: string;
  current_performance: number;
  commentary: string;
  investment_strategy: string;
  timing_analysis: TimingAnalysis;
}

interface FundRecommendation {
  fund_name: string;
  full_name: string;
  current_performance: number;
  commentary: string;
  investment_strategy: string;
  investment_type: string;
  sip_recommended: boolean;
  monthly_sip_amount: number;
  sip_duration_months: number;
  total_investment_amount: number;
  entry_date: string;
  minimum_investment: number;
  expense_ratio: number;
  fund_manager: string;
  risk_level: string;
}

interface GoldRecommendation {
  investment_type: string;
  current_performance: number;
  commentary: string;
  investment_strategy: string;
  entry_date: string;
  recommended_allocation: number;
  holding_period_days: number;
  volatility: number;
  liquidity_rating: string;
  current_price: number;
  expected_return: number;
  risk_level: string;
  minimum_investment: number;
  storage_required: boolean;
  tax_implications: string;
}

interface TimingAnalysis {
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  expected_return: number;
  support_level: number;
  resistance_level: number;
  stop_loss: number;
  target_price: number;
  holding_period: number;
  volatility: number;
}

const investmentOptions: InvestmentOption[] = [
  {
    id: 'stocks',
    name: 'Stocks',
    icon: TrendingUp,
    description: 'Individual company shares with high growth potential'
  },
  {
    id: 'mutualFunds',
    name: 'Mutual Funds',
    icon: Building2,
    description: 'Diversified portfolios managed by professionals'
  },
  {
    id: 'gold',
    name: 'Gold',
    icon: Coins,
    description: 'Precious metal investment for portfolio stability'
  }
];

const sectorOptions: SectorOption[] = [
  { id: 'technology', name: 'Information Technology' },
  { id: 'finance', name: 'Banking & Finance' },
  { id: 'energy', name: 'Energy & Oil' },
  { id: 'healthcare', name: 'Healthcare & Pharma' },
  { id: 'automotive', name: 'Automotive' },
  { id: 'fmcg', name: 'FMCG & Consumer Goods' },
  { id: 'infrastructure', name: 'Infrastructure' },
  { id: 'metals', name: 'Metals & Mining' },
  { id: 'telecom', name: 'Telecommunications' },
  { id: 'realestate', name: 'Real Estate' }
];

function App() {
  // Basic Investment Inputs
  const [investmentAmount, setInvestmentAmount] = useState<string>('');
  const [riskAppetite, setRiskAppetite] = useState<string>('Medium');
  const [selectedSectors, setSelectedSectors] = useState<string[]>([]);
  
  // Time Horizon
  const [investmentDuration, setInvestmentDuration] = useState<string>('Medium-term');
  
  // Asset Preferences
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);
  
  // Strategy Preferences
  const [portfolioStrategy, setPortfolioStrategy] = useState<string>('Balanced Strategy');
  const [useMlPrediction, setUseMlPrediction] = useState<boolean>(true);
  
  // Date Range for Backtesting
  const [fromDate, setFromDate] = useState<string>('');
  const [toDate, setToDate] = useState<string>('');
  
  // Advanced Filters
  const [minRoiExpectation, setMinRoiExpectation] = useState<number>(8);
  const [excludeVolatileAssets, setExcludeVolatileAssets] = useState<boolean>(false);
  const [includeOnlyLiquidAssets, setIncludeOnlyLiquidAssets] = useState<boolean>(false);
  
  // Optional Sections Toggle
  const [showBacktesting, setShowBacktesting] = useState<boolean>(false);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState<boolean>(false);
  
  // Collapsible Sections for Results
  const [showStocksSection, setShowStocksSection] = useState<boolean>(true);
  const [showMutualFundsSection, setShowMutualFundsSection] = useState<boolean>(true);
  const [showGoldSection, setShowGoldSection] = useState<boolean>(true);
  
  // UI State
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [comprehensiveData, setComprehensiveData] = useState<ComprehensiveRecommendation | null>(null);

  const formatCurrency = (value: string) => {
    const numericValue = value.replace(/[^0-9]/g, '');
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(parseInt(numericValue) || 0);
  };

  const handleAmountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.replace(/[^0-9]/g, '');
    setInvestmentAmount(value);
  };

  const toggleOption = (optionId: string) => {
    setSelectedOptions(prev => 
      prev.includes(optionId) 
        ? prev.filter(id => id !== optionId)
        : [...prev, optionId]
    );
  };

  const toggleSector = (sectorId: string) => {
    setSelectedSectors(prev => 
      prev.includes(sectorId) 
        ? prev.filter(id => id !== sectorId)
        : [...prev, sectorId]
    );
  };

  const mapRiskToNumeric = (risk: string): number => {
    switch (risk) {
      case 'Low': return 3;
      case 'High': return 8;
      default: return 5; // Medium
    }
  };

  const runPrediction = async () => {
    if (!investmentAmount || selectedOptions.length === 0) return;

    setIsLoading(true);
    
    try {
      const requestData: any = {
        amount: parseFloat(investmentAmount),
        risk_tolerance: mapRiskToNumeric(riskAppetite),
        expected_return: minRoiExpectation,
        selected_options: selectedOptions,
        investment_duration: investmentDuration,
        portfolio_strategy: portfolioStrategy,
        use_ml_prediction: useMlPrediction,
        preferred_sectors: selectedSectors
      };

      // Only include advanced filters if the section is expanded
      if (showAdvancedFilters) {
        requestData.exclude_volatile_assets = excludeVolatileAssets;
        requestData.include_only_liquid_assets = includeOnlyLiquidAssets;
        requestData.min_roi_expectation = minRoiExpectation;
      }

      // Only include backtesting dates if the section is expanded and dates are provided
      if (showBacktesting && fromDate && toDate) {
        requestData.from_date = fromDate;
        requestData.to_date = toDate;
      }

      console.log('Sending request to API:', requestData);

      const response = await axios.post(`${API_BASE_URL}/api/portfolio/predict`, requestData, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      console.log('API Response:', response.data);

      const apiResult = response.data;
      
      const transformedPrediction: PredictionResult = {
        recommendedAllocation: {
          stocks: apiResult.recommended_allocation?.stocks || 0,
          mutualFunds: apiResult.recommended_allocation?.mutualFunds || 0,
          gold: apiResult.recommended_allocation?.gold || 0
        },
        expectedReturn: apiResult.expected_return || minRoiExpectation,
        riskScore: apiResult.risk_score || mapRiskToNumeric(riskAppetite),
        recommendations: apiResult.recommendations || [],
        detailedInvestments: apiResult.detailed_investments || {},
        backtestResults: apiResult.backtest_results || null,
        sectorAnalysis: apiResult.sector_analysis || null
      };

      setPrediction(transformedPrediction);
      
      // Store comprehensive data if available
      if (apiResult.detailed_investments) {
        setComprehensiveData({
          status: 'success',
          message: 'Comprehensive recommendations loaded',
          recommendations: {
            stocks: apiResult.detailed_investments.stocks?.comprehensive_data || [],
            mutual_funds: apiResult.detailed_investments.mutualFunds?.comprehensive_data || [],
            gold: apiResult.detailed_investments.gold?.comprehensive_data || []
          },
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      console.error('API Error:', error);
      
      // Fallback to mock data if API fails
      const riskMultiplier = mapRiskToNumeric(riskAppetite) / 10;
      const returnMultiplier = minRoiExpectation / 10;

      let stocksPercent = 0;
      let mutualFundsPercent = 0;
      let goldPercent = 0;

      if (selectedOptions.includes('stocks')) {
        stocksPercent = Math.min(70, 30 + (riskMultiplier * 40));
      }
      if (selectedOptions.includes('mutualFunds')) {
        mutualFundsPercent = Math.min(50, 20 + (returnMultiplier * 30));
      }
      if (selectedOptions.includes('gold')) {
        goldPercent = Math.min(30, 10 + ((10 - mapRiskToNumeric(riskAppetite)) * 2));
      }

      const total = stocksPercent + mutualFundsPercent + goldPercent;
      if (total > 0) {
        stocksPercent = (stocksPercent / total) * 100;
        mutualFundsPercent = (mutualFundsPercent / total) * 100;
        goldPercent = (goldPercent / total) * 100;
      }

      const fallbackPrediction: PredictionResult = {
        recommendedAllocation: {
          stocks: Math.round(stocksPercent),
          mutualFunds: Math.round(mutualFundsPercent),
          gold: Math.round(goldPercent)
        },
        expectedReturn: Math.round(minRoiExpectation * 0.9 + mapRiskToNumeric(riskAppetite) * 0.3),
        riskScore: mapRiskToNumeric(riskAppetite),
        recommendations: [
          `⚠️ Using fallback recommendations (API connection failed)`,
          `Based on your ${riskAppetite} risk appetite, we recommend a ${portfolioStrategy.toLowerCase()}`,
          `Your ${minRoiExpectation}% return expectation aligns with a diversified portfolio`,
          `Investment duration: ${investmentDuration} - Consider rebalancing as needed`
        ]
      };

      setPrediction(fallbackPrediction);
    } finally {
      setIsLoading(false);
    }
  };

  const isFormValid = investmentAmount && parseFloat(investmentAmount) > 0 && selectedOptions.length > 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-950 to-indigo-950">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          {/* User Profile Image */}
          <div className="mb-6">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gray-800/60 backdrop-blur-sm rounded-full border-2 border-purple-800/50 overflow-hidden">
              {/* Replace "/profile.jpg" with your actual image path */}
              <img 
                src="/profile.jpg" 
                alt="Profile" 
                className="w-full h-full object-cover"
                onError={(e) => {
                  // Fallback to icon if image fails to load
                  const target = e.target as HTMLImageElement;
                  target.style.display = 'none';
                  target.nextElementSibling?.classList.remove('hidden');
                }}
              />
              <BarChart3 className="w-8 h-8 text-purple-400 hidden" />
            </div>
          </div>
          
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Investment Advisor
          </h1>
          <p className="text-xl text-purple-300 max-w-2xl mx-auto leading-relaxed">
            Get personalized investment recommendations tailored to your financial goals and risk profile
          </p>
        </div>

        <div className="max-w-full mx-auto px-4">
          <div className="grid lg:grid-cols-6 gap-6">
            {/* Input Form - Narrower Left Column */}
            <div className="lg:col-span-1 bg-gray-900/40 backdrop-blur-lg rounded-2xl p-4 border border-purple-800/40 shadow-2xl h-fit sticky top-4">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Target className="w-4 h-4 mr-2 text-purple-400" />
                Parameters
              </h2>

              {/* 1. Investment Preferences */}
              <div className="mb-6">
                <h3 className="text-lg font-medium text-white mb-3">💰 Investment Preferences</h3>
                
                {/* Investment Amount */}
                <div className="mb-4">
                  <label className="block text-purple-300 text-sm font-medium mb-2">
                    Total Investment Amount (₹)
                  </label>
                  <div className="relative">
                    <span className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400">₹</span>
                    <input
                      type="text"
                      value={investmentAmount ? formatCurrency(investmentAmount) : ''}
                      onChange={handleAmountChange}
                      placeholder="₹1,00,000"
                      className="w-full pl-8 pr-4 py-3 bg-gray-800/50 border border-purple-800/50 rounded-xl text-white placeholder-purple-400 focus:outline-none focus:ring-2 focus:ring-purple-700 focus:border-purple-600 transition-all duration-200 shadow-inner"
                    />
                  </div>
                </div>

                {/* Risk Appetite */}
                <div className="mb-4">
                  <label className="block text-purple-300 text-sm font-medium mb-2">
                    Risk Appetite
                  </label>
                  <div className="grid grid-cols-3 gap-3">
                    {['Low', 'Medium', 'High'].map((risk) => (
                      <div
                        key={risk}
                        onClick={() => setRiskAppetite(risk)}
                        className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 text-center ${
                          riskAppetite === risk
                            ? 'bg-purple-900/40 border-purple-700 ring-2 ring-purple-800/60'
                            : 'bg-gray-800/30 border-purple-800/40 hover:bg-gray-800/50'
                        }`}
                      >
                        <span className="text-white font-medium">{risk}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Preferred Sectors */}
                <div className="mb-4">
                  <label className="block text-purple-300 text-sm font-medium mb-2">
                    Preferred Sectors (Optional)
                  </label>
                  <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto">
                    {sectorOptions.map((sector) => (
                      <div
                        key={sector.id}
                        onClick={() => toggleSector(sector.id)}
                        className={`p-2 rounded-lg border cursor-pointer transition-all duration-200 text-sm ${
                          selectedSectors.includes(sector.id)
                            ? 'bg-purple-900/40 border-purple-700 text-white'
                            : 'bg-gray-800/30 border-purple-800/40 text-purple-300 hover:bg-gray-800/50'
                        }`}
                      >
                        {sector.name}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 2. Time Horizon */}
              <div className="mb-6">
                <h3 className="text-lg font-medium text-white mb-3">⏱️ Time Horizon</h3>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { id: 'Short-term', label: 'Short-term', desc: '0-3 months' },
                    { id: 'Medium-term', label: 'Medium-term', desc: '3-12 months' },
                    { id: 'Long-term', label: 'Long-term', desc: '1+ year' }
                  ].map((duration) => (
                    <div
                      key={duration.id}
                      onClick={() => setInvestmentDuration(duration.id)}
                      className={`p-4 rounded-lg border cursor-pointer transition-all duration-200 ${
                        investmentDuration === duration.id
                          ? 'bg-purple-900/40 border-purple-700 ring-2 ring-purple-800/60'
                          : 'bg-gray-800/30 border-purple-800/40 hover:bg-gray-800/50'
                      }`}
                    >
                      <div className="text-white font-medium">{duration.label}</div>
                      <div className="text-purple-300 text-xs mt-1">{duration.desc}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 3. Asset Preferences */}
              <div className="mb-6">
                <h3 className="text-lg font-medium text-white mb-3">📊 Asset Classes</h3>
                <div className="space-y-4">
                  {investmentOptions.map((option) => {
                    const IconComponent = option.icon;
                    const isSelected = selectedOptions.includes(option.id);
                    
                    return (
                      <div
                        key={option.id}
                        onClick={() => toggleOption(option.id)}
                        className={`p-3 rounded-xl border transition-all duration-200 cursor-pointer ${
                          isSelected
                            ? 'bg-purple-900/40 border-purple-700 ring-2 ring-purple-800/60 shadow-lg'
                            : 'bg-gray-800/30 border-purple-800/40 hover:bg-gray-800/50 hover:border-purple-700/60 hover:shadow-lg'
                        }`}
                      >
                        <div className="flex items-start space-x-3">
                          <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
                            isSelected ? 'bg-purple-800 shadow-lg' : 'bg-gray-800/60 border border-purple-800/40'
                          }`}>
                            {isSelected ? (
                              <CheckCircle className="w-4 h-4 text-white" />
                            ) : (
                              <IconComponent className="w-4 h-4 text-purple-400" />
                            )}
                          </div>
                          <div className="flex-1">
                            <h3 className="text-white font-medium text-sm">{option.name}</h3>
                            <p className="text-purple-300 text-xs mt-1">{option.description}</p>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* 4. Strategy Preferences */}
              <div className="mb-6">
                <h3 className="text-lg font-medium text-white mb-3">🎯 Strategy Preferences</h3>
                
                {/* Portfolio Strategy */}
                <div className="mb-4">
                  <label className="block text-purple-300 text-sm font-medium mb-3">
                    Portfolio Strategy
                  </label>
                  <div className="grid grid-cols-3 gap-3">
                    {['Maximize ROI', 'Minimize Risk', 'Balanced Strategy'].map((strategy) => (
                      <div
                        key={strategy}
                        onClick={() => setPortfolioStrategy(strategy)}
                        className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 text-center ${
                          portfolioStrategy === strategy
                            ? 'bg-purple-900/40 border-purple-700 ring-2 ring-purple-800/60'
                            : 'bg-gray-800/30 border-purple-800/40 hover:bg-gray-800/50'
                        }`}
                      >
                        <span className="text-white font-medium text-sm">{strategy}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* ML Prediction Toggle */}
                <div className="flex items-center justify-between p-4 bg-gray-800/30 rounded-lg border border-purple-800/40">
                  <div>
                    <span className="text-white font-medium">AI/ML Predictions</span>
                    <p className="text-purple-300 text-sm">Enable machine learning predictions</p>
                  </div>
                  <button
                    onClick={() => setUseMlPrediction(!useMlPrediction)}
                    className={`w-12 h-6 rounded-full transition-all duration-200 ${
                      useMlPrediction ? 'bg-purple-700' : 'bg-gray-600'
                    }`}
                  >
                    <div className={`w-5 h-5 bg-white rounded-full transition-all duration-200 ${
                      useMlPrediction ? 'translate-x-6' : 'translate-x-0.5'
                    }`} />
                  </button>
                </div>
              </div>

              {/* 5. Date Range for Backtesting (Optional) */}
              <div className="mb-6">
                <button
                  onClick={() => setShowBacktesting(!showBacktesting)}
                  className="w-full flex items-center justify-between p-3 bg-gray-800/30 rounded-lg border border-purple-800/40 hover:bg-gray-800/50 transition-colors"
                >
                  <h3 className="text-base font-medium text-white">📅 Backtesting Period (Optional)</h3>
                  <ChevronDown className={`w-5 h-5 text-purple-400 transition-transform ${showBacktesting ? 'rotate-180' : ''}`} />
                </button>
                
                {showBacktesting && (
                  <div className="mt-4 p-4 bg-gray-800/20 rounded-lg border border-purple-800/30">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-purple-300 text-sm font-medium mb-2">From Date</label>
                        <input
                          type="date"
                          value={fromDate}
                          onChange={(e) => setFromDate(e.target.value)}
                          className="w-full px-4 py-3 bg-gray-800/50 border border-purple-800/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-700"
                        />
                      </div>
                      <div>
                        <label className="block text-purple-300 text-sm font-medium mb-2">To Date</label>
                        <input
                          type="date"
                          value={toDate}
                          onChange={(e) => setToDate(e.target.value)}
                          className="w-full px-4 py-3 bg-gray-800/50 border border-purple-800/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-700"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* 6. Advanced Filters (Optional) */}
              <div className="mb-6">
                <button
                  onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
                  className="w-full flex items-center justify-between p-3 bg-gray-800/30 rounded-lg border border-purple-800/40 hover:bg-gray-800/50 transition-colors"
                >
                  <h3 className="text-base font-medium text-white">⚙️ Advanced Filters (Optional)</h3>
                  <ChevronDown className={`w-5 h-5 text-purple-400 transition-transform ${showAdvancedFilters ? 'rotate-180' : ''}`} />
                </button>
                
                {showAdvancedFilters && (
                  <div className="mt-4 p-4 bg-gray-800/20 rounded-lg border border-purple-800/30">
                    {/* Minimum ROI Expectation */}
                    <div className="mb-4">
                      <label className="block text-purple-300 text-sm font-medium mb-3">
                        Minimum ROI Expectation: {minRoiExpectation}%
                      </label>
                      <input
                        type="range"
                        min="3"
                        max="25"
                        value={minRoiExpectation}
                        onChange={(e) => setMinRoiExpectation(parseInt(e.target.value))}
                        className="w-full h-2 bg-gray-800/60 rounded-lg appearance-none cursor-pointer slider"
                      />
                      <div className="flex justify-between text-xs text-purple-400 mt-2">
                        <span>3%</span>
                        <span>25%</span>
                      </div>
                    </div>

                    {/* Advanced Options */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg border border-purple-800/40">
                        <span className="text-white">Exclude Volatile Assets</span>
                        <input
                          type="checkbox"
                          checked={excludeVolatileAssets}
                          onChange={(e) => setExcludeVolatileAssets(e.target.checked)}
                          className="w-5 h-5 text-purple-700 bg-gray-800 border-purple-800 rounded focus:ring-purple-700"
                        />
                      </div>
                      <div className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg border border-purple-800/40">
                        <span className="text-white">Include Only Liquid Assets</span>
                        <input
                          type="checkbox"
                          checked={includeOnlyLiquidAssets}
                          onChange={(e) => setIncludeOnlyLiquidAssets(e.target.checked)}
                          className="w-5 h-5 text-purple-700 bg-gray-800 border-purple-800 rounded focus:ring-purple-700"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Get Recommendations Button */}
              <button
                onClick={runPrediction}
                disabled={!isFormValid || isLoading}
                className={`w-full py-4 px-6 rounded-xl font-semibold text-white transition-all duration-200 flex items-center justify-center space-x-2 ${
                  isFormValid && !isLoading
                    ? 'bg-gradient-to-r from-purple-800 to-indigo-800 hover:from-purple-700 hover:to-indigo-700 shadow-xl hover:shadow-purple-800/40 transform hover:scale-[1.02] border border-purple-700/60'
                    : 'bg-gray-800/60 cursor-not-allowed opacity-50 border border-gray-700/60'
                }`}
              >
                {isLoading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    <span>Analyzing Your Portfolio...</span>
                  </>
                ) : (
                  <>
                    <span>Get Recommendations</span>
                    <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </button>
            </div>

            {/* Results Panel - Much Wider Right Column */}
            <div className="lg:col-span-5 bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-6 flex items-center">
                <Award className="w-5 h-5 mr-3 text-purple-400" />
                Investment Recommendations
              </h2>

              {!prediction ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-800/60 backdrop-blur-sm rounded-full flex items-center justify-center mx-auto mb-4 border border-purple-800/40">
                    <BarChart3 className="w-8 h-8 text-purple-400" />
                  </div>
                  <p className="text-purple-300 text-lg mb-4">
                    Configure your investment parameters and click "Run Prediction" to get AI-powered recommendations
                  </p>
                  <div className="text-sm text-purple-400 bg-gray-800/30 rounded-lg p-3 border border-purple-800/40">
                    <div className="flex items-center justify-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span>Connected to AI Investment Engine</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Summary Stats */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-800/50 rounded-xl p-4 border border-purple-800/40 shadow-lg">
                      <div className="text-purple-400 text-sm font-medium">Expected Return</div>
                      <div className="text-2xl font-bold text-white">{prediction.expectedReturn}%</div>
                    </div>
                    <div className="bg-gray-800/50 rounded-xl p-4 border border-purple-800/40 shadow-lg">
                      <div className="text-purple-400 text-sm font-medium">Risk Score</div>
                      <div className="text-2xl font-bold text-white">{prediction.riskScore}/10</div>
                    </div>
                  </div>

                  {/* Allocation Chart */}
                  <div>
                    <h3 className="text-white font-semibold mb-4">Recommended Allocation</h3>
                    <div className="space-y-3">
                      {prediction.recommendedAllocation.stocks > 0 && (
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <TrendingUp className="w-5 h-5 text-purple-400" />
                            <span className="text-white">Stocks</span>
                          </div>
                          <div className="flex items-center space-x-3">
                            <div className="w-24 h-2 bg-gray-800/60 rounded-full overflow-hidden border border-purple-800/30">
                              <div 
                                className="h-full bg-gradient-to-r from-purple-700 to-indigo-700 transition-all duration-500"
                                style={{ width: `${prediction.recommendedAllocation.stocks}%` }}
                              ></div>
                            </div>
                            <span className="text-white font-medium w-12 text-right">{prediction.recommendedAllocation.stocks}%</span>
                          </div>
                        </div>
                      )}
                      
                      {prediction.recommendedAllocation.mutualFunds > 0 && (
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <Building2 className="w-5 h-5 text-purple-400" />
                            <span className="text-white">Mutual Funds</span>
                          </div>
                          <div className="flex items-center space-x-3">
                            <div className="w-24 h-2 bg-gray-800/60 rounded-full overflow-hidden border border-purple-800/30">
                              <div 
                                className="h-full bg-gradient-to-r from-purple-700 to-indigo-700 transition-all duration-500"
                                style={{ width: `${prediction.recommendedAllocation.mutualFunds}%` }}
                              ></div>
                            </div>
                            <span className="text-white font-medium w-12 text-right">{prediction.recommendedAllocation.mutualFunds}%</span>
                          </div>
                        </div>
                      )}

                      {prediction.recommendedAllocation.gold > 0 && (
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <Coins className="w-5 h-5 text-purple-400" />
                            <span className="text-white">Gold</span>
                          </div>
                          <div className="flex items-center space-x-3">
                            <div className="w-24 h-2 bg-gray-800/60 rounded-full overflow-hidden border border-purple-800/30">
                              <div 
                                className="h-full bg-gradient-to-r from-purple-700 to-indigo-700 transition-all duration-500"
                                style={{ width: `${prediction.recommendedAllocation.gold}%` }}
                              ></div>
                            </div>
                            <span className="text-white font-medium w-12 text-right">{prediction.recommendedAllocation.gold}%</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Recommendations */}
                  <div>
                    <h3 className="text-white font-semibold mb-4">AI Recommendations</h3>
                    <div className="space-y-3">
                      {prediction.recommendations.map((rec, index) => (
                        <div key={index} className="flex items-start space-x-3 p-3 bg-gray-800/30 rounded-lg border border-purple-800/40 shadow-md">
                          <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                          <p className="text-purple-200 text-sm leading-relaxed">{rec}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Comprehensive Investment Analysis */}
                  {comprehensiveData && (
                    <div className="mt-8 space-y-6">
                      <h3 className="text-white font-semibold text-lg mb-4 flex items-center">
                        <Award className="w-5 h-5 mr-2 text-purple-400" />
                        Comprehensive Investment Analysis
                      </h3>

                      {/* Stocks Section */}
                      {comprehensiveData.recommendations.stocks.length > 0 && (
                        <div className="bg-gray-800/50 rounded-xl border border-purple-800/40">
                          <button
                            onClick={() => setShowStocksSection(!showStocksSection)}
                            className="w-full flex items-center justify-between p-4 hover:bg-gray-800/30 transition-colors rounded-t-xl"
                          >
                            <h4 className="text-purple-400 font-medium flex items-center">
                              <TrendingUp className="w-4 h-4 mr-2" />
                              📊 STOCKS RECOMMENDATIONS ({comprehensiveData.recommendations.stocks.length} sectors)
                            </h4>
                            <ChevronDown className={`w-5 h-5 text-purple-400 transition-transform ${showStocksSection ? 'rotate-180' : ''}`} />
                          </button>
                          
                          {showStocksSection && (
                            <div className="p-6 pt-0">
                          {comprehensiveData.recommendations.stocks.map((sector, index) => (
                            <div key={index} className="mb-6 p-4 bg-gray-700/30 rounded-lg border border-purple-700/30">
                              {/* Sector Header */}
                              <div className="flex justify-between items-start mb-3">
                                <div>
                                  <h5 className="text-white font-medium text-lg">{sector.sector} Sector</h5>
                                  <div className="text-sm text-purple-300 mt-1">
                                    🎯 Predicted Return: <span className="text-green-400 font-semibold">{sector.predicted_return.toFixed(2)}%</span> | 
                                    📊 ML Score: <span className="text-blue-400 font-semibold">{sector.ml_score.toFixed(2)}</span> | 
                                    🏢 Investment Opportunities: <span className="text-yellow-400 font-semibold">{sector.investment_count}</span>
                                  </div>
                                </div>
                                <div className="text-right">
                                  <div className="text-white font-bold text-lg">₹{sector.avg_price.toFixed(2)}</div>
                                  <div className="text-xs text-purple-400">Average Price</div>
                                </div>
                              </div>
                              
                              {/* Sector Commentary */}
                              <div className="text-purple-200 text-sm mb-4 p-3 bg-purple-900/20 rounded-lg border border-purple-800/30">
                                💬 <strong>SECTOR ANALYSIS:</strong> {sector.commentary}
                              </div>

                              {/* Individual Stocks */}
                              {sector.stocks && sector.stocks.length > 0 && (
                                <div>
                                  <h6 className="text-purple-400 font-medium mb-3 flex items-center">
                                    📈 <span className="ml-1">SPECIFIC STOCKS TO BUY:</span>
                                  </h6>
                                  {sector.stocks.map((stock, stockIndex) => (
                                    <div key={stockIndex} className="mb-4 p-4 bg-gray-600/20 rounded-lg border border-gray-600/40">
                                      {/* Stock Header */}
                                      <div className="flex justify-between items-start mb-3">
                                        <div>
                                          <span className="text-white font-bold text-lg">{stock.symbol}</span>
                                          <span className="text-purple-300 ml-2">- {stock.company_name}</span>
                                        </div>
                                        <div className="text-right">
                                          <div className="text-green-400 font-bold text-lg">
                                            {stock.current_performance.toFixed(2)}%
                                          </div>
                                          <div className="text-xs text-green-300">Current Performance</div>
                                        </div>
                                      </div>
                                      
                                      {/* Stock Commentary */}
                                      <div className="text-purple-200 text-sm mb-3 p-2 bg-purple-900/10 rounded">
                                        💬 <strong>Performance Analysis:</strong> {stock.commentary}
                                      </div>
                                      
                                      {/* Investment Strategy */}
                                      <div className="text-blue-200 text-sm mb-4 p-3 bg-blue-900/20 rounded border border-blue-800/30">
                                        📝 <strong>INVESTMENT STRATEGY:</strong> {stock.investment_strategy}
                                      </div>
                                      
                                      {/* Timing Analysis Details */}
                                      {stock.timing_analysis && (
                                        <div className="bg-gray-800/40 p-3 rounded-lg border border-gray-700/50">
                                          <div className="text-yellow-400 font-medium mb-2 block">⏰ DETAILED TIMING ANALYSIS:</div>
                                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                                            <div className="bg-green-900/20 p-2 rounded border border-green-800/30">
                                              <div className="text-green-300 font-medium">📅 Entry Date</div>
                                              <div className="text-white font-semibold">{stock.timing_analysis.entry_date}</div>
                                            </div>
                                            <div className="bg-green-900/20 p-2 rounded border border-green-800/30">
                                              <div className="text-green-300 font-medium">💰 Entry Price</div>
                                              <div className="text-white font-semibold">₹{stock.timing_analysis.entry_price}</div>
                                            </div>
                                            <div className="bg-blue-900/20 p-2 rounded border border-blue-800/30">
                                              <div className="text-blue-300 font-medium">📅 Exit Date</div>
                                              <div className="text-white font-semibold">{stock.timing_analysis.exit_date}</div>
                                            </div>
                                            <div className="bg-blue-900/20 p-2 rounded border border-blue-800/30">
                                              <div className="text-blue-300 font-medium">💰 Exit Price</div>
                                              <div className="text-white font-semibold">₹{stock.timing_analysis.exit_price}</div>
                                            </div>
                                            <div className="bg-purple-900/20 p-2 rounded border border-purple-800/30">
                                              <div className="text-purple-300 font-medium">📊 Expected Return</div>
                                              <div className="text-white font-semibold">{stock.timing_analysis.expected_return.toFixed(2)}%</div>
                                            </div>
                                            <div className="bg-red-900/20 p-2 rounded border border-red-800/30">
                                              <div className="text-red-300 font-medium">🛑 Stop Loss</div>
                                              <div className="text-white font-semibold">₹{stock.timing_analysis.stop_loss}</div>
                                            </div>
                                            <div className="bg-yellow-900/20 p-2 rounded border border-yellow-800/30">
                                              <div className="text-yellow-300 font-medium">🎯 Target Price</div>
                                              <div className="text-white font-semibold">₹{stock.timing_analysis.target_price}</div>
                                            </div>
                                            <div className="bg-indigo-900/20 p-2 rounded border border-indigo-800/30">
                                              <div className="text-indigo-300 font-medium">⏳ Holding Period</div>
                                              <div className="text-white font-semibold">{stock.timing_analysis.holding_period} days</div>
                                            </div>
                                          </div>
                                          <div className="mt-2 bg-orange-900/20 p-2 rounded border border-orange-800/30">
                                            <div className="text-orange-300 font-medium text-xs">📊 Volatility</div>
                                            <div className="text-white font-semibold">{stock.timing_analysis.volatility.toFixed(2)}%</div>
                                          </div>
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Mutual Funds Section */}
                      {comprehensiveData.recommendations.mutual_funds.length > 0 && (
                        <div className="bg-gray-800/50 rounded-xl border border-purple-800/40">
                          <button
                            onClick={() => setShowMutualFundsSection(!showMutualFundsSection)}
                            className="w-full flex items-center justify-between p-4 hover:bg-gray-800/30 transition-colors rounded-t-xl"
                          >
                            <h4 className="text-purple-400 font-medium flex items-center">
                              <Building2 className="w-4 h-4 mr-2" />
                              💼 MUTUAL FUNDS RECOMMENDATIONS ({comprehensiveData.recommendations.mutual_funds.length} sectors)
                            </h4>
                            <ChevronDown className={`w-5 h-5 text-purple-400 transition-transform ${showMutualFundsSection ? 'rotate-180' : ''}`} />
                          </button>
                          
                          {showMutualFundsSection && (
                            <div className="p-6 pt-0">
                          {comprehensiveData.recommendations.mutual_funds.map((sector, index) => (
                            <div key={index} className="mb-6 p-4 bg-gray-700/30 rounded-lg border border-purple-700/30">
                              {/* Sector Header */}
                              <div className="flex justify-between items-start mb-3">
                                <div>
                                  <h5 className="text-white font-medium text-lg">{sector.sector} Sector</h5>
                                  <div className="text-sm text-purple-300 mt-1">
                                    🎯 Predicted Return: <span className="text-green-400 font-semibold">{sector.predicted_return.toFixed(2)}%</span> | 
                                    📊 ML Score: <span className="text-blue-400 font-semibold">{sector.ml_score.toFixed(2)}</span>
                                  </div>
                                </div>
                              </div>
                              
                              {/* Sector Commentary */}
                              <div className="text-purple-200 text-sm mb-4 p-3 bg-purple-900/20 rounded-lg border border-purple-800/30">
                                💬 <strong>SECTOR ANALYSIS:</strong> {sector.commentary}
                              </div>

                              {/* Individual Funds */}
                              {sector.funds && sector.funds.length > 0 && (
                                <div>
                                  <h6 className="text-purple-400 font-medium mb-3 flex items-center">
                                    💼 <span className="ml-1">SPECIFIC FUNDS TO INVEST:</span>
                                  </h6>
                                  {sector.funds.map((fund, fundIndex) => (
                                    <div key={fundIndex} className="mb-4 p-4 bg-gray-600/20 rounded-lg border border-gray-600/40">
                                      {/* Fund Header */}
                                      <div className="flex justify-between items-start mb-3">
                                        <div className="flex-1">
                                          <div className="text-white font-bold text-lg">{fund.fund_name}</div>
                                          <div className="text-purple-300 text-sm">{fund.full_name}</div>
                                          <div className="text-gray-400 text-xs mt-1">Managed by: {fund.fund_manager}</div>
                                        </div>
                                        <div className="text-right ml-4">
                                          <div className="text-green-400 font-bold text-lg">
                                            {fund.current_performance.toFixed(2)}%
                                          </div>
                                          <div className="text-xs text-green-300">Current Performance</div>
                                        </div>
                                      </div>
                                      
                                      {/* Fund Details Grid */}
                                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3 text-xs">
                                        <div className="bg-blue-900/20 p-2 rounded border border-blue-800/30">
                                          <div className="text-blue-300 font-medium">💰 Min Investment</div>
                                          <div className="text-white font-semibold">₹{fund.minimum_investment}</div>
                                        </div>
                                        <div className="bg-orange-900/20 p-2 rounded border border-orange-800/30">
                                          <div className="text-orange-300 font-medium">📊 Expense Ratio</div>
                                          <div className="text-white font-semibold">{fund.expense_ratio}%</div>
                                        </div>
                                        <div className="bg-purple-900/20 p-2 rounded border border-purple-800/30">
                                          <div className="text-purple-300 font-medium">⚠️ Risk Level</div>
                                          <div className="text-white font-semibold">{fund.risk_level}</div>
                                        </div>
                                        <div className="bg-green-900/20 p-2 rounded border border-green-800/30">
                                          <div className="text-green-300 font-medium">📅 Entry Date</div>
                                          <div className="text-white font-semibold">{fund.entry_date}</div>
                                        </div>
                                      </div>
                                      
                                      {/* Fund Commentary */}
                                      <div className="text-purple-200 text-sm mb-3 p-2 bg-purple-900/10 rounded">
                                        💬 <strong>Fund Analysis:</strong> {fund.commentary}
                                      </div>
                                      
                                      {/* SIP vs Lump Sum Analysis */}
                                      <div className="bg-gray-800/40 p-3 rounded-lg border border-gray-700/50">
                                        <div className="flex items-center justify-between mb-2">
                                          <div className="text-yellow-400 font-medium">🎯 INVESTMENT STRATEGY:</div>
                                          <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                            fund.sip_recommended 
                                              ? 'bg-green-900/40 text-green-300 border border-green-800' 
                                              : 'bg-blue-900/40 text-blue-300 border border-blue-800'
                                          }`}>
                                            {fund.investment_type}
                                          </span>
                                        </div>
                                        
                                        {fund.sip_recommended ? (
                                          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                                            <div className="bg-green-900/20 p-2 rounded border border-green-800/30">
                                              <div className="text-green-300 font-medium text-xs">💰 Monthly SIP</div>
                                              <div className="text-white font-bold">₹{fund.monthly_sip_amount.toFixed(0)}</div>
                                            </div>
                                            <div className="bg-blue-900/20 p-2 rounded border border-blue-800/30">
                                              <div className="text-blue-300 font-medium text-xs">⏰ SIP Duration</div>
                                              <div className="text-white font-bold">{fund.sip_duration_months} months</div>
                                            </div>
                                            <div className="bg-purple-900/20 p-2 rounded border border-purple-800/30">
                                              <div className="text-purple-300 font-medium text-xs">💎 Total Investment</div>
                                              <div className="text-white font-bold">₹{fund.total_investment_amount.toFixed(0)}</div>
                                            </div>
                                          </div>
                                        ) : (
                                          <div className="bg-blue-900/20 p-2 rounded border border-blue-800/30 mb-3">
                                            <div className="text-blue-300 font-medium text-xs">💰 Lump Sum Investment</div>
                                            <div className="text-white font-bold">₹{fund.total_investment_amount.toFixed(0)}</div>
                                          </div>
                                        )}
                                        
                                        <div className="text-blue-200 text-sm p-2 bg-blue-900/10 rounded">
                                          📝 <strong>Strategy:</strong> {fund.investment_strategy}
                                        </div>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Gold Section */}
                      {comprehensiveData.recommendations.gold.length > 0 && (
                        <div className="bg-gray-800/50 rounded-xl border border-purple-800/40">
                          <button
                            onClick={() => setShowGoldSection(!showGoldSection)}
                            className="w-full flex items-center justify-between p-4 hover:bg-gray-800/30 transition-colors rounded-t-xl"
                          >
                            <h4 className="text-purple-400 font-medium flex items-center">
                              <Coins className="w-4 h-4 mr-2" />
                              🥇 GOLD INVESTMENTS RECOMMENDATIONS
                            </h4>
                            <ChevronDown className={`w-4 h-4 transition-transform ${showGoldSection ? 'rotate-180' : ''}`} />
                          </button>
                          
                          {showGoldSection && (
                            <div className="p-6 pt-0">
                              {comprehensiveData.recommendations.gold.map((sector, index) => (
                            <div key={index} className="mb-6 p-4 bg-gray-700/30 rounded-lg border border-purple-700/30">
                              {/* Sector Header */}
                              <div className="flex justify-between items-start mb-3">
                                <div>
                                  <h5 className="text-white font-medium text-lg">{sector.sector}</h5>
                                  <div className="text-sm text-purple-300 mt-1">
                                    🎯 Predicted Return: <span className="text-green-400 font-semibold">{sector.predicted_return.toFixed(2)}%</span> | 
                                    📊 ML Score: <span className="text-blue-400 font-semibold">{sector.ml_score.toFixed(2)}</span>
                                  </div>
                                </div>
                              </div>
                              
                              {/* Sector Commentary */}
                              <div className="text-purple-200 text-sm mb-4 p-3 bg-purple-900/20 rounded-lg border border-purple-800/30">
                                💬 <strong>GOLD MARKET ANALYSIS:</strong> {sector.commentary}
                              </div>

                              {/* Gold Investment Options */}
                              {sector.gold_investments && sector.gold_investments.length > 0 && (
                                <div>
                                  <h6 className="text-purple-400 font-medium mb-3 flex items-center">
                                    🥇 <span className="ml-1">SPECIFIC GOLD INVESTMENTS:</span>
                                  </h6>
                                  {sector.gold_investments.map((gold, goldIndex) => (
                                    <div key={goldIndex} className="mb-4 p-4 bg-gray-600/20 rounded-lg border border-gray-600/40">
                                      {/* Gold Investment Header */}
                                      <div className="flex justify-between items-start mb-3">
                                        <div className="flex-1">
                                          <div className="text-white font-bold text-lg">{gold.investment_type}</div>
                                          <div className="text-purple-300 text-sm">Current Price: ₹{gold.current_price}/gram</div>
                                        </div>
                                        <div className="text-right ml-4">
                                          <div className="text-green-400 font-bold text-lg">
                                            {gold.current_performance.toFixed(2)}%
                                          </div>
                                          <div className="text-xs text-green-300">Current Performance</div>
                                        </div>
                                      </div>
                                      
                                      {/* Gold Investment Details Grid */}
                                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3 text-xs">
                                        <div className="bg-yellow-900/20 p-2 rounded border border-yellow-800/30">
                                          <div className="text-yellow-300 font-medium">💰 Recommended Amount</div>
                                          <div className="text-white font-semibold">₹{gold.recommended_allocation.toFixed(0)}</div>
                                        </div>
                                        <div className="bg-green-900/20 p-2 rounded border border-green-800/30">
                                          <div className="text-green-300 font-medium">📅 Entry Date</div>
                                          <div className="text-white font-semibold">{gold.entry_date}</div>
                                        </div>
                                        <div className="bg-blue-900/20 p-2 rounded border border-blue-800/30">
                                          <div className="text-blue-300 font-medium">⏳ Holding Period</div>
                                          <div className="text-white font-semibold">{gold.holding_period_days} days</div>
                                        </div>
                                        <div className="bg-purple-900/20 p-2 rounded border border-purple-800/30">
                                          <div className="text-purple-300 font-medium">📊 Expected Return</div>
                                          <div className="text-white font-semibold">{gold.expected_return.toFixed(2)}%</div>
                                        </div>
                                        <div className="bg-orange-900/20 p-2 rounded border border-orange-800/30">
                                          <div className="text-orange-300 font-medium">📊 Volatility</div>
                                          <div className="text-white font-semibold">{gold.volatility.toFixed(2)}%</div>
                                        </div>
                                        <div className="bg-indigo-900/20 p-2 rounded border border-indigo-800/30">
                                          <div className="text-indigo-300 font-medium">💧 Liquidity</div>
                                          <div className="text-white font-semibold">{gold.liquidity_rating}</div>
                                        </div>
                                        <div className="bg-red-900/20 p-2 rounded border border-red-800/30">
                                          <div className="text-red-300 font-medium">⚠️ Risk Level</div>
                                          <div className="text-white font-semibold">{gold.risk_level}</div>
                                        </div>
                                        <div className="bg-teal-900/20 p-2 rounded border border-teal-800/30">
                                          <div className="text-teal-300 font-medium">💎 Min Investment</div>
                                          <div className="text-white font-semibold">₹{gold.minimum_investment}</div>
                                        </div>
                                      </div>
                                      
                                      {/* Storage & Tax Info */}
                                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                                        <div className={`p-2 rounded border text-xs ${
                                          gold.storage_required 
                                            ? 'bg-orange-900/20 border-orange-800/30' 
                                            : 'bg-green-900/20 border-green-800/30'
                                        }`}>
                                          <div className={`font-medium ${gold.storage_required ? 'text-orange-300' : 'text-green-300'}`}>
                                            🏠 Storage Required
                                          </div>
                                          <div className="text-white font-semibold">
                                            {gold.storage_required ? 'Yes - Physical Storage Needed' : 'No - Digital/Demat Form'}
                                          </div>
                                        </div>
                                        <div className="bg-gray-900/20 p-2 rounded border border-gray-700/30 text-xs">
                                          <div className="text-gray-300 font-medium">💸 Tax Implications</div>
                                          <div className="text-white font-semibold text-xs">{gold.tax_implications}</div>
                                        </div>
                                      </div>
                                      
                                      {/* Gold Commentary */}
                                      <div className="text-purple-200 text-sm mb-3 p-2 bg-purple-900/10 rounded">
                                        💬 <strong>Investment Analysis:</strong> {gold.commentary}
                                      </div>
                                      
                                      {/* Investment Strategy */}
                                      <div className="text-yellow-200 text-sm p-3 bg-yellow-900/20 rounded border border-yellow-800/30">
                                        📝 <strong>INVESTMENT STRATEGY:</strong> {gold.investment_strategy}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Investment Breakdown */}
                  <div>
                    <h3 className="text-white font-semibold mb-4">Investment Breakdown</h3>
                    <div className="space-y-3">
                      {prediction.recommendedAllocation.stocks > 0 && (
                        <div>
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-purple-300">Stocks</span>
                            <span className="text-white font-semibold">
                              {formatCurrency(((prediction.recommendedAllocation.stocks / 100) * parseInt(investmentAmount || '0')).toString())}
                            </span>
                          </div>
                          {prediction.detailedInvestments?.stocks && (
                            <div className="ml-4 text-sm space-y-1">
                              <div className="text-purple-400">
                                Expected Return: {prediction.detailedInvestments.stocks.average_expected_return?.toFixed(1)}%
                              </div>
                              {prediction.detailedInvestments.stocks.top_picks?.length > 0 && (
                                <div className="text-purple-400">
                                  Top Pick: {prediction.detailedInvestments.stocks.top_picks[0]?.name || 'N/A'}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                      {prediction.recommendedAllocation.mutualFunds > 0 && (
                        <div>
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-purple-300">Mutual Funds</span>
                            <span className="text-white font-semibold">
                              {formatCurrency(((prediction.recommendedAllocation.mutualFunds / 100) * parseInt(investmentAmount || '0')).toString())}
                            </span>
                          </div>
                          {prediction.detailedInvestments?.mutualFunds && (
                            <div className="ml-4 text-sm space-y-1">
                              <div className="text-purple-400">
                                Expected Return: {prediction.detailedInvestments.mutualFunds.average_expected_return?.toFixed(1)}%
                              </div>
                              {prediction.detailedInvestments.mutualFunds.top_picks?.length > 0 && (
                                <div className="text-purple-400">
                                  Top Pick: {prediction.detailedInvestments.mutualFunds.top_picks[0]?.name || 'N/A'}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                      {prediction.recommendedAllocation.gold > 0 && (
                        <div>
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-purple-300">Gold</span>
                            <span className="text-white font-semibold">
                              {formatCurrency(((prediction.recommendedAllocation.gold / 100) * parseInt(investmentAmount || '0')).toString())}
                            </span>
                          </div>
                          {prediction.detailedInvestments?.gold && (
                            <div className="ml-4 text-sm space-y-1">
                              <div className="text-purple-400">
                                Expected Return: {prediction.detailedInvestments.gold.average_expected_return?.toFixed(1)}%
                              </div>
                              {prediction.detailedInvestments.gold.top_picks?.length > 0 && (
                                <div className="text-purple-400">
                                  Type: {prediction.detailedInvestments.gold.top_picks[0]?.type || 'ETF'}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                      <div className="border-t border-purple-800/40 pt-3 mt-3">
                        <div className="flex justify-between items-center">
                          <span className="text-white font-semibold">Total Investment</span>
                          <span className="text-white font-bold text-lg">
                            {formatCurrency(investmentAmount || '0')}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          background: linear-gradient(135deg, #6B46C1, #7C3AED);
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 4px 12px rgba(107, 70, 193, 0.4);
          transition: all 0.2s ease;
        }
        
        .slider::-webkit-slider-thumb:hover {
          transform: scale(1.1);
          box-shadow: 0 6px 16px rgba(107, 70, 193, 0.6);
        }
        
        .slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          background: linear-gradient(135deg, #6B46C1, #7C3AED);
          border-radius: 50%;
          cursor: pointer;
          border: none;
          box-shadow: 0 4px 12px rgba(107, 70, 193, 0.4);
          transition: all 0.2s ease;
        }
        
        .slider::-moz-range-thumb:hover {
          transform: scale(1.1);
          box-shadow: 0 6px 16px rgba(107, 70, 193, 0.6);
        }
        
        .slider::-webkit-slider-track {
          background: rgba(31, 41, 55, 0.6);
          border-radius: 4px;
          border: 1px solid rgba(107, 70, 193, 0.3);
        }
        
        .slider::-moz-range-track {
          background: rgba(31, 41, 55, 0.6);
          border-radius: 4px;
          border: 1px solid rgba(107, 70, 193, 0.3);
        }
      `}</style>
    </div>
  );
}

export default App;
