import React, { useState } from 'react';
import axios from 'axios';
import { TrendingUp, DollarSign, Target, BarChart3, Coins, Building2, Award, ArrowRight, CheckCircle, ChevronDown } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8002';

interface InvestmentOption {
  id: string;
  name: string;
  icon: React.ComponentType<any>;
  description: string;
}

interface StockRecommendation {
  symbol: string;
  company_name: string;
  current_performance: number;
  investment_strategy: string;
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  expected_return: number;
  stop_loss: number;
  target_price: number;
  holding_period: number;
  volatility: number;
}

interface MutualFundRecommendation {
  fund_name: string;
  fund_manager: string;
  current_performance: number;
  investment_strategy: string;
  expected_return: number;
  is_sip_recommended: boolean;
  sip_amount: number;
  sip_duration_months: number;
  lump_sum_amount: number;
  expense_ratio: number;
  risk_level: string;
  minimum_investment: number;
}

interface GoldRecommendation {
  investment_type: string;
  current_performance: number;
  investment_strategy: string;
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  expected_return: number;
  holding_period: number;
  volatility: number;
  liquidity_rating: string;
  storage_required: boolean;
  tax_implications: string;
}

interface SectorRecommendation {
  sector: string;
  predicted_return: number;
  investment_opportunities: number;
  stocks?: StockRecommendation[];
  mutual_funds?: MutualFundRecommendation[];
  gold?: GoldRecommendation[];
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

function App() {
  // Form state
  const [investmentAmount, setInvestmentAmount] = useState<string>('');
  const [riskAppetite, setRiskAppetite] = useState<string>('Medium');
  const [minRoiExpectation, setMinRoiExpectation] = useState<number>(10);
  const [investmentDuration, setInvestmentDuration] = useState<string>('');
  const [portfolioStrategy, setPortfolioStrategy] = useState<string>('balanced');
  const [useMlPrediction, setUseMlPrediction] = useState<boolean>(true);
  const [selectedSectors, setSelectedSectors] = useState<string[]>([]);
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);
  
  // Advanced filters
  const [showAdvancedFilters, setShowAdvancedFilters] = useState<boolean>(false);
  const [excludeVolatileAssets, setExcludeVolatileAssets] = useState<boolean>(false);
  const [includeOnlyLiquidAssets, setIncludeOnlyLiquidAssets] = useState<boolean>(false);
  
  // Backtesting
  const [showBacktesting, setShowBacktesting] = useState<boolean>(false);
  const [fromDate, setFromDate] = useState<string>('');
  const [toDate, setToDate] = useState<string>('');
  
  // Results and UI state
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [comprehensiveData, setComprehensiveData] = useState<ComprehensiveRecommendation | null>(null);
  
  // Collapsible Sections for Results
  const [showStocksSection, setShowStocksSection] = useState<boolean>(true);
  const [showMutualFundsSection, setShowMutualFundsSection] = useState<boolean>(true);
  const [showGoldSection, setShowGoldSection] = useState<boolean>(true);

  const formatCurrency = (value: string) => {
    const numericValue = value.replace(/[^0-9]/g, '');
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(parseInt(numericValue) || 0);
  };

  const investmentOptions: InvestmentOption[] = [
    { id: 'stocks', name: 'Stocks', icon: TrendingUp, description: 'Direct equity investments' },
    { id: 'mutualFunds', name: 'Mutual Funds', icon: BarChart3, description: 'Diversified fund investments' },
    { id: 'gold', name: 'Gold', icon: Coins, description: 'Precious metal investments' },
  ];

  const fetchComprehensiveData = async () => {
    try {
      const duration = investmentDuration && !isNaN(parseInt(investmentDuration)) ? parseInt(investmentDuration) : 12;
      const response = await axios.get(`${API_BASE_URL}/recommend/comprehensive`, {
        params: {
          investment_amount: parseFloat(investmentAmount || '0'),
          risk_tolerance: riskAppetite.toLowerCase(),
          investment_horizon: duration,
          strategy_type: portfolioStrategy,
          enable_ml: useMlPrediction,
          asset_classes: selectedOptions.join(','),
          sectors: selectedSectors.join(','),
          min_roi_expectation: minRoiExpectation,
          exclude_volatile_assets: excludeVolatileAssets,
          include_only_liquid_assets: includeOnlyLiquidAssets,
          backtest_start: fromDate,
          backtest_end: toDate
        }
      });
      
      if (response.data.status === 'success') {
        setComprehensiveData(response.data.data);
      }
    } catch (error) {
      console.error('Error fetching comprehensive data:', error);
    }
  };

  const runPrediction = async () => {
    if (!investmentAmount || !investmentDuration || selectedOptions.length === 0) return;

    setIsLoading(true);
    
    try {
      // Fetch comprehensive data with user parameters
      await fetchComprehensiveData();
      
    } catch (error) {
      console.error('API Error:', error);
      
      // Show error message to user
      alert('Failed to fetch recommendations. Please try again.');
      
    } finally {
      setIsLoading(false);
    }
  };

  const handleInvestmentOptionChange = (optionId: string) => {
    setSelectedOptions(prev => 
      prev.includes(optionId)
        ? prev.filter(id => id !== optionId)
        : [...prev, optionId]
    );
  };

  const handleSectorChange = (sector: string) => {
    setSelectedSectors(prev => 
      prev.includes(sector)
        ? prev.filter(s => s !== sector)
        : [...prev, sector]
    );
  };

  const isFormValid = investmentAmount && investmentDuration && selectedOptions.length > 0;

  return (
        <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-purple-200 mb-8 text-center flex items-center justify-center">
            <img 
              src="/profile.jpg" 
              alt="Profile" 
              className="w-12 h-12 mr-4 rounded-full border-2 border-purple-400 shadow-lg object-cover"
            />
            Investment Recommender
          </h1>
          <p className="text-purple-200 text-lg">AI-powered investment recommendations tailored for you</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
          {/* Input Panel - Left Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Investment Amount */}
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <DollarSign className="w-6 h-6 mr-2 text-green-400" />
                Investment Details
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-purple-200 text-sm font-medium mb-2">Investment Amount</label>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {/* Quick Amount Selector */}
                    <div>
                      <label className="block text-purple-300 text-xs font-medium mb-1">Quick Select</label>
                      <select
                        value=""
                        onChange={(e) => setInvestmentAmount(e.target.value)}
                        className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                      >
                        <option value="">Select Amount</option>
                        <option value="1000">₹1,000</option>
                        <option value="10000">₹10,000</option>
                        <option value="100000">₹1,00,000</option>
                      </select>
                    </div>
                    
                    {/* Custom Amount Input */}
                    <div>
                      <label className="block text-purple-300 text-xs font-medium mb-1">Custom Amount</label>
                      <input
                        type="text"
                        value={investmentAmount}
                        onChange={(e) => setInvestmentAmount(e.target.value)}
                        placeholder="Enter amount in INR"
                        className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white placeholder-purple-300/60 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                      />
                    </div>
                  </div>
                  {investmentAmount && (
                    <p className="text-green-400 text-sm mt-2">{formatCurrency(investmentAmount)}</p>
                  )}
                </div>
              </div>
            </div>

            {/* Risk and Returns */}
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <Target className="w-6 h-6 mr-2 text-red-400" />
                Risk & Returns
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-purple-200 text-sm font-medium mb-2">Risk Appetite</label>
                  <select
                    value={riskAppetite}
                    onChange={(e) => setRiskAppetite(e.target.value)}
                    className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                  >
                    <option value="Low">Low Risk</option>
                    <option value="Medium">Medium Risk</option>
                    <option value="High">High Risk</option>
                  </select>
                </div>

                <div>
                  <label className="block text-purple-200 text-sm font-medium mb-2">
                    Expected Return: {minRoiExpectation}% annually
                  </label>
                  <input
                    type="range"
                    min="5"
                    max="25"
                    value={minRoiExpectation}
                    onChange={(e) => setMinRoiExpectation(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <div className="flex justify-between text-xs text-purple-300 mt-1">
                    <span>5%</span>
                    <span>25%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Investment Options */}
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <Building2 className="w-6 h-6 mr-2 text-blue-400" />
                Investment Options
              </h2>

              <div className="space-y-3">
                {investmentOptions.map((option) => (
                  <label key={option.id} className="flex items-center p-3 rounded-xl bg-gray-800/40 border border-gray-700/50 hover:border-purple-600/50 cursor-pointer transition-all">
                    <input
                      type="checkbox"
                      checked={selectedOptions.includes(option.id)}
                      onChange={() => handleInvestmentOptionChange(option.id)}
                      className="sr-only"
                    />
                    <div className={`w-5 h-5 rounded border-2 mr-3 flex items-center justify-center transition-all ${
                      selectedOptions.includes(option.id)
                        ? 'bg-purple-600 border-purple-600'
                        : 'border-gray-600'
                    }`}>
                      {selectedOptions.includes(option.id) && (
                        <CheckCircle className="w-3 h-3 text-white" />
                      )}
                    </div>
                    <option.icon className="w-5 h-5 mr-3 text-purple-300" />
                    <div>
                      <div className="text-white font-medium">{option.name}</div>
                      <div className="text-purple-300 text-sm">{option.description}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Time Horizon */}
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <Target className="w-6 h-6 mr-2 text-yellow-400" />
                Time Horizon
              </h2>
              
              <div>
                <label className="block text-purple-200 text-sm font-medium mb-2">Investment Duration</label>
                <select
                  value={investmentDuration}
                  onChange={(e) => setInvestmentDuration(e.target.value)}
                  className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                >
                  <option value="">Select Duration</option>
                  <option value="1">Short-term (0-3 months)</option>
                  <option value="6">Medium-term (3-12 months)</option>
                  <option value="12">Long-term (1+ year)</option>
                </select>
              </div>
            </div>

            {/* Sector Preferences */}
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <BarChart3 className="w-6 h-6 mr-2 text-cyan-400" />
                Preferred Sectors (Optional)
              </h2>
              
              <div className="grid grid-cols-2 gap-3">
                {[
                  'Technology', 'Finance', 'Healthcare', 'Energy',
                  'Consumer Goods', 'Telecommunications', 'Real Estate', 'Utilities'
                ].map((sector) => (
                  <label key={sector} className="flex items-center p-3 rounded-xl bg-gray-800/40 border border-gray-700/50 hover:border-purple-600/50 cursor-pointer transition-all">
                    <input
                      type="checkbox"
                      checked={selectedSectors.includes(sector)}
                      onChange={() => handleSectorChange(sector)}
                      className="sr-only"
                    />
                    <div className={`w-4 h-4 rounded border-2 mr-3 flex items-center justify-center transition-all ${
                      selectedSectors.includes(sector)
                        ? 'bg-purple-600 border-purple-600'
                        : 'border-gray-600'
                    }`}>
                      {selectedSectors.includes(sector) && (
                        <CheckCircle className="w-2.5 h-2.5 text-white" />
                      )}
                    </div>
                    <span className="text-white text-sm">{sector}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Strategy Preferences */}
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <Award className="w-6 h-6 mr-2 text-orange-400" />
                Strategy Preferences
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-purple-200 text-sm font-medium mb-2">Portfolio Strategy</label>
                  <select
                    value={portfolioStrategy}
                    onChange={(e) => setPortfolioStrategy(e.target.value)}
                    className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                  >
                    <option value="balanced">Balanced Strategy</option>
                    <option value="maximize_roi">Maximize ROI</option>
                    <option value="minimize_risk">Minimize Risk</option>
                  </select>
                </div>

                <div>
                  <label className="flex items-center p-3 rounded-xl bg-gray-800/40 border border-gray-700/50 cursor-pointer transition-all hover:border-purple-600/50">
                    <input
                      type="checkbox"
                      checked={useMlPrediction}
                      onChange={(e) => setUseMlPrediction(e.target.checked)}
                      className="sr-only"
                    />
                    <div className={`w-5 h-5 rounded border-2 mr-3 flex items-center justify-center transition-all ${
                      useMlPrediction
                        ? 'bg-purple-600 border-purple-600'
                        : 'border-gray-600'
                    }`}>
                      {useMlPrediction && (
                        <CheckCircle className="w-3 h-3 text-white" />
                      )}
                    </div>
                    <div>
                      <div className="text-white font-medium">Enable ML Predictions</div>
                      <div className="text-purple-300 text-sm">Use machine learning models for better recommendations</div>
                    </div>
                  </label>
                </div>
              </div>
            </div>

            {/* Date Range for Backtesting */}
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <BarChart3 className="w-6 h-6 mr-2 text-green-400" />
                Historical Analysis (Optional)
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="flex items-center p-3 rounded-xl bg-gray-800/40 border border-gray-700/50 cursor-pointer transition-all hover:border-purple-600/50">
                    <input
                      type="checkbox"
                      checked={showBacktesting}
                      onChange={(e) => setShowBacktesting(e.target.checked)}
                      className="sr-only"
                    />
                    <div className={`w-5 h-5 rounded border-2 mr-3 flex items-center justify-center transition-all ${
                      showBacktesting
                        ? 'bg-purple-600 border-purple-600'
                        : 'border-gray-600'
                    }`}>
                      {showBacktesting && (
                        <CheckCircle className="w-3 h-3 text-white" />
                      )}
                    </div>
                    <div>
                      <div className="text-white font-medium">Show Backtesting Results</div>
                      <div className="text-purple-300 text-sm">Display historical performance analysis</div>
                    </div>
                  </label>
                </div>

                {showBacktesting && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-purple-200 text-sm font-medium mb-2">From Date</label>
                      <input
                        type="date"
                        value={fromDate}
                        onChange={(e) => setFromDate(e.target.value)}
                        className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                      />
                    </div>
                    <div>
                      <label className="block text-purple-200 text-sm font-medium mb-2">To Date</label>
                      <input
                        type="date"
                        value={toDate}
                        onChange={(e) => setToDate(e.target.value)}
                        className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Advanced Filters */}
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center cursor-pointer" onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}>
                <Target className="w-6 h-6 mr-2 text-red-400" />
                Advanced Filters (Optional)
                <ChevronDown className={`w-5 h-5 ml-auto transition-transform ${showAdvancedFilters ? 'rotate-180' : ''}`} />
              </h2>
              
              {showAdvancedFilters && (
                <div className="space-y-4">
                  <div>
                    <label className="flex items-center p-3 rounded-xl bg-gray-800/40 border border-gray-700/50 cursor-pointer transition-all hover:border-purple-600/50">
                      <input
                        type="checkbox"
                        checked={excludeVolatileAssets}
                        onChange={(e) => setExcludeVolatileAssets(e.target.checked)}
                        className="sr-only"
                      />
                      <div className={`w-5 h-5 rounded border-2 mr-3 flex items-center justify-center transition-all ${
                        excludeVolatileAssets
                          ? 'bg-purple-600 border-purple-600'
                          : 'border-gray-600'
                      }`}>
                        {excludeVolatileAssets && (
                          <CheckCircle className="w-3 h-3 text-white" />
                        )}
                      </div>
                      <div>
                        <div className="text-white font-medium">Exclude Volatile Assets</div>
                        <div className="text-purple-300 text-sm">Avoid high-volatility investments</div>
                      </div>
                    </label>
                  </div>

                  <div>
                    <label className="flex items-center p-3 rounded-xl bg-gray-800/40 border border-gray-700/50 cursor-pointer transition-all hover:border-purple-600/50">
                      <input
                        type="checkbox"
                        checked={includeOnlyLiquidAssets}
                        onChange={(e) => setIncludeOnlyLiquidAssets(e.target.checked)}
                        className="sr-only"
                      />
                      <div className={`w-5 h-5 rounded border-2 mr-3 flex items-center justify-center transition-all ${
                        includeOnlyLiquidAssets
                          ? 'bg-purple-600 border-purple-600'
                          : 'border-gray-600'
                      }`}>
                        {includeOnlyLiquidAssets && (
                          <CheckCircle className="w-3 h-3 text-white" />
                        )}
                      </div>
                      <div>
                        <div className="text-white font-medium">Include Only Liquid Assets</div>
                        <div className="text-purple-300 text-sm">Focus on easily tradeable investments</div>
                      </div>
                    </label>
                  </div>
                </div>
              )}
            </div>

            {/* Futures and Options */}
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <TrendingUp className="w-6 h-6 mr-2 text-purple-400" />
                Futures and Options
              </h2>
              
              <div className="space-y-4">
                <p className="text-purple-200 text-sm mb-4">
                  Advanced trading strategies and F&O predictions powered by machine learning
                </p>
                
                <a 
                  href="https://ml-fno-prediction.vercel.app/" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="block w-full px-4 py-3 rounded-xl bg-gradient-to-r from-purple-700 to-indigo-700 hover:from-purple-600 hover:to-indigo-600 text-white text-center font-medium transition-all duration-200 border border-purple-600/50 hover:border-purple-500 shadow-lg hover:shadow-purple-700/40 transform hover:scale-[1.02]"
                >
                  <div className="flex items-center justify-center space-x-2">
                    <span>Open F&O Predictor</span>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </div>
                </a>
                
                <div className="text-xs text-purple-300 text-center">
                  External ML-powered F&O analysis platform
                </div>
              </div>
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

          {/* Results Panel - Right Column */}
          <div className="lg:col-span-3 bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
            <h2 className="text-xl font-semibold text-white mb-6 flex items-center">
              <TrendingUp className="w-6 h-6 mr-2 text-green-400" />
              Investment Analysis & Recommendations
            </h2>

            <div className="h-full overflow-y-auto pr-2">
              {!comprehensiveData ? (
                <div className="text-center py-12">
                  <div className="text-purple-300 mb-4">
                    <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  </div>
                  <p className="text-purple-200 text-lg mb-2">Ready to analyze your portfolio</p>
                  <p className="text-purple-300 text-sm">
                    Configure your investment parameters and click "Get Recommendations" to get AI-powered recommendations
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                
                {/* Stock Recommendations */}
                {comprehensiveData.recommendations?.stocks && comprehensiveData.recommendations.stocks.length > 0 && (
                  <div className="bg-gray-800/50 rounded-xl p-6 border border-purple-700/50">
                    <div className="flex items-center justify-between mb-4 cursor-pointer" onClick={() => setShowStocksSection(!showStocksSection)}>
                      <h3 className="text-lg font-semibold text-white flex items-center">
                        <TrendingUp className="w-5 h-5 mr-2 text-yellow-400" />
                        Stock Recommendations ({comprehensiveData.recommendations.stocks.reduce((acc, sector) => acc + sector.investment_opportunities, 0)} opportunities)
                      </h3>
                      <ChevronDown className={`w-5 h-5 text-white transition-transform ${showStocksSection ? 'rotate-180' : ''}`} />
                    </div>
                    
                    {showStocksSection && comprehensiveData.recommendations.stocks.map((sector, sectorIndex) => (
                      <div key={sectorIndex} className="mb-6 last:mb-0">
                        <div className="flex items-center justify-between mb-4 p-3 bg-purple-900/20 rounded-lg">
                          <h4 className="text-md font-semibold text-purple-200">
                            {sector.sector} Sector
                          </h4>
                          <div className="text-right">
                            <div className="text-green-400 font-bold">+{sector.predicted_return.toFixed(1)}%</div>
                            <div className="text-purple-300 text-sm">{sector.investment_opportunities} stocks</div>
                          </div>
                        </div>
                        
                        {sector.stocks && sector.stocks.map((stock, stockIndex) => (
                          <div key={stockIndex} className="mb-4 p-4 bg-gray-900/50 rounded-lg border border-gray-700/50">
                            <div className="flex items-center justify-between mb-3">
                              <div>
                                <h5 className="text-white font-semibold">{stock.symbol}</h5>
                                <p className="text-purple-300 text-sm">{stock.company_name}</p>
                              </div>
                              <div className="text-right">
                                <div className="text-green-400 font-bold">+{stock.expected_return.toFixed(1)}%</div>
                                <div className="text-gray-400 text-sm">Expected Return</div>
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                              <div className="text-center p-2 bg-blue-900/20 rounded">
                                <div className="text-blue-300 font-bold">₹{stock.entry_price.toFixed(2)}</div>
                                <div className="text-blue-200 text-xs">Entry Price</div>
                              </div>
                              <div className="text-center p-2 bg-green-900/20 rounded">
                                <div className="text-green-300 font-bold">₹{stock.target_price.toFixed(2)}</div>
                                <div className="text-green-200 text-xs">Target Price</div>
                              </div>
                              <div className="text-center p-2 bg-red-900/20 rounded">
                                <div className="text-red-300 font-bold">₹{stock.stop_loss.toFixed(2)}</div>
                                <div className="text-red-200 text-xs">Stop Loss</div>
                              </div>
                              <div className="text-center p-2 bg-yellow-900/20 rounded">
                                <div className="text-yellow-300 font-bold">{stock.volatility.toFixed(1)}%</div>
                                <div className="text-purple-200 text-xs">Volatility</div>
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-2 gap-4 mb-3">
                              <div className="text-sm">
                                <span className="text-gray-400">Entry Date:</span>
                                <span className="text-white ml-2">{stock.entry_date}</span>
                              </div>
                              <div className="text-sm">
                                <span className="text-gray-400">Exit Date:</span>
                                <span className="text-white ml-2">{stock.exit_date}</span>
                              </div>
                              <div className="text-sm">
                                <span className="text-gray-400">Holding Period:</span>
                                <span className="text-white ml-2">{stock.holding_period} days</span>
                              </div>
                              <div className="text-sm">
                                <span className="text-gray-400">Current Performance:</span>
                                <span className="text-green-400 ml-2">+{stock.current_performance.toFixed(1)}%</span>
                              </div>
                            </div>
                            
                            <div className="p-3 bg-purple-900/10 rounded border-l-4 border-purple-500">
                              <p className="text-purple-200 text-sm">
                                <strong>Strategy:</strong> {stock.investment_strategy}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                )}

                {/* Mutual Fund Recommendations */}
                {comprehensiveData.recommendations?.mutual_funds && comprehensiveData.recommendations.mutual_funds.length > 0 && (
                  <div className="bg-gray-800/50 rounded-xl p-6 border border-blue-700/50">
                    <div className="flex items-center justify-between mb-4 cursor-pointer" onClick={() => setShowMutualFundsSection(!showMutualFundsSection)}>
                      <h3 className="text-lg font-semibold text-white flex items-center">
                        <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
                        Mutual Fund Recommendations ({comprehensiveData.recommendations.mutual_funds.reduce((acc, sector) => acc + sector.investment_opportunities, 0)} opportunities)
                      </h3>
                      <ChevronDown className={`w-5 h-5 text-white transition-transform ${showMutualFundsSection ? 'rotate-180' : ''}`} />
                    </div>
                    
                    {showMutualFundsSection && comprehensiveData.recommendations.mutual_funds.map((sector, sectorIndex) => (
                      <div key={sectorIndex} className="mb-6 last:mb-0">
                        <div className="flex items-center justify-between mb-4 p-3 bg-blue-900/20 rounded-lg">
                          <h4 className="text-md font-semibold text-blue-200">
                            {sector.sector} Funds
                          </h4>
                          <div className="text-right">
                            <div className="text-green-400 font-bold">+{sector.predicted_return.toFixed(1)}%</div>
                            <div className="text-blue-300 text-sm">{sector.investment_opportunities} funds</div>
                          </div>
                        </div>
                        
                        {sector.mutual_funds && sector.mutual_funds.map((fund, fundIndex) => (
                          <div key={fundIndex} className="mb-4 p-4 bg-gray-900/50 rounded-lg border border-gray-700/50">
                            <div className="flex items-center justify-between mb-3">
                              <div>
                                <h5 className="text-white font-semibold">{fund.fund_name}</h5>
                                <p className="text-blue-300 text-sm">Fund Manager: {fund.fund_manager}</p>
                              </div>
                              <div className="text-right">
                                <div className="text-green-400 font-bold">+{fund.expected_return.toFixed(1)}%</div>
                                <div className="text-gray-400 text-sm">Expected Return</div>
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                              <div className="text-center p-2 bg-blue-900/20 rounded">
                                <div className="text-blue-300 font-bold">{fund.expense_ratio.toFixed(2)}%</div>
                                <div className="text-blue-200 text-xs">Expense Ratio</div>
                              </div>
                              <div className="text-center p-2 bg-green-900/20 rounded">
                                <div className="text-green-300 font-bold">+{fund.current_performance.toFixed(1)}%</div>
                                <div className="text-green-200 text-xs">Current Performance</div>
                              </div>
                              <div className="text-center p-2 bg-purple-900/20 rounded">
                                <div className="text-purple-300 font-bold">{fund.risk_level}</div>
                                <div className="text-purple-200 text-xs">Risk Level</div>
                              </div>
                              <div className="text-center p-2 bg-yellow-900/20 rounded">
                                <div className="text-yellow-300 font-bold">₹{fund.minimum_investment}</div>
                                <div className="text-purple-200 text-xs">Min Investment</div>
                              </div>
                            </div>
                            
                            {fund.is_sip_recommended ? (
                              <div className="grid grid-cols-2 gap-4 mb-3">
                                <div className="text-sm">
                                  <span className="text-gray-400">SIP Amount:</span>
                                  <span className="text-white ml-2">₹{fund.sip_amount.toFixed(0)}/month</span>
                                </div>
                                <div className="text-sm">
                                  <span className="text-gray-400">SIP Duration:</span>
                                  <span className="text-white ml-2">{fund.sip_duration_months} months</span>
                                </div>
                              </div>
                            ) : (
                              <div className="grid grid-cols-1 gap-4 mb-3">
                                <div className="text-sm">
                                  <span className="text-gray-400">Lump Sum Amount:</span>
                                  <span className="text-white ml-2">₹{fund.lump_sum_amount.toFixed(0)}</span>
                                </div>
                              </div>
                            )}
                            
                            <div className="p-3 bg-blue-900/10 rounded border-l-4 border-blue-500">
                              <p className="text-blue-200 text-sm">
                                <strong>Strategy:</strong> {fund.investment_strategy}
                              </p>
                              <p className="text-blue-200 text-sm mt-1">
                                <strong>Recommendation:</strong> {fund.is_sip_recommended ? 'SIP (Systematic Investment Plan)' : 'Lump Sum Investment'}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                )}

                {/* Gold Recommendations */}
                {comprehensiveData.recommendations?.gold && comprehensiveData.recommendations.gold.length > 0 && (
                  <div className="bg-gray-800/50 rounded-xl p-6 border border-yellow-700/50">
                    <div className="flex items-center justify-between mb-4 cursor-pointer" onClick={() => setShowGoldSection(!showGoldSection)}>
                      <h3 className="text-lg font-semibold text-white flex items-center">
                        <Coins className="w-5 h-5 mr-2 text-yellow-400" />
                        Gold Recommendations ({comprehensiveData.recommendations.gold.reduce((acc, sector) => acc + sector.investment_opportunities, 0)} opportunities)
                      </h3>
                      <ChevronDown className={`w-5 h-5 text-white transition-transform ${showGoldSection ? 'rotate-180' : ''}`} />
                    </div>
                    
                    {showGoldSection && comprehensiveData.recommendations.gold.map((sector, sectorIndex) => (
                      <div key={sectorIndex} className="mb-6 last:mb-0">
                        <div className="flex items-center justify-between mb-4 p-3 bg-yellow-900/20 rounded-lg">
                          <h4 className="text-md font-semibold text-purple-200">
                            {sector.sector} Investment
                          </h4>
                          <div className="text-right">
                            <div className="text-green-400 font-bold">+{sector.predicted_return.toFixed(1)}%</div>
                            <div className="text-yellow-300 text-sm">{sector.investment_opportunities} options</div>
                          </div>
                        </div>
                        
                        {sector.gold && sector.gold.map((gold, goldIndex) => (
                          <div key={goldIndex} className="mb-4 p-4 bg-gray-900/50 rounded-lg border border-gray-700/50">
                            <div className="flex items-center justify-between mb-3">
                              <div>
                                <h5 className="text-white font-semibold">{gold.investment_type}</h5>
                                <p className="text-yellow-300 text-sm">Liquidity: {gold.liquidity_rating}</p>
                              </div>
                              <div className="text-right">
                                <div className="text-green-400 font-bold">+{gold.expected_return.toFixed(1)}%</div>
                                <div className="text-gray-400 text-sm">Expected Return</div>
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                              <div className="text-center p-2 bg-blue-900/20 rounded">
                                <div className="text-blue-300 font-bold">₹{gold.entry_price.toFixed(2)}</div>
                                <div className="text-blue-200 text-xs">Entry Price</div>
                              </div>
                              <div className="text-center p-2 bg-green-900/20 rounded">
                                <div className="text-green-300 font-bold">₹{gold.exit_price.toFixed(2)}</div>
                                <div className="text-green-200 text-xs">Exit Price</div>
                              </div>
                              <div className="text-center p-2 bg-yellow-900/20 rounded">
                                <div className="text-yellow-300 font-bold">{gold.volatility.toFixed(1)}%</div>
                                <div className="text-purple-200 text-xs">Volatility</div>
                              </div>
                              <div className="text-center p-2 bg-purple-900/20 rounded">
                                <div className="text-purple-300 font-bold">{gold.storage_required ? 'Yes' : 'No'}</div>
                                <div className="text-purple-200 text-xs">Storage Needed</div>
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-2 gap-4 mb-3">
                              <div className="text-sm">
                                <span className="text-gray-400">Entry Date:</span>
                                <span className="text-white ml-2">{gold.entry_date}</span>
                              </div>
                              <div className="text-sm">
                                <span className="text-gray-400">Exit Date:</span>
                                <span className="text-white ml-2">{gold.exit_date}</span>
                              </div>
                              <div className="text-sm">
                                <span className="text-gray-400">Holding Period:</span>
                                <span className="text-white ml-2">{gold.holding_period} days</span>
                              </div>
                              <div className="text-sm">
                                <span className="text-gray-400">Current Performance:</span>
                                <span className="text-green-400 ml-2">+{gold.current_performance.toFixed(1)}%</span>
                              </div>
                            </div>
                            
                            <div className="p-3 bg-yellow-900/10 rounded border-l-4 border-yellow-500">
                              <p className="text-purple-200 text-sm">
                                <strong>Strategy:</strong> {gold.investment_strategy}
                              </p>
                              <p className="text-purple-200 text-sm mt-1">
                                <strong>Tax Implications:</strong> {gold.tax_implications}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
