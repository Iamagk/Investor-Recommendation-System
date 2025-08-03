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
  
  // UI State
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

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

      const response = await axios.post(`${API_BASE_URL}/api/portfolio/predict`, requestData);
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
          <div className="mb-6">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gray-800/60 backdrop-blur-sm rounded-full border-2 border-purple-800/50 overflow-hidden">
              <img 
                src="/profile.jpg" 
                alt="Profile" 
                className="w-full h-full object-cover"
                onError={(e) => {
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

              {/* Investment Amount */}
              <div className="mb-6">
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

              {/* Asset Preferences */}
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

              {/* Backtesting Section */}
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

              {/* Advanced Filters */}
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

            {/* Results Panel - Wider Right Column */}
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
                    Configure your investment parameters and click "Get Recommendations" to get AI-powered recommendations
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
        
        .slider::-webkit-slider-track {
          background: rgba(31, 41, 55, 0.6);
          height: 8px;
          border-radius: 4px;
          border: 1px solid rgba(107, 70, 193, 0.3);
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
        
        .slider::-moz-range-track {
          background: rgba(31, 41, 55, 0.6);
          height: 8px;
          border-radius: 4px;
          border: 1px solid rgba(107, 70, 193, 0.3);
        }
      `}</style>
    </div>
  );
}

export default App;
