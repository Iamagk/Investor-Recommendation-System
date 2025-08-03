import React, { useState } from 'react';
import axios from 'axios';
import { TrendingUp, DollarSign, Target, BarChart3, Coins, Building2, Award, ArrowRight, CheckCircle, ChevronDown } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

interface InvestmentOption {
  id: string;
  name: string;
  icon: React.ComponentType<any>;
  description: string;
}

interface DetailedInvestment {
  amount: number;
  percentage: number;
  recommendations: string[];
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
  fund_type: string;
  current_performance: number;
  investment_strategy: string;
  sip_amount: number;
  sip_frequency: string;
  expected_return: number;
  expense_ratio: number;
  risk_rating: string;
  entry_date: string;
  exit_date: string;
  volatility: number;
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
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
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

  const sectorOptions = [
    'Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer Goods',
    'Industrial', 'Real Estate', 'Utilities', 'Materials', 'Telecommunications'
  ];

  const mapRiskToNumeric = (risk: string): number => {
    switch (risk.toLowerCase()) {
      case 'low': return 3;
      case 'medium': return 5;
      case 'high': return 8;
      default: return 5; // Medium
    }
  };

  const fetchComprehensiveData = async () => {
    try {
      const duration = investmentDuration && !isNaN(parseInt(investmentDuration)) ? parseInt(investmentDuration) : 12;
      const response = await axios.get(`${API_BASE_URL}/recommend/comprehensive`, {
        params: {
          investment_amount: parseFloat(investmentAmount || '0'),
          risk_tolerance: riskAppetite.toLowerCase(),
          investment_horizon: duration
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
      
      // Fetch comprehensive data separately
      await fetchComprehensiveData();
      
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

      const mockPrediction: PredictionResult = {
        recommendedAllocation: {
          stocks: stocksPercent,
          mutualFunds: mutualFundsPercent,
          gold: goldPercent
        },
        expectedReturn: minRoiExpectation + (riskMultiplier * 2),
        riskScore: mapRiskToNumeric(riskAppetite),
        recommendations: [
          `Based on your ${riskAppetite.toLowerCase()} risk appetite, we recommend a diversified portfolio.`,
          `Target return: ${minRoiExpectation}% annually`,
          `Investment duration: ${investmentDuration} months`
        ]
      };

      setPrediction(mockPrediction);
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

  const isFormValid = investmentAmount && selectedOptions.length > 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-4 flex items-center justify-center">
            <Award className="w-10 h-10 mr-3 text-yellow-400" />
            Smart Investment Advisor
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
                  <input
                    type="text"
                    value={investmentAmount}
                    onChange={(e) => setInvestmentAmount(e.target.value)}
                    placeholder="Enter amount in INR"
                    className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white placeholder-purple-300/60 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                  />
                  {investmentAmount && (
                    <p className="text-green-400 text-sm mt-1">{formatCurrency(investmentAmount)}</p>
                  )}
                </div>

                <div>
                  <label className="block text-purple-200 text-sm font-medium mb-2">Investment Duration (months)</label>
                  <input
                    type="number"
                    value={investmentDuration}
                    onChange={(e) => setInvestmentDuration(e.target.value)}
                    placeholder="e.g., 12"
                    className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white placeholder-purple-300/60 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                  />
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
              Investment Analysis
            </h2>

            {!prediction ? (
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
                {/* Investment Breakdown */}
                <div className="bg-gray-800/50 rounded-xl p-6 border border-purple-700/50">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
                    Investment Breakdown
                  </h3>
                  
                  <div className="space-y-4">
                    {prediction.recommendedAllocation.stocks > 0 && (
                      <div className="flex items-center justify-between p-3 bg-purple-900/20 rounded-lg border border-purple-800/30">
                        <div className="flex items-center">
                          <TrendingUp className="w-5 h-5 mr-3 text-purple-400" />
                          <span className="text-white font-medium">Stocks</span>
                        </div>
                        <div className="text-right">
                          <div className="text-purple-300 font-bold">{prediction.recommendedAllocation.stocks.toFixed(1)}%</div>
                          <div className="text-green-400 text-sm">
                            {formatCurrency((parseFloat(investmentAmount) * prediction.recommendedAllocation.stocks / 100).toString())}
                          </div>
                        </div>
                      </div>
                    )}

                    {prediction.recommendedAllocation.mutualFunds > 0 && (
                      <div className="flex items-center justify-between p-3 bg-blue-900/20 rounded-lg border border-blue-800/30">
                        <div className="flex items-center">
                          <BarChart3 className="w-5 h-5 mr-3 text-blue-400" />
                          <span className="text-white font-medium">Mutual Funds</span>
                        </div>
                        <div className="text-right">
                          <div className="text-blue-300 font-bold">{prediction.recommendedAllocation.mutualFunds.toFixed(1)}%</div>
                          <div className="text-green-400 text-sm">
                            {formatCurrency((parseFloat(investmentAmount) * prediction.recommendedAllocation.mutualFunds / 100).toString())}
                          </div>
                        </div>
                      </div>
                    )}

                    {prediction.recommendedAllocation.gold > 0 && (
                      <div className="flex items-center justify-between p-3 bg-yellow-900/20 rounded-lg border border-yellow-800/30">
                        <div className="flex items-center">
                          <Coins className="w-5 h-5 mr-3 text-yellow-400" />
                          <span className="text-white font-medium">Gold</span>
                        </div>
                        <div className="text-right">
                          <div className="text-yellow-300 font-bold">{prediction.recommendedAllocation.gold.toFixed(1)}%</div>
                          <div className="text-green-400 text-sm">
                            {formatCurrency((parseFloat(investmentAmount) * prediction.recommendedAllocation.gold / 100).toString())}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Expected Returns */}
                <div className="bg-gray-800/50 rounded-xl p-6 border border-green-700/50">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <Target className="w-5 h-5 mr-2 text-green-400" />
                    Expected Performance
                  </h3>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-4 bg-green-900/20 rounded-lg border border-green-800/30">
                      <div className="text-green-400 text-2xl font-bold">{prediction.expectedReturn.toFixed(1)}%</div>
                      <div className="text-green-300 text-sm">Annual Return</div>
                    </div>
                    <div className="text-center p-4 bg-blue-900/20 rounded-lg border border-blue-800/30">
                      <div className="text-blue-400 text-2xl font-bold">{prediction.riskScore}/10</div>
                      <div className="text-blue-300 text-sm">Risk Score</div>
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                {prediction.recommendations && prediction.recommendations.length > 0 && (
                  <div className="bg-gray-800/50 rounded-xl p-6 border border-blue-700/50">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                      <Award className="w-5 h-5 mr-2 text-yellow-400" />
                      AI Recommendations
                    </h3>
                    
                    <div className="space-y-3">
                      {prediction.recommendations.map((recommendation, index) => (
                        <div key={index} className="p-3 bg-blue-900/20 rounded-lg border border-blue-800/30">
                          <p className="text-blue-200 text-sm">{recommendation}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
