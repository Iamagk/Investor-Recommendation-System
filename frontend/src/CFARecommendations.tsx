import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Target, DollarSign, Shield, Clock, Zap, BarChart3, AlertTriangle, RefreshCw } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8002';

interface CFARecommendation {
  symbol: string;
  company_name: string;
  sector: string;
  current_price: number;
  target_price: number;
  stop_loss: number;
  investment_recommendation: string;
  risk_rating: string;
  suggested_allocation_percent: number;
  entry_strategy: string;
  entry_price_range: [number, number];
  volume_to_buy: number;
  investment_horizon: string;
  ml_predicted_price_1w: number;
  ml_predicted_price_1m: number;
  ml_predicted_price_3m: number;
  ml_confidence_score: number;
  pe_ratio?: number;
  pb_ratio?: number;
  dividend_yield?: number;
  beta?: number;
  volatility_30d: number;
  intrinsic_value: number;
  margin_of_safety: number;
  last_updated: string;
  next_review_date: string;
  technical_indicators: any;
  support_levels: number[];
  resistance_levels: number[];
}

interface CFAResponse {
  status: string;
  message: string;
  data_source: string;
  analysis_type: string;
  recommendations: CFARecommendation[];
  total_analyzed: number;
  timestamp: string;
}

function CFARecommendations() {
  const [cfaData, setCfaData] = useState<CFAResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedStock, setSelectedStock] = useState<CFARecommendation | null>(null);
  const [showDetails, setShowDetails] = useState<boolean>(false);
  const [portfolioSize, setPortfolioSize] = useState<string>('100000');
  const [riskPreference, setRiskPreference] = useState<string>('MODERATE');

  const fetchCFARecommendations = async () => {
    setIsLoading(true);
    try {
      console.log('Fetching CFA-style recommendations...');
      
      const response = await fetch(`${API_BASE_URL}/cfa/cfa-recommendations?top_n=10&portfolio_size=${portfolioSize}&risk_preference=${riskPreference}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('CFA Response data:', data);
      
      if (data && data.status === 'success') {
        setCfaData(data);
      } else {
        console.error('Invalid CFA response structure:', data);
      }
    } catch (error) {
      console.error('Error fetching CFA recommendations:', error);
      alert(`Failed to fetch CFA recommendations: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchCFARecommendations();
  }, []);

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case 'STRONG_BUY': return 'text-green-400 bg-green-900/20 border-green-500';
      case 'BUY': return 'text-green-300 bg-green-900/20 border-green-400';
      case 'HOLD': return 'text-yellow-400 bg-yellow-900/20 border-yellow-500';
      case 'SELL': return 'text-red-300 bg-red-900/20 border-red-400';
      case 'STRONG_SELL': return 'text-red-400 bg-red-900/20 border-red-500';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-500';
    }
  };

  const getRecommendationIcon = (recommendation: string) => {
    switch (recommendation) {
      case 'STRONG_BUY': return <TrendingUp className="w-5 h-5" />;
      case 'BUY': return <TrendingUp className="w-4 h-4" />;
      case 'HOLD': return <Target className="w-4 h-4" />;
      case 'SELL': return <TrendingDown className="w-4 h-4" />;
      case 'STRONG_SELL': return <TrendingDown className="w-5 h-5" />;
      default: return <BarChart3 className="w-4 h-4" />;
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'CONSERVATIVE': return 'text-blue-400 bg-blue-900/20';
      case 'MODERATE': return 'text-yellow-400 bg-yellow-900/20';
      case 'AGGRESSIVE': return 'text-red-400 bg-red-900/20';
      default: return 'text-gray-400 bg-gray-900/20';
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-purple-200 mb-4 flex items-center justify-center">
            <Shield className="w-8 h-8 mr-3 text-gold-400" />
            CFA Professional Investment Analysis
          </h1>
          <p className="text-purple-200 text-lg">Real-time market data with comprehensive fundamental & technical analysis</p>
        </div>

        {/* Controls */}
        <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl mb-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className="block text-purple-200 text-sm font-medium mb-2">Portfolio Size</label>
              <select
                value={portfolioSize}
                onChange={(e) => setPortfolioSize(e.target.value)}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-purple-500"
              >
                <option value="50000">₹50,000</option>
                <option value="100000">₹1,00,000</option>
                <option value="500000">₹5,00,000</option>
                <option value="1000000">₹10,00,000</option>
                <option value="5000000">₹50,00,000</option>
              </select>
            </div>
            <div>
              <label className="block text-purple-200 text-sm font-medium mb-2">Risk Preference</label>
              <select
                value={riskPreference}
                onChange={(e) => setRiskPreference(e.target.value)}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-purple-500"
              >
                <option value="CONSERVATIVE">Conservative</option>
                <option value="MODERATE">Moderate</option>
                <option value="AGGRESSIVE">Aggressive</option>
              </select>
            </div>
            <div className="flex items-end">
              <button
                onClick={fetchCFARecommendations}
                disabled={isLoading}
                className={`w-full px-6 py-2 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center space-x-2 ${
                  isLoading
                    ? 'bg-gray-800/60 cursor-not-allowed opacity-50'
                    : 'bg-gradient-to-r from-purple-700 to-blue-700 hover:from-purple-600 hover:to-blue-600 text-white shadow-lg'
                }`}
              >
                <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                <span>Analyze Portfolio</span>
              </button>
            </div>
          </div>
        </div>

        {/* Analysis Overview */}
        {cfaData && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-xl p-4 border border-purple-800/40">
              <div className="text-purple-300 text-sm mb-1">Analysis Type</div>
              <div className="text-lg font-bold text-white">{cfaData.analysis_type}</div>
            </div>
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-xl p-4 border border-purple-800/40">
              <div className="text-purple-300 text-sm mb-1">Data Source</div>
              <div className="text-lg font-bold text-green-400">{cfaData.data_source}</div>
            </div>
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-xl p-4 border border-purple-800/40">
              <div className="text-purple-300 text-sm mb-1">Stocks Analyzed</div>
              <div className="text-lg font-bold text-white">{cfaData.total_analyzed}</div>
            </div>
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-xl p-4 border border-purple-800/40">
              <div className="text-purple-300 text-sm mb-1">Portfolio Size</div>
              <div className="text-lg font-bold text-white">{formatCurrency(parseInt(portfolioSize))}</div>
            </div>
          </div>
        )}

        {/* CFA Recommendations */}
        <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
          <h2 className="text-xl font-semibold text-white mb-6 flex items-center">
            <Zap className="w-6 h-6 mr-2 text-yellow-400" />
            Professional Investment Recommendations
          </h2>

          {isLoading ? (
            <div className="text-center py-12">
              <div className="w-8 h-8 border-2 border-purple-300/30 border-t-purple-300 rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-purple-200">Analyzing market data with CFA methodology...</p>
            </div>
          ) : !cfaData ? (
            <div className="text-center py-12">
              <Shield className="w-16 h-16 mx-auto mb-4 opacity-50 text-purple-300" />
              <p className="text-purple-200 text-lg mb-2">No data available</p>
              <p className="text-purple-300 text-sm">Click "Analyze Portfolio" to load CFA recommendations</p>
            </div>
          ) : !cfaData.recommendations || cfaData.recommendations.length === 0 ? (
            <div className="text-center py-12">
              <Shield className="w-16 h-16 mx-auto mb-4 opacity-50 text-purple-300" />
              <p className="text-purple-200 text-lg mb-2">No recommendations found</p>
              <p className="text-purple-300 text-sm">Try adjusting your portfolio size or risk preference</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-6">
              {cfaData.recommendations.map((stock, index) => (
                <div
                  key={index}
                  className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50 hover:border-purple-600/50 transition-all cursor-pointer"
                  onClick={() => {
                    setSelectedStock(stock);
                    setShowDetails(true);
                  }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-white font-semibold text-lg">{stock.symbol}</h3>
                      <p className="text-purple-300 text-sm">{stock.company_name}</p>
                      <p className="text-gray-400 text-xs">{stock.sector}</p>
                    </div>
                    <div className={`px-4 py-2 rounded-lg border ${getRecommendationColor(stock.investment_recommendation)} flex items-center space-x-2`}>
                      {getRecommendationIcon(stock.investment_recommendation)}
                      <span className="font-semibold">{stock.investment_recommendation.replace('_', ' ')}</span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-4">
                    <div className="text-center p-3 bg-blue-900/20 rounded">
                      <div className="text-blue-300 font-bold">{formatCurrency(stock.current_price)}</div>
                      <div className="text-blue-200 text-xs">Current Price</div>
                    </div>
                    <div className="text-center p-3 bg-green-900/20 rounded">
                      <div className="text-green-300 font-bold">{formatCurrency(stock.target_price)}</div>
                      <div className="text-green-200 text-xs">Target Price</div>
                    </div>
                    <div className="text-center p-3 bg-red-900/20 rounded">
                      <div className="text-red-300 font-bold">{formatCurrency(stock.stop_loss)}</div>
                      <div className="text-red-200 text-xs">Stop Loss</div>
                    </div>
                    <div className="text-center p-3 bg-purple-900/20 rounded">
                      <div className="text-purple-300 font-bold">{stock.volume_to_buy}</div>
                      <div className="text-purple-200 text-xs">Shares to Buy</div>
                    </div>
                    <div className="text-center p-3 bg-yellow-900/20 rounded">
                      <div className="text-yellow-300 font-bold">{stock.suggested_allocation_percent.toFixed(1)}%</div>
                      <div className="text-yellow-200 text-xs">Portfolio %</div>
                    </div>
                    <div className="text-center p-3 bg-orange-900/20 rounded">
                      <div className="text-orange-300 font-bold">{stock.margin_of_safety.toFixed(1)}%</div>
                      <div className="text-orange-200 text-xs">Safety Margin</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4 text-sm">
                    <div>
                      <span className="text-gray-400">Strategy:</span>
                      <span className="text-white ml-2">{stock.entry_strategy.replace('_', ' ')}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Horizon:</span>
                      <span className="text-white ml-2">{stock.investment_horizon.replace('_', ' ')}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">ML Confidence:</span>
                      <span className="text-white ml-2">{(stock.ml_confidence_score * 100).toFixed(0)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Intrinsic Value:</span>
                      <span className="text-white ml-2">{formatCurrency(stock.intrinsic_value)}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className={`px-3 py-1 rounded text-xs ${getRiskColor(stock.risk_rating)}`}>
                        Risk: {stock.risk_rating}
                      </div>
                      <div className="text-xs text-gray-400">
                        Volatility: {stock.volatility_30d.toFixed(1)}%
                      </div>
                      {stock.pe_ratio && (
                        <div className="text-xs text-gray-400">
                          P/E: {stock.pe_ratio.toFixed(1)}
                        </div>
                      )}
                    </div>
                    <div className="text-sm text-purple-300 hover:text-purple-200 transition-colors">
                      Click for detailed analysis →
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Detailed Analysis Modal */}
        {showDetails && selectedStock && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-gray-900/95 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold text-white flex items-center">
                  <Shield className="w-6 h-6 mr-2 text-purple-400" />
                  {selectedStock.symbol} - CFA Analysis
                </h3>
                <button
                  onClick={() => setShowDetails(false)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Investment Analysis */}
                <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                  <h4 className="text-white font-semibold mb-3 flex items-center">
                    <Target className="w-4 h-4 mr-2 text-green-400" />
                    Investment Analysis
                  </h4>
                  <div className="space-y-3">
                    <div className={`p-3 rounded-lg border ${getRecommendationColor(selectedStock.investment_recommendation)}`}>
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">Recommendation: {selectedStock.investment_recommendation.replace('_', ' ')}</span>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-blue-900/20 p-3 rounded border border-blue-500">
                        <div className="text-blue-300 text-sm">Current Price</div>
                        <div className="text-blue-300 font-bold">{formatCurrency(selectedStock.current_price)}</div>
                      </div>
                      <div className="bg-green-900/20 p-3 rounded border border-green-500">
                        <div className="text-green-300 text-sm">Target Price</div>
                        <div className="text-green-300 font-bold">{formatCurrency(selectedStock.target_price)}</div>
                      </div>
                      <div className="bg-red-900/20 p-3 rounded border border-red-500">
                        <div className="text-red-300 text-sm">Stop Loss</div>
                        <div className="text-red-300 font-bold">{formatCurrency(selectedStock.stop_loss)}</div>
                      </div>
                      <div className="bg-purple-900/20 p-3 rounded border border-purple-500">
                        <div className="text-purple-300 text-sm">Intrinsic Value</div>
                        <div className="text-purple-300 font-bold">{formatCurrency(selectedStock.intrinsic_value)}</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Position Sizing */}
                <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                  <h4 className="text-white font-semibold mb-3 flex items-center">
                    <DollarSign className="w-4 h-4 mr-2 text-yellow-400" />
                    Position Sizing
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Shares to Buy:</span>
                      <span className="text-white font-bold">{selectedStock.volume_to_buy}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Portfolio Allocation:</span>
                      <span className="text-white">{selectedStock.suggested_allocation_percent.toFixed(2)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Investment Amount:</span>
                      <span className="text-white">{formatCurrency(selectedStock.volume_to_buy * selectedStock.current_price)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Entry Strategy:</span>
                      <span className="text-white">{selectedStock.entry_strategy.replace('_', ' ')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Entry Range:</span>
                      <span className="text-white">{formatCurrency(selectedStock.entry_price_range[0])} - {formatCurrency(selectedStock.entry_price_range[1])}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Investment Horizon:</span>
                      <span className="text-white">{selectedStock.investment_horizon.replace('_', ' ')}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* ML Predictions */}
              <div className="mt-6 bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                <h4 className="text-white font-semibold mb-3 flex items-center">
                  <BarChart3 className="w-4 h-4 mr-2 text-blue-400" />
                  ML Model Predictions
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-blue-900/20 rounded">
                    <div className="text-blue-300 font-bold">{formatCurrency(selectedStock.ml_predicted_price_1w || 0)}</div>
                    <div className="text-blue-200 text-xs">1 Week Target</div>
                  </div>
                  <div className="text-center p-3 bg-green-900/20 rounded">
                    <div className="text-green-300 font-bold">{formatCurrency(selectedStock.ml_predicted_price_1m || 0)}</div>
                    <div className="text-green-200 text-xs">1 Month Target</div>
                  </div>
                  <div className="text-center p-3 bg-purple-900/20 rounded">
                    <div className="text-purple-300 font-bold">{formatCurrency(selectedStock.ml_predicted_price_3m || 0)}</div>
                    <div className="text-purple-200 text-xs">3 Month Target</div>
                  </div>
                  <div className="text-center p-3 bg-yellow-900/20 rounded">
                    <div className="text-yellow-300 font-bold">{(selectedStock.ml_confidence_score * 100).toFixed(0)}%</div>
                    <div className="text-yellow-200 text-xs">ML Confidence</div>
                  </div>
                </div>
              </div>

              {/* Review Information */}
              <div className="mt-6 bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                <h4 className="text-white font-semibold mb-3 flex items-center">
                  <Clock className="w-4 h-4 mr-2 text-orange-400" />
                  Review Schedule
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Last Updated:</span>
                    <span className="text-white ml-2">{new Date(selectedStock.last_updated).toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Next Review:</span>
                    <span className="text-white ml-2">{new Date(selectedStock.next_review_date).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default CFARecommendations;
