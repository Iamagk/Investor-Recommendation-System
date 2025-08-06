import React, { useState, useEffect } from 'react';
import StockChart from './StockChart';
import { TrendingUp, TrendingDown, Clock, Target, AlertTriangle, RefreshCw, BarChart3, Activity, Zap } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8002';

interface IntradayStock {
  symbol: string;
  company_name: string;
  current_price: number;
  predicted_price: number;
  expected_return: number;
  volatility: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  signal_strength: number;
  entry_time: string;
  exit_time: string;
  target_price: number;
  stop_loss: number;
  volume: number;
  technical_indicators: {
    rsi: number;
    macd: number;
    moving_average_20: number;
    moving_average_50: number;
    bollinger_upper: number;
    bollinger_lower: number;
  };
  risk_score: number;
  confidence: number;
  price_history?: number[];
  dates?: string[];
  previous_close?: number;
  day_change?: number;
  day_change_percent?: number;
  market_cap?: number;
  sector?: string;
  risk_level?: string;
  last_updated?: string;
}

interface IntradayRecommendation {
  recommendations: IntradayStock[];
  total_stocks_analyzed: number;
  market_status: string;
  last_updated: string;
  data_source: string;
}

function IntradayTrading() {
  const [intradayData, setIntradayData] = useState<IntradayRecommendation | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedStock, setSelectedStock] = useState<IntradayStock | null>(null);
  const [showChart, setShowChart] = useState<boolean>(false);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  // Debug log on every render
  console.log('Component render - intradayData:', intradayData);
  console.log('Component render - isLoading:', isLoading);

  // Global test function for debugging
  (window as any).testAPI = async () => {
    try {
      console.log('Testing API directly...');
      const response = await fetch('http://localhost:8002/intraday/recommendations');
      const data = await response.json();
      console.log('Direct API test result:', data);
      return data;
    } catch (error) {
      console.error('Direct API test failed:', error);
      return error;
    }
  };

  const fetchIntradayRecommendations = async () => {
    setIsLoading(true);
    try {
      console.log('Fetching intraday recommendations...');
      
      // More explicit fetch with headers
      const response = await fetch(`${API_BASE_URL}/intraday/recommendations`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        mode: 'cors',
      });
      
      console.log('Fetch response status:', response.status);
      console.log('Fetch response ok:', response.ok);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Response data:', data);
      console.log('Type of data:', typeof data);
      console.log('Has recommendations?', data && data.recommendations);
      console.log('Recommendations length:', data?.recommendations?.length);
      
      if (data && data.recommendations) {
        console.log('Setting intraday data:', data);
        console.log('Recommendations array:', data.recommendations);
        setIntradayData(data);
        setLastUpdated(new Date().toLocaleTimeString());
        console.log('Data set successfully');
      } else {
        console.error('Invalid response structure:', data);
        console.log('Full data object:', data);
      }
    } catch (error) {
      console.error('Error fetching intraday recommendations:', error);
      console.error('Error type:', typeof error);
      console.error('Error name:', error instanceof Error ? error.name : 'Unknown');
      console.error('Error message:', error instanceof Error ? error.message : 'Unknown error');
      
      // More user-friendly error handling
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        alert('Network error: Unable to connect to the backend server. Please check if the server is running on port 8002.');
      } else {
        alert(`Failed to fetch intraday recommendations: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const openStockChart = (stock: IntradayStock) => {
    setSelectedStock(stock);
    setShowChart(true);
  };

  useEffect(() => {
    fetchIntradayRecommendations();
  }, []);

  useEffect(() => {
    console.log('intradayData state changed:', intradayData);
  }, [intradayData]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (autoRefresh) {
      interval = setInterval(fetchIntradayRecommendations, 60000); // Refresh every minute
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'text-green-400 bg-green-900/20 border-green-500';
      case 'SELL': return 'text-red-400 bg-red-900/20 border-red-500';
      case 'HOLD': return 'text-yellow-400 bg-yellow-900/20 border-yellow-500';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-500';
    }
  };

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'BUY': return <TrendingUp className="w-4 h-4" />;
      case 'SELL': return <TrendingDown className="w-4 h-4" />;
      case 'HOLD': return <Target className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-purple-200 mb-4 flex items-center justify-center">
            <Activity className="w-8 h-8 mr-3 text-orange-400" />
            Intraday Trading Dashboard
          </h1>
          <p className="text-purple-200 text-lg">AI-powered real-time stock recommendations with buy/sell signals</p>
        </div>

        {/* Controls */}
        <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl mb-8">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center space-x-4">
              <button
                onClick={fetchIntradayRecommendations}
                disabled={isLoading}
                className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 flex items-center space-x-2 ${
                  isLoading
                    ? 'bg-gray-800/60 cursor-not-allowed opacity-50'
                    : 'bg-gradient-to-r from-orange-700 to-red-700 hover:from-orange-600 hover:to-red-600 text-white shadow-lg hover:shadow-orange-700/40 transform hover:scale-[1.02]'
                }`}
              >
                <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                <span>Refresh Data</span>
              </button>

              <button
                onClick={() => {
                  console.log('Manual test button clicked');
                  console.log('Current intradayData:', intradayData);
                  fetchIntradayRecommendations();
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg"
              >
                Debug Test
              </button>

              <label className="flex items-center space-x-2 text-purple-200">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="rounded bg-gray-800 border-purple-600"
                />
                <span className="text-sm">Auto Refresh (1 min)</span>
              </label>
            </div>

            {lastUpdated && (
              <div className="flex items-center space-x-2 text-purple-300">
                <Clock className="w-4 h-4" />
                <span className="text-sm">Last updated: {lastUpdated}</span>
              </div>
            )}
          </div>
        </div>

        {/* Market Overview */}
        {intradayData && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-xl p-4 border border-purple-800/40">
              <div className="text-purple-300 text-sm mb-1">Market Status</div>
              <div className={`text-lg font-bold ${intradayData.market_status === 'OPEN' ? 'text-green-400' : 'text-red-400'}`}>
                {intradayData.market_status}
              </div>
            </div>
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-xl p-4 border border-purple-800/40">
              <div className="text-purple-300 text-sm mb-1">Market Sentiment</div>
              <div className="text-lg font-bold text-purple-400">
                ACTIVE
              </div>
            </div>
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-xl p-4 border border-purple-800/40">
              <div className="text-purple-300 text-sm mb-1">Total Opportunities</div>
              <div className="text-lg font-bold text-white">{intradayData.total_stocks_analyzed}</div>
            </div>
            <div className="bg-gray-900/40 backdrop-blur-lg rounded-xl p-4 border border-purple-800/40">
              <div className="text-purple-300 text-sm mb-1">Generated At</div>
              <div className="text-lg font-bold text-white">
                {intradayData.last_updated || new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
        )}

        {/* Stock Recommendations */}
        <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl">
          <h2 className="text-xl font-semibold text-white mb-6 flex items-center">
            <Zap className="w-6 h-6 mr-2 text-yellow-400" />
            Today's Intraday Recommendations
          </h2>

          {isLoading ? (
            <div className="text-center py-12">
              <div className="w-8 h-8 border-2 border-purple-300/30 border-t-purple-300 rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-purple-200">Loading intraday recommendations...</p>
            </div>
          ) : !intradayData ? (
            <div className="text-center py-12">
              <Activity className="w-16 h-16 mx-auto mb-4 opacity-50 text-purple-300" />
              <p className="text-purple-200 text-lg mb-2">No data available</p>
              <p className="text-purple-300 text-sm">Click "Refresh Data" to load today's recommendations</p>
              <p className="text-purple-300 text-xs mt-2">Debug: intradayData = {JSON.stringify(intradayData)}</p>
            </div>
          ) : !intradayData.recommendations || intradayData.recommendations.length === 0 ? (
            <div className="text-center py-12">
              <Activity className="w-16 h-16 mx-auto mb-4 opacity-50 text-purple-300" />
              <p className="text-purple-200 text-lg mb-2">No recommendations found</p>
              <p className="text-purple-300 text-sm">No trading signals available at this time</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4">
              {intradayData.recommendations.map((stock, index) => (
                <div
                  key={index}
                  className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50 hover:border-purple-600/50 transition-all cursor-pointer"
                  onClick={() => openStockChart(stock)}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-white font-semibold text-lg">{stock.symbol}</h3>
                      <p className="text-purple-300 text-sm">{stock.company_name}</p>
                    </div>
                    <div className={`px-3 py-1 rounded-lg border ${getSignalColor(stock.signal)} flex items-center space-x-2`}>
                      {getSignalIcon(stock.signal)}
                      <span className="font-semibold">{stock.signal}</span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-4">
                    <div className="text-center p-2 bg-purple-900/20 rounded">
                      <div className="text-purple-300 font-bold">₹{stock.current_price.toFixed(2)}</div>
                      <div className="text-purple-200 text-xs">Current Price</div>
                    </div>
                    <div className="text-center p-2 bg-green-900/20 rounded">
                      <div className="text-green-300 font-bold">₹{stock.target_price.toFixed(2)}</div>
                      <div className="text-green-200 text-xs">Target</div>
                    </div>
                    <div className="text-center p-2 bg-red-900/20 rounded">
                      <div className="text-red-300 font-bold">₹{stock.stop_loss.toFixed(2)}</div>
                      <div className="text-red-200 text-xs">Stop Loss</div>
                    </div>
                    <div className="text-center p-2 bg-blue-900/20 rounded">
                      <div className="text-blue-300 font-bold">{stock.expected_return.toFixed(1)}%</div>
                      <div className="text-blue-200 text-xs">Expected Return</div>
                    </div>
                    <div className="text-center p-2 bg-yellow-900/20 rounded">
                      <div className="text-yellow-300 font-bold">{stock.signal_strength}/10</div>
                      <div className="text-yellow-200 text-xs">Signal Strength</div>
                    </div>
                    <div className="text-center p-2 bg-orange-900/20 rounded">
                      <div className="text-orange-300 font-bold">{stock.confidence.toFixed(0)}%</div>
                      <div className="text-orange-200 text-xs">Confidence</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="text-sm">
                      <span className="text-gray-400">Entry Time:</span>
                      <span className="text-white ml-2">{stock.entry_time}</span>
                    </div>
                    <div className="text-sm">
                      <span className="text-gray-400">Exit Time:</span>
                      <span className="text-white ml-2">{stock.exit_time}</span>
                    </div>
                    <div className="text-sm">
                      <span className="text-gray-400">RSI:</span>
                      <span className={`ml-2 ${stock.technical_indicators.rsi > 70 ? 'text-red-400' : stock.technical_indicators.rsi < 30 ? 'text-green-400' : 'text-white'}`}>
                        {stock.technical_indicators.rsi.toFixed(1)}
                      </span>
                    </div>
                    <div className="text-sm">
                      <span className="text-gray-400">Volume:</span>
                      <span className="text-white ml-2">{stock.volume.toLocaleString()}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className={`px-2 py-1 rounded text-xs ${stock.risk_score <= 3 ? 'bg-green-900/20 text-green-300' : stock.risk_score <= 6 ? 'bg-yellow-900/20 text-yellow-300' : 'bg-red-900/20 text-red-300'}`}>
                        Risk: {stock.risk_score}/10
                      </div>
                      <div className="text-xs text-gray-400">
                        Volatility: {stock.volatility.toFixed(1)}%
                      </div>
                    </div>
                    <div className="text-sm text-purple-300 hover:text-purple-200 transition-colors">
                      Click to view chart →
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Stock Chart Modal */}
        {showChart && selectedStock && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-gray-900/95 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40 shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold text-white flex items-center">
                  <BarChart3 className="w-6 h-6 mr-2 text-purple-400" />
                  {selectedStock.symbol} - {selectedStock.company_name}
                </h3>
                <button
                  onClick={() => setShowChart(false)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Chart */}
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50 mb-6">
                <StockChart 
                  symbol={selectedStock.symbol}
                  technicalIndicators={{
                    rsi: selectedStock.technical_indicators.rsi,
                    macd: selectedStock.technical_indicators.macd,
                    moving_averages: {
                      ma_20: selectedStock.technical_indicators.moving_average_20,
                      ma_50: selectedStock.technical_indicators.moving_average_50
                    }
                  }}
                  priceHistory={selectedStock.price_history || []}
                  dates={selectedStock.dates || []}
                />
              </div>

              {/* Technical Analysis */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                  <h4 className="text-white font-semibold mb-3 flex items-center">
                    <Target className="w-4 h-4 mr-2 text-blue-400" />
                    Technical Indicators
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">RSI:</span>
                      <span className={selectedStock.technical_indicators.rsi > 70 ? 'text-red-400' : selectedStock.technical_indicators.rsi < 30 ? 'text-green-400' : 'text-white'}>
                        {selectedStock.technical_indicators.rsi.toFixed(1)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">MACD:</span>
                      <span className="text-white">{selectedStock.technical_indicators.macd.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">MA(20):</span>
                      <span className="text-white">₹{selectedStock.technical_indicators.moving_average_20.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">MA(50):</span>
                      <span className="text-white">₹{selectedStock.technical_indicators.moving_average_50.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Bollinger Upper:</span>
                      <span className="text-white">₹{selectedStock.technical_indicators.bollinger_upper.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Bollinger Lower:</span>
                      <span className="text-white">₹{selectedStock.technical_indicators.bollinger_lower.toFixed(2)}</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                  <h4 className="text-white font-semibold mb-3 flex items-center">
                    <AlertTriangle className="w-4 h-4 mr-2 text-orange-400" />
                    Trading Plan
                  </h4>
                  <div className="space-y-3">
                    <div className={`p-3 rounded-lg border ${getSignalColor(selectedStock.signal)}`}>
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">Signal: {selectedStock.signal}</span>
                        <span>Strength: {selectedStock.signal_strength}/10</span>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-green-900/20 p-2 rounded border border-green-500">
                        <div className="text-green-300 text-sm">Target Price</div>
                        <div className="text-green-300 font-bold">₹{selectedStock.target_price.toFixed(2)}</div>
                      </div>
                      <div className="bg-red-900/20 p-2 rounded border border-red-500">
                        <div className="text-red-300 text-sm">Stop Loss</div>
                        <div className="text-red-300 font-bold">₹{selectedStock.stop_loss.toFixed(2)}</div>
                      </div>
                    </div>
                    <div className="text-sm text-gray-300">
                      <div>Entry Time: {selectedStock.entry_time}</div>
                      <div>Exit Time: {selectedStock.exit_time}</div>
                      <div>Expected Return: {selectedStock.expected_return.toFixed(1)}%</div>
                      <div>Confidence: {selectedStock.confidence.toFixed(0)}%</div>
                    </div>
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

export default IntradayTrading;
