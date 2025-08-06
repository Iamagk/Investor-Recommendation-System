import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  TrendingUp, 
  TrendingDown, 
  Clock, 
  Target, 
  Shield, 
  Volume2,
  AlertTriangle,
  CheckCircle,
  Eye,
  Calculator,
  RefreshCw
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8002';

interface DailySignal {
  symbol: string;
  company_name: string;
  signal: string;
  signal_strength: number;
  confidence: number;
  current_price: number;
  entry_price: number;
  target_price: number;
  stop_loss: number;
  expected_return: number;
  risk_reward_ratio: number;
  volume_strength: number;
  time_sensitive: boolean;
  best_entry_time: string;
  best_exit_time: string;
  technical_reasons: string[];
  day_performance: {
    open: number;
    high: number;
    low: number;
    current: number;
    gap_percent: number;
  };
  support_resistance: {
    support1: number;
    resistance1: number;
  };
  market_session: string;
  last_updated: string;
}

interface MarketSummary {
  total_analyzed: number;
  buy_signals: number;
  sell_signals: number;
  hold_signals: number;
  market_session: string;
  best_trading_time: string;
  market_sentiment: string;
}

interface ScannerResult {
  symbol: string;
  company_name: string;
  current_price: number;
  day_change_percent: number;
  volume_ratio: number;
  rsi: number;
  scanner_reason: string;
  support: number;
  resistance: number;
  sector: string;
}

const DailyAnalysis: React.FC = () => {
  const [signals, setSignals] = useState<DailySignal[]>([]);
  const [marketSummary, setMarketSummary] = useState<MarketSummary | null>(null);
  const [scannerResults, setScannerResults] = useState<ScannerResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'signals' | 'scanner' | 'calculator'>('signals');
  const [signalFilter, setSignalFilter] = useState<string>('');
  const [scanType, setScanType] = useState<string>('breakouts');
  const [lastUpdated, setLastUpdated] = useState<string>('');
  
  // Position Calculator State
  const [capital, setCapital] = useState<string>('100000');
  const [riskPercent, setRiskPercent] = useState<string>('2');
  const [entryPrice, setEntryPrice] = useState<string>('');
  const [stopLoss, setStopLoss] = useState<string>('');
  const [positionResult, setPositionResult] = useState<any>(null);

  const fetchDailySignals = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/daily/daily-signals`, {
        params: {
          top_n: 20,
          signal_type: signalFilter || undefined,
          min_confidence: 60
        }
      });
      
      if (response.data.status === 'success') {
        setSignals(response.data.signals);
        setMarketSummary(response.data.market_summary);
        setLastUpdated(response.data.timestamp);
      }
    } catch (error) {
      console.error('Error fetching daily signals:', error);
      alert('Failed to fetch daily signals. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchScannerResults = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/daily/market-scanner`, {
        params: { scan_type: scanType }
      });
      
      if (response.data.status === 'success') {
        setScannerResults(response.data.results);
      }
    } catch (error) {
      console.error('Error fetching scanner results:', error);
      alert('Failed to fetch scanner results. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const calculatePosition = async () => {
    if (!capital || !entryPrice || !stopLoss) {
      alert('Please fill in all required fields');
      return;
    }

    try {
      const response = await axios.get(`${API_BASE_URL}/daily/position-calculator`, {
        params: {
          capital: parseFloat(capital),
          risk_percent: parseFloat(riskPercent),
          entry_price: parseFloat(entryPrice),
          stop_loss: parseFloat(stopLoss)
        }
      });
      
      if (response.data.status === 'success') {
        setPositionResult(response.data);
      }
    } catch (error: any) {
      console.error('Error calculating position:', error);
      alert(error.response?.data?.detail || 'Failed to calculate position size');
    }
  };

  useEffect(() => {
    fetchDailySignals();
  }, [signalFilter]);

  useEffect(() => {
    if (activeTab === 'scanner') {
      fetchScannerResults();
    }
  }, [activeTab, scanType]);

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'STRONG_BUY': return 'text-green-400 bg-green-900/20';
      case 'BUY': return 'text-green-300 bg-green-900/10';
      case 'STRONG_SELL': return 'text-red-400 bg-red-900/20';
      case 'SELL': return 'text-red-300 bg-red-900/10';
      default: return 'text-yellow-300 bg-yellow-900/10';
    }
  };

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'STRONG_BUY':
      case 'BUY':
        return <TrendingUp className="w-4 h-4" />;
      case 'STRONG_SELL':
      case 'SELL':
        return <TrendingDown className="w-4 h-4" />;
      default:
        return <Target className="w-4 h-4" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-purple-200 mb-4 flex items-center justify-center">
            <TrendingUp className="w-10 h-10 mr-4 text-green-400" />
            Daily Trading Analysis
          </h1>
          <p className="text-purple-200 text-lg mb-6">
            Buy Low, Sell High - Advanced Day Trading Signals & Analysis
          </p>
          
          {lastUpdated && (
            <div className="flex items-center justify-center text-purple-300 text-sm">
              <Clock className="w-4 h-4 mr-2" />
              Last updated: {lastUpdated}
            </div>
          )}
        </div>

        {/* Market Summary */}
        {marketSummary && (
          <div className="mb-8 bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
            <h2 className="text-xl font-semibold text-white mb-4">Market Summary</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <div className="text-center p-3 bg-blue-900/20 rounded-lg">
                <div className="text-blue-300 font-bold text-lg">{marketSummary.total_analyzed}</div>
                <div className="text-blue-200 text-sm">Stocks Analyzed</div>
              </div>
              <div className="text-center p-3 bg-green-900/20 rounded-lg">
                <div className="text-green-300 font-bold text-lg">{marketSummary.buy_signals}</div>
                <div className="text-green-200 text-sm">Buy Signals</div>
              </div>
              <div className="text-center p-3 bg-red-900/20 rounded-lg">
                <div className="text-red-300 font-bold text-lg">{marketSummary.sell_signals}</div>
                <div className="text-red-200 text-sm">Sell Signals</div>
              </div>
              <div className="text-center p-3 bg-yellow-900/20 rounded-lg">
                <div className="text-yellow-300 font-bold text-lg">{marketSummary.hold_signals}</div>
                <div className="text-yellow-200 text-sm">Hold Signals</div>
              </div>
              <div className="text-center p-3 bg-purple-900/20 rounded-lg">
                <div className="text-purple-300 font-bold text-lg">{marketSummary.market_session}</div>
                <div className="text-purple-200 text-sm">Market Session</div>
              </div>
              <div className="text-center p-3 bg-indigo-900/20 rounded-lg">
                <div className="text-indigo-300 font-bold text-lg">{marketSummary.market_sentiment}</div>
                <div className="text-indigo-200 text-sm">Sentiment</div>
              </div>
            </div>
            <div className="mt-4 text-center">
              <div className="text-purple-200 text-sm">
                ⏰ <strong>Best Trading Times:</strong> {marketSummary.best_trading_time}
              </div>
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="mb-6">
          <div className="flex justify-center">
            <div className="bg-gray-900/60 backdrop-blur-lg rounded-xl p-2 border border-purple-800/40">
              <button
                onClick={() => setActiveTab('signals')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  activeTab === 'signals'
                    ? 'bg-gradient-to-r from-purple-700 to-indigo-700 text-white shadow-lg'
                    : 'text-purple-300 hover:text-white hover:bg-purple-800/30'
                }`}
              >
                📊 Trading Signals
              </button>
              <button
                onClick={() => setActiveTab('scanner')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  activeTab === 'scanner'
                    ? 'bg-gradient-to-r from-purple-700 to-indigo-700 text-white shadow-lg'
                    : 'text-purple-300 hover:text-white hover:bg-purple-800/30'
                }`}
              >
                🔍 Market Scanner
              </button>
              <button
                onClick={() => setActiveTab('calculator')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  activeTab === 'calculator'
                    ? 'bg-gradient-to-r from-purple-700 to-indigo-700 text-white shadow-lg'
                    : 'text-purple-300 hover:text-white hover:bg-purple-800/30'
                }`}
              >
                🧮 Position Calculator
              </button>
            </div>
          </div>
        </div>

        {/* Trading Signals Tab */}
        {activeTab === 'signals' && (
          <div>
            {/* Filters and Controls */}
            <div className="mb-6 bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center space-x-4">
                  <label className="text-purple-200 text-sm font-medium">Filter by Signal:</label>
                  <select
                    value={signalFilter}
                    onChange={(e) => setSignalFilter(e.target.value)}
                    className="px-4 py-2 rounded-lg bg-gray-800/60 border border-purple-600/40 text-white focus:border-purple-500"
                  >
                    <option value="">All Signals</option>
                    <option value="STRONG_BUY">Strong Buy</option>
                    <option value="BUY">Buy</option>
                    <option value="STRONG_SELL">Strong Sell</option>
                    <option value="SELL">Sell</option>
                    <option value="HOLD">Hold</option>
                  </select>
                </div>
                <button
                  onClick={fetchDailySignals}
                  disabled={loading}
                  className="flex items-center space-x-2 px-4 py-2 bg-purple-700 hover:bg-purple-600 text-white rounded-lg transition-all duration-200 disabled:opacity-50"
                >
                  <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            {/* Trading Signals */}
            {loading ? (
              <div className="text-center py-12">
                <div className="w-8 h-8 border-2 border-purple-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-purple-200">Loading trading signals...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {signals.map((signal, index) => (
                  <div key={index} className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-semibold text-white">{signal.symbol}</h3>
                        <p className="text-purple-300 text-sm">{signal.company_name}</p>
                      </div>
                      <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${getSignalColor(signal.signal)}`}>
                        {getSignalIcon(signal.signal)}
                        <span className="font-medium">{signal.signal.replace('_', ' ')}</span>
                      </div>
                    </div>

                    {/* Price Information */}
                    <div className="grid grid-cols-3 gap-4 mb-4">
                      <div className="text-center p-3 bg-blue-900/20 rounded-lg">
                        <div className="text-blue-300 font-bold">₹{signal.current_price}</div>
                        <div className="text-blue-200 text-xs">Current</div>
                      </div>
                      <div className="text-center p-3 bg-green-900/20 rounded-lg">
                        <div className="text-green-300 font-bold">₹{signal.target_price}</div>
                        <div className="text-green-200 text-xs">Target</div>
                      </div>
                      <div className="text-center p-3 bg-red-900/20 rounded-lg">
                        <div className="text-red-300 font-bold">₹{signal.stop_loss}</div>
                        <div className="text-red-200 text-xs">Stop Loss</div>
                      </div>
                    </div>

                    {/* Performance Metrics */}
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="flex items-center justify-between p-2 bg-gray-800/40 rounded">
                        <span className="text-gray-400 text-sm">Expected Return:</span>
                        <span className="text-green-400 font-medium">{signal.expected_return.toFixed(1)}%</span>
                      </div>
                      <div className="flex items-center justify-between p-2 bg-gray-800/40 rounded">
                        <span className="text-gray-400 text-sm">Risk:Reward:</span>
                        <span className="text-yellow-400 font-medium">1:{signal.risk_reward_ratio.toFixed(1)}</span>
                      </div>
                      <div className="flex items-center justify-between p-2 bg-gray-800/40 rounded">
                        <span className="text-gray-400 text-sm">Confidence:</span>
                        <span className="text-blue-400 font-medium">{signal.confidence.toFixed(0)}%</span>
                      </div>
                      <div className="flex items-center justify-between p-2 bg-gray-800/40 rounded">
                        <span className="text-gray-400 text-sm">Volume:</span>
                        <div className="flex items-center">
                          <Volume2 className="w-3 h-3 mr-1 text-purple-400" />
                          <span className="text-purple-400 font-medium">{signal.volume_strength}/10</span>
                        </div>
                      </div>
                    </div>

                    {/* Day Performance */}
                    <div className="mb-4 p-3 bg-gray-800/20 rounded-lg">
                      <div className="text-gray-400 text-xs mb-2">Day Performance:</div>
                      <div className="grid grid-cols-4 gap-2 text-xs">
                        <div><span className="text-gray-400">Open:</span> <span className="text-white">₹{signal.day_performance.open}</span></div>
                        <div><span className="text-gray-400">High:</span> <span className="text-green-400">₹{signal.day_performance.high}</span></div>
                        <div><span className="text-gray-400">Low:</span> <span className="text-red-400">₹{signal.day_performance.low}</span></div>
                        <div><span className="text-gray-400">Gap:</span> 
                          <span className={signal.day_performance.gap_percent >= 0 ? 'text-green-400' : 'text-red-400'}>
                            {signal.day_performance.gap_percent > 0 ? '+' : ''}{signal.day_performance.gap_percent.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Timing */}
                    <div className="mb-4 p-3 bg-yellow-900/10 rounded-lg border-l-4 border-yellow-500">
                      <div className="flex items-center text-yellow-200 text-sm mb-1">
                        <Clock className="w-4 h-4 mr-2" />
                        Timing Recommendation
                      </div>
                      <div className="text-yellow-100 text-xs">
                        <div>Entry: {signal.best_entry_time}</div>
                        <div>Exit: {signal.best_exit_time}</div>
                      </div>
                    </div>

                    {/* Technical Reasons */}
                    {signal.technical_reasons.length > 0 && (
                      <div className="p-3 bg-purple-900/10 rounded-lg border-l-4 border-purple-500">
                        <div className="text-purple-200 text-sm font-medium mb-2">Technical Analysis:</div>
                        <ul className="text-purple-100 text-xs space-y-1">
                          {signal.technical_reasons.slice(0, 3).map((reason, i) => (
                            <li key={i} className="flex items-center">
                              <CheckCircle className="w-3 h-3 mr-2 text-purple-400" />
                              {reason}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Support/Resistance */}
                    <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                      <div className="text-center p-2 bg-red-900/10 rounded">
                        <div className="text-red-300">Support: ₹{signal.support_resistance.support1}</div>
                      </div>
                      <div className="text-center p-2 bg-green-900/10 rounded">
                        <div className="text-green-300">Resistance: ₹{signal.support_resistance.resistance1}</div>
                      </div>
                    </div>

                    {signal.time_sensitive && (
                      <div className="mt-3 flex items-center justify-center p-2 bg-orange-900/20 rounded-lg border border-orange-600/40">
                        <AlertTriangle className="w-4 h-4 mr-2 text-orange-400" />
                        <span className="text-orange-200 text-sm font-medium">Time Sensitive Opportunity</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Market Scanner Tab */}
        {activeTab === 'scanner' && (
          <div>
            {/* Scanner Controls */}
            <div className="mb-6 bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <label className="text-purple-200 text-sm font-medium">Scan Type:</label>
                  <select
                    value={scanType}
                    onChange={(e) => setScanType(e.target.value)}
                    className="px-4 py-2 rounded-lg bg-gray-800/60 border border-purple-600/40 text-white focus:border-purple-500"
                  >
                    <option value="breakouts">Breakouts</option>
                    <option value="oversold">Oversold</option>
                    <option value="overbought">Overbought</option>
                    <option value="high_volume">High Volume</option>
                  </select>
                </div>
                <button
                  onClick={fetchScannerResults}
                  disabled={loading}
                  className="flex items-center space-x-2 px-4 py-2 bg-purple-700 hover:bg-purple-600 text-white rounded-lg transition-all duration-200 disabled:opacity-50"
                >
                  <Eye className={`w-4 h-4 ${loading ? 'animate-pulse' : ''}`} />
                  <span>Scan Market</span>
                </button>
              </div>
            </div>

            {/* Scanner Results */}
            {loading ? (
              <div className="text-center py-12">
                <div className="w-8 h-8 border-2 border-purple-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-purple-200">Scanning market...</p>
              </div>
            ) : (
              <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
                <h3 className="text-xl font-semibold text-white mb-4">Scanner Results - {scanType.replace('_', ' ')}</h3>
                
                {scannerResults.length === 0 ? (
                  <div className="text-center py-8">
                    <Eye className="w-12 h-12 mx-auto mb-4 text-purple-400 opacity-50" />
                    <p className="text-purple-300">No matches found for current scan criteria</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {scannerResults.map((result, index) => (
                      <div key={index} className="p-4 bg-gray-800/40 rounded-lg border border-gray-700/50">
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <h4 className="text-white font-semibold">{result.symbol}</h4>
                            <p className="text-purple-300 text-sm">{result.company_name}</p>
                          </div>
                          <div className="text-right">
                            <div className="text-white font-bold">₹{result.current_price}</div>
                            <div className={`text-sm ${result.day_change_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {result.day_change_percent > 0 ? '+' : ''}{result.day_change_percent.toFixed(1)}%
                            </div>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2 mb-3 text-xs">
                          <div className="text-center p-2 bg-blue-900/20 rounded">
                            <div className="text-blue-300">Vol: {result.volume_ratio.toFixed(1)}x</div>
                          </div>
                          <div className="text-center p-2 bg-purple-900/20 rounded">
                            <div className="text-purple-300">RSI: {result.rsi.toFixed(0)}</div>
                          </div>
                          <div className="text-center p-2 bg-red-900/20 rounded">
                            <div className="text-red-300">S: ₹{result.support}</div>
                          </div>
                          <div className="text-center p-2 bg-green-900/20 rounded">
                            <div className="text-green-300">R: ₹{result.resistance}</div>
                          </div>
                        </div>
                        
                        <div className="p-2 bg-yellow-900/10 rounded border-l-4 border-yellow-500">
                          <p className="text-yellow-200 text-xs">{result.scanner_reason}</p>
                        </div>
                        
                        <div className="mt-2 text-center">
                          <span className="text-gray-400 text-xs">{result.sector}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Position Calculator Tab */}
        {activeTab === 'calculator' && (
          <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
            <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
              <Calculator className="w-6 h-6 mr-2 text-purple-400" />
              Position Size Calculator
            </h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Input Section */}
              <div className="space-y-4">
                <div>
                  <label className="block text-purple-200 text-sm font-medium mb-2">
                    Available Capital (₹)
                  </label>
                  <input
                    type="text"
                    value={capital}
                    onChange={(e) => setCapital(e.target.value)}
                    placeholder="100000"
                    className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white placeholder-purple-300/60 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20"
                  />
                </div>
                
                <div>
                  <label className="block text-purple-200 text-sm font-medium mb-2">
                    Risk Per Trade (%) - Max 5%
                  </label>
                  <input
                    type="text"
                    value={riskPercent}
                    onChange={(e) => setRiskPercent(e.target.value)}
                    placeholder="2"
                    className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white placeholder-purple-300/60 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20"
                  />
                </div>
                
                <div>
                  <label className="block text-purple-200 text-sm font-medium mb-2">
                    Entry Price (₹)
                  </label>
                  <input
                    type="text"
                    value={entryPrice}
                    onChange={(e) => setEntryPrice(e.target.value)}
                    placeholder="100.00"
                    className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white placeholder-purple-300/60 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20"
                  />
                </div>
                
                <div>
                  <label className="block text-purple-200 text-sm font-medium mb-2">
                    Stop Loss Price (₹)
                  </label>
                  <input
                    type="text"
                    value={stopLoss}
                    onChange={(e) => setStopLoss(e.target.value)}
                    placeholder="95.00"
                    className="w-full px-4 py-3 rounded-xl bg-gray-800/60 border border-purple-600/40 text-white placeholder-purple-300/60 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20"
                  />
                </div>
                
                <button
                  onClick={calculatePosition}
                  className="w-full py-3 px-6 bg-gradient-to-r from-purple-700 to-indigo-700 hover:from-purple-600 hover:to-indigo-600 text-white rounded-xl font-medium transition-all duration-200 flex items-center justify-center space-x-2"
                >
                  <Calculator className="w-5 h-5" />
                  <span>Calculate Position</span>
                </button>
              </div>

              {/* Results Section */}
              <div>
                {positionResult ? (
                  <div className="space-y-4">
                    <div className="p-4 bg-green-900/20 rounded-lg border border-green-700/50">
                      <h4 className="text-green-300 font-semibold mb-3">Position Calculation</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Recommended Quantity:</span>
                          <span className="text-white font-medium">{positionResult.position_calculation.recommended_quantity} shares</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Position Value:</span>
                          <span className="text-white font-medium">₹{positionResult.position_calculation.position_value.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Capital Used:</span>
                          <span className="text-white font-medium">{positionResult.position_calculation.capital_used_percent}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Risk Amount:</span>
                          <span className="text-red-300 font-medium">₹{positionResult.position_calculation.risk_amount.toLocaleString()}</span>
                        </div>
                      </div>
                    </div>

                    <div className="p-4 bg-blue-900/20 rounded-lg border border-blue-700/50">
                      <h4 className="text-blue-300 font-semibold mb-3">Risk Management</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Max Loss if Stopped:</span>
                          <span className="text-red-300 font-medium">₹{positionResult.risk_management.max_loss_if_stopped.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Remaining Capital:</span>
                          <span className="text-white font-medium">₹{positionResult.risk_management.remaining_capital.toLocaleString()}</span>
                        </div>
                      </div>
                      
                      <div className="mt-4">
                        <div className="text-gray-400 text-xs mb-2">Suggested Targets:</div>
                        <div className="grid grid-cols-3 gap-2">
                          {positionResult.risk_management.suggested_targets.map((target: number, index: number) => (
                            <div key={index} className="text-center p-2 bg-green-900/20 rounded">
                              <div className="text-green-300 text-sm">₹{target}</div>
                              <div className="text-green-200 text-xs">{index + 1}:1 R:R</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="p-4 bg-yellow-900/10 rounded-lg border-l-4 border-yellow-500">
                      <p className="text-yellow-200 text-sm">{positionResult.position_calculation.recommendation}</p>
                    </div>

                    <div className="p-4 bg-purple-900/10 rounded-lg">
                      <h4 className="text-purple-300 font-semibold mb-2">Trading Rules</h4>
                      <ul className="text-purple-200 text-xs space-y-1">
                        {positionResult.trading_rules.map((rule: string, index: number) => (
                          <li key={index} className="flex items-start">
                            <Shield className="w-3 h-3 mr-2 mt-0.5 text-purple-400 flex-shrink-0" />
                            {rule}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Calculator className="w-16 h-16 mx-auto mb-4 text-purple-400 opacity-50" />
                    <p className="text-purple-300">Enter your trade details to calculate optimal position size</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DailyAnalysis;
