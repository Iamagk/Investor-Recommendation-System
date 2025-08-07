import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  TrendingUp, 
  TrendingDown, 
  Settings, 
  Target, 
  Clock,
  BarChart3,
  Zap,
  Award,
  Activity,
  AlertCircle,
  CheckCircle,
  DollarSign,
  Calendar,
  BookOpen
} from 'lucide-react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
  Bar
} from 'recharts';

const API_BASE_URL = 'http://localhost:8002';

interface ProfessionalSignal {
  symbol: string;
  company_name: string;
  confidence_score: number;
  direction: string;
  entry_price: number;
  stop_loss: number;
  target_price_1: number;
  target_price_2: number;
  expected_roi: number;
  risk_reward_ratio: number;
  signal_time: string;
  volatility_level: string;
  sector: string;
  chart_data: any;
  signal_log: string[];
  // LLM Explanations
  market_analysis?: string;
  action_plan?: string;
  risk_management?: string;
  timing?: string;
}

interface TradingPreferences {
  capital_allocation: string;
  risk_appetite: string;
  preferred_sectors: string;
  max_trades_per_day: string;
  trading_style: string;
  holding_duration_minutes: string;
}

interface BacktestResult {
  total_signals: number;
  profitable_signals: number;
  loss_signals: number;
  accuracy_rate: number;
  avg_profit: number;
  avg_loss: number;
  risk_reward_achieved: number;
}

interface EODReview {
  trading_summary: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_pnl: number;
    roi_for_day: number;
  };
  performance_metrics: {
    avg_holding_time_minutes: number;
    largest_win: number;
    largest_loss: number;
    risk_reward_realized: number;
    accuracy_vs_predicted: number;
  };
  lessons_learned: string[];
  recommendations_for_tomorrow: string[];
}

const ProfessionalTrading: React.FC = () => {
  const [signals, setSignals] = useState<ProfessionalSignal[]>([]);
  const [enhancedSignals, setEnhancedSignals] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'signals' | 'enhanced' | 'backtest' | 'review'>('signals');
  const [backtestResults, setBacktestResults] = useState<BacktestResult | null>(null);
  const [eodReview, setEODReview] = useState<EODReview | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [currentPrices, setCurrentPrices] = useState<{[key: string]: number}>({});
  
  // Trading Preferences
  const [preferences, setPreferences] = useState<TradingPreferences>({
    capital_allocation: '10000',
    risk_appetite: 'MEDIUM',
    preferred_sectors: '',
    max_trades_per_day: '3',
    trading_style: 'MOMENTUM',
    holding_duration_minutes: '30'
  });

  const fetchProfessionalSignals = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/pro/enhanced-signals`, null, {
        params: {
          capital_allocation: parseFloat(preferences.capital_allocation),
          risk_appetite: preferences.risk_appetite,
          max_trades_per_day: parseInt(preferences.max_trades_per_day),
          trading_style: preferences.trading_style,
          holding_duration_minutes: parseInt(preferences.holding_duration_minutes)
        }
      });
      
      if (response.data.signals) {
        // Convert enhanced signals to include additional fields for display
        const convertedSignals = response.data.signals.map((signal: any) => ({
          ...signal,
          chart_data: signal.chart_data || {},
          shares_recommended: signal.shares_recommended || 0,
          investment_amount: signal.investment_amount || 0,
          profit_potential_1: signal.profit_potential_1 || 0,
          profit_potential_2: signal.profit_potential_2 || 0
        }));
        setSignals(convertedSignals);
      }
    } catch (error) {
      console.error('Error fetching professional signals:', error);
      alert('Failed to fetch professional signals. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchBacktestResults = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/pro/backtest-signals`, {
        params: { trading_style: preferences.trading_style }
      });
      
      if (response.data.status === 'success') {
        setBacktestResults(response.data.summary);
      }
    } catch (error) {
      console.error('Error fetching backtest results:', error);
      alert('Failed to fetch backtest results. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchEODReview = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/pro/end-of-day-review`);
      
      if (response.data.status === 'success') {
        setEODReview({
          trading_summary: response.data.trading_summary,
          performance_metrics: response.data.performance_metrics,
          lessons_learned: response.data.lessons_learned,
          recommendations_for_tomorrow: response.data.recommendations_for_tomorrow
        });
      }
    } catch (error) {
      console.error('Error fetching EOD review:', error);
      alert('Failed to fetch EOD review. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchEnhancedSignals = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/pro/enhanced-signals`, null, {
        params: {
          capital_allocation: parseFloat(preferences.capital_allocation),
          risk_appetite: preferences.risk_appetite,
          max_trades_per_day: parseInt(preferences.max_trades_per_day),
          trading_style: preferences.trading_style,
          holding_duration_minutes: parseInt(preferences.holding_duration_minutes)
        }
      });
      
      if (response.data.signals) {
        // Convert enhanced signals to include additional fields for display
        const convertedSignals = response.data.signals.map((signal: any) => ({
          ...signal,
          chart_data: signal.chart_data || {},
          shares_recommended: signal.shares_recommended || 0,
          investment_amount: signal.investment_amount || 0,
          profit_potential_1: signal.profit_potential_1 || 0,
          profit_potential_2: signal.profit_potential_2 || 0
        }));
        setEnhancedSignals(convertedSignals);
      }
    } catch (error) {
      console.error('Error fetching enhanced signals:', error);
      alert('Failed to fetch enhanced signals. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchCurrentPrices = async (symbols: string[]) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/pro/current-prices`, {
        symbols: symbols
      });
      
      if (response.data.prices) {
        setCurrentPrices(response.data.prices);
        setLastUpdated(new Date().toLocaleTimeString());
        
        // Update chart data with latest price points and recalculate ROI
        setSignals(prevSignals => prevSignals.map(signal => {
          const currentPrice = response.data.prices[signal.symbol];
          let updatedSignal = { ...signal };
          
          // Update chart data if available
          if (currentPrice && signal.chart_data && signal.chart_data.timestamps) {
            // Add new price point to chart data
            const newTimestamp = new Date().toLocaleTimeString('en-US', { 
              hour12: false, 
              hour: '2-digit', 
              minute: '2-digit' 
            });
            
            const updatedChartData = {
              ...signal.chart_data,
              timestamps: [...signal.chart_data.timestamps, newTimestamp],
              ohlc: {
                ...signal.chart_data.ohlc,
                close: [...signal.chart_data.ohlc.close, currentPrice],
                open: [...signal.chart_data.ohlc.open, currentPrice],
                high: [...signal.chart_data.ohlc.high, currentPrice],
                low: [...signal.chart_data.ohlc.low, currentPrice]
              },
              volume: signal.chart_data.volume ? 
                [...signal.chart_data.volume, signal.chart_data.volume[signal.chart_data.volume.length - 1]] : 
                null
            };
            
            // Keep only last 50 data points to prevent chart from becoming too cluttered
            if (updatedChartData.timestamps.length > 50) {
              updatedChartData.timestamps = updatedChartData.timestamps.slice(-50);
              updatedChartData.ohlc.close = updatedChartData.ohlc.close.slice(-50);
              updatedChartData.ohlc.open = updatedChartData.ohlc.open.slice(-50);
              updatedChartData.ohlc.high = updatedChartData.ohlc.high.slice(-50);
              updatedChartData.ohlc.low = updatedChartData.ohlc.low.slice(-50);
              if (updatedChartData.volume) {
                updatedChartData.volume = updatedChartData.volume.slice(-50);
              }
            }
            
            updatedSignal.chart_data = updatedChartData;
          }
          
          // Recalculate ROI if price has changed
          if (currentPrice && currentPrice !== signal.entry_price) {
            const newROI = ((currentPrice - signal.entry_price) / signal.entry_price) * 100;
            updatedSignal.expected_roi = newROI;
          }
          
          return updatedSignal;
        }));
      }
    } catch (error) {
      console.error('Error fetching current prices:', error);
    }
  };

  const refreshSignals = async () => {
    if (signals.length > 0) {
      const symbols = signals.map(signal => signal.symbol);
      await fetchCurrentPrices(symbols);
    }
  };

  useEffect(() => {
    if (activeTab === 'signals') {
      fetchProfessionalSignals();
    } else if (activeTab === 'enhanced') {
      fetchEnhancedSignals();
    } else if (activeTab === 'backtest') {
      fetchBacktestResults();
    } else if (activeTab === 'review') {
      fetchEODReview();
    }
  }, [activeTab, preferences]);

  // Auto-refresh prices every 30 seconds
  useEffect(() => {
    let interval: number;
    
    if (autoRefresh && activeTab === 'signals' && signals.length > 0) {
      interval = setInterval(() => {
        refreshSignals();
      }, 30000); // 30 seconds
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [autoRefresh, activeTab, signals.length]);

  // Fetch initial prices after signals are loaded
  useEffect(() => {
    if (signals.length > 0) {
      const symbols = signals.map(signal => signal.symbol);
      fetchCurrentPrices(symbols);
    }
  }, [signals.length > 0 ? signals.map(s => s.symbol).join(',') : '']);

  const getDirectionColor = (direction: string) => {
    return direction === 'BUY' ? 'text-green-400 bg-green-900/20' : 'text-red-400 bg-red-900/20';
  };

  const getVolatilityColor = (level: string) => {
    switch (level) {
      case 'LOW': return 'text-blue-400 bg-blue-900/20';
      case 'MEDIUM': return 'text-yellow-400 bg-yellow-900/20';
      case 'HIGH': return 'text-red-400 bg-red-900/20';
      default: return 'text-gray-400 bg-gray-900/20';
    }
  };

  const getTradingStyleIcon = (style: string) => {
    switch (style) {
      case 'SCALPING': return <Zap className="w-4 h-4" />;
      case 'MOMENTUM': return <TrendingUp className="w-4 h-4" />;
      case 'SWING': return <Activity className="w-4 h-4" />;
      default: return <BarChart3 className="w-4 h-4" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-purple-200 mb-4 flex items-center justify-center">
            <Award className="w-10 h-10 mr-4 text-gold-400" />
            CFA Professional Trading System
          </h1>
          <p className="text-purple-200 text-lg mb-4">
            Professional-grade trading signals with comprehensive risk management
          </p>
          
          {/* Real-time Status */}
          <div className="flex items-center justify-center space-x-6 mb-6">
            <div className="flex items-center space-x-2 px-4 py-2 bg-gray-900/60 rounded-lg border border-purple-800/40">
              <div className={`w-3 h-3 rounded-full ${autoRefresh ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`}></div>
              <span className="text-purple-200 text-sm">
                {autoRefresh ? 'Real-time Updates ON' : 'Real-time Updates OFF'}
              </span>
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className="text-purple-300 hover:text-white transition-colors"
              >
                <Settings className="w-4 h-4" />
              </button>
            </div>
            
            {lastUpdated && (
              <div className="flex items-center space-x-2 px-4 py-2 bg-gray-900/60 rounded-lg border border-purple-800/40">
                <Clock className="w-4 h-4 text-blue-400" />
                <span className="text-purple-200 text-sm">Last Updated: {lastUpdated}</span>
              </div>
            )}
            
            {signals.length > 0 && (
              <button
                onClick={refreshSignals}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-900/60 rounded-lg border border-blue-600/40 hover:bg-blue-900/80 transition-colors"
              >
                <Activity className="w-4 h-4 text-blue-400" />
                <span className="text-blue-200 text-sm">Refresh Now</span>
              </button>
            )}
          </div>
        </div>

        {/* Trading Preferences Panel */}
        <div className="mb-8 bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
          <div className="flex items-center mb-4">
            <Settings className="w-6 h-6 mr-2 text-purple-400" />
            <h2 className="text-xl font-semibold text-white">Trading Preferences</h2>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div>
              <label className="block text-purple-200 text-sm font-medium mb-2">Capital (₹)</label>
              <input
                type="text"
                value={preferences.capital_allocation}
                onChange={(e) => setPreferences({...preferences, capital_allocation: e.target.value})}
                className="w-full px-3 py-2 rounded-lg bg-gray-800/60 border border-purple-600/40 text-white text-sm focus:border-purple-500"
              />
            </div>
            
            <div>
              <label className="block text-purple-200 text-sm font-medium mb-2">Risk Appetite</label>
              <select
                value={preferences.risk_appetite}
                onChange={(e) => setPreferences({...preferences, risk_appetite: e.target.value})}
                className="w-full px-3 py-2 rounded-lg bg-gray-800/60 border border-purple-600/40 text-white text-sm focus:border-purple-500"
              >
                <option value="LOW">Low</option>
                <option value="MEDIUM">Medium</option>
                <option value="HIGH">High</option>
              </select>
            </div>
            
            <div>
              <label className="block text-purple-200 text-sm font-medium mb-2">Trading Style</label>
              <select
                value={preferences.trading_style}
                onChange={(e) => setPreferences({...preferences, trading_style: e.target.value})}
                className="w-full px-3 py-2 rounded-lg bg-gray-800/60 border border-purple-600/40 text-white text-sm focus:border-purple-500"
              >
                <option value="SCALPING">Scalping (1-5min)</option>
                <option value="MOMENTUM">Momentum (15-60min)</option>
                <option value="SWING">Swing (1-4hrs)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-purple-200 text-sm font-medium mb-2">Max Trades/Day</label>
              <select
                value={preferences.max_trades_per_day}
                onChange={(e) => setPreferences({...preferences, max_trades_per_day: e.target.value})}
                className="w-full px-3 py-2 rounded-lg bg-gray-800/60 border border-purple-600/40 text-white text-sm focus:border-purple-500"
              >
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="5">5</option>
                <option value="8">8</option>
              </select>
            </div>
            
            <div>
              <label className="block text-purple-200 text-sm font-medium mb-2">Hold Duration (min)</label>
              <input
                type="text"
                value={preferences.holding_duration_minutes}
                onChange={(e) => setPreferences({...preferences, holding_duration_minutes: e.target.value})}
                className="w-full px-3 py-2 rounded-lg bg-gray-800/60 border border-purple-600/40 text-white text-sm focus:border-purple-500"
              />
            </div>
            
            <div>
              <label className="block text-purple-200 text-sm font-medium mb-2">Sectors</label>
              <input
                type="text"
                value={preferences.preferred_sectors}
                onChange={(e) => setPreferences({...preferences, preferred_sectors: e.target.value})}
                placeholder="IT,Finance,Pharma"
                className="w-full px-3 py-2 rounded-lg bg-gray-800/60 border border-purple-600/40 text-white text-sm focus:border-purple-500"
              />
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="mb-6">
          <div className="flex justify-center">
            <div className="bg-gray-900/60 backdrop-blur-lg rounded-xl p-2 border border-purple-800/40">
              <button
                onClick={() => setActiveTab('signals')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2 ${
                  activeTab === 'signals'
                    ? 'bg-gradient-to-r from-purple-700 to-indigo-700 text-white shadow-lg'
                    : 'text-purple-300 hover:text-white hover:bg-purple-800/30'
                }`}
              >
                <Target className="w-4 h-4" />
                <span>Professional Signals</span>
              </button>
              <button
                onClick={() => setActiveTab('enhanced')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2 ${
                  activeTab === 'enhanced'
                    ? 'bg-gradient-to-r from-green-700 to-emerald-700 text-white shadow-lg'
                    : 'text-green-300 hover:text-white hover:bg-green-800/30'
                }`}
              >
                <DollarSign className="w-4 h-4" />
                <span>Small Capital ⚡</span>
              </button>
              <button
                onClick={() => setActiveTab('backtest')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2 ${
                  activeTab === 'backtest'
                    ? 'bg-gradient-to-r from-purple-700 to-indigo-700 text-white shadow-lg'
                    : 'text-purple-300 hover:text-white hover:bg-purple-800/30'
                }`}
              >
                <BarChart3 className="w-4 h-4" />
                <span>Backtest Results</span>
              </button>
              <button
                onClick={() => setActiveTab('review')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2 ${
                  activeTab === 'review'
                    ? 'bg-gradient-to-r from-purple-700 to-indigo-700 text-white shadow-lg'
                    : 'text-purple-300 hover:text-white hover:bg-purple-800/30'
                }`}
              >
                <Calendar className="w-4 h-4" />
                <span>EOD Review</span>
              </button>
            </div>
          </div>
        </div>

        {/* Professional Signals Tab */}
        {activeTab === 'signals' && (
          <div>
            {loading ? (
              <div className="text-center py-12">
                <div className="w-8 h-8 border-2 border-purple-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-purple-200">Generating professional signals...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {signals.map((signal, index) => (
                  <div key={index} className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
                    {/* Signal Header */}
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-semibold text-white flex items-center">
                          {signal.symbol}
                          {getTradingStyleIcon(preferences.trading_style)}
                        </h3>
                        <p className="text-purple-300 text-sm">{signal.company_name}</p>
                        <p className="text-gray-400 text-xs">{signal.sector}</p>
                      </div>
                      <div className="text-right">
                        <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${getDirectionColor(signal.direction)}`}>
                          {signal.direction === 'BUY' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                          <span className="font-medium">{signal.direction}</span>
                        </div>
                        <div className="flex items-center justify-end space-x-2 mt-1">
                          <div className="text-white font-bold text-lg">₹{signal.entry_price.toFixed(2)}</div>
                          {currentPrices[signal.symbol] && currentPrices[signal.symbol] !== signal.entry_price && (
                            <div className={`flex items-center text-xs px-2 py-1 rounded ${
                              currentPrices[signal.symbol] > signal.entry_price 
                                ? 'text-green-300 bg-green-900/20' 
                                : 'text-red-300 bg-red-900/20'
                            }`}>
                              {currentPrices[signal.symbol] > signal.entry_price ? '↗' : '↘'}
                              ₹{Math.abs(currentPrices[signal.symbol] - signal.entry_price).toFixed(2)}
                            </div>
                          )}
                        </div>
                        {currentPrices[signal.symbol] && (
                          <div className="text-xs text-gray-400 mt-1">
                            Live: ₹{currentPrices[signal.symbol].toFixed(2)}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Confidence and Metrics */}
                    <div className="grid grid-cols-3 gap-3 mb-4">
                      <div className="text-center p-3 bg-blue-900/20 rounded-lg">
                        <div className="text-blue-300 font-bold text-lg">{signal.confidence_score.toFixed(0)}%</div>
                        <div className="text-blue-200 text-xs">Confidence</div>
                      </div>
                      <div className="text-center p-3 bg-green-900/20 rounded-lg">
                        <div className="text-green-300 font-bold text-lg">{signal.expected_roi.toFixed(1)}%</div>
                        <div className="text-green-200 text-xs">Expected ROI</div>
                      </div>
                      <div className="text-center p-3 bg-yellow-900/20 rounded-lg">
                        <div className="text-yellow-300 font-bold text-lg">1:{signal.risk_reward_ratio.toFixed(1)}</div>
                        <div className="text-yellow-200 text-xs">Risk:Reward</div>
                      </div>
                    </div>

                    {/* Price Levels */}
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      <div className="p-3 bg-green-900/10 rounded-lg border-l-4 border-green-500">
                        <div className="text-green-300 font-semibold">Targets</div>
                        <div className="text-green-200 text-sm">T1: ₹{signal.target_price_1.toFixed(2)}</div>
                        <div className="text-green-200 text-sm">T2: ₹{signal.target_price_2.toFixed(2)}</div>
                      </div>
                      <div className="p-3 bg-red-900/10 rounded-lg border-l-4 border-red-500">
                        <div className="text-red-300 font-semibold">Risk Management</div>
                        <div className="text-red-200 text-sm">Stop Loss: ₹{signal.stop_loss.toFixed(2)}</div>
                        <div className={`text-sm ${getVolatilityColor(signal.volatility_level).replace('bg-', 'text-').replace('/20', '')}`}>
                          {signal.volatility_level} Volatility
                        </div>
                      </div>
                    </div>

                    {/* Signal Timing */}
                    <div className="flex items-center justify-between p-3 bg-purple-900/10 rounded-lg border-l-4 border-purple-500 mb-4">
                      <div className="flex items-center text-purple-200">
                        <Clock className="w-4 h-4 mr-2" />
                        Signal Time: {signal.signal_time}
                      </div>
                      <div className="text-purple-300 text-sm">
                        Hold: {preferences.holding_duration_minutes}min
                      </div>
                    </div>

                    {/* Signal Log */}
                    <div className="p-3 bg-gray-800/20 rounded-lg">
                      <div className="text-gray-400 text-xs mb-2 flex items-center">
                        <BookOpen className="w-3 h-3 mr-1" />
                        Signal Log:
                      </div>
                      <div className="space-y-1">
                        {signal.signal_log.slice(0, 3).map((log, i) => (
                          <div key={i} className="text-gray-200 text-xs font-mono">{log}</div>
                        ))}
                      </div>
                    </div>

                    {/* LLM Professional Analysis */}
                    {(signal.market_analysis || signal.action_plan || signal.risk_management || signal.timing) && (
                      <div className="mt-4 space-y-3">
                        <div className="text-center">
                          <h4 className="text-purple-200 text-sm font-semibold mb-3 flex items-center justify-center">
                            <BookOpen className="w-4 h-4 mr-2" />
                            Professional Trading Analysis
                          </h4>
                        </div>
                        
                        {signal.market_analysis && (
                          <div className="p-3 bg-blue-900/10 rounded-lg border-l-4 border-blue-500">
                            <div className="text-blue-300 font-semibold text-xs mb-1 flex items-center">
                              <BarChart3 className="w-3 h-3 mr-1" />
                              Market Analysis
                            </div>
                            <div className="text-blue-100 text-xs leading-relaxed">{signal.market_analysis}</div>
                          </div>
                        )}
                        
                        {signal.action_plan && (
                          <div className="p-3 bg-green-900/10 rounded-lg border-l-4 border-green-500">
                            <div className="text-green-300 font-semibold text-xs mb-1 flex items-center">
                              <Target className="w-3 h-3 mr-1" />
                              Action Plan
                            </div>
                            <div className="text-green-100 text-xs leading-relaxed whitespace-pre-line">{signal.action_plan}</div>
                          </div>
                        )}
                        
                        {signal.risk_management && (
                          <div className="p-3 bg-red-900/10 rounded-lg border-l-4 border-red-500">
                            <div className="text-red-300 font-semibold text-xs mb-1 flex items-center">
                              <AlertCircle className="w-3 h-3 mr-1" />
                              Risk Management
                            </div>
                            <div className="text-red-100 text-xs leading-relaxed whitespace-pre-line">{signal.risk_management}</div>
                          </div>
                        )}
                        
                        {signal.timing && (
                          <div className="p-3 bg-purple-900/10 rounded-lg border-l-4 border-purple-500">
                            <div className="text-purple-300 font-semibold text-xs mb-1 flex items-center">
                              <Clock className="w-3 h-3 mr-1" />
                              Timing Strategy
                            </div>
                            <div className="text-purple-100 text-xs leading-relaxed whitespace-pre-line">{signal.timing}</div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Real-time Chart */}
                    {signal.chart_data && (
                      <div className="mt-4 p-4 bg-gray-900/30 rounded-lg border border-gray-700/50">
                        <div className="flex items-center justify-between mb-3">
                          <h4 className="text-white font-semibold text-sm flex items-center">
                            <BarChart3 className="w-4 h-4 mr-2 text-blue-400" />
                            Live Price Chart - {signal.symbol}
                          </h4>
                          <div className="text-xs text-gray-400">
                            Last updated: {lastUpdated || 'Loading...'}
                          </div>
                        </div>
                        
                        <div className="h-64 w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart
                              data={(() => {
                                const chartData = signal.chart_data;
                                if (!chartData || !chartData.timestamps || !chartData.ohlc) return [];
                                
                                return chartData.timestamps.map((time: string, index: number) => ({
                                  time,
                                  price: chartData.ohlc.close[index],
                                  volume: chartData.volume ? chartData.volume[index] : 0,
                                  sma: chartData.indicators?.SMA_20 ? chartData.indicators.SMA_20[index] : null,
                                  rsi: chartData.indicators?.RSI ? chartData.indicators.RSI[index] : null,
                                  entry: signal.entry_price,
                                  stopLoss: signal.stop_loss,
                                  target1: signal.target_price_1,
                                  target2: signal.target_price_2,
                                  // Add current live price to latest data point
                                  currentPrice: index === chartData.timestamps.length - 1 ? 
                                    (currentPrices[signal.symbol] || signal.entry_price) : null
                                }));
                              })()}
                              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                              <XAxis 
                                dataKey="time" 
                                tick={{ fontSize: 10, fill: '#9CA3AF' }}
                                axisLine={{ stroke: '#4B5563' }}
                              />
                              <YAxis 
                                tick={{ fontSize: 10, fill: '#9CA3AF' }}
                                axisLine={{ stroke: '#4B5563' }}
                                domain={['dataMin - 1', 'dataMax + 1']}
                              />
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: '#1F2937', 
                                  border: '1px solid #374151',
                                  borderRadius: '8px',
                                  color: '#F9FAFB'
                                }}
                                labelStyle={{ color: '#D1D5DB' }}
                              />
                              
                              {/* Price Line */}
                              <Line 
                                type="monotone" 
                                dataKey="price" 
                                stroke="#3B82F6" 
                                strokeWidth={2}
                                dot={false}
                                name="Current Price"
                              />
                              
                              {/* Moving Average */}
                              <Line 
                                type="monotone" 
                                dataKey="sma" 
                                stroke="#F59E0B" 
                                strokeWidth={1}
                                strokeDasharray="5 5"
                                dot={false}
                                name="SMA 20"
                              />
                              
                              {/* Entry Price Line */}
                              <Line 
                                type="monotone" 
                                dataKey="entry" 
                                stroke="#10B981" 
                                strokeWidth={2}
                                strokeDasharray="3 3"
                                dot={false}
                                name="Entry Price"
                              />
                              
                              {/* Stop Loss Line */}
                              <Line 
                                type="monotone" 
                                dataKey="stopLoss" 
                                stroke="#EF4444" 
                                strokeWidth={1}
                                strokeDasharray="2 2"
                                dot={false}
                                name="Stop Loss"
                              />
                              
                              {/* Target Lines */}
                              <Line 
                                type="monotone" 
                                dataKey="target1" 
                                stroke="#8B5CF6" 
                                strokeWidth={1}
                                strokeDasharray="4 2"
                                dot={false}
                                name="Target 1"
                              />
                              <Line 
                                type="monotone" 
                                dataKey="target2" 
                                stroke="#EC4899" 
                                strokeWidth={1}
                                strokeDasharray="4 2"
                                dot={false}
                                name="Target 2"
                              />
                              
                              {/* Volume Bars */}
                              <Bar 
                                dataKey="volume" 
                                fill="#6366F1" 
                                opacity={0.3}
                                yAxisId="volume"
                                name="Volume"
                              />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                        
                        {/* Chart Legend */}
                        <div className="mt-3 flex flex-wrap gap-4 text-xs">
                          <div className="flex items-center">
                            <div className="w-3 h-0.5 bg-blue-500 mr-1"></div>
                            <span className="text-blue-300">Current Price</span>
                          </div>
                          <div className="flex items-center">
                            <div className="w-3 h-0.5 bg-yellow-500 mr-1" style={{borderTop: '1px dashed'}}></div>
                            <span className="text-yellow-300">SMA 20</span>
                          </div>
                          <div className="flex items-center">
                            <div className="w-3 h-0.5 bg-green-500 mr-1" style={{borderTop: '1px dashed'}}></div>
                            <span className="text-green-300">Entry: ${signal.entry_price}</span>
                          </div>
                          <div className="flex items-center">
                            <div className="w-3 h-0.5 bg-red-500 mr-1" style={{borderTop: '1px dashed'}}></div>
                            <span className="text-red-300">Stop: ${signal.stop_loss}</span>
                          </div>
                          <div className="flex items-center">
                            <div className="w-3 h-0.5 bg-purple-500 mr-1" style={{borderTop: '1px dashed'}}></div>
                            <span className="text-purple-300">T1: ${signal.target_price_1}</span>
                          </div>
                          <div className="flex items-center">
                            <div className="w-3 h-0.5 bg-pink-500 mr-1" style={{borderTop: '1px dashed'}}></div>
                            <span className="text-pink-300">T2: ${signal.target_price_2}</span>
                          </div>
                        </div>
                        
                        {/* Current Price Badge */}
                        {currentPrices[signal.symbol] && (
                          <div className="mt-3 flex justify-center">
                            <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
                              currentPrices[signal.symbol] > signal.entry_price 
                                ? 'bg-green-900/30 text-green-300 border border-green-600/50' 
                                : 'bg-red-900/30 text-red-300 border border-red-600/50'
                            }`}>
                              Live: ${currentPrices[signal.symbol]?.toFixed(2)} 
                              {currentPrices[signal.symbol] > signal.entry_price ? ' ↗' : ' ↘'}
                              <span className="ml-1 text-xs opacity-75">
                                ({((currentPrices[signal.symbol] - signal.entry_price) / signal.entry_price * 100).toFixed(2)}%)
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Professional Badge */}
                    <div className="mt-4 flex items-center justify-center p-2 bg-gradient-to-r from-gold-900/20 to-yellow-900/20 rounded-lg border border-gold-600/40">
                      <Award className="w-4 h-4 mr-2 text-gold-400" />
                      <span className="text-gold-200 text-sm font-medium">CFA Professional Grade</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Enhanced Small Capital Tab */}
        {activeTab === 'enhanced' && (
          <div>
            {loading ? (
              <div className="text-center py-12">
                <div className="w-8 h-8 border-2 border-green-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-green-200">Generating optimized signals for small capital...</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Capital Optimization Summary */}
                <div className="bg-green-900/20 backdrop-blur-lg rounded-2xl p-6 border border-green-600/40">
                  <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                    <DollarSign className="w-6 h-6 mr-2 text-green-400" />
                    Small Capital Optimization
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center p-3 bg-green-800/20 rounded-lg">
                      <div className="text-green-300 font-bold text-lg">₹{preferences.capital_allocation}</div>
                      <div className="text-green-200 text-xs">Total Capital</div>
                    </div>
                    <div className="text-center p-3 bg-emerald-800/20 rounded-lg">
                      <div className="text-emerald-300 font-bold text-lg">
                        ₹{parseInt(preferences.capital_allocation) < 15000 ? '300' : 
                           parseInt(preferences.capital_allocation) < 30000 ? '500' : '800'}+
                      </div>
                      <div className="text-emerald-200 text-xs">Min Profit Target</div>
                    </div>
                    <div className="text-center p-3 bg-lime-800/20 rounded-lg">
                      <div className="text-lime-300 font-bold text-lg">2.5-5%</div>
                      <div className="text-lime-200 text-xs">Target Moves</div>
                    </div>
                    <div className="text-center p-3 bg-green-700/20 rounded-lg">
                      <div className="text-green-300 font-bold text-lg">₹{Math.round(parseInt(preferences.capital_allocation) * 0.3)}</div>
                      <div className="text-green-200 text-xs">Max Per Trade</div>
                    </div>
                  </div>
                </div>

                {/* Enhanced Signals Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {enhancedSignals.map((signal, index) => (
                    <div key={index} className="bg-green-900/20 backdrop-blur-lg rounded-2xl p-6 border border-green-600/40">
                      {/* Signal Header */}
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h3 className="text-xl font-semibold text-white flex items-center">
                            {signal.symbol}
                            <span className="ml-2 text-green-400">⚡</span>
                          </h3>
                          <p className="text-green-300 text-sm">{signal.company_name}</p>
                          <p className="text-gray-400 text-xs">{signal.sector}</p>
                        </div>
                        <div className="text-right">
                          <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
                            signal.direction === 'BUY' 
                              ? 'bg-green-900/50 text-green-300 border border-green-600' 
                              : 'bg-red-900/50 text-red-300 border border-red-600'
                          }`}>
                            {signal.direction}
                          </div>
                          <div className="text-green-400 text-xs mt-1">
                            Confidence: {signal.confidence_score}%
                          </div>
                        </div>
                      </div>

                      {/* Price Information */}
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="bg-gray-800/30 rounded-lg p-3">
                          <div className="text-gray-400 text-xs">Entry Price</div>
                          <div className="text-white font-semibold">₹{signal.entry_price}</div>
                        </div>
                        <div className="bg-gray-800/30 rounded-lg p-3">
                          <div className="text-gray-400 text-xs">Stop Loss</div>
                          <div className="text-red-300 font-semibold">₹{signal.stop_loss}</div>
                        </div>
                      </div>

                      {/* Targets and Profits */}
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="bg-green-800/20 rounded-lg p-3 border border-green-600/30">
                          <div className="text-green-400 text-xs">Target 1</div>
                          <div className="text-green-300 font-semibold">₹{signal.target_price_1}</div>
                          <div className="text-green-200 text-xs">Profit: ₹{signal.profit_potential_1}</div>
                        </div>
                        <div className="bg-emerald-800/20 rounded-lg p-3 border border-emerald-600/30">
                          <div className="text-emerald-400 text-xs">Target 2</div>
                          <div className="text-emerald-300 font-semibold">₹{signal.target_price_2}</div>
                          <div className="text-emerald-200 text-xs">Profit: ₹{signal.profit_potential_2}</div>
                        </div>
                      </div>

                      {/* Position Details */}
                      <div className="bg-blue-900/20 rounded-lg p-4 mb-4 border border-blue-600/30">
                        <h4 className="text-blue-300 font-semibold text-sm mb-2">Position Details</h4>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-400">Shares: </span>
                            <span className="text-white font-semibold">{signal.shares_recommended}</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Investment: </span>
                            <span className="text-blue-300 font-semibold">₹{signal.investment_amount}</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Expected ROI: </span>
                            <span className="text-green-300 font-semibold">{signal.expected_roi.toFixed(1)}%</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Risk:Reward: </span>
                            <span className="text-purple-300 font-semibold">1:{signal.risk_reward_ratio.toFixed(1)}</span>
                          </div>
                        </div>
                      </div>

                      {/* LLM Explanations */}
                      <div className="space-y-3">
                        {signal.market_analysis && (
                          <div className="p-3 bg-blue-900/10 rounded-lg border-l-4 border-blue-500">
                            <div className="text-blue-300 font-semibold text-xs mb-1">Market Analysis</div>
                            <div className="text-blue-100 text-xs">{signal.market_analysis}</div>
                          </div>
                        )}
                        
                        {signal.action_plan && (
                          <div className="p-3 bg-green-900/10 rounded-lg border-l-4 border-green-500">
                            <div className="text-green-300 font-semibold text-xs mb-1">Action Plan</div>
                            <div className="text-green-100 text-xs">{signal.action_plan}</div>
                          </div>
                        )}
                      </div>

                      {/* Enhanced Badge */}
                      <div className="mt-4 flex items-center justify-center p-2 bg-gradient-to-r from-green-900/20 to-emerald-900/20 rounded-lg border border-green-600/40">
                        <DollarSign className="w-4 h-4 mr-2 text-green-400" />
                        <span className="text-green-200 text-sm font-medium">Small Capital Optimized ⚡</span>
                      </div>
                    </div>
                  ))}
                </div>

                {enhancedSignals.length === 0 && !loading && (
                  <div className="text-center py-12">
                    <DollarSign className="w-16 h-16 mx-auto mb-4 text-green-400 opacity-50" />
                    <p className="text-green-300">No profitable opportunities found for your capital range</p>
                    <p className="text-green-400 text-sm mt-2">Try adjusting your risk appetite or capital allocation</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Backtest Results Tab */}
        {activeTab === 'backtest' && (
          <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
            <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
              <BarChart3 className="w-6 h-6 mr-2 text-purple-400" />
              Signal Accuracy Backtest
            </h3>
            
            {loading ? (
              <div className="text-center py-12">
                <div className="w-8 h-8 border-2 border-purple-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-purple-200">Running backtest analysis...</p>
              </div>
            ) : backtestResults ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="text-center p-4 bg-blue-900/20 rounded-lg">
                  <div className="text-blue-300 font-bold text-2xl">{backtestResults.total_signals}</div>
                  <div className="text-blue-200 text-sm">Total Signals</div>
                </div>
                <div className="text-center p-4 bg-green-900/20 rounded-lg">
                  <div className="text-green-300 font-bold text-2xl">{backtestResults.accuracy_rate.toFixed(0)}%</div>
                  <div className="text-green-200 text-sm">Accuracy Rate</div>
                </div>
                <div className="text-center p-4 bg-purple-900/20 rounded-lg">
                  <div className="text-purple-300 font-bold text-2xl">{backtestResults.avg_profit.toFixed(1)}%</div>
                  <div className="text-purple-200 text-sm">Avg Profit</div>
                </div>
                <div className="text-center p-4 bg-yellow-900/20 rounded-lg">
                  <div className="text-yellow-300 font-bold text-2xl">1:{backtestResults.risk_reward_achieved.toFixed(1)}</div>
                  <div className="text-yellow-200 text-sm">Risk:Reward</div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <BarChart3 className="w-16 h-16 mx-auto mb-4 text-purple-400 opacity-50" />
                <p className="text-purple-300">No backtest data available</p>
              </div>
            )}
          </div>
        )}

        {/* End of Day Review Tab */}
        {activeTab === 'review' && (
          <div className="space-y-6">
            {loading ? (
              <div className="text-center py-12">
                <div className="w-8 h-8 border-2 border-purple-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-purple-200">Generating EOD review...</p>
              </div>
            ) : eodReview ? (
              <>
                {/* Trading Summary */}
                <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
                  <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                    <DollarSign className="w-6 h-6 mr-2 text-green-400" />
                    Trading Performance Summary
                  </h3>
                  
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                    <div className="text-center p-3 bg-blue-900/20 rounded-lg">
                      <div className="text-blue-300 font-bold text-lg">{eodReview.trading_summary.total_trades}</div>
                      <div className="text-blue-200 text-sm">Total Trades</div>
                    </div>
                    <div className="text-center p-3 bg-green-900/20 rounded-lg">
                      <div className="text-green-300 font-bold text-lg">{eodReview.trading_summary.win_rate.toFixed(0)}%</div>
                      <div className="text-green-200 text-sm">Win Rate</div>
                    </div>
                    <div className="text-center p-3 bg-purple-900/20 rounded-lg">
                      <div className="text-purple-300 font-bold text-lg">₹{eodReview.trading_summary.total_pnl.toFixed(0)}</div>
                      <div className="text-purple-200 text-sm">Total P&L</div>
                    </div>
                    <div className="text-center p-3 bg-yellow-900/20 rounded-lg">
                      <div className="text-yellow-300 font-bold text-lg">{eodReview.trading_summary.roi_for_day.toFixed(1)}%</div>
                      <div className="text-yellow-200 text-sm">ROI Today</div>
                    </div>
                    <div className="text-center p-3 bg-indigo-900/20 rounded-lg">
                      <div className="text-indigo-300 font-bold text-lg">{eodReview.performance_metrics.avg_holding_time_minutes}</div>
                      <div className="text-indigo-200 text-sm">Avg Hold (min)</div>
                    </div>
                    <div className="text-center p-3 bg-teal-900/20 rounded-lg">
                      <div className="text-teal-300 font-bold text-lg">1:{eodReview.performance_metrics.risk_reward_realized.toFixed(1)}</div>
                      <div className="text-teal-200 text-sm">R:R Realized</div>
                    </div>
                  </div>
                </div>

                {/* Lessons Learned */}
                <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
                  <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                    <BookOpen className="w-6 h-6 mr-2 text-blue-400" />
                    Key Insights & Lessons
                  </h3>
                  
                  <div className="space-y-3">
                    {eodReview.lessons_learned.map((lesson, index) => (
                      <div key={index} className="flex items-start p-3 bg-blue-900/10 rounded-lg border-l-4 border-blue-500">
                        <CheckCircle className="w-4 h-4 mr-3 mt-0.5 text-blue-400 flex-shrink-0" />
                        <span className="text-blue-200 text-sm">{lesson}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Tomorrow's Recommendations */}
                <div className="bg-gray-900/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-800/40">
                  <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                    <Target className="w-6 h-6 mr-2 text-orange-400" />
                    Tomorrow's Strategy
                  </h3>
                  
                  <div className="space-y-3">
                    {eodReview.recommendations_for_tomorrow.map((recommendation, index) => (
                      <div key={index} className="flex items-start p-3 bg-orange-900/10 rounded-lg border-l-4 border-orange-500">
                        <AlertCircle className="w-4 h-4 mr-3 mt-0.5 text-orange-400 flex-shrink-0" />
                        <span className="text-orange-200 text-sm">{recommendation}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="text-center py-12">
                <Calendar className="w-16 h-16 mx-auto mb-4 text-purple-400 opacity-50" />
                <p className="text-purple-300">No EOD review data available</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ProfessionalTrading;
