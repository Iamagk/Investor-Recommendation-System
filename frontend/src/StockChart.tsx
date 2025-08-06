import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ChartDataPoint {
  date: string;
  price: number;
  signal?: 'BUY' | 'SELL' | null;
}

interface TechnicalIndicators {
  rsi: number;
  macd: number;
  moving_averages: {
    ma_20: number;
    ma_50: number;
  };
}

interface StockChartProps {
  symbol: string;
  technicalIndicators: TechnicalIndicators;
  priceHistory?: number[];
  dates?: string[];
}

const StockChart: React.FC<StockChartProps> = ({ symbol, technicalIndicators, priceHistory = [], dates = [] }) => {
  // Use real price data if available, otherwise generate sample data
  const useRealData = priceHistory.length > 0 && dates.length > 0;
  
  const generateChartData = (): ChartDataPoint[] => {
    if (useRealData) {
      // Use real price history
      return priceHistory.map((price, index) => {
        let signal: 'BUY' | 'SELL' | null = null;
        
        // Add some buy/sell signals based on price movement
        if (index > 0) {
          const prevPrice = priceHistory[index - 1];
          if (price > prevPrice * 1.03 && Math.random() > 0.85) {
            signal = 'SELL';
          } else if (price < prevPrice * 0.97 && Math.random() > 0.85) {
            signal = 'BUY';
          }
        }
        
        return {
          date: dates[index] || new Date().toISOString().split('T')[0],
          price,
          signal
        };
      });
    } else {
      // Generate sample data as fallback
      const data: ChartDataPoint[] = [];
      const startPrice = technicalIndicators.moving_averages.ma_20 || 100;
      const days = 30;
      
      for (let i = 0; i < days; i++) {
        const date = new Date();
        date.setDate(date.getDate() - (days - i));
        
        const price = startPrice + (Math.random() - 0.5) * 20 + Math.sin(i / 5) * 10;
        let signal: 'BUY' | 'SELL' | null = null;
        
        // Add some buy/sell signals based on price movement
        if (i > 0 && data[i - 1]) {
          const prevPrice = data[i - 1].price;
          if (price > prevPrice * 1.02 && Math.random() > 0.8) {
            signal = 'SELL';
          } else if (price < prevPrice * 0.98 && Math.random() > 0.8) {
            signal = 'BUY';
          }
        }
        
        data.push({
          date: date.toISOString().split('T')[0],
          price,
          signal
        });
      }
      
      return data;
    }
  };

  const chartData = generateChartData();
  const labels = chartData.map(d => new Date(d.date).toLocaleDateString());
  const prices = chartData.map(d => d.price);
  
  // Create signal points
  const buyPoints: number[] = new Array(chartData.length).fill(null);
  const sellPoints: number[] = new Array(chartData.length).fill(null);
  
  chartData.forEach((point, index) => {
    if (point.signal === 'BUY') {
      buyPoints[index] = point.price;
    } else if (point.signal === 'SELL') {
      sellPoints[index] = point.price;
    }
  });

  const data = {
    labels,
    datasets: [
      {
        label: 'Price',
        data: prices,
        borderColor: 'rgb(139, 92, 246)',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.1,
      },
      {
        label: 'MA(20)',
        data: new Array(chartData.length).fill(technicalIndicators.moving_averages.ma_20),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        borderWidth: 1,
        fill: false,
        borderDash: [5, 5],
      },
      {
        label: 'MA(50)',
        data: new Array(chartData.length).fill(technicalIndicators.moving_averages.ma_50),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderWidth: 1,
        fill: false,
        borderDash: [10, 5],
      },
      {
        label: 'Buy Signals',
        data: buyPoints,
        backgroundColor: 'rgb(34, 197, 94)',
        borderColor: 'rgb(34, 197, 94)',
        pointRadius: 8,
        pointHoverRadius: 10,
        showLine: false,
        fill: false,
      },
      {
        label: 'Sell Signals',
        data: sellPoints,
        backgroundColor: 'rgb(239, 68, 68)',
        borderColor: 'rgb(239, 68, 68)',
        pointRadius: 8,
        pointHoverRadius: 10,
        showLine: false,
        fill: false,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: 'rgb(196, 181, 253)',
        },
      },
      title: {
        display: true,
        text: `${symbol} - Price Chart with Buy/Sell Signals`,
        color: 'rgb(196, 181, 253)',
        font: {
          size: 16,
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(17, 24, 39, 0.9)',
        titleColor: 'rgb(196, 181, 253)',
        bodyColor: 'rgb(196, 181, 253)',
        borderColor: 'rgb(139, 92, 246)',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Date',
          color: 'rgb(156, 163, 175)',
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
        },
        grid: {
          color: 'rgba(75, 85, 99, 0.3)',
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Price (₹)',
          color: 'rgb(156, 163, 175)',
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
        },
        grid: {
          color: 'rgba(75, 85, 99, 0.3)',
        },
      },
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false,
    },
  };

  return (
    <div className="w-full h-96">
      <Line data={data} options={options} />
    </div>
  );
};

export default StockChart;
