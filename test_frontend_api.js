// Test script to verify the comprehensive backtest API call
const axios = require('axios');

async function testComprehensiveBacktest() {
  try {
    console.log('Testing comprehensive backtest API call...');
    
    const response = await axios.post('http://localhost:8000/backtest/comprehensive', {
      budget: 100000,
      risk_profile: 'moderate',
      investment_horizon: 'long_term',
      top_n: 3
    });
    
    console.log('Status:', response.data.status);
    console.log('Has backtest_output:', !!response.data.backtest_output);
    console.log('Analysis type:', response.data.analysis_type);
    
    if (response.data.backtest_output) {
      console.log('\n--- First 1000 characters of backtest_output ---');
      console.log(response.data.backtest_output.substring(0, 1000));
      console.log('\n--- Contains SECTOR ANALYSIS? ---');
      console.log(response.data.backtest_output.includes('💬 SECTOR ANALYSIS:'));
      console.log('--- Contains INVESTMENT STRATEGY? ---');
      console.log(response.data.backtest_output.includes('📝 INVESTMENT STRATEGY:'));
      console.log('--- Contains TIMING ANALYSIS? ---');
      console.log(response.data.backtest_output.includes('⏰ TIMING ANALYSIS:'));
    }
    
  } catch (error) {
    console.error('Error:', error.message);
  }
}

testComprehensiveBacktest();
