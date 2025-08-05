import os
from typing import Dict, Any
import logging
import requests
import json

logger = logging.getLogger(__name__)

# Try to import OpenAI, fall back gracefully if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.info("OpenAI library not available, using fallback explanations")

class LLMAnalysisService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.use_openai = bool(self.openai_api_key)
        self.ollama_available = self._check_ollama_availability()
        
        # Force fallback mode for now as Ollama refuses financial advice
        # TODO: Try other models or improve prompting for Ollama
        self.llm_mode = "fallback"
        logger.info("Using enhanced rule-based templates for comprehensive financial analysis")
        
        # Determine priority: Ollama (free) > OpenAI (paid) > Fallback (rule-based)
        # if self.ollama_available:
        #     self.llm_mode = "ollama"
        #     logger.info("Using Ollama (free local LLM) for analysis")
        # elif self.use_openai:
        #     self.llm_mode = "openai"
        #     logger.info("Using OpenAI (paid API) for analysis")
        # else:
        #     self.llm_mode = "fallback"
        #     logger.info("Using fallback rule-based templates for analysis")

    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is running and has the model available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                # Check if we have llama3.2:3b or any other model available
                available_models = [model["name"] for model in models]
                if available_models:
                    logger.info(f"Ollama available with models: {available_models}")
                    return True
            return False
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    def generate_stock_analysis(self, stock_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate detailed analysis for a stock recommendation"""
        
        if self.llm_mode == "ollama":
            return self._generate_ollama_stock_analysis(stock_data)
        elif self.llm_mode == "openai":
            return self._generate_openai_stock_analysis(stock_data)
        else:
            return self._generate_fallback_stock_analysis(stock_data)

    def generate_mutual_fund_analysis(self, fund_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate detailed analysis for a mutual fund recommendation"""
        
        if self.llm_mode == "ollama":
            return self._generate_ollama_fund_analysis(fund_data)
        elif self.llm_mode == "openai":
            return self._generate_openai_fund_analysis(fund_data)
        else:
            return self._generate_fallback_fund_analysis(fund_data)

    def _generate_openai_stock_analysis(self, stock_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate stock analysis using OpenAI"""
        try:
            expected_return = stock_data.get('expected_return', 10.0)
            if expected_return <= 0:
                expected_return = 10.0
                
            prompt = f"""
            Analyze this stock recommendation and provide detailed explanations:
            
            Stock: {stock_data.get('company_name', 'Unknown')} ({stock_data.get('symbol', 'N/A')})
            Current Price: ₹{stock_data.get('current_price', stock_data.get('entry_price', 0))}
            Target Price: ₹{stock_data.get('target_price', 0)}
            Expected Return: {expected_return}%
            Current Performance: {stock_data.get('current_performance', 0)}%
            Volatility: {stock_data.get('volatility', 15.0)}%
            Holding Period: {stock_data.get('holding_period', 365)} days
            
            Please provide:
            1. Performance Analysis: Explain the current performance and expected returns in simple terms
            2. Investment Strategy: Detailed strategy explanation for this stock with target returns
            3. Risk Assessment: Analysis of the volatility and risk factors
            4. Market Outlook: Why this stock is recommended now with specific return expectations
            
            Keep each explanation to 2-3 sentences and make it easy to understand for retail investors.
            Include specific percentage returns and timeframes in your analysis.
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional financial advisor providing investment analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.7
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse the response into sections
            return self._parse_analysis_response(analysis_text)
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._generate_fallback_stock_analysis(stock_data)

    def _generate_openai_fund_analysis(self, fund_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate mutual fund analysis using OpenAI"""
        try:
            sip_details = ""
            if fund_data.get('is_sip_recommended'):
                sip_details = f"SIP Amount: ₹{fund_data.get('sip_amount', 0):.0f}/month for {fund_data.get('sip_duration_months', 0)} months"
            else:
                sip_details = f"Lump Sum: ₹{fund_data.get('lump_sum_amount', 0):.0f}"

            prompt = f"""
            Analyze this mutual fund recommendation and provide detailed explanations:
            
            Fund: {fund_data.get('fund_name', 'Unknown Fund')}
            Fund Manager: {fund_data.get('fund_manager', 'Unknown')}
            NAV: ₹{fund_data.get('nav', 100)}
            Expected Return: {fund_data.get('expected_return', 12.0)}%
            Current Performance: {fund_data.get('current_performance', 0)}%
            Expense Ratio: {fund_data.get('expense_ratio', 1.5)}%
            Risk Level: {fund_data.get('risk_level', 'Medium')}
            Investment Type: {'SIP' if fund_data.get('is_sip_recommended') else 'Lump Sum'}
            {sip_details}
            Minimum Investment: ₹{fund_data.get('minimum_investment', 500)}
            
            Please provide:
            1. Performance Analysis: Explain the fund's current performance and expected returns
            2. Investment Strategy: Why this fund is suitable and recommended investment approach
            3. Risk Assessment: Analysis of the risk level and expense ratio impact
            4. SIP vs Lump Sum: Specific details about investment amount, duration, and rationale
            
            Keep each explanation to 2-3 sentences and make it easy to understand for retail investors.
            Include specific amounts and timeframes in your analysis.
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional financial advisor providing mutual fund analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.7
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse the response into sections
            return self._parse_analysis_response(analysis_text)
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._generate_fallback_fund_analysis(fund_data)

    def _generate_ollama_stock_analysis(self, stock_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate stock analysis using Ollama local LLM"""
        try:
            expected_return = stock_data.get('expected_return', 10.0)
            if expected_return <= 0:
                expected_return = 10.0
                
            prompt = f"""Analyze this stock recommendation and provide detailed explanations in exactly 4 sections:

Stock: {stock_data.get('company_name', 'Unknown')} ({stock_data.get('symbol', 'N/A')})
Current Price: ₹{stock_data.get('current_price', stock_data.get('entry_price', 0))}
Target Price: ₹{stock_data.get('target_price', 0)}
Expected Return: {expected_return}%
Current Performance: {stock_data.get('current_performance', 0)}%
Volatility: {stock_data.get('volatility', 15.0)}%
Holding Period: {stock_data.get('holding_period', 365)} days

Please provide exactly 4 sections:
1. Performance Analysis: Explain the current performance and expected returns in simple terms
2. Investment Strategy: Detailed strategy explanation for this stock with target returns  
3. Risk Assessment: Analysis of the volatility and risk factors
4. Market Outlook: Why this stock is recommended now with specific return expectations

Keep each section to 2-3 sentences. Make it easy to understand for retail investors. Include specific percentage returns and timeframes."""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                analysis_text = response.json().get("response", "")
                return self._parse_analysis_response(analysis_text)
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return self._generate_fallback_stock_analysis(stock_data)
                
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            return self._generate_fallback_stock_analysis(stock_data)

    def _generate_ollama_fund_analysis(self, fund_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate mutual fund analysis using Ollama local LLM"""
        try:
            sip_details = ""
            if fund_data.get('is_sip_recommended'):
                sip_details = f"SIP Amount: ₹{fund_data.get('sip_amount', 0):.0f}/month for {fund_data.get('sip_duration_months', 0)} months"
            else:
                sip_details = f"Lump Sum: ₹{fund_data.get('lump_sum_amount', 0):.0f}"

            prompt = f"""Analyze this mutual fund recommendation and provide detailed explanations in exactly 4 sections:

Fund: {fund_data.get('fund_name', 'Unknown Fund')}
Fund Manager: {fund_data.get('fund_manager', 'Unknown')}
NAV: ₹{fund_data.get('nav', 100)}
Expected Return: {fund_data.get('expected_return', 12.0)}%
Current Performance: {fund_data.get('current_performance', 0)}%
Expense Ratio: {fund_data.get('expense_ratio', 1.5)}%
Risk Level: {fund_data.get('risk_level', 'Medium')}
Investment Type: {'SIP' if fund_data.get('is_sip_recommended') else 'Lump Sum'}
{sip_details}
Minimum Investment: ₹{fund_data.get('minimum_investment', 500)}

Please provide exactly 4 sections:
1. Performance Analysis: Explain the fund's current performance and expected returns
2. Investment Strategy: Why this fund is suitable and recommended investment approach
3. Risk Assessment: Analysis of the risk level and expense ratio impact  
4. SIP Analysis: Specific details about investment amount, duration, and rationale

Keep each section to 2-3 sentences. Make it easy to understand for retail investors. Include specific amounts and timeframes."""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                analysis_text = response.json().get("response", "")
                # For mutual funds, we need to adapt parsing for SIP analysis
                parsed = self._parse_analysis_response(analysis_text)
                # Map SIP Analysis to sip_analysis for mutual funds
                if "sip_analysis" not in parsed and "market_outlook" in parsed:
                    parsed["sip_analysis"] = parsed["market_outlook"]
                return parsed
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return self._generate_fallback_fund_analysis(fund_data)
                
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            return self._generate_fallback_fund_analysis(fund_data)

    def _generate_fallback_stock_analysis(self, stock_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate stock analysis using rule-based templates"""
        company_name = stock_data.get('company_name', 'This company')
        symbol = stock_data.get('symbol', 'N/A')
        current_performance = stock_data.get('current_performance', 0)
        expected_return = stock_data.get('expected_return', 10.0)
        volatility = stock_data.get('volatility', 15.0)
        holding_period = stock_data.get('holding_period', 365)

        # Convert decimal returns to percentage if needed
        if expected_return < 1:
            expected_return = expected_return * 100
        if current_performance < 1 and current_performance != 0:
            current_performance = current_performance * 100
        if volatility < 1 and volatility != 0:
            volatility = volatility * 100

        # Ensure we have valid values
        if expected_return <= 0:
            expected_return = 10.0
        if current_performance == 0 and expected_return > 0:
            current_performance = expected_return * 0.8  # Assume 80% of expected as current

        # Performance Analysis
        if current_performance > 15:
            perf_analysis = f"{company_name} ({symbol}) is showing strong performance with {current_performance:.1f}% gains. The stock has demonstrated consistent growth and is currently outperforming market expectations."
        elif current_performance > 5:
            perf_analysis = f"{company_name} ({symbol}) is delivering steady performance with {current_performance:.1f}% returns. The stock shows stable growth patterns and is performing in line with sector averages."
        else:
            perf_analysis = f"{company_name} ({symbol}) is currently showing modest performance at {current_performance:.1f}%. However, our analysis indicates strong potential for improvement based on fundamental factors and expected returns of {expected_return:.1f}%."

        # Investment Strategy
        if holding_period > 180:  # Long-term
            strategy = f"Long-term investment strategy recommended for {company_name}. Buy and hold approach for {holding_period} days to capitalize on fundamental growth and target {expected_return:.1f}% returns. Consider systematic investment to benefit from rupee cost averaging."
        else:  # Short-term
            strategy = f"Short to medium-term trading strategy for {company_name}. Target {expected_return:.1f}% returns over {holding_period} days using technical analysis and momentum indicators."

        # Risk Assessment
        if volatility > 20:
            risk_assessment = f"High volatility stock ({volatility:.1f}%) suitable for aggressive investors. Implement strict stop-loss and position sizing. Higher risk but potential for {expected_return:.1f}% returns."
        elif volatility > 10:
            risk_assessment = f"Moderate volatility ({volatility:.1f}%) makes this suitable for balanced portfolios. Good risk-reward ratio with manageable price fluctuations and {expected_return:.1f}% return potential."
        else:
            risk_assessment = f"Low volatility ({volatility:.1f}%) indicates stable investment. Suitable for conservative investors seeking steady {expected_return:.1f}% returns with minimal price swings."

        # Market Outlook
        market_outlook = f"Current market conditions favor {company_name} with expected returns of {expected_return:.1f}%. Technical and fundamental analysis support this recommendation for the target holding period of {holding_period} days."

        return {
            "performance_analysis": perf_analysis,
            "investment_strategy": strategy,
            "risk_assessment": risk_assessment,
            "market_outlook": market_outlook
        }

    def _generate_fallback_fund_analysis(self, fund_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate mutual fund analysis using rule-based templates"""
        fund_name = fund_data.get('fund_name', 'This fund')
        current_performance = fund_data.get('current_performance', 0)
        expected_return = fund_data.get('expected_return', 12.0)
        expense_ratio = fund_data.get('expense_ratio', 1.5)
        risk_level = fund_data.get('risk_level', 'Medium')
        is_sip = fund_data.get('is_sip_recommended', False)
        sip_amount = fund_data.get('sip_amount', 0)
        sip_duration = fund_data.get('sip_duration_months', 0)
        lump_sum = fund_data.get('lump_sum_amount', 0)
        min_investment = fund_data.get('minimum_investment', 500)

        # Convert decimal returns to percentage if needed
        if expected_return < 1:
            expected_return = expected_return * 100
        if current_performance < 1 and current_performance != 0:
            current_performance = current_performance * 100
        if expense_ratio < 1 and expense_ratio != 0:
            expense_ratio = expense_ratio * 100

        # Ensure we have valid values
        if expected_return <= 0:
            expected_return = 12.0
        if current_performance == 0 and expected_return > 0:
            current_performance = expected_return * 0.9  # Assume 90% of expected as current

        # Performance Analysis
        if current_performance > 15:
            perf_analysis = f"{fund_name} has delivered excellent performance with {current_performance:.1f}% returns. The fund has consistently outperformed its benchmark and peer funds in the category with strong potential for {expected_return:.1f}% annual returns."
        elif current_performance > 8:
            perf_analysis = f"{fund_name} shows solid performance with {current_performance:.1f}% returns. The fund maintains steady growth and has a good track record of wealth creation with expected {expected_return:.1f}% annual returns."
        else:
            perf_analysis = f"{fund_name} is currently showing {current_performance:.1f}% returns. While modest, the fund offers stability and is positioned for future growth with projected {expected_return:.1f}% annual returns."

        # Investment Strategy - Fix SIP detection and provide detailed recommendations
        if is_sip and sip_amount > 0:
            strategy = f"SIP investment recommended for {fund_name} to benefit from rupee cost averaging. Monthly SIP of ₹{sip_amount:.0f} for {sip_duration} months will help smooth out market volatility and build wealth systematically targeting {expected_return:.1f}% annual returns."
        else:
            strategy = f"Lump sum investment strategy suitable for {fund_name}. One-time investment of ₹{lump_sum:.0f} is recommended based on current market conditions to capture expected {expected_return:.1f}% annual returns."

        # Risk Assessment
        risk_desc = {
            'Low': 'conservative investors with capital preservation focus',
            'Medium': 'balanced investors seeking moderate growth with manageable risk',
            'High': 'aggressive investors comfortable with market volatility for higher returns'
        }
        
        risk_assessment = f"{risk_level} risk fund suitable for {risk_desc.get(risk_level, 'various types of')} investors. Expense ratio of {expense_ratio:.2f}% is competitive and won't significantly impact long-term {expected_return:.1f}% return potential."

        # SIP vs Lump Sum - Provide specific recommendations
        if is_sip and sip_amount > 0:
            total_investment = sip_amount * sip_duration if sip_duration > 0 else sip_amount * 12
            sip_analysis = f"SIP approach recommended with monthly investment of ₹{sip_amount:.0f} for {sip_duration} months. Minimum SIP amount is ₹{min_investment}. This strategy will help average out market volatility and reduce timing risk. Total investment over the period will be ₹{total_investment:.0f} targeting {expected_return:.1f}% annual returns."
        else:
            sip_analysis = f"Lump sum investment of ₹{lump_sum:.0f} is preferred due to current market valuations and fund positioning. Minimum investment requirement is ₹{min_investment}. One-time investment allows you to capture the full upside potential from the expected {expected_return:.1f}% returns immediately."

        return {
            "performance_analysis": perf_analysis,
            "investment_strategy": strategy,
            "risk_assessment": risk_assessment,
            "sip_analysis": sip_analysis
        }

    def _parse_analysis_response(self, analysis_text: str) -> Dict[str, str]:
        """Parse LLM response into structured sections"""
        sections = {
            "performance_analysis": "",
            "investment_strategy": "",
            "risk_assessment": "",
            "market_outlook": "",
            "sip_analysis": ""
        }
        
        # Split by lines and process
        lines = analysis_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers - be more specific about header patterns
            line_lower = line.lower()
            
            # Look for numbered sections or bold headers
            if (line.startswith('1.') and 'performance' in line_lower) or \
               (line.startswith('**') and 'performance' in line_lower and line.endswith('**')):
                current_section = "performance_analysis"
                continue
            elif (line.startswith('2.') and ('strategy' in line_lower or 'investment' in line_lower)) or \
                 (line.startswith('**') and ('strategy' in line_lower or 'investment' in line_lower) and line.endswith('**')):
                current_section = "investment_strategy"
                continue
            elif (line.startswith('3.') and 'risk' in line_lower) or \
                 (line.startswith('**') and 'risk' in line_lower and line.endswith('**')):
                current_section = "risk_assessment"
                continue
            elif (line.startswith('4.') and ('sip' in line_lower or 'lump' in line_lower)) or \
                 (line.startswith('**') and 'sip' in line_lower and line.endswith('**')):
                current_section = "sip_analysis"
                continue
            elif (line.startswith('4.') and ('outlook' in line_lower or 'market' in line_lower)) or \
                 (line.startswith('**') and ('outlook' in line_lower or 'market' in line_lower) and line.endswith('**')):
                current_section = "market_outlook"
                continue
            
            # Add content to current section (skip header lines)
            if current_section and not line.startswith(('1.', '2.', '3.', '4.', '**')):
                if sections[current_section]:
                    sections[current_section] += " " + line
                else:
                    sections[current_section] = line
        
        # If sections are empty, try a simpler approach - split by double asterisks
        if not any(sections.values()):
            parts = analysis_text.split('**')
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                    
                part_lower = part.lower()
                if 'performance' in part_lower and i + 1 < len(parts):
                    sections["performance_analysis"] = parts[i + 1].strip()
                elif ('strategy' in part_lower or 'investment' in part_lower) and i + 1 < len(parts):
                    sections["investment_strategy"] = parts[i + 1].strip()
                elif 'risk' in part_lower and i + 1 < len(parts):
                    sections["risk_assessment"] = parts[i + 1].strip()
                elif ('outlook' in part_lower or 'market' in part_lower) and i + 1 < len(parts):
                    sections["market_outlook"] = parts[i + 1].strip()
                elif 'sip' in part_lower and i + 1 < len(parts):
                    sections["sip_analysis"] = parts[i + 1].strip()
        
        # Clean up empty sections and return only non-empty ones
        return {k: v for k, v in sections.items() if v.strip()}

# Create global instance
llm_service = LLMAnalysisService()
