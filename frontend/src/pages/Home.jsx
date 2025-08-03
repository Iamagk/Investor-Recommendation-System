import { useState } from "react";
import InvestmentForm from "../components/InvestmentForm";
import ResultsDisplay from "../components/ResultsDisplay";
import { getRecommendation } from "../api/recommend";

export default function Home() {
  const [results, setResults] = useState(null);

  const handleSubmit = async (data) => {
    const res = await getRecommendation(data);
    setResults(res);
  };

  return (
    <div className="max-w-3xl mx-auto mt-6">
      <InvestmentForm onSubmit={handleSubmit} />
      <ResultsDisplay results={results} />
    </div>
  );
}