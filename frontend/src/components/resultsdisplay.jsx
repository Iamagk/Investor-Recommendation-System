export default function ResultsDisplay({ results }) {
  if (!results || !results.stocks) return null;

  return (
    <div className="mt-4 p-4 bg-gray-100 rounded shadow">
      <h2 className="text-xl font-semibold mb-2">Recommended Portfolio</h2>
      <ul className="space-y-2">
        {["stocks", "mutual_funds", "gold"].map((type) =>
          results[type]?.map((item, i) => (
            <li key={`${type}-${i}`} className="bg-white p-3 rounded shadow">
              <strong>{type.toUpperCase()}</strong>: {item.name} — ₹{item.amount} — {item.reason}
            </li>
          ))
        )}
      </ul>
    </div>
  );
}