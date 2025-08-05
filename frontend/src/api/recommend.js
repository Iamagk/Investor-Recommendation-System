import axios from "axios";

const BASE_URL = "http://localhost:8002"; // Updated to match the FastAPI server port

export const getRecommendation = async (data) => {
  const response = await axios.post(`${BASE_URL}/recommend`, data);
  return response.data;
};