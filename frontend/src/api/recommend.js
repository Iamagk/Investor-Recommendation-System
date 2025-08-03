import axios from "axios";

const BASE_URL = "http://localhost:8000"; // or your FastAPI deployed URL

export const getRecommendation = async (data) => {
  const response = await axios.post(`${BASE_URL}/recommend`, data);
  return response.data;
};