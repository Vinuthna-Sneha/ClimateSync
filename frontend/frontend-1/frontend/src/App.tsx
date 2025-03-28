import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ZoneMap from "./components/ZoneMap";
import StrategyPage from "./components/StrategyPage";

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        {/* Remove selectedZone prop since ZoneMap manages it internally */}
        <Route path="/" element={<ZoneMap />} />
        {/* Standardize route casing to lowercase for consistency */}
        <Route path="/strategy" element={<StrategyPage />} />
      </Routes>
    </Router>
  );
};

export default App;