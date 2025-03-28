import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { useNavigate, useLocation } from "react-router-dom";
import '@fortawesome/fontawesome-free/css/all.css';

// Function to parse Markdown to plain text with styled JSX, including tables
const parseMarkdownToText = (text: string): JSX.Element => {
  let processedText = text
    .replace(/Â°/g, "°")
    .replace(/â‚„/g, "₄")
    .replace(/â‚‚/g, "₂")
    .replace(/â‚ƒ/g, "₃")
    .replace(/\bCH₄\b/g, "CH₄")
    .replace(/\bNO₂\b/g, "NO₂")
    .replace(/\bO₃\b/g, "O₃")
    .replace(/\bSO₂\b/g, "SO₂")
    .replace(/\bH₂SO₄\b/g, "H₂SO₄")
    .replace(/\bCO₂\b/g, "CO₂")
    .replace(/\bH₂O\b/g, "H₂O");

  const lines = processedText.split("\n");
  const elements: JSX.Element[] = [];
  let inTable = false;
  let tableHeaders: string[] = [];
  let tableRows: string[][] = [];

  lines.forEach((line, index) => {
    const trimmedLine = line.trim();

    if (!trimmedLine) {
      if (inTable) {
        elements.push(renderTable(tableHeaders, tableRows, index));
        inTable = false;
        tableHeaders = [];
        tableRows = [];
      }
      elements.push(<br key={index} />);
      return;
    }

    // Modified table detection logic
    if (trimmedLine.startsWith("|") && trimmedLine.includes("|")) {
      const cells = trimmedLine
        .split("|")
        .map(cell => cell.trim())
        .filter(cell => cell !== ''); // Remove empty cells from split

      if (!inTable && cells.length >= 2) {
        // First row with multiple cells is the header
        inTable = true;
        tableHeaders = cells;
        return;
      }

      if (inTable && cells.length >= 2) {
        // Subsequent rows with multiple cells are data rows
        tableRows.push(cells);
        return;
      }
    } else if (inTable) {
      // If we were in a table and hit a non-table line, render the table
      elements.push(renderTable(tableHeaders, tableRows, index));
      inTable = false;
      tableHeaders = [];
      tableRows = [];
    }

    if (trimmedLine.match(/^#{1,6}\s/)) {
      const level = trimmedLine.match(/^#+/)![0].length;
      const content = trimmedLine.replace(/^#{1,6}\s/, "");
      elements.push(
        <div
          key={index}
          className={`font-bold text-gray-800 mb-2 ${
            level === 1 ? "text-2xl" : level === 2 ? "text-xl" : "text-lg"
          }`}
        >
          {parseInlineMarkdown(content)}
        </div>
      );
      return;
    }

    if (trimmedLine.match(/^\s*[-*]\s/)) {
      const content = trimmedLine.replace(/^\s*[-*]\s/, "");
      elements.push(
        <div key={index} className="flex items-start mb-1 text-gray-700 text-sm">
          <span className="mr-2">•</span>
          <span>{parseInlineMarkdown(content)}</span>
        </div>
      );
      return;
    }

    if (trimmedLine.match(/^```/)) {
      const codeLines: string[] = [];
      let i = index + 1;
      while (i < lines.length && !lines[i].match(/^```/)) {
        codeLines.push(lines[i]);
        i++;
      }
      elements.push(
        <pre
          key={index}
          className="bg-gray-100 p-2 rounded-md text-sm text-gray-800 mb-2 overflow-x-auto"
        >
          {codeLines.join("\n")}
        </pre>
      );
      lines.splice(index + 1, i - index);
      return;
    }

    elements.push(
      <p key={index} className="text-gray-700 text-sm mb-2 leading-relaxed">
        {parseInlineMarkdown(trimmedLine)}
      </p>
    );
  });

  if (inTable) {
    elements.push(renderTable(tableHeaders, tableRows, lines.length));
  }

  return <div>{elements}</div>;
};// Helper to render a table
const renderTable = (headers: string[], rows: string[][], keyBase: number): JSX.Element => {
  return (
    <table key={`table-${keyBase}`} className="w-full border-collapse mb-4 text-sm text-gray-700">
      <thead>
        <tr className="bg-gray-200">
          {headers.map((header, i) => (
            <th key={i} className="border border-gray-300 p-2 font-semibold text-left">
              {parseInlineMarkdown(header)}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, rowIndex) => (
          <tr key={rowIndex} className={rowIndex % 2 === 0 ? "bg-gray-50" : "bg-white"}>
            {row.map((cell, cellIndex) => (
              <td key={cellIndex} className="border border-gray-300 p-2">
                {parseInlineMarkdown(cell)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
};

// Helper to parse inline Markdown
const parseInlineMarkdown = (text: string): JSX.Element => {
  let parts: (string | JSX.Element)[] = [text];

  parts = parts.flatMap((part) => {
    if (typeof part !== "string") return part;
    const boldRegex = /(\*\*[^*]+\*\*|__[^_]+__)/g;
    const splitParts = part.split(boldRegex);
    return splitParts.map((subPart, i) => {
      if (subPart.match(/^\*\*[^*]+\*\*$/)) {
        return (
          <span key={i} className="font-bold">
            {subPart.slice(2, -2)}
          </span>
        );
      }
      if (subPart.match(/^__[^_]+__$/)) {
        return (
          <span key={i} className="font-bold">
            {subPart.slice(2, -2)}
          </span>
        );
      }
      return subPart;
    });
  });

  parts = parts.flatMap((part) => {
    if (typeof part !== "string") return part;
    const italicRegex = /(\*[^*]+\*|_[^_]+_)/g;
    const splitParts = part.split(italicRegex);
    return splitParts.map((subPart, i) => {
      if (subPart.match(/^\*[^*]+\*$/)) {
        return (
          <span key={i} className="italic">
            {subPart.slice(1, -1)}
          </span>
        );
      }
      if (subPart.match(/^_[^_]+_$/)) {
        return (
          <span key={i} className="italic">
            {subPart.slice(1, -1)}
          </span>
        );
      }
      return subPart;
    });
  });

  return <>{parts}</>;
};

const calculatePopupPosition = (rect: DOMRect, windowHeight: number, popupHeight = 300) => {
  const spaceBelow = windowHeight - rect.bottom;
  const spaceAbove = rect.top;

  if (spaceBelow < popupHeight && spaceAbove > popupHeight) {
    return { x: rect.left, y: rect.top - popupHeight - 10 };
  }

  const yPos = Math.min(rect.bottom + window.scrollY, windowHeight - popupHeight - 10 + window.scrollY);
  return { x: rect.left, y: yPos };
};

// Helper to extract zone ID from the formatted string "Zone {id} ({zone_name})"
const extractZoneId = (zoneString: string): string => {
  const match = zoneString.match(/Zone (\d+)/);
  return match ? match[1] : "";
};

const StrategyPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [zones, setZones] = useState<string[]>([]);
  const [selectedZone, setSelectedZone] = useState<string>("");
  const [strategyZone, setStrategyZone] = useState<string>("");
  const [idea, setIdea] = useState<string>("");
  const [originalIdea, setOriginalIdea] = useState<string>("");
  const [strategy, setStrategy] = useState<{ strategy: string } | null>(null);
  const [pastStrategies, setPastStrategies] = useState<
    { version: number; timestamp: string; strategy: string; idea: string }[]
  >([]);
  const [selectedText, setSelectedText] = useState<string>("");
  const [popupVisible, setPopupVisible] = useState(false);
  const [popupPosition, setPopupPosition] = useState({ x: 0, y: 0 });
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [isFinalized, setIsFinalized] = useState(false);
  const [progress, setProgress] = useState({ emissions: 0, cost: 0 });
  const [actionType, setActionType] = useState<"question" | "modify" | null>(null);
  const [answers, setAnswers] = useState<{ question: string; answer: string }[]>([]);
  const [isPastStrategiesCollapsed, setIsPastStrategiesCollapsed] = useState(false);
  const [isQnAVisible, setIsQnAVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const strategyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (location.state) {
      const { selectedText, zoneId } = location.state as { selectedText?: string; zoneId?: string };
      if (selectedText) setIdea(selectedText);
      if (zoneId) setSelectedZone(zoneId);
    }
  }, [location.state]);

  useEffect(() => {
    const fetchZones = async () => {
      try {
        const response = await fetch("https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/zone_ids");
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        setZones(data);
        if (data.length > 0 && !selectedZone && !location.state?.zoneId) {
          const firstZoneId = extractZoneId(data[0]);
          setSelectedZone(firstZoneId);
        }
      } catch (error) {
        console.error("Error fetching zones:", error);
        setZones([]);
      }
    };
    fetchZones();
  }, [location.state]);

  useEffect(() => {
    const fetchPastStrategies = async () => {
      if (!selectedZone) return;
      try {
        const response = await fetch(`https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/get-all-strategies/${selectedZone}`);
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
        }
        const data = await response.json();
        console.log(`Fetched ${data.strategies.length} strategies for Zone ${selectedZone}:`, data.strategies);
        setPastStrategies(
          data.strategies.map((s: any) => ({
            version: s.version,
            timestamp: s.timestamp,
            strategy: s.strategy,
            idea: s.idea,
          }))
        );
      } catch (error) {
        console.error("Error fetching past strategies:", error);
        setPastStrategies([]);
      }
    };
    fetchPastStrategies();
  }, [selectedZone]);

  const handleGenerateStrategy = async () => {
    if (!selectedZone || !idea) {
      alert("Please select a zone and provide an idea.");
      return;
    }
    setIsLoading(true);
    try {
      const response = await fetch("https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/generate-strategy/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ zone_id: parseInt(selectedZone), idea }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setStrategy({ strategy: data.strategy });
      setStrategyZone(selectedZone);
      setOriginalIdea(idea);
      setIsFinalized(false);
      setIdea("");
    } catch (error) {
      console.error("Error fetching strategy:", error);
      setStrategy({ strategy: "Failed to load strategy." });
    } finally {
      setIsLoading(false);
    }
  };

  const handleTextSelection = () => {
    const selection = window.getSelection();
    const text = selection?.toString();
    if (text && text.length > 0) {
      setSelectedText(text);
      setAnswer("");
      const range = selection?.getRangeAt(0);
      const rect = range?.getBoundingClientRect();
      if (rect) {
        const newPosition = calculatePopupPosition(rect, window.innerHeight);
        setPopupPosition(newPosition);
        setPopupVisible(true);
      }
    } else {
      setPopupVisible(false);
      setAnswer("");
    }
  };

  const handleQuestionSubmit = async () => {
    if (!strategy || !idea) return;
    try {
      const response = await fetch("https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/ask-question/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: strategy.strategy, question: idea }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setAnswers([...answers, { question: idea, answer: data.answer }]);
      setIdea("");
      setIsQnAVisible(true);
    } catch (error) {
      console.error("Error submitting question:", error);
      setAnswers([...answers, { question: idea, answer: "Failed to get an answer." }]);
      setIdea("");
      setIsQnAVisible(true);
    }
  };

  const handleModifyOverall = async () => {
    if (!strategy) return;
    setIsLoading(true);
    try {
      const response = await fetch("https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/modify-strategy/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          zone_id: parseInt(strategyZone),
          strategy: strategy.strategy,
          modification_request: idea,
        }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setStrategy({ strategy: data.modified_strategy });
      setOriginalIdea(idea);
      setIdea("");
      setIsFinalized(false);
    } catch (error) {
      console.error("Error modifying strategy:", error);
      setStrategy({ ...strategy, strategy: "Failed to modify strategy." });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFinalize = async () => {
    if (!strategy || typeof strategy.strategy !== "string" || !strategyZone || isNaN(parseInt(strategyZone)) || !originalIdea) {
      console.error("Cannot finalize: Invalid strategy, zone, or idea", { strategy, strategyZone, originalIdea });
      return;
    }
    const timestamp = new Date().toLocaleString();
    const strategyData = {
      zone_id: parseInt(strategyZone),
      strategy: strategy.strategy,
      timestamp: timestamp,
      idea: originalIdea,
    };
    try {
      const response = await fetch("https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/finalize-strategy/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(strategyData),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
      }
      const result = await response.json();
      const updatedResponse = await fetch(`https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/get-all-strategies/${strategyZone}`);
      if (!updatedResponse.ok) {
        if (updatedResponse.status === 404 || updatedResponse.status === 200) {
          setPastStrategies([{ version: result.version, timestamp, strategy: strategy.strategy, idea: originalIdea }]);
        } else {
          throw new Error(`HTTP error! status: ${updatedResponse.status}`);
        }
      } else {
        const updatedData = await updatedResponse.json();
        setPastStrategies(
          updatedData.strategies.map((s: any) => ({
            version: s.version,
            timestamp: s.timestamp,
            strategy: s.strategy,
            idea: s.idea,
          }))
        );
      }
      setIsFinalized(true);
    } catch (error) {
      console.error("Error finalizing strategy:", error);
      setPastStrategies((prev) => [
        ...prev,
        { version: pastStrategies.length + 1, timestamp, strategy: strategy.strategy, idea: originalIdea },
      ]);
      setIsFinalized(true);
    }
  };

  const handleVersionClick = async (version: number) => {
    if (!selectedZone) {
      alert("Please select a zone first.");
      return;
    }
    try {
      const response = await fetch(`https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/get-strategy/${selectedZone}/${version}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setStrategy({ strategy: data.strategy });
      setStrategyZone(selectedZone);
      setIsFinalized(true);
    } catch (error) {
      console.error("Error fetching strategy version:", error);
      setStrategy({ strategy: "Failed to load strategy version." });
    }
  };

  const handleSend = () => {
    if (!idea || !selectedZone) {
      alert("Please select a zone and provide an idea.");
      return;
    }
    if (!strategy && !selectedText) {
      handleGenerateStrategy();
    } else if (actionType === "question") {
      handleQuestionSubmit();
    } else if (actionType === "modify") {
      handleModifyOverall();
    } else {
      handleGenerateStrategy();
    }
  };

  const toggleActionType = (type: "question" | "modify") => {
    setActionType(actionType === type ? null : type);
  };

  return (
    <div className="h-screen w-screen bg-gray-100 flex flex-col overflow-hidden">
      <div className="flex items-center bg-[#034c53] text-white p-4 z-10 shadow-md rounded-t-lg fixed top-0 left-0 right-0">
        <button
          onClick={() => navigate("/")}
          className="bg-transparent border-none text-white text-base cursor-pointer mr-5"
        >
          ←
        </button>
        <h2 className="m-0 text-2xl font-medium">Climate Strategy Planner</h2>
      </div>

      <div className="flex flex-1 overflow-hidden mt-[60px] relative">
        <div
          className={`bg-gray-50 border-r border-gray-200 overflow-y-auto h-[calc(100vh-60px)] transition-all duration-300 ${
            isPastStrategiesCollapsed ? "w-0 overflow-hidden" : "w-1/4 p-5"
          }`}
        >
          {!isPastStrategiesCollapsed && (
            <>
              <div className="flex justify-between items-center mb-5">
                <h2 className="text-xl font-semibold text-[#034c53]">Past Strategies</h2>
                <button
                  onClick={() => setIsPastStrategiesCollapsed(true)}
                  className="bg-gray-300 text-gray-700 p-1 rounded-full w-8 h-8 flex items-center justify-center hover:bg-gray-400 transition-colors"
                >
                  <i className="fas fa-history"></i>
                </button>
              </div>
              {pastStrategies.length > 0 ? (
                <ul className="list-none p-0">
                  {pastStrategies.map((strat) => (
                    <li
                      key={strat.version}
                      className="bg-white p-3.5 rounded-md mb-3.5 text-gray-700 text-sm shadow-sm cursor-pointer hover:bg-gray-100 relative"
                      onClick={() => handleVersionClick(strat.version)}
                    >
                      <span className="absolute top-2 right-2 text-xs font-bold text-gray-500">
                        Version {strat.version}
                      </span>
                      <h3 className="text-lg font-medium text-gray-800 mt-1 mb-2">{strat.idea}</h3>
                      <div>
                        <strong className="font-bold">Strategy:</strong>{" "}
                        {parseMarkdownToText(strat.strategy.slice(0, 50) + "...")}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">{strat.timestamp}</div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-gray-500 text-sm">No past strategies available.</p>
              )}
            </>
          )}
        </div>

        {isPastStrategiesCollapsed && (
          <button
            onClick={() => setIsPastStrategiesCollapsed(false)}
            className="absolute top-[70px] left-2 bg-gray-300 text-gray-700 p-1 rounded-full w-8 h-8 flex items-center justify-center hover:bg-gray-400 transition-colors z-10"
          >
            <i className="fas fa-history"></i>
          </button>
        )}

        <div className="flex-1 p-10 overflow-y-auto text-gray-800 flex flex-col relative">
          <div className="flex-1 overflow-y-auto">
            {strategy && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="bg-white p-5 rounded-lg border border-gray-200 mb-5 shadow-sm"
                ref={strategyRef}
                onMouseUp={handleTextSelection}
              >
                <h2 className="text-xl font-semibold text-black mb-5">
                  Current Strategy for Zone {strategyZone}
                </h2>
                <div className="mb-5">
                  {Object.entries(strategy).map(([section, content]) => (
                    <div key={section} className="mb-5">
                      <h3 className="text-base font-medium text-black capitalize mb-2.5">{section}</h3>
                      <div className="leading-relaxed text-gray-700 text-sm">
                        {parseMarkdownToText(content)}
                      </div>
                    </div>
                  ))}
                </div>
                <button
                  onClick={handleFinalize}
                  disabled={isFinalized}
                  className={`bg-green-500 text-white border-none py-2.5 px-5 rounded-md cursor-pointer text-sm font-medium transition-colors ${
                    isFinalized ? "opacity-60" : "hover:bg-green-600"
                  }`}
                >
                  Finalize Strategy
                </button>
              </motion.div>
            )}

            {isFinalized && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="bg-white p-5 rounded-lg border border-gray-200 mb-5 shadow-sm"
              >
                <h2 className="text-xl font-semibold text-black mb-5">Progress Monitoring</h2>
                <p className="text-gray-700 mb-2.5 text-sm">
                  Emissions Reduced: {progress.emissions} tons CO2e
                </p>
                <p className="text-gray-700 mb-3.5 text-sm">Cost Spent: ${progress.cost}M</p>
                <button className="bg-green-500 text-white border-none py-2.5 px-5 rounded-md cursor-pointer text-sm font-medium hover:bg-green-600 transition-colors">
                  Get Assistance
                </button>
              </motion.div>
            )}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-white p-2.5 rounded-lg border border-gray-200 shadow-sm w-4/5 max-w-4xl self-center sticky bottom-2.5 z-10"
          >
            <div className="flex gap-2.5 mb-2">
              <div className="flex-1">
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Your Idea
                </label>
                <textarea
                  value={idea}
                  onChange={(e) => setIdea(e.target.value)}
                  placeholder="Enter your idea, question, or modification..."
                  className="w-full p-2 bg-gray-50 text-gray-700 border border-gray-200 rounded-md h-16 resize-none text-sm outline-none"
                />
              </div>
              <div className="w-56">
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Select Zone
                </label>
                <select
                  value={selectedZone}
                  onChange={(e) => setSelectedZone(e.target.value)}
                  className="w-full p-2 bg-gray-50 text-gray-700 border border-gray-200 rounded-md text-sm outline-none"
                >
                  <option value="" disabled>
                    Select a zone
                  </option>
                  {zones.length > 0 ? (
                    zones.map((zone) => {
                      const zoneId = extractZoneId(zone);
                      return (
                        <option key={zoneId} value={zoneId}>
                          {zone}
                        </option>
                      );
                    })
                  ) : (
                    <option value="" disabled>
                      No zones available
                    </option>
                  )}
                </select>
              </div>
            </div>
            <div className="flex justify-end items-center gap-2">
              {strategy && (
                <div className="flex gap-1.5">
                  <button
                    onClick={() => toggleActionType("question")}
                    className={`py-1.5 px-2.5 rounded-md cursor-pointer text-xs font-medium transition-colors ${
                      actionType === "question" ? "bg-green-500 text-white" : "bg-gray-300 text-gray-700 hover:bg-gray-400"
                    }`}
                  >
                    Question
                  </button>
                  <button
                    onClick={() => toggleActionType("modify")}
                    className={`py-1.5 px-2.5 rounded-md cursor-pointer text-xs font-medium transition-colors ${
                      actionType === "modify" ? "bg-green-500 text-white" : "bg-gray-300 text-gray-700 hover:bg-gray-400"
                    }`}
                  >
                    Modify
                  </button>
                </div>
              )}
              <button
                onClick={handleSend}
                disabled={!idea || !selectedZone || isLoading}
                className={`bg-green-500 text-white border-none py-2 px-4 rounded-md cursor-pointer text-sm font-medium transition-colors ${
                  (!idea || !selectedZone || isLoading) ? "opacity-60" : "hover:bg-green-600"
                }`}
              >
                {isLoading ? (
                  <span className="flex items-center">
                    <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Loading...
                  </span>
                ) : (
                  "Send"
                )}
              </button>
            </div>
          </motion.div>
        </div>
      </div>

      {popupVisible && selectedText && (
        <motion.div
          className="fixed bg-white p-3.5 rounded-lg border border-gray-200 shadow-md z-50 w-[300px] max-h-[400px] overflow-y-auto overflow-x-hidden"
          style={{ top: popupPosition.y, left: Math.min(popupPosition.x, window.innerWidth - 320) }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
        >
          <h2 className="text-base font-medium text-black mb-2.5 break-words">
            Selected: "{selectedText}"
          </h2>
          <div className="mb-2.5">
            <input
              className="w-full p-2 bg-gray-50 text-gray-700 border border-gray-200 rounded-md text-sm box-border"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about the selected text..."
            />
            <button
              onClick={async () => {
                if (!strategy || !question) return;
                try {
                  const response = await fetch("https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/ask-question/", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                      strategy: strategy.strategy,
                      question,
                      selected_text: selectedText,
                    }),
                  });
                  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                  const data = await response.json();
                  setAnswer(data.answer);
                  setQuestion("");
                } catch (error) {
                  console.error("Error submitting question:", error);
                  setAnswer("Failed to get an answer.");
                }
              }}
              disabled={!question}
              className={`w-full bg-green-500 text-white border-none p-2 rounded-md cursor-pointer text-sm font-medium transition-colors mt-1.5 ${
                !question ? "opacity-60" : "hover:bg-green-600"
              }`}
            >
              Submit Question
            </button>
          </div>
          {answer && (
            <div className="bg-gray-50 p-2.5 rounded-md mb-2.5 text-gray-700 text-sm break-words">
              {parseMarkdownToText(answer)}
            </div>
          )}
          <button
            onClick={() => {
              setPopupVisible(false);
              setAnswer("");
              setQuestion("");
            }}
            className="w-full bg-red-500 text-white border-none p-2 rounded-md cursor-pointer text-sm font-medium hover:bg-red-600 transition-colors"
          >
            Close
          </button>
        </motion.div>
      )}

      {isQnAVisible && (
        <motion.div
          className="fixed bottom-20 right-5 w-96 bg-white rounded-lg shadow-lg border border-gray-200 z-50 flex flex-col"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
        >
          <div className="flex justify-between items-center bg-gray-100 p-2 rounded-t-lg">
            <h2 className="text-sm font-semibold text-gray-800">Questions & Answers</h2>
            <div className="flex gap-1">
              <button
                onClick={() => setIsQnAVisible(false)}
                className="bg-gray-300 text-gray-700 p-1 rounded-full w-6 h-6 flex items-center justify-center hover:bg-gray-400 transition-colors"
              >
                ×
              </button>
            </div>
          </div>
          <div className="flex-1 p-3 overflow-y-auto max-h-80">
            {answers.length > 0 ? (
              answers.map((qa, index) => (
                <div key={index} className="mb-3">
                  <p className="font-medium text-gray-700 text-xs mb-1">Q: {qa.question}</p>
                  <div className="text-gray-600 text-xs">{parseMarkdownToText(qa.answer)}</div>
                </div>
              ))
            ) : (
              <p className="text-gray-500 text-xs">No questions yet.</p>
            )}
          </div>
        </motion.div>
      )}

      {answers.length > 0 && !isQnAVisible && (
        <button
          onClick={() => setIsQnAVisible(true)}
          className="fixed bottom-5 right-5 bg-green-500 text-white p-3 rounded-full shadow-lg hover:bg-green-600 transition-colors z-50"
        >
          Q&A
        </button>
      )}
    </div>
  );
};

export default StrategyPage;