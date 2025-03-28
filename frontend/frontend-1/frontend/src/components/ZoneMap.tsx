import React, { useEffect, useState, useRef } from "react";
import { APIProvider, Map, AdvancedMarker } from "@vis.gl/react-google-maps";
import { Pie, Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
} from "chart.js";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title);

interface ZoneData {
  zone_id: string;
  coordinates: { latitude: number; longitude: number }[];
}

interface ZoneInfo {
  zone_id: string;
  details: string[];
}

interface ZoneDemographics {
  zone_id: string;
  demographics: {
    latitude: number;
    longitude: number;
    bare: number;
    built: number;
    crops: number;
    flooded_vegetation: number;
    grass: number;
    shrub_and_scrub: number;
    snow_and_ice: number;
    trees: number;
    water: number;
    temperature: number;
    humidity: number;
    precipitation: number;
    wind_speed: number;
    BCCMASS: number;
    CH4_column_volume_mixing_ratio_dry_air_bias_corrected: number;
    CO_column_number_density: number;
    DUCMASS: number;
    NO2_column_number_density: number;
    O3_column_number_density: number;
    SO2_column_number_density: number;
    absorbing_aerosol_index: number;
    tropospheric_HCHO_column_number_density: number;
    Health_Risk_Index: number;
    Urban_Heat_Index: number;
    Real_Estate_Risk: number;
    Green_Score: number;
    H: number;
    E: number;
    V: number;
    CRI: number;
    Risk_Category: string;
  };
}

type Zone = {
  id: string; // Raw zone_id (e.g., "0")
  displayName: string; // Formatted name (e.g., "Zone 0 (Santhosha Puram)")
};

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

    if (trimmedLine.startsWith("|") && trimmedLine.endsWith("|")) {
      const cells = trimmedLine
        .slice(1, -1)
        .split("|")
        .map((cell) => cell.trim());

      if (!inTable && cells.length > 1) {
        inTable = true;
        tableHeaders = cells;
        return;
      }

      if (inTable && trimmedLine.match(/^\s*\|[-:\s]+\|[-:\s]+\|\s*$/)) {
        return;
      }

      if (inTable) {
        tableRows.push(cells);
        return;
      }
    } else if (inTable) {
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
};

// Helper to render a table
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

const generateRandomColor = (): string => {
  const letters = "0123456789ABCDEF";
  let color = "#";
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
};

const generateColorArray = (numZones: number): string[] => {
  const colors = new Set<string>();
  while (colors.size < numZones) {
    colors.add(generateRandomColor());
  }
  return Array.from(colors);
};

// Function to extract sections and their positions
const extractSection = (
  text: string,
  startHeading: string,
  nextHeading: string | null
): { content: string | null; start: number; end: number } => {
  const startMatch = new RegExp(startHeading).exec(text);
  if (!startMatch) {
    return { content: null, start: -1, end: -1 };
  }

  const startPos = startMatch.index + startMatch[0].length;
  let endPos: number;

  if (nextHeading) {
    const nextMatch = new RegExp(nextHeading).exec(text.slice(startPos));
    endPos = nextMatch ? startPos + nextMatch.index : text.length;
  } else {
    endPos = text.length;
  }

  return {
    content: text.slice(startPos, endPos).trim(),
    start: startMatch.index,
    end: endPos,
  };
};

const ZoneMap: React.FC = () => {
  const [zones, setZones] = useState<Zone[]>([]);
  const [zoneData, setZoneData] = useState<ZoneData[]>([]);
  const [selectedZone, setSelectedZone] = useState<ZoneInfo | null>(null);
  const [selectedDemographics, setSelectedDemographics] = useState<ZoneDemographics | null>(null);
  const [loading, setLoading] = useState(true);
  const [zoneColors, setZoneColors] = useState<string[]>([]);
  const [mapWidth, setMapWidth] = useState(50);
  const [isLegendMinimized, setIsLegendMinimized] = useState(false);
  const [activeTab, setActiveTab] = useState<"Report" | "Demographics">("Report");
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchZoneIds = async () => {
      try {
        setLoading(true);
        const res = await fetch("https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/zone_ids");
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const zoneIds: string[] = await res.json();
        const parsedZones: Zone[] = zoneIds.map((zoneStr) => {
          const match = zoneStr.match(/Zone (\d+) \((.+)\)/);
          if (!match) throw new Error(`Invalid zone format: ${zoneStr}`);
          return {
            id: match[1],
            displayName: zoneStr,
          };
        });
        setZones(parsedZones);
        setZoneColors(generateColorArray(parsedZones.length));
        await fetchAllZoneData(parsedZones);
      } catch (err) {
        console.error("Error fetching zone IDs:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchZoneIds();
  }, []);

  const fetchAllZoneData = async (zoneList: Zone[]) => {
    const data: ZoneData[] = [];
    for (const zone of zoneList) {
      try {
        const res = await fetch(`https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/zones/${zone.id}`);
        if (!res.ok) continue;
        const zoneData: ZoneData = await res.json();
        data.push(zoneData);
      } catch (err) {
        console.error(`Error fetching zone ${zone.id}:`, err);
      }
    }
    setZoneData(data);
  };

  const fetchZoneDetails = async (zoneId: string) => {
    try {
      const res = await fetch(`https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/zone_info/${zoneId}`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data: ZoneInfo = await res.json();
      setSelectedZone(data);
    } catch (error) {
      console.error("Error fetching zone details:", error);
    }
  };

  const fetchZoneDemographics = async (zoneId: string) => {
    try {
      const res = await fetch(`https://backend-dot-inspired-rock-450806-r5.uc.r.appspot.com/zone_demographics/${zoneId}`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data: ZoneDemographics = await res.json();
      setSelectedDemographics(data);
    } catch (error) {
      console.error("Error fetching zone demographics:", error);
    }
  };

  const handleZoneClick = (zoneId: string) => {
    fetchZoneDetails(zoneId);
    fetchZoneDemographics(zoneId);
  };

  const handleMouseDown = () => {
    isDragging.current = true;
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging.current && containerRef.current) {
      const containerWidth = containerRef.current.getBoundingClientRect().width;
      const newWidth = (e.clientX / containerWidth) * 100;
      setMapWidth(Math.max(20, Math.min(80, newWidth)));
    }
  };

  useEffect(() => {
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  const Legend = () => (
    <div
      style={{
        position: "absolute",
        bottom: "30px",
        left: "10px",
        background: "white",
        color: "black",
        padding: "10px",
        borderRadius: "5px",
        boxShadow: "0 0 5px rgba(0,0,0,0.3)",
        zIndex: 1000,
        minWidth: "150px",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: isLegendMinimized ? 0 : "10px",
        }}
      >
        <h4 style={{ margin: 0, color: "#000" }}>Zone Legend</h4>
        <button
          onClick={() => setIsLegendMinimized((prev) => !prev)}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontSize: "16px",
            color: "#000",
            padding: "0",
            lineHeight: "1",
          }}
          aria-label={isLegendMinimized ? "Expand legend" : "Minimize legend"}
        >
          {isLegendMinimized ? "▲" : "▼"}
        </button>
      </div>
      {!isLegendMinimized &&
        zones.map((zone, index) => (
          <div key={zone.id} style={{ display: "flex", alignItems: "center" }}>
            <div
              style={{
                width: "15px",
                height: "15px",
                backgroundColor: zoneColors[index],
                marginRight: "5px",
                opacity: 0.6,
              }}
            />
            <span style={{ color: "#000" }}>{zone.displayName}</span>
          </div>
        ))}
    </div>
  );

  const DemographicsSection = () => {
    if (!selectedDemographics || !selectedDemographics.demographics) {
      return (
        <p style={{ color: "#666", fontStyle: "italic", textAlign: "center", marginTop: "20px" }}>
          Select a zone to view demographics.
        </p>
      );
    }

    const { demographics } = selectedDemographics;

    const landUseData = {
      labels: [
        "Bare Land",
        "Built-up Area",
        "Crops",
        "Flooded Vegetation",
        "Grass",
        "Shrub and Scrub",
        "Snow and Ice",
        "Trees",
        "Water",
      ],
      datasets: [
        {
          data: [
            demographics.bare * 100,
            demographics.built * 100,
            demographics.crops * 100,
            demographics.flooded_vegetation * 100,
            demographics.grass * 100,
            demographics.shrub_and_scrub * 100,
            demographics.snow_and_ice * 100,
            demographics.trees * 100,
            demographics.water * 100,
          ],
          backgroundColor: [
            "#FF6384",
            "#36A2EB",
            "#FFCE56",
            "#4BC0C0",
            "#9966FF",
            "#FF9F40",
            "#C9CBCF",
            "#2ECC71",
            "#3498DB",
          ],
          hoverBackgroundColor: [
            "#FF6384",
            "#36A2EB",
            "#FFCE56",
            "#4BC0C0",
            "#9966FF",
            "#FF9F40",
            "#C9CBCF",
            "#2ECC71",
            "#3498DB",
          ],
        },
      ],
    };

    const envFactorsData = {
      labels: ["Temperature", "Humidity", "Precipitation", "Wind Speed"],
      datasets: [
        {
          label: "Environmental Factors",
          data: [
            demographics.temperature,
            demographics.humidity,
            demographics.precipitation,
            demographics.wind_speed,
          ],
          backgroundColor: "rgba(75, 192, 192, 0.6)",
          borderColor: "rgba(75, 192, 192, 1)",
          borderWidth: 1,
        },
      ],
    };

    const airQualityData = {
      labels: [
        "BC Mass",
        "CH₄ Column",
        "CO Column",
        "DUCMASS",
        "NO₂ Column",
        "O₃ Column",
        "SO₂ Column",
        "Absorbing Aerosol Index",
        "Tropospheric HCHO",
      ],
      datasets: [
        {
          label: "Air Quality Metrics",
          data: [
            demographics.BCCMASS,
            demographics.CH4_column_volume_mixing_ratio_dry_air_bias_corrected,
            demographics.CO_column_number_density,
            demographics.DUCMASS,
            demographics.NO2_column_number_density,
            demographics.O3_column_number_density,
            demographics.SO2_column_number_density,
            demographics.absorbing_aerosol_index,
            demographics.tropospheric_HCHO_column_number_density,
          ],
          backgroundColor: "rgba(255, 99, 132, 0.6)",
          borderColor: "rgba(255, 99, 132, 1)",
          borderWidth: 1,
        },
      ],
    };

    const riskMetricsData = {
      labels: ["Health Risk", "Urban Heat", "Real Estate", "Green Score", "CRI"],
      datasets: [
        {
          label: "Risk Metrics",
          data: [
            demographics.Health_Risk_Index,
            demographics.Urban_Heat_Index,
            demographics.Real_Estate_Risk,
            demographics.Green_Score,
            demographics.CRI,
          ],
          backgroundColor: "rgba(54, 162, 235, 0.6)",
          borderColor: "rgba(54, 162, 235, 1)",
          borderWidth: 1,
        },
      ],
    };

    const additionalMetricsData = {
      labels: ["Hazard Score (H)", "Exposure Score (E)", "Vulnerability Score (V)"],
      datasets: [
        {
          label: "Additional Metrics",
          data: [demographics.H, demographics.E, demographics.V],
          backgroundColor: "rgba(153, 102, 255, 0.6)",
          borderColor: "rgba(153, 102, 255, 1)",
          borderWidth: 1,
        },
      ],
    };

    return (
      <motion.div
        initial={{ opacity: 0, x: 100 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -100 }}
        transition={{ duration: 0.3 }}
        style={{ padding: "20px" }}
      >
        <h2 style={{ marginBottom: "20px", color: "#000", fontSize: "24px" }}>
          Demographics for Zone {selectedDemographics.zone_id}
        </h2>
        <div style={{ marginBottom: "20px" }}>
          <h3 style={{ color: "#000", fontSize: "18px" }}>Coordinates</h3>
          <p style={{ color: "#000" }}>
            <strong>Latitude:</strong> {demographics.latitude.toFixed(5)}°N
          </p>
          <p style={{ color: "#000" }}>
            <strong>Longitude:</strong> {demographics.longitude.toFixed(5)}°E
          </p>
          <div>
          
          <p style={{ color: "#000" }}>
            <strong>Risk Category:</strong>{" "}
            <span
              style={{
                color:
                  demographics.Risk_Category === "Moderate"
                    ? "orange"
                    : demographics.Risk_Category === "Very High"
                    ? "red"
                    : demographics.Risk_Category === "High"
                    ? "darkred"
                    : "green",
                fontWeight: "bold",
              }}
            >
              {demographics.Risk_Category}
            </span>
          </p>
        </div>
        </div>
        
        <div style={{ marginBottom: "20px" }}>
          <h3 style={{ color: "#000", fontSize: "18px" }}>Land Use Distribution</h3>
          <div style={{ height: "300px", width: "100%" }}>
            <Pie
              data={landUseData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: { position: "right", labels: { color: "#000" } },
                  tooltip: {
                    callbacks: { label: (context) => `${context.label}: ${context.raw.toFixed(2)}%` },
                    backgroundColor: "rgba(0, 0, 0, 0.8)",
                    bodyColor: "#fff",
                    titleColor: "#fff",
                  },
                },
              }}
            />
          </div>
        </div>
        <div style={{ marginBottom: "20px" }}>
          <h3 style={{ color: "#000", fontSize: "18px" }}>Environmental Factors</h3>
          <div style={{ height: "300px", width: "100%" }}>
            <Bar
              data={envFactorsData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  y: { beginAtZero: true, title: { display: true, text: "Value", color: "#000" }, ticks: { color: "#000" } },
                  x: { title: { display: true, text: "Factor", color: "#000" }, ticks: { color: "#000" } },
                },
                plugins: {
                  legend: { labels: { color: "#000" } },
                  tooltip: { backgroundColor: "rgba(0, 0, 0, 0.8)", bodyColor: "#fff", titleColor: "#fff" },
                },
              }}
            />
          </div>
        </div>
        <div style={{ marginBottom: "20px" }}>
          <h3 style={{ color: "#000", fontSize: "18px" }}>Air Quality Metrics</h3>
          <div style={{ height: "300px", width: "100%" }}>
            <Bar
              data={airQualityData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  y: { beginAtZero: true, title: { display: true, text: "Value", color: "#000" }, ticks: { color: "#000" } },
                  x: { title: { display: true, text: "Metric", color: "#000" }, ticks: { color: "#000" } },
                },
                plugins: {
                  legend: { labels: { color: "#000" } },
                  tooltip: { backgroundColor: "rgba(0, 0, 0, 0.8)", bodyColor: "#fff", titleColor: "#fff" },
                },
              }}
            />
          </div>
        </div>
        <div style={{ marginBottom: "20px" }}>
          <h3 style={{ color: "#000", fontSize: "18px" }}>Risk Metrics</h3>
          <div style={{ height: "300px", width: "100%" }}>
            <Bar
              data={riskMetricsData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  y: { beginAtZero: true, title: { display: true, text: "Value", color: "#000" }, ticks: { color: "#000" } },
                  x: { title: { display: true, text: "Metric", color: "#000" }, ticks: { color: "#000" } },
                },
                plugins: {
                  legend: { labels: { color: "#000" } },
                  tooltip: { backgroundColor: "rgba(0, 0, 0, 0.8)", bodyColor: "#fff", titleColor: "#fff" },
                },
              }}
            />
          </div>
        </div>
        <div style={{ marginBottom: "20px" }}>
          <h3 style={{ color: "#000", fontSize: "18px" }}>Additional Metrics (H, E, V)</h3>
          <div style={{ height: "300px", width: "100%" }}>
            <Bar
              data={additionalMetricsData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  y: { beginAtZero: true, title: { display: true, text: "Value", color: "#000" }, ticks: { color: "#000" } },
                  x: { title: { display: true, text: "Metric", color: "#000" }, ticks: { color: "#000" } },
                },
                plugins: {
                  legend: { labels: { color: "#000" } },
                  tooltip: { backgroundColor: "rgba(0, 0, 0, 0.8)", bodyColor: "#fff", titleColor: "#fff" },
                },
              }}
            />
          </div>
        </div>
        
      </motion.div>
    );
  };

  const ReportSection = () => {
    if (!selectedZone) {
      return (
        <p style={{ color: "#666", fontStyle: "italic", textAlign: "center", marginTop: "20px" }}>
          Click on a zone to view the report.
        </p>
      );
    }

    const fullText = selectedZone.details.join("\n");

    const headings = {
      "Missing Pieces & Next Steps": "#### 6\\. Missing Pieces & Next Steps",
      "Step-by-Step Improvement Plan": "#### 7\\. Step-by-Step Improvement Plan",
    };

    const missingPieces = extractSection(
      fullText,
      headings["Missing Pieces & Next Steps"],
      headings["Step-by-Step Improvement Plan"]
    );
    const improvementPlan = extractSection(
      fullText,
      headings["Step-by-Step Improvement Plan"],
      null
    );

    let filteredText = fullText;
    const sectionsToRemove = [
      { start: missingPieces.start, end: missingPieces.end },
      { start: improvementPlan.start, end: improvementPlan.end },
    ].filter((section) => section.start !== -1 && section.end !== -1);

    sectionsToRemove.sort((a, b) => b.start - a.start);
    for (const section of sectionsToRemove) {
      filteredText = filteredText.slice(0, section.start) + filteredText.slice(section.end);
    }

    const filteredDetails = filteredText.split("\n").filter((line) => line.trim() !== "");

    const handleImplementClick = (sectionName: string, sectionContent: string) => {
      console.log("Navigating to /strategy with:", { selectedText: sectionContent, zoneId: selectedZone.zone_id });
      navigate("/strategy", {
        state: {
          selectedText: sectionContent,
          zoneId: selectedZone.zone_id,
        },
      });
    };

    return (
      <motion.div
        initial={{ opacity: 0, x: -100 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 100 }}
        transition={{ duration: 0.3 }}
        style={{ padding: "20px" }}
      >
        {filteredDetails.length > 0 ? (
          <div>
            <div
              style={{
                background: "#fff",
                padding: "15px",
                borderRadius: "8px",
                border: "1px solid #ddd",
                marginTop: "10px",
                boxShadow: "0 2px 4px rgba(0,0,0,0.05)",
              }}
            >
              {filteredDetails.map((item, index) => (
                <div key={index} style={{ margin: "5px 0", lineHeight: "1.6", color: "#333" }}>
                  {parseMarkdownToText(item)}
                </div>
              ))}
            </div>

            <div style={{ marginTop: "20px" }}>
              {missingPieces.content && (
                <div style={{ marginBottom: "20px" }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: "10px",
                    }}
                  >
                    <h3 style={{ color: "#000", fontSize: "18px", margin: 0 ,fontWeight: "bold" }}>
                      Missing Pieces & Next Steps
                    </h3>
                    <button
                      onClick={() =>
                        handleImplementClick("Missing Pieces & Next Steps", missingPieces.content!)
                      }
                      style={{
                        background: "#25D366",
                        color: "#fff",
                        border: "none",
                        padding: "8px 16px",
                        borderRadius: "5px",
                        cursor: "pointer",
                        fontSize: "14px",
                        fontWeight: "500",
                        transition: "background 0.3s",
                      }}
                      onMouseOver={(e) => (e.currentTarget.style.background = "#20B858")}
                      onMouseOut={(e) => (e.currentTarget.style.background = "#25D366")}
                    >
                      Implement
                    </button>
                  </div>
                  <div
                    style={{
                      background: "#ffffff",
                      padding: "15px",
                      borderRadius: "5px",
                      border: "1px solid #eee",
                    }}
                  >
                    {parseMarkdownToText(missingPieces.content)}
                  </div>
                </div>
              )}
              {improvementPlan.content && (
                <div style={{ marginBottom: "20px" }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: "10px",
                    }}
                  >
                    <h3 style={{ color: "#000", fontSize: "18px", margin: 0  ,fontWeight: "bold"}}>
                      Step-by-Step Improvement Plan
                    </h3>
                    <button
                      onClick={() =>
                        handleImplementClick("Step-by-Step Improvement Plan", improvementPlan.content!)
                      }
                      style={{
                        background: "#25D366",
                        color: "#fff",
                        border: "none",
                        padding: "8px 16px",
                        borderRadius: "5px",
                        cursor: "pointer",
                        fontSize: "14px",
                        fontWeight: "500",
                        transition: "background 0.3s",
                      }}
                      onMouseOver={(e) => (e.currentTarget.style.background = "#20B858")}
                      onMouseOut={(e) => (e.currentTarget.style.background = "#25D366")}
                    >
                      Implement
                    </button>
                  </div>
                  <div
                    style={{
                      background: "#ffffff",
                      padding: "15px",
                      borderRadius: "5px",
                      border: "1px solid #eee",
                    }}
                  >
                    {parseMarkdownToText(improvementPlan.content)}
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <p style={{ color: "#666", fontStyle: "italic" }}>No report details available.</p>
        )}
      </motion.div>
    );
  };

  return (
    <>
      <div
        ref={containerRef}
        style={{
          display: "flex",
          height: "100vh",
          width: "100vw",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div style={{ width: `${mapWidth}%`, height: "100%", position: "relative" }}>
          <APIProvider apiKey="AIzaSyBB0sfpf3LB1TvQnaRka9fQdjbeu3jtSnk">
            <Map
              style={{ height: "100%", width: "100%" }}
              defaultZoom={10}
              defaultCenter={{ lat: 18.5, lng: 84.0 }}
              mapId="c917f6678ebdcd6d"
            >
              {!loading &&
                zoneData.map((zone) =>
                  zone.coordinates.map((coord, idx) => {
                    const zoneIndex = zones.findIndex((z) => z.id === zone.zone_id); // Updated to use findIndex
                    const color = zoneIndex !== -1 ? zoneColors[zoneIndex] : "#808080";
                    return (
                      <AdvancedMarker
                        key={`${zone.zone_id}-${idx}`}
                        position={{ lat: coord.latitude, lng: coord.longitude }}
                        onClick={() => handleZoneClick(zone.zone_id)}
                      >
                        <div
                          style={{
                            width: "10px",
                            height: "10px",
                            backgroundColor: color,
                            borderRadius: "50%",
                            cursor: "pointer",
                            opacity: 0.6,
                          }}
                        />
                      </AdvancedMarker>
                    );
                  })
                )}
              {!loading && <Legend />}
            </Map>
          </APIProvider>
          {loading && (
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                background: "rgba(0, 0, 0, 0.5)",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                zIndex: 2000,
              }}
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                style={{
                  width: "50px",
                  height: "50px",
                  border: "5px solid #075E54",
                  borderTop: "5px solid #25D366",
                  borderRadius: "50%",
                }}
              />
            </div>
          )}
        </div>
        <div
          style={{
            width: "5px",
            background: "#ccc",
            cursor: "col-resize",
            height: "100%",
            flexShrink: 0,
          }}
          onMouseDown={handleMouseDown}
        />
        <div
          style={{
            width: `${100 - mapWidth}%`,
            height: "100%",
            background: "#f5f5f5",
            overflowY: "auto",
            color: "#333",
            boxSizing: "border-box",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              display: "flex",
              background: "#075E54",
              color: "#fff",
              padding: "10px 0",
              position: "sticky",
              top: 0,
              zIndex: 100,
              boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
            }}
          >
            <button
              onClick={() => setActiveTab("Report")}
              style={{
                flex: 1,
                padding: "10px",
                background: activeTab === "Report" ? "#128C7E" : "transparent",
                color: "#fff",
                border: "none",
                cursor: "pointer",
                fontSize: "16px",
                fontWeight: "500",
                position: "relative",
                transition: "background 0.3s",
              }}
            >
              Report
              {activeTab === "Report" && (
                <motion.div
                  layoutId="underline"
                  style={{
                    position: "absolute",
                    bottom: 0,
                    left: 0,
                    right: 0,
                    height: "3px",
                    background: "#fff",
                  }}
                />
              )}
            </button>
            <button
              onClick={() => setActiveTab("Demographics")}
              style={{
                flex: 1,
                padding: "10px",
                background: activeTab === "Demographics" ? "#128C7E" : "transparent",
                color: "#fff",
                border: "none",
                cursor: "pointer",
                fontSize: "16px",
                fontWeight: "500",
                position: "relative",
                transition: "background 0.3s",
              }}
            >
              Demographics
              {activeTab === "Demographics" && (
                <motion.div
                  layoutId="underline"
                  style={{
                    position: "absolute",
                    bottom: 0,
                    left: 0,
                    right: 0,
                    height: "3px",
                    background: "#fff",
                  }}
                />
              )}
            </button>
          </div>

          <div
            style={{
              padding: "15px",
              background: "#fff",
              borderBottom: "1px solid #ddd",
              position: "sticky",
              top: "50px",
              zIndex: 99,
            }}
          >
            {selectedZone ? (
              <div>
                <p style={{ fontWeight: "bold", marginBottom: "10px", fontSize: "16px", color: "#000" }}>
                  Zone ID: {selectedZone.zone_id}
                </p>
              </div>
            ) : (
              <p style={{ color: "#666", fontStyle: "italic" }}>Click on a zone to view details.</p>
            )}
          </div>

          <div style={{ flex: 1, overflowY: "auto" }}>
            <AnimatePresence mode="wait">
              {activeTab === "Report" ? (
                <ReportSection key="report" />
              ) : (
                <DemographicsSection key="demographics" />
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </>
  );
};

export default ZoneMap;