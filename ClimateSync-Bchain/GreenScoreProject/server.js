require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const ethers = require("ethers");
const admin = require("firebase-admin");

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });

admin.initializeApp({
  credential: admin.credential.cert(require("./firebaseConfig.json")),
  storageBucket: "your-firebase-bucket.appspot.com"
});

const db = admin.firestore();
const storage = admin.storage().bucket();

// Blockchain Setup
const provider = new ethers.JsonRpcProvider(process.env.POLYGON_RPC);
const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);
const contractABI = require("./contractABI.json");
const contractAddress = "YOUR_DEPLOYED_CONTRACT_ADDRESS";
const contract = new ethers.Contract(contractAddress, contractABI, wallet);

// Upload Travel Data API
app.post("/upload", upload.single("travelData"), async (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).send("No file uploaded.");

  const metadata = {
    mode: req.body.mode,
    timestamp: new Date().toISOString(),
  };

  await db.collection("travelLogs").doc(file.filename).set(metadata);
  res.json({ success: true, fileId: file.filename });
});

// Get Score API
app.get("/get-score/:user", async (req, res) => {
  const score = await contract.getScore(req.params.user);
  res.json({ score: score.toString() });
});

// Award Green Score API
app.post("/award-score", async (req, res) => {
  try {
    const { userAddress, points } = req.body;
    const tx = await contract.updateScore(points);
    await tx.wait();
    res.json({ success: true, txHash: tx.hash });
  } catch (error) {
    res.json({ success: false, error: error.message });
  }
});

app.listen(3000, () => console.log("Server running on port 3000"));
