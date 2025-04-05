require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

module.exports = {
  solidity: "0.8.28",  // Use the latest Solidity version you need
  networks: {
    sepolia: {
      url: process.env.SEPOLIA_RPC,   // RPC URL from Infura/Alchemy
      accounts: [process.env.PRIVATE_KEY]  // Private Key from .env
    },
    localhost: { // Local Hardhat network
      url: "http://127.0.0.1:8545", // Hardhat node RPC
      accounts: [process.env.L_PRIVATE_KEY]
    },
  }
};
