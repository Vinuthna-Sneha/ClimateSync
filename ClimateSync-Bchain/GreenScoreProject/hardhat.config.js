// // /** @type import('hardhat/config').HardhatUserConfig */
// // module.exports = {
// //   solidity: "0.8.28",
// // };
// require("@nomicfoundation/hardhat-toolbox");
// require("dotenv").config();

// module.exports = {
//   solidity: "0.8.0",
//   networks: {
//     polygon: {
//       url: process.env.POLYGON_RPC,
//       accounts: [process.env.PRIVATE_KEY]
//     }
//   }
// };
// Load Hardhat plugins and dotenv
require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.28", // Solidity version
  networks: {
    sepolia: {
      url: process.env.SEPOLIA_RPC, // Sepolia RPC URL from .env
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : []
    }
  }
};

