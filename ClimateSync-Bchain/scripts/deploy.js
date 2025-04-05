const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();

  console.log("Deploying contract with the account:", deployer.address);

  // ✅ Correct balance retrieval
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", hre.ethers.formatEther(balance), "ETH");

  // Compile and deploy the contract
  const Contract = await hre.ethers.getContractFactory("GreenScore");
  const contract = await Contract.deploy();

  await contract.waitForDeployment(); // Wait for deployment confirmation

  console.log("Contract deployed at:", contract.target); // Correct address property
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
