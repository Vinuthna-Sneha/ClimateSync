require("dotenv").config();
const { ethers } = require("hardhat");

async function main() {
    const [deployer] = await ethers.getSigners();

    console.log(`Deploying contract with account: ${deployer.address}`);
    // console.log(`Account balance: ${(await deployer.getBalance()).toString()}`);

    // Load the contract factory
    const GreenScore = await ethers.getContractFactory("GreenScore");

    // Deploy the contract
    const contract = await GreenScore.deploy();

    // Wait for deployment to complete
    await contract.deployed();

    console.log(`✅ GreenScore deployed at: ${contract.address}`);
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
